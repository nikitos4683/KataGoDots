"""Gradients of a transformer model with flex attention, compiled the way train.py compiles it, against eager FP32
SDPA. Inductor's layout optimization made the compiled flex attention backward give q/k projection gradients
8-35% off (see trainloop_helpers.configure_model_for_compile); this guards against that coming back."""

import copy

import pytest
import torch
from torch.amp import autocast

from katago.train import modelconfigs, trainloop_helpers
from katago.train.model_pytorch import Game, Model


def _compile_available():
    if not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return trainloop_helpers.ensure_msvc_for_compile()


@pytest.mark.skipif(not _compile_available(), reason="Needs CUDA, triton and (on Windows) MSVC for torch.compile")
def test_compiled_flex_attention_gradients_match_sdpa(monkeypatch):
    import torch._dynamo
    import torch._inductor.config as inductor_config

    monkeypatch.setenv("KATAGO_FLEX_ATTENTION", "1")
    # Restored after the test: configure_model_for_compile changes it for flex attention models
    monkeypatch.setattr(inductor_config, "layout_optimization", inductor_config.layout_optimization)
    torch._dynamo.reset()

    device = torch.device("cuda")
    config = copy.deepcopy(modelconfigs.get_config("b5c384h6nbttflrs-fson-silu-rsnh", dots=False))
    config["block_kind"] = config["block_kind"][:1]
    torch.manual_seed(0)
    model = Model(config, pos_len_x=19, pos_len_y=19, games=[Game.GO]).to(device)
    model.initialize()
    model.train()
    assert model.use_flex_attention

    generator = torch.Generator().manual_seed(1)
    batch_size = 16
    spatial = (torch.rand((batch_size, 22, 19, 19), generator=generator) < 0.2).float()
    spatial[:, 0] = 1.0
    spatial[::4, :, -3:, :] = 0.0  # smaller boards in the batch
    spatial[::4, :, :, -3:] = 0.0
    global_input = torch.randn((batch_size, 19), generator=generator)
    spatial, global_input = spatial.to(device), global_input.to(device)

    def gradients(run, use_flex, fp16):
        model.use_flex_attention = use_flex
        model.zero_grad(set_to_none=True)
        with autocast("cuda", dtype=torch.float16, enabled=fp16):
            outputs = run(spatial, global_input)[0]
        loss = torch.square(outputs[0].float()).mean() + torch.square(outputs[1].float()).mean()
        scale = 65536.0 if fp16 else 1.0  # like GradScaler, so that small fp16 gradients don't underflow
        (loss * scale).backward()
        return {name: p.grad.float() / scale for name, p in model.named_parameters()
                if p.grad is not None and ("q_proj" in name or "k_proj" in name)}

    # Wrapped like train.py does, while the model uses flex attention
    compiled = trainloop_helpers.wrap_model_for_training(model, device, 1, False)
    reference = gradients(model, use_flex=False, fp16=False)
    got = gradients(compiled, use_flex=True, fp16=True)
    torch._dynamo.reset()

    assert reference.keys() == got.keys() and reference
    errors = {name: float((got[name] - reference[name]).norm() / reference[name].norm()) for name in reference}
    # With the bug the mean error was ~0.23-0.34 on this model, without it ~0.004
    assert sum(errors.values()) / len(errors) < 0.08, errors
