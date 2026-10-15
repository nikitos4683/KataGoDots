# Losses and metrics of the sparse Dots model (model version 18), see docs/DotsSparseNN.md 6.
# The scalar heads use the same losses as the dense model. Policy is a softmax over the M point candidates plus
# grounding, ownership is a sigmoid per POINT token and futurepos is per point candidate.

from typing import Dict

import torch
import torch.nn.functional

from ..train.metrics_pytorch import Metrics, cross_entropy
from ..train.model_sparse_dots import count_mask


def sparse_policy_targets(batch: Dict[str, torch.Tensor]):
    """Returns the visit counts of the player and of the opponent's reply [N, M+1] each (the last is grounding)."""
    cand = batch["candPolicyTargetsNM"].to(torch.float32)
    ground = batch["groundPolicyTargetsN"].to(torch.float32)
    player = torch.cat([cand[:, :, 0], ground[:, 0:1]], dim=1)
    opponent = torch.cat([cand[:, :, 1], ground[:, 1:2]], dim=1)
    return player, opponent


def score_distribution(batch: Dict[str, torch.Tensor], length: int) -> torch.Tensor:
    """The score distribution target [N, length] (weights summing to 100) from its two bins: their scores (bin minus
    the middle bin length / 2) and their weights. Scores beyond the ends go to the end bins."""
    bins = (batch["scoreDistrScoresN"].long() + length // 2).clamp(0, length - 1)
    weights = batch["scoreDistrWeightsN"].to(torch.float32)
    return torch.zeros((bins.shape[0], length), dtype=torch.float32, device=bins.device).scatter_add_(1, bins, weights)


def sparse_cand_mask(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    num_cands = batch["candCoordsNM"].shape[1]
    return torch.cat([count_mask(batch["candCountN"], num_cands), batch["groundN"][:, 0:1].to(torch.float32)], dim=1)


class SparseDotsMetrics(Metrics):
    def __init__(self, world_size: int, raw_model):
        super().__init__(world_size, raw_model)
        self.policy_len = None  # Variable: M+1 candidates

    def loss_policy_player_samplewise(self, pred_logits, target_probs, weight, global_weight):
        assert target_probs.shape == pred_logits.shape
        return global_weight * weight * cross_entropy(pred_logits, target_probs, dim=1)

    def loss_policy_opponent_samplewise(self, pred_logits, target_probs, weight, global_weight):
        assert target_probs.shape == pred_logits.shape
        return 0.15 * global_weight * weight * cross_entropy(pred_logits, target_probs, dim=1)

    def loss_ownership_sparse_samplewise(self, pred_logits, target, weight, point_mask, global_weight):
        # Each batch element cares about its average loss over its dots, like the dense model
        loss = torch.sum(
            torch.nn.functional.binary_cross_entropy_with_logits(pred_logits, target, reduction="none") * point_mask,
            dim=1,
        ) / point_mask.sum(dim=1).clamp(min=1.0)
        return 1.5 * global_weight * weight * loss

    def loss_futurepos_sparse_samplewise(self, pred_pretanh, target, weight, cand_mask, board_area, global_weight):
        # pred_pretanh, target [N,2,M]. Normalized by sqrt of the real board area like the dense model.
        loss = torch.square(torch.tanh(pred_pretanh) - target) * cand_mask.unsqueeze(1)
        # Horizon weights 1 and 0.25 as scalars: a constant tensor would keep the compiled loss out of the
        # AOTAutograd cache
        loss = torch.sum(loss[:, 0], dim=1) + 0.25 * torch.sum(loss[:, 1], dim=1)
        loss = loss / torch.sqrt(board_area)
        return 0.25 * global_weight * weight * loss

    def metrics_dict_batchwise_single_heads_output(
        self,
        raw_model,
        model_output_postprocessed,
        batch,
        is_training,
        soft_policy_weight_scale,
        disable_optimistic_policy,
        meta_kata_only_soft_policy,
        value_loss_scale,
        td_value_loss_scales,
        variance_time_loss_scale,
        is_intermediate,
        include_model_norms=True,
    ):
        assert not meta_kata_only_soft_policy, "Sparse Dots models have no metadata encoder"
        (
            policy_logits,
            value_logits,
            td_value_logits,
            pred_td_score,
            ownership_logits,
            futurepos_pretanh,
            pred_scoremean,
            pred_scorestdev,
            pred_lead,
            pred_variance_time,
            pred_shortterm_value_error,
            pred_shortterm_score_error,
            scorebelief_logits,
        ) = model_output_postprocessed

        target_global_nc = batch["globalTargetsNC"]
        score_distribution_ns = score_distribution(batch, scorebelief_logits.shape[1])
        board_size = batch["boardSizeN"].to(torch.float32)
        board_area = board_size[:, 0] * board_size[:, 1]

        policymask = sparse_cand_mask(batch)
        num_cands = batch["candCoordsNM"].shape[1]
        point_mask = count_mask(batch["pointCountN"], batch["pointCoordsNP"].shape[1])

        visits_player, visits_opponent = sparse_policy_targets(batch)
        visits_player = visits_player * policymask
        visits_opponent = visits_opponent * policymask
        sum_player = visits_player.sum(dim=1, keepdim=True)
        sum_opponent = visits_opponent.sum(dim=1, keepdim=True)
        target_policy_player = visits_player / sum_player.clamp(min=1e-30)
        # The opponent's reply is projected onto the current candidates; nothing left means no target
        target_policy_opponent = visits_opponent / sum_opponent.clamp(min=1e-30)
        target_policy_player_soft = torch.pow((target_policy_player + 1e-7) * policymask, 0.25)
        target_policy_player_soft = target_policy_player_soft / torch.sum(target_policy_player_soft, dim=1, keepdim=True)
        target_policy_opponent_soft = torch.pow((target_policy_opponent + 1e-7) * policymask, 0.25)
        target_policy_opponent_soft = target_policy_opponent_soft / torch.sum(target_policy_opponent_soft, dim=1, keepdim=True)

        target_weight_policy_player = target_global_nc[:, 26] * (sum_player.squeeze(1) > 0).to(torch.float32)
        target_weight_policy_opponent = target_global_nc[:, 28] * (sum_opponent.squeeze(1) > 0).to(torch.float32)

        target_value = target_global_nc[:, 0:3]
        target_scoremean = target_global_nc[:, 3]
        target_td_value = torch.stack(
            (target_global_nc[:, 4:7], target_global_nc[:, 8:11], target_global_nc[:, 12:15]), dim=1
        )
        target_td_score = torch.cat(
            (target_global_nc[:, 7:8], target_global_nc[:, 11:12], target_global_nc[:, 15:16]), dim=1
        )
        target_lead = target_global_nc[:, 21]
        target_variance_time = target_global_nc[:, 22]
        global_weight = target_global_nc[:, 25]
        target_weight_ownership = target_global_nc[:, 27]
        target_weight_lead = target_global_nc[:, 29]
        target_weight_futurepos = target_global_nc[:, 33]
        target_weight_value = 1.0 - target_global_nc[:, 35]
        target_weight_td_value = 1.0 - target_global_nc[:, 24]

        target_score_distribution = score_distribution_ns / 100.0
        target_ownership = batch["pointOwnershipTargetsNP"].to(torch.float32)
        target_futurepos = batch["candFutureposTargetsNM"].to(torch.float32).permute(0, 2, 1)

        policy_opt_loss_scale = 0.930
        long_policy_opt_loss_scale = 0.100
        short_policy_opt_loss_scale = 0.200

        loss_policy_player = self.loss_policy_player_samplewise(
            policy_logits[:, 0, :], target_policy_player, target_weight_policy_player, global_weight,
        ).sum()
        loss_policy_opponent = self.loss_policy_opponent_samplewise(
            policy_logits[:, 1, :], target_policy_opponent, target_weight_policy_opponent, global_weight,
        ).sum()
        loss_policy_player_soft = self.loss_policy_player_samplewise(
            policy_logits[:, 2, :], target_policy_player_soft, target_weight_policy_player, global_weight,
        ).sum()
        loss_policy_opponent_soft = self.loss_policy_opponent_samplewise(
            policy_logits[:, 3, :], target_policy_opponent_soft, target_weight_policy_opponent, global_weight,
        ).sum()

        if disable_optimistic_policy:
            target_weight_longoptimistic_policy = target_weight_policy_player * 0.5
            target_weight_shortoptimistic_policy = target_weight_policy_player * 0.5
        else:
            # Long-term optimistic policy: weighted by the final game win squared or the score outcome
            # around 1.5 sigma more than expected
            win_squared = torch.square(target_global_nc[:, 0] + 0.5 * target_global_nc[:, 2])
            longterm_score_stdevs_excess = (target_global_nc[:, 3] - pred_scoremean.detach()) / torch.sqrt(torch.square(pred_scorestdev.detach()) + 0.25)
            target_weight_longoptimistic_policy = torch.clamp(
                win_squared + torch.sigmoid((longterm_score_stdevs_excess - 1.5) * 3.0), min=0.0, max=1.0,
            ) * target_weight_policy_player * target_weight_ownership
            # Short-term optimistic policy: weighted by the short-term value or score outcome around 1.5 sigma more than expected
            shortterm_value_actual = target_global_nc[:, 12] - target_global_nc[:, 13]
            shortterm_value_pred = torch.nn.functional.softmax(td_value_logits[:, 2, :].detach(), dim=1)
            shortterm_value_pred = shortterm_value_pred[:, 0] - shortterm_value_pred[:, 1]
            shortterm_value_stdevs_excess = (shortterm_value_actual - shortterm_value_pred) / torch.sqrt(pred_shortterm_value_error.detach() + 0.0001)
            shortterm_score_stdevs_excess = (target_global_nc[:, 15] - pred_td_score[:, 2].detach()) / torch.sqrt(pred_shortterm_score_error.detach() + 0.25)
            target_weight_shortoptimistic_policy = torch.clamp(
                torch.sigmoid((shortterm_value_stdevs_excess - 1.5) * 3.0) + torch.sigmoid((shortterm_score_stdevs_excess - 1.5) * 3.0),
                min=0.0, max=1.0,
            ) * target_weight_policy_player * target_weight_ownership

        loss_longoptimistic_policy = self.loss_policy_player_samplewise(
            policy_logits[:, 4, :], target_policy_player, target_weight_longoptimistic_policy, global_weight,
        ).sum()
        loss_shortoptimistic_policy = self.loss_policy_player_samplewise(
            policy_logits[:, 5, :], target_policy_player, target_weight_shortoptimistic_policy, global_weight,
        ).sum()
        target_weight_longoptimistic_policy_sum = (global_weight * target_weight_longoptimistic_policy).sum()
        target_weight_shortoptimistic_policy_sum = (global_weight * target_weight_shortoptimistic_policy).sum()

        loss_value = self.loss_value_samplewise(value_logits, target_value, target_weight_value, global_weight).sum()
        loss_td_value_unsummed = self.loss_td_value_samplewise(td_value_logits, target_td_value, target_weight_td_value, global_weight)
        loss_td_value1 = loss_td_value_unsummed[:, 0].sum()
        loss_td_value2 = loss_td_value_unsummed[:, 1].sum()
        loss_td_value3 = loss_td_value_unsummed[:, 2].sum()
        loss_td_score = self.loss_td_score_samplewise(pred_td_score, target_td_score, target_weight_ownership, global_weight).sum()

        loss_ownership = self.loss_ownership_sparse_samplewise(
            ownership_logits, target_ownership, target_weight_ownership, point_mask, global_weight,
        ).sum()
        loss_futurepos = self.loss_futurepos_sparse_samplewise(
            futurepos_pretanh, target_futurepos, target_weight_futurepos, policymask[:, :num_cands], board_area, global_weight,
        ).sum()
        loss_scoremean = self.loss_scoremean_samplewise(pred_scoremean, target_scoremean, target_weight_ownership, global_weight).sum()
        loss_scorebelief_cdf = self.loss_scorebelief_cdf_samplewise(scorebelief_logits, target_score_distribution, target_weight_ownership, global_weight).sum()
        loss_scorebelief_pdf = self.loss_scorebelief_pdf_samplewise(scorebelief_logits, target_score_distribution, target_weight_ownership, global_weight).sum()
        loss_scorestdev = self.loss_scorestdev_samplewise(pred_scorestdev, scorebelief_logits, global_weight).sum()
        loss_lead = self.loss_lead_samplewise(pred_lead, target_lead, target_weight_lead, global_weight).sum()
        loss_variance_time = self.loss_variance_time_samplewise(pred_variance_time, target_variance_time, target_weight_ownership, global_weight).sum()
        loss_shortterm_value_error = self.loss_shortterm_value_error_samplewise(
            pred_shortterm_value_error, td_value_logits, target_td_value, target_weight_ownership, global_weight,
        ).sum()
        loss_shortterm_score_error = self.loss_shortterm_score_error_samplewise(
            pred_shortterm_score_error, pred_td_score, target_td_score, target_weight_ownership, global_weight,
        ).sum()
        loss_zero = torch.zeros_like(loss_policy_player)

        loss_sum = (
            loss_policy_player * policy_opt_loss_scale
            + loss_policy_opponent
            + loss_policy_player_soft * soft_policy_weight_scale
            + loss_policy_opponent_soft * soft_policy_weight_scale
            + loss_longoptimistic_policy * long_policy_opt_loss_scale
            + loss_shortoptimistic_policy * short_policy_opt_loss_scale
            + loss_value * value_loss_scale
            + loss_td_value1 * td_value_loss_scales[0]
            + loss_td_value2 * td_value_loss_scales[1]
            + loss_td_value3 * td_value_loss_scales[2]
            + loss_td_score
            + loss_ownership
            + loss_futurepos
            + loss_scoremean
            + loss_scorebelief_cdf
            + loss_scorebelief_pdf
            + loss_scorestdev
            + loss_lead
            + loss_variance_time * variance_time_loss_scale
            + loss_shortterm_value_error
            + loss_shortterm_score_error
        )

        results = {
            "p0loss_sum": loss_policy_player,
            "p1loss_sum": loss_policy_opponent,
            "p0softloss_sum": loss_policy_player_soft,
            "p1softloss_sum": loss_policy_opponent_soft,
            "p0lopt_sum": loss_longoptimistic_policy,
            "p0loptw_sum": target_weight_longoptimistic_policy_sum,
            "p0sopt_sum": loss_shortoptimistic_policy,
            "p0soptw_sum": target_weight_shortoptimistic_policy_sum,
            "vloss_sum": loss_value,
            "tdvloss1_sum": loss_td_value1,
            "tdvloss2_sum": loss_td_value2,
            "tdvloss3_sum": loss_td_value3,
            "tdsloss_sum": loss_td_score,
            "oloss_sum": loss_ownership,
            "fploss_sum": loss_futurepos,
            "smloss_sum": loss_scoremean,
            "sbcdfloss_sum": loss_scorebelief_cdf,
            "sbpdfloss_sum": loss_scorebelief_pdf,
            "sdregloss_sum": loss_scorestdev,
            "leadloss_sum": loss_lead,
            "vtimeloss_sum": loss_variance_time,
            "evstloss_sum": loss_shortterm_value_error,
            "esstloss_sum": loss_shortterm_score_error,
            "qwlloss_sum": loss_zero,
            "qscloss_sum": loss_zero,
            "loss_sum": loss_sum,
            "pacc1_sum": self.accuracy1(policy_logits[:, 0, :], target_policy_player, target_weight_policy_player, global_weight),
            "vsquare_sum": self.square_value(value_logits, global_weight),
        }

        if is_intermediate:
            return results
        extra_results = {
            "wsum": global_weight.sum() * self.world_size,
            "nsamp": int(global_weight.shape[0]) * self.world_size,
            "ptentr_sum": self.target_entropy(target_policy_player, target_weight_policy_player, global_weight),
            "ptsoftentr_sum": self.target_entropy(target_policy_player_soft, target_weight_policy_player, global_weight),
        }
        if include_model_norms:
            extra_results.update(self.get_model_norm_metrics(raw_model))
        results.update(extra_results)
        return results
