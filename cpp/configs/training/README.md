# Self-play Training Configs

This directory contains configuration files for self-play data generation and model gating.

See `selfplay1.cfg` for inline comments describing search and game initialization parameters.

---

## Dots Game Configurations

* `selfplay1_dots.cfg`: Default selfplay data generation configuration for the game of Dots. Configured for a standard 39x32 rectangular board, `bbs` or `notago` rulesets, and appropriate visit counts and batching for Dots.
* `gatekeeper1_dots.cfg`: Default gatekeeper configuration for evaluating candidate Dots neural nets against the current accepted best model.
* `match_dots.cfg` (in `cpp/configs/`): Configuration for running automated matches between different Dots models or search parameters.

---

## Go Configurations (Reference)

* `selfplay1.cfg`, `selfplay2.cfg`, `selfplay8a.cfg`: Go selfplay configs for early training on 1, 2, or 8 GPUs.
* `selfplay1_maxsize9.cfg`: Config restricted to boards of size 9x9 and smaller.
* `selfplay8b.cfg`, `selfplay8b20.cfg`: Go configs for mid-stage training with larger visit counts.
* `selfplay8midrun.cfg`, `selfplay8mainb18.cfg`: Advanced Go training configs used in later stages of public distributed training.
* `gatekeeper*.cfg`: Go gatekeeper configurations.
