<div align="center">

<h1>Causal Flow Q-Learning</h1>

<h3>Robust Offline Reinforcement Learning under Confounding</h3>

<a href="https://arxiv.org/abs/2602.02847">Paper</a>

</div>

## Overview

**Causal Flow Q-Learning (CFQL)** is an offline reinforcement learning algorithm
designed for pixel-based settings where the demonstrator and the learner may
observe the scene differently — for example, when parts of the image are masked,
occluded, or otherwise corrupted. Such mismatches act as unobserved
confounders: a standard offline RL agent that ignores them can be fooled into
preferring actions that look optimal in the data but transfer poorly to the
learner's own observation channel.

CFQL addresses this through two ideas built on top of
[Flow Q-Learning (FQL)](https://arxiv.org/abs/2502.02538):

1. **An action discriminator** that learns to distinguish actions drawn from
   the BC flow policy (treated as in-support) from actions produced by the
   distilled one-step policy. Its output is interpreted as a *factual weight*
   — how likely a candidate action is to be well-supported by the demonstrator
   distribution.
2. **A causal-robust Q target** for the actor update: a Q-ensemble produces
   both a mean and a per-(state, action) worst-case estimate, and the actor
   maximises a factual-weight–blend of the two. When the discriminator is
   confident the action is supported, the actor follows the mean Q; otherwise
   it falls back on the worst-case Q.

During offline-to-online fine-tuning the discriminator and worst-case bound are
switched off, recovering a standard FQL update on the growing replay buffer.

The code base builds on the reference implementations in
[OGBench](https://github.com/seohongpark/ogbench) and the original FQL
release; the baseline agents (FQL, IFQL, IQL, ReBRAC, SAC/RLPD) are kept in
[agents/](agents/) for direct comparison.

## Installation

CFQL requires Python 3.9+ and JAX. To install the full dependencies:

```bash
pip install -r requirements.txt
```

The main pinned dependencies are `jax >= 0.4.26`, `ogbench == 1.1.0`, and
`gymnasium == 0.29.1`.

> [!NOTE]
> To use D4RL environments, you also need to set up MuJoCo 2.1.0. For V-D4RL
> (Cheetah-run) experiments, install `torchrl` and point `--dataset_root` at
> a directory where the V-D4RL datasets can be downloaded.

## Usage

The CFQL implementations live in [agents/en_cfd_fql.py](agents/en_cfd_fql.py)
(`robust_en_fql`, the ensemble variant used in the paper) and
[agents/cfd_fql.py](agents/cfd_fql.py) (`robust_fql`, the simpler
two-critic variant). All experiments are launched through a single entry
point, [main.py](main.py).

### Offline RL with CFQL

```bash
# CFQL on OGBench visual-cube-double with the left half of the frame masked.
python main.py \
    --env_name=visual-cube-double-play-singletask-task1-v0 \
    --agent=agents/en_cfd_fql.py \
    --offline_steps=500000 \
    --agent.alpha=300 \
    --agent.disc_coef=15.0 \
    --agent.encoder=impala_small \
    --p_aug=0.5 --frame_stack=3 \
    --confound_mode=1
```

### Offline-to-online fine-tuning

```bash
# Pretrain offline, then continue with online interaction.
python main.py \
    --env_name=visual-cube-single-play-singletask-task1-v0 \
    --agent=agents/en_cfd_fql.py \
    --offline_steps=500000 --online_steps=500000 \
    --agent.alpha=100 \
    --agent.disc_coef=15.0 \
    --agent.encoder=impala_small \
    --p_aug=0.5 --frame_stack=3
```

### Baselines

Swap in any of the bundled baselines through `--agent`:

```bash
# FQL baseline (no causal correction).
python main.py --env_name=visual-cube-double-play-singletask-task1-v0 \
    --agent=agents/fql.py --offline_steps=500000 --agent.alpha=300 \
    --agent.encoder=impala_small --p_aug=0.5 --frame_stack=3

# IQL / ReBRAC / IFQL baselines work the same way.
python main.py --env_name=... --agent=agents/iql.py    ...
python main.py --env_name=... --agent=agents/rebrac.py ...
python main.py --env_name=... --agent=agents/ifql.py   ...
```

### Confounder modes

`--confound_mode` injects an observation-space confounder by zeroing a fixed
region of every (64×64) pixel observation:

| Value | Region masked            |
| ----- | ------------------------ |
| `0`   | No masking (default)     |
| `1`   | Left half                |
| `2`   | Lower half               |
| `3`   | Lower-left quadrant      |

These mismatches are the offline-RL confounders studied in the paper.

### Key hyperparameters

| Flag                                         | Meaning                                                                                |
| -------------------------------------------- | -------------------------------------------------------------------------------------- |
| `--agent.alpha`                              | BC-distillation coefficient; needs to be tuned per environment.                        |
| `--agent.disc_coef`                          | Weight of the action-discriminator loss. Decayed during training if `disc_decay=True`. |
| `--agent.num_ensembles` (`en_cfd_fql` only)  | Size of the Q-ensemble used to build the worst-case target.                            |
| `--agent.normalize_q_loss`                   | Scale-invariant Q loss; recommended for new tasks.                                     |
| `--agent.encoder`, `--p_aug`, `--frame_stack`| Visual-input settings for pixel observations.                                          |

## Code layout

```
agents/        # Agent implementations (CFQL + baselines)
envs/          # Environment wrappers (OGBench, D4RL, V-D4RL/DMC)
utils/         # Networks, dataset/replay buffer, evaluation, logging
main.py        # Single training entry point
requirements.txt
```

## Citation

If you find this work useful in your research, please cite:

```bibtex
@article{li2026cfql,
  title   = {Causal Flow Q-Learning for Robust Offline Reinforcement Learning},
  author  = {Li, Mingxuan and Zhang, Junzhe and Bareinboim, Elias},
  journal = {arXiv preprint arXiv:2602.02847},
  year    = {2026}
}
```

## Acknowledgments

This code base is built on top of [FQL](https://github.com/seohongpark/fql)
and [OGBench](https://github.com/seohongpark/ogbench)'s reference
implementations.
