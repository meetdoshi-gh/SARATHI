# SARATHI — Pseudocode Package

**What Can't Be Measured Must Be Remembered**  
Vision-only autonomous driving · SAC + RSAC · POMDP

---

## What This Package Contains

Pseudocode at abstraction level **4–5/10** — structured to communicate
architecture and design reasoning without being a runnable implementation.

**What IS present:**
- Class names and method signatures
- Data shapes and flow between components
- Plain-English step descriptions inside every method body
- Key design decisions with rationale (GroupNorm, episode buffer, Jacobian correction, etc.)
- Algorithm ordering and what-feeds-what across the full training pipeline

**What is NOT present:**
- Actual tensor operations (`torch.cat`, `F.relu`, etc.)
- Loss computation implementation
- Optimizer calls (`loss.backward()`, `optimizer.step()`)
- Gradient clipping code
- Exact deque / numpy stacking operations
- Any runnable library calls

You can understand the full system from these files.  
You cannot run them without significant additional engineering work.

Full implementation: `https://github.com/meetdoshi-gh/sarathi`  
Portfolio page: `meetdoshi.me/sarathi`

---

## File Guide

```
SARATHI_Pseudocode/
├── src/
│   ├── sarathi_env_wrapper.py      # MetaDrive config, reward shaping, obs pipeline, action rescaling
│   ├── sarathi_networks.py         # StereoEncoder, SAC/RSAC actor & critic class skeletons
│   ├── sarathi_replay_buffer.py    # TransitionBuffer (SAC) + EpisodeSequenceBuffer (RSAC)
│   ├── sarathi_sac_training.py     # sac_update_step, rsac_update_step, soft_update — step-by-step
│   └── sarathi_train.py            # Warmup, epoch loop, episode collection, evaluation
└── diagrams/                       # 8 standalone HTML portfolio diagrams
    ├── diagram_system_architecture.html
    ├── diagram_observation_gap.html
    ├── diagram_cnn_encoder.html
    ├── diagram_sac_vs_rsac.html
    ├── diagram_replay_buffer.html
    ├── diagram_training_loop.html
    ├── diagram_stereo_geometry.html
    └── diagram_results.html
```

---

## Key Design Decisions

| Component | Decision | Rationale |
|---|---|---|
| Normalisation | GroupNorm over BatchNorm | BatchNorm is undefined at batch size 1 (inference); GroupNorm is batch-size agnostic |
| Replay buffer | Episode-indexed deque | LSTM needs contiguous sequences — random i.i.d. transitions give wrong temporal context |
| Exploration | SAC entropy bonus, auto-tuned α | Keeps policy stochastic until the visual encoder has learned useful features |
| Temporal context | LSTM in actor + critic (RSAC) | Speed and heading rate are unobservable; LSTM accumulates evidence from the visual stream |
| Action space | Steering + engine force, bounded | Removes unreachable action corners; eliminates a major source of early-training instability |
| Reward | Sidewalk penalty + speed bonus | Discovered via training video inspection — invisible in scalar reward curves alone |

---

## On Hyperparameters

All learning rates, τ (soft update rate), batch size, buffer capacity, LSTM hidden size,
and sequence length are **intentionally omitted** from this package.

These are axes a practitioner must sweep to reproduce or extend this work.
`sarathi_sac_training.py` contains a qualitative description of each axis's
sensitivity behaviour to guide that sweep.

Best result on the primary task: **99.23% route completion**  
(deterministic evaluation, 5 episodes, SAC with tuned configuration)

---

*SARATHI · meetdoshi.me*