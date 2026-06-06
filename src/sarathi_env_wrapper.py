"""
sarathi_env_wrapper.py  —  SARATHI Pseudocode  (abstraction level 4-5/10)
═══════════════════════════════════════════════════════════════════════════
PURPOSE: Wrap MetaDrive to expose the SARATHI observation space and reward.
WHAT THIS FILE SHOWS: env config structure, reward design choices, obs pipeline.
WHAT IS MISSING: actual MetaDrive import calls, exact gym wrapper boilerplate.
"""


# ── Environment configuration ─────────────────────────────────────────────────

def build_env_config():
    """
    Returns a config dict for MetaDrive.

    Key decisions reflected here:
    - Single 3-block map ("SSS"), zero traffic — isolates the perception problem
    - decision_repeat=5 physics steps per agent step → ~10 Hz control frequency
    - Two cameras: left at (-0.2, 0.225, 0.55), right at (+0.2, 0.225, 0.55)
    - Both cameras: 120×160, grayscale, 65° FOV
    - norm_pixel=True → pixel values already in [0, 1] from the simulator
    """
    config = {
        "map": "SSS",
        "traffic_density": 0.0,
        "decision_repeat": 5,
        "image_observation": True,
        "norm_pixel": True,
        # ... vehicle_config with left and right camera specs
    }
    return config


# ── Observation preprocessing ─────────────────────────────────────────────────

def preprocess_observation(raw_obs):
    """
    Converts raw MetaDrive observation dict into the three tensors the policy uses.

    Steps:
      1. Extract left RGB image  (H, W, 3)  from raw_obs["image"]
      2. Extract right RGB image (H, W, 3)  from raw_obs["image_right"]
      3. Convert both to grayscale (H, W) via luminosity weights
         → Grayscale chosen over RGB to reduce memory 3×; sufficient for depth cues
      4. Add channel dimension → (1, H, W)  as float32
      5. Extract navigation vector (10,)  from raw_obs["navigation"]

    Returns: left_gray (1,120,160), right_gray (1,120,160), nav_vec (10,)
    """
    ...


# ── Reward shaping ────────────────────────────────────────────────────────────

def compute_reward(info, current_speed):
    """
    Replaces MetaDrive's default reward with a shaped version.

    Reward components (additive):
      +100   destination reached   (terminal, one-time)
        -5   crash or out-of-road  (terminal, one-time)
       -10   sidewalk contact       (dense, each step)
      +Δd    forward progress       (dense, MetaDrive's built-in step_reward)
      +0.5 × (speed / max_speed)   speed bonus, capped at 1.0

    Design note: the sidewalk penalty was discovered necessary by watching
    training videos — the agent drifted to the road edge without it, a failure
    mode invisible in the scalar reward curve.

    Returns: float scalar reward for this timestep
    """
    reward = 0.0
    # add each component based on info flags and current_speed
    ...
    return reward


# ── Action rescaling ──────────────────────────────────────────────────────────

def rescale_action(tanh_action):
    """
    Maps the policy's tanh output ∈ [-1, 1]² to physical actuator bounds.

    Steering: tanh[0] → [-15°, +15°]
    Force:    tanh[1] → [0.1,   0.125]

    Why these bounds: removes physically unreachable corners of the action space
    that would otherwise waste policy capacity and destabilize early training.

    Returns: numpy array [steering_deg, engine_force]
    """
    ...
