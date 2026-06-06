"""
sarathi_replay_buffer.py  —  SARATHI Pseudocode  (abstraction level 4-5/10)
═══════════════════════════════════════════════════════════════════════════════
PURPOSE: Two replay buffer designs — one for SAC, one for RSAC.
WHAT THIS FILE SHOWS: structural design, sampling logic, capacity management.
WHAT IS MISSING: actual deque/list operations, numpy stacking, index arithmetic.
"""


# ── Standard Transition Buffer (SAC) ─────────────────────────────────────────

class TransitionBuffer:
    """
    Flat circular buffer of independent (s, a, r, s', done) tuples.
    Used by SAC — random i.i.d. sampling is correct for feedforward policies.

    Capacity: fixed max number of transitions (FIFO eviction when full).
    Sampling: random.sample of size B — no ordering requirement.
    """

    def __init__(self, max_transitions):
        # initialise a fixed-capacity deque
        ...

    def push(self, obs_left, obs_right, nav, action, reward,
             next_obs_left, next_obs_right, next_nav, done):
        # append the transition tuple to the deque
        # deque automatically evicts oldest when full
        ...

    def sample(self, batch_size):
        """
        Draw batch_size transitions uniformly at random.
        Returns a dict of stacked numpy arrays:
          left, right, nav, action, reward, next_left, next_right, next_nav, done
          shapes: (B, 1, H, W), (B, 1, H, W), (B, 10), (B, 2), (B, 1), ...
        """
        ...

    def __len__(self):
        # return current number of stored transitions
        ...


# ── Episode-Indexed Sequence Buffer (RSAC) ────────────────────────────────────

class EpisodeSequenceBuffer:
    """
    Stores full episodes; samples contiguous T-step fragments for LSTM training.

    WHY A DIFFERENT BUFFER IS NEEDED
    ─────────────────────────────────
    An LSTM cannot build valid temporal context from randomly shuffled transitions.
    If you zero-initialise the hidden state at a random mid-episode timestep,
    the LSTM receives incorrect context for every subsequent step in the fragment.
    Contiguous sequences are required so the LSTM's context is causally correct.

    Standard RL libraries (at the time) provided no recurrent SAC implementation,
    so this buffer was designed from scratch.

    DESIGN
    ──────
    Storage unit:  full episode  (a list of per-step transition tuples)
    Container:     a deque of episodes
    Capacity:      tracked as total transitions across all episodes
    Eviction:      when total transitions exceed max, pop the oldest WHOLE episode
                   (never evict individual transitions — breaks episode integrity)

    SAMPLING
    ─────────
    1. Pick a random episode from the deque
    2. Pick a random start index within that episode
    3. Extract T consecutive transitions starting at that index
    4. LSTM is zero-initialised at the fragment start (approximation — valid in practice)
    """

    def __init__(self, max_transitions):
        # initialise deque of episodes and a counter for total transitions
        ...

    def start_episode(self):
        # open a new empty list for the current episode
        ...

    def push_step(self, obs_left, obs_right, nav, action, reward,
                  next_obs_left, next_obs_right, next_nav, done):
        # append this step's tuple to the current open episode list
        ...

    def end_episode(self):
        # 1. close the current episode → append to the deque
        # 2. add episode length to total_transitions counter
        # 3. while total_transitions > max_transitions:
        #       pop the oldest episode from deque front
        #       subtract its length from the counter
        ...

    def sample_sequences(self, batch_size, seq_len):
        """
        Returns batch_size contiguous fragments, each of length seq_len.

        For each element in the batch:
          - filter to episodes with length >= seq_len
          - pick a random valid episode
          - pick a random start index in [0, episode_len - seq_len]
          - slice seq_len consecutive transitions

        Returns a dict of stacked numpy arrays — all shaped (B, T, ...):
          left (B,T,1,H,W), right (B,T,1,H,W), nav (B,T,10),
          action (B,T,2), reward (B,T,1), ..., done (B,T,1)
        """
        ...

    def __len__(self):
        # return total_transitions counter
        ...
