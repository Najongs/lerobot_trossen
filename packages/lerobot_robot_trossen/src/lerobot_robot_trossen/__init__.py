from .bi_widowxai_follower import BiWidowXAIFollowerRobot
from .config_bi_widowxai_follower import BiWidowXAIFollowerRobotConfig
from .config_widowxai_follower import WidowXAIFollowerConfig
from .widowxai_follower import WidowXAIFollower
from .mobileai import MobileAIRobot
from .config_mobileai import MobileAIRobotConfig
from .fast_obs_patch import apply_fast_observation_patch
from .loop_rate_log import apply_loop_rate_logging_patch
from .chunk_execution_patch import apply_chunk_execution_patch
from .task_onehot_patch import apply_task_onehot_patch
from .smolvla_action_query import install_load_hook as _install_action_query_hook

# On by default; set LEROBOT_FAST_OBS=0 to opt out. lerobot imports this package
# on startup (lerobot.utils.import_utils discovers "lerobot_robot_*"), which is
# early enough to rebind the inference path before any policy runs.
apply_fast_observation_patch()

# On by default; set LEROBOT_LOOP_HZ_LOG=0 to opt out (which also restores
# upstream's per-frame fps warning). Same import timing as above, which is early
# enough to wrap record_loop before record() resolves it.
apply_loop_rate_logging_patch()

# Off by default; installed only when a base latency switch is set
# (LEROBOT_CHUNK_PREFETCH_TICKS, LEROBOT_BASE_LEAD_TICKS, LEROBOT_REPLAY_DATASET,
# LEROBOT_CHUNK_EXECUTION_LOG and their companions). Must run after the loop
# rate patch: it wraps that wrapper rather than replacing it.
apply_chunk_execution_patch()

# Off by default; installed only when LEROBOT_TASK_ONEHOT=<stage>/<K> is set.
# Widens the policy state to [arms, 0, 0, one_hot] for multi-stage ACT
# checkpoints trained with a stage one-hot (trossen-ai-simulation train_multi.py).
apply_task_onehot_patch()

# Always installed; a no-op unless a checkpoint's model.safetensors contains
# `model.action_queries` (SmolVLA + ActionQuery, trossen-ai-simulation J38).
# lerobot loads policies with strict=False, so without this hook those query
# weights would be dropped with a single "Unexpected key(s)" warning and the
# policy would run without them. A J38 run must print
# "[action_query] model.action_queries (64) ..." at load -- if it doesn't, stop.
# The module is byte-identical to trossen-ai-simulation/scripts/smolvla_action_query.py.
# Not wrapped in try/except on purpose (codex 10/12): a failed install must be loud.
# The hook also verifies the loaded queries against the file; scripts/record_ensemble.py
# installs it again for its spawn workers, which never import this plugin.
_install_action_query_hook()
