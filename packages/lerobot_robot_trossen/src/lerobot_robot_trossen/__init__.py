# SmolVLA ActionQuery load hook (trossen-ai-simulation J38) -- FIRST, before any robot
# class is registered: if installing it fails, the whole plugin import fails, so
# `--robot.type` is not found and the run dies loudly instead of silently running a J38
# checkpoint without its queries (lerobot's attempt_import only logs plugin errors).
# A no-op for checkpoints without `model.action_queries` (all ACT, J18/J37 SmolVLA).
# lerobot loads with strict=False, so without it the queries are dropped with a single
# "Unexpected key(s)" warning. After load the hook verifies the queries against the file.
# scripts/record_ensemble.py installs it again for its spawn workers (they never import
# this plugin). Byte-identical to trossen-ai-simulation/scripts/smolvla_action_query.py.
from .smolvla_action_query import install_load_hook as _install_action_query_hook

_install_action_query_hook()

# ACT per-stage heads (trossen-ai-simulation J52/J53, §94.63) -- same rules as above:
# installed FIRST so a failure fails the plugin import. A no-op unless the checkpoint's
# `model.action_head.weight` is 3-D (K stage heads); then it attaches the heads and the
# per-stage decoder query offsets before load and verifies them against the file after.
# Without the hook such a checkpoint fails to load (shape mismatch), never silently.
# The stage is read from the last K state slots, i.e. LEROBOT_TASK_ONEHOT=<i>/<K>.
# Byte-identical to trossen-ai-simulation/scripts/act_stage_heads.py.
from .act_stage_heads import install_load_hook as _install_stage_heads_hook

_install_stage_heads_hook()

# Relative-EEF ACT (trossen-ai-simulation J57/J58, `action_space: rel_eef`) -- same rules:
# installed FIRST so a failure fails the plugin import. A no-op unless the checkpoint's
# multi_manifest.json says rel_eef; then every predicted chunk is converted back to joint
# angles (FK/IK, eef_kin.py) before the postprocessor, and a checkpoint whose action stats
# look relative but carries no manifest is refused at load. rel_eef_patch.py has the rest.
# eef_kin.py is byte-identical to trossen-ai-simulation/scripts/eef_kin.py.
from .rel_eef_patch import install_load_hook as _install_rel_eef_hook

_install_rel_eef_hook()

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
from .parallel_arms import apply_record_pairing_patch

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

# On by default (LEROBOT_PARALLEL_ARMS=0 turns it off): the leader arms connect and go to
# sleep at the same time as the follower arms in lerobot-record (parallel_arms.py).
apply_record_pairing_patch()
