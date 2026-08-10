from .bi_widowxai_follower import BiWidowXAIFollowerRobot
from .config_bi_widowxai_follower import BiWidowXAIFollowerRobotConfig
from .config_widowxai_follower import WidowXAIFollowerConfig
from .widowxai_follower import WidowXAIFollower
from .mobileai import MobileAIRobot
from .config_mobileai import MobileAIRobotConfig
from .fast_obs_patch import apply_fast_observation_patch

# On by default; set LEROBOT_FAST_OBS=0 to opt out. lerobot imports this package
# on startup (lerobot.utils.import_utils discovers "lerobot_robot_*"), which is
# early enough to rebind the inference path before any policy runs.
apply_fast_observation_patch()
