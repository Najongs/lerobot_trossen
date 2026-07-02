from dataclasses import dataclass

from lerobot_robot_trossen.config_bi_widowxai_follower import (
    BiWidowXAIFollowerRobotConfig,
)

from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("mobileai_robot")
@dataclass
class MobileAIRobotConfig(BiWidowXAIFollowerRobotConfig):
    # Mobile AI uses the same configuration as BiWidowXAIFollowerRobotConfig.
    # The base of the kit does not require any configuration parameters.
    enable_base_motor_torque: bool = False

    # When False, the base velocity (x.vel, theta.vel) is dropped from the
    # observation, yielding a 14-dim observation.state (arms only). Match this to
    # the policy checkpoint: True for base-in-state models, False for the
    # _nobasestate models. Base is still commanded via action_features regardless.
    include_base_in_state: bool = True
