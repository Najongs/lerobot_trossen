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

    # When False (the default), the base velocity (x.vel, theta.vel) is dropped
    # from the observation, yielding a 14-dim observation.state (arms only). Match
    # this to the policy checkpoint: False for the _14D datasets and the models
    # trained on them, True for the older base-in-state ones. Base is still
    # commanded via action_features regardless.
    include_base_in_state: bool = False

    # When False, connect() no longer refuses to start with the base in emergency
    # stop. Only that hard error is disabled; the warning get_observation() logs
    # on each base state change stays on either way. Set False for runs that
    # deliberately keep the base immobilized; eval reaches this check too, since
    # it goes through the same robot.connect() path.
    estop_check: bool = True
