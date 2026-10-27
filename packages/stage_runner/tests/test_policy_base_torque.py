"""`release_base_torque` on a policy stage: the policy's base command moves nothing.

Drives the real cli.main against tests/configs/smoke_mock.yaml with the flag set
on its first stage, the mock robot given a fake base that tracks its torque.

What is asserted, and why each matters on the robot:

* every command the flagged stage sends reaches a base whose torque is OFF, and
  the next stage's commands reach it ON again -- stage 6's pi0 starts driving
  to the fridge before the stage ends (the reason for the flag);
* each switch comes after a zero velocity command, so the base does not
  re-energise holding a stale velocity;
* a switch that refuses to go off ends the stage before the policy runs, and
  one that refuses to come back on ends the trial before the next stage;
* a robot connected without base torque is left alone;
* preflight refuses the flag on a non-policy stage, and on a powered base it
  cannot switch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from stage_runner import cli, config, preflight
from stage_runner.config import StageConfig
from test_leader_reset import _assert_log_is_closed, _FakeBase, _of, _run

SMOKE = Path(__file__).parent / "configs" / "smoke_mock.yaml"


class _TrackedBase(_FakeBase):
    def __init__(self, release_ok: bool = True, restore_ok: bool = True) -> None:
        super().__init__(restore_ok=restore_ok)
        self.release_ok = release_ok
        self.torque_on = True

    def enable_motor_torque(self, enable: bool) -> tuple[bool, str]:
        self.calls.append(("enable_motor_torque", enable))
        if not enable and not self.release_ok:
            return False, "mock release refusal"
        if enable and not self.restore_ok:
            return False, "mock refusal"
        self.torque_on = enable
        return True, ""


def _flagged_config(tmp_path: Path, **flags: bool) -> Path:
    data = yaml.safe_load(SMOKE.read_text())
    for stage in data["stages"]:
        if flags.get(stage["id"]):
            stage["release_base_torque"] = True
    path = tmp_path / "flagged.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def _with_tracked_base(
    holder: dict[str, Any], *, torque: bool = True, **base_options: bool
) -> Any:
    def instrument(robot: Any, captured: dict[str, Any]) -> None:
        base = _TrackedBase(**base_options)
        robot.base = base
        robot.config.enable_base_motor_torque = torque
        holder["base"] = base
        holder["sent"] = sent = []
        send_action = robot.send_action

        def tracked_send_action(action: Any) -> Any:
            sent.append(base.torque_on)
            return send_action(action)

        robot.send_action = tracked_send_action

    return instrument


def _stage_frames(result: dict[str, Any]) -> dict[str, int]:
    return {end["stage_id"]: end["frames"] for end in _of(result, "stage_end")}


def test_the_flagged_stage_runs_with_base_torque_off_and_the_next_with_it_on(
    tmp_path: Path,
) -> None:
    holder: dict[str, Any] = {}
    result = _run(
        tmp_path,
        config_path=_flagged_config(tmp_path, s4=True),
        instrument=_with_tracked_base(holder),
    )
    assert result["exit_code"] == cli.EXIT_OK, result["raised"]
    _assert_log_is_closed(result)
    base = holder["base"]
    torque = [value for name, value in base.calls if name == "enable_motor_torque"]
    assert torque == [False, True]
    for index, (name, _) in enumerate(base.calls):
        if name == "enable_motor_torque":
            assert base.calls[index - 1] == ("set_cmd_vel", (0.0, 0.0))
    # s4's frames were sent with the torque off, s5's with it on.
    frames = _stage_frames(result)
    sent = holder["sent"]
    assert frames["s4"] > 0 and frames["s5"] > 0
    assert sent[: frames["s4"]] == [False] * frames["s4"]
    assert sent[-frames["s5"] :] == [True] * frames["s5"]
    (line,) = _of(result, "stage_base_torque")
    assert line["stage_id"] == "s4"
    assert line["base_torque"] == "released"
    assert line["base_torque_error"] is None


def test_a_stage_without_the_flag_never_touches_the_torque(tmp_path: Path) -> None:
    holder: dict[str, Any] = {}
    result = _run(
        tmp_path,
        config_path=_flagged_config(tmp_path),
        instrument=_with_tracked_base(holder),
    )
    assert result["exit_code"] == cli.EXIT_OK
    assert not any(name == "enable_motor_torque" for name, _ in holder["base"].calls)
    assert _of(result, "stage_base_torque") == []


def test_a_base_that_will_not_release_ends_the_stage_before_the_policy_runs(
    tmp_path: Path,
) -> None:
    holder: dict[str, Any] = {}
    result = _run(
        tmp_path,
        config_path=_flagged_config(tmp_path, s4=True),
        instrument=_with_tracked_base(holder, release_ok=False),
    )
    assert result["exit_code"] == cli.EXIT_ABORTED
    _assert_log_is_closed(result)
    assert _stage_frames(result) == {"s4": 0}
    (line,) = _of(result, "stage_base_torque")
    assert line["base_torque"] == "release_failed"
    assert [start["stage_id"] for start in _of(result, "stage_start")] == ["s4"]


def test_a_base_whose_torque_cannot_be_restored_ends_the_trial(
    tmp_path: Path,
) -> None:
    holder: dict[str, Any] = {}
    result = _run(
        tmp_path,
        config_path=_flagged_config(tmp_path, s4=True),
        instrument=_with_tracked_base(holder, restore_ok=False),
    )
    assert result["exit_code"] == cli.EXIT_ABORTED
    _assert_log_is_closed(result)
    (line,) = _of(result, "stage_base_torque")
    assert line["base_torque"] == "restore_failed"
    assert "mock refusal" in line["base_torque_error"]
    assert [start["stage_id"] for start in _of(result, "stage_start")] == ["s4"]


def test_a_watcher_that_fails_to_start_still_gets_the_torque_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The watcher starts after the torque goes off; if starting it raises, the
    # restore must still run, or the next base stage drives an unpowered base.
    from stage_runner import completion, executors

    class _BrokenWatcher:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def start(self) -> None:
            raise RuntimeError("mock watcher thread failure")

        def stop(self) -> None:
            pass

        def offer(self, observation: Any) -> None:
            pass

    monkeypatch.setattr(
        executors,
        "WATCHED_TERMINATORS",
        (*executors.WATCHED_TERMINATORS, config.TERMINATOR_TIMEOUT),
    )
    monkeypatch.setattr(completion, "CompletionWatcher", _BrokenWatcher)
    monkeypatch.setattr(completion, "resolve_settings", lambda *args: None)
    holder: dict[str, Any] = {}
    result = _run(
        tmp_path,
        config_path=_flagged_config(tmp_path, s4=True),
        instrument=_with_tracked_base(holder),
    )
    assert result["exit_code"] != cli.EXIT_OK
    torque = [v for name, v in holder["base"].calls if name == "enable_motor_torque"]
    assert torque == [False, True]


def test_a_robot_connected_without_base_torque_is_left_alone(tmp_path: Path) -> None:
    holder: dict[str, Any] = {}
    result = _run(
        tmp_path,
        config_path=_flagged_config(tmp_path, s4=True),
        instrument=_with_tracked_base(holder, torque=False),
    )
    assert result["exit_code"] == cli.EXIT_OK
    assert not any(name == "enable_motor_torque" for name, _ in holder["base"].calls)
    (line,) = _of(result, "stage_base_torque")
    assert line["base_torque"] == "not_released"


# --------------------------------------------------------------------------
# Preflight.


RESET_CONFIG = Path(__file__).parent / "configs" / "smoke_mock_leader_reset.yaml"


def test_preflight_refuses_the_flag_on_a_non_policy_stage() -> None:
    parsed = config.parse_config(["--config_path", str(RESET_CONFIG)])
    assert parsed.stages[0].executor == config.EXECUTOR_LEADER_RESET
    parsed.stages[0].release_base_torque = True
    with pytest.raises(preflight.PreflightError, match="release_base_torque"):
        preflight.check_stage_definitions(parsed)


def _config_with(stage: StageConfig) -> Any:
    parsed = config.parse_config(["--config_path", str(SMOKE)])
    parsed.stages = [stage]
    return parsed


class _Robot:
    name = "fake"

    def __init__(self, torque: bool, base: Any) -> None:
        self.config = type("C", (), {"enable_base_motor_torque": torque})()
        self.base = base


def _flagged() -> config.StageRunnerConfig:
    return _config_with(
        StageConfig(id="s6", policy_path="mock://hold", release_base_torque=True)
    )


def test_preflight_refuses_a_powered_base_it_cannot_switch() -> None:
    with pytest.raises(preflight.PreflightError, match="powered base"):
        preflight.check_released_base_torque(_flagged(), _Robot(True, object()))


def test_preflight_passes_a_switchable_or_unpowered_base() -> None:
    preflight.check_released_base_torque(_flagged(), _Robot(True, _TrackedBase()))
    preflight.check_released_base_torque(_flagged(), _Robot(False, object()))
