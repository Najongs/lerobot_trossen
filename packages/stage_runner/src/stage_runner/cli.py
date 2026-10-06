"""Composition root: parse, preflight, construct, run one trial, tear down.

``python -m stage_runner --config_path configs/chain_task45.yaml`` lands in
:func:`main`; ``dataset.repo_id`` is set in that YAML and changes every trial. The order of the
calls below is load-bearing and each is numbered against the design's call
sequence; nothing before step 12 (``robot.connect()``) can move the robot.
"""

import logging
from collections.abc import Sequence

from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.control_utils import is_headless
from lerobot.utils.utils import init_logging
from lerobot.utils.visualization_utils import init_rerun

from stage_runner import (
    __version__,
    config,
    policies,
    preflight,
    record_adapter,
    runner,
)
from stage_runner.chain_params import ChainParamsError, load_chain_params
from stage_runner.context import ChainRuntime, StageContext
from stage_runner.events import EventLog
from stage_runner.preflight import PreflightError
from stage_runner.results import ABORTING_TERMINATORS, TrialOutcome

# ACCEPTED RISK 1 (2026-09-08, decided) -- joint UNITS are never asserted, here
# or anywhere else in this package. preflight.check_stage_dimensions compares
# dimensions only. The dataset's meta/info.json carries no unit annotation, so
# there is nothing to compare a checkpoint against: radians are assumed
# throughout, and config.source.yaml is the artifact that carries that
# assumption forward with the run rather than leaving it in someone's head.
# Re-check when an asset that DOES annotate units (a different robot, a
# different acquisition) is mixed into the same chain.
#
# ACCEPTED RISK 2 (2026-09-08, decided) -- a hard kill leaves the base moving.
# SIGKILL or a power cut skips the finally block in main(), and the base holds
# its last velocity command indefinitely because it is velocity-controlled (the
# arms are position-controlled, so "last command held" already means stopped
# there). The prescription that would cover it is a command timeout in the base
# firmware, which is not a layer this runner can add. P1's safety net is a
# person standing beside the robot with the e-stop, and in P1 that person is the
# operator running this command. Re-check when trials run unattended: repeated
# trials with nobody present, or a remote demo.

logger = logging.getLogger(__name__)

EXIT_OK: int = 0
EXIT_ABORTED: int = 1
EXIT_PREFLIGHT: int = 2
# The base was left driving. Every stage may have run and trial_end may say
# reason="completed" -- what failed is the hardware half, at a boundary where BOTH
# stop paths failed, and until this code existed such a run exited 0 and was
# indistinguishable from a clean one to anything that does not parse the JSONL.
# It is a different instruction from 1 and 2: those say "this trial is not a
# measurement", this says "stop the batch and go look at the robot".
EXIT_BASE_STOP_FAILED: int = 3
# The chain broke: a stage ended with a terminator its required_terminator does
# not allow -- a policy stage that timed out instead of completing, or a boundary
# reset that never arrived. Its own code, separate from 1: code 1 means a HUMAN
# stopped the run and the trial is not a measurement, this means the MODEL did
# not finish the chain, which IS the measurement the run exists to produce. A
# batch script retries neither, but a chain report counts them in different
# columns.
EXIT_CHAIN_FAILED: int = 4


def main(argv: Sequence[str] | None = None) -> int:
    """Run one trial and return a process exit code.

    0 on a clean run, 1 when a stage terminated in an aborting way (operator
    abort or a stage error), 2 when preflight refused to start, 3 when a stage
    boundary failed to stop the base. 3 outranks 1: a trial that both aborted and
    left the base driving is the base's problem first.

    A trial that RAISES returns no code at all -- the exception propagates so the
    operator gets the traceback, and the process exits with the interpreter's own
    status. A failed base stop on that path is therefore reported by the ERROR
    line transitions.stop_base already logs and by the last transition event, not
    by code 3. Widening 3 to cover it would mean swallowing the traceback here.
    """
    # Step 2. FIRST statement, before anything parses a config:
    # register_third_party_plugins() imports every installed lerobot_robot_* /
    # lerobot_camera_* / lerobot_teleoperator_* / lerobot_policy_* distribution,
    # and that is what puts `mobileai_robot` into RobotConfig's draccus choice
    # registry. Parse first and step 3 dies in type resolution instead.
    # (`stage_runner_mock_robot` does NOT come from here and no longer rides on
    # importing this package either: __init__ is lazy so stage_runner.aggregate
    # stays stdlib-only, and config.parse_config imports mock_robot itself.)
    record_adapter.register_plugins()

    # lerobot's record() calls this and we do not go through it, so without this
    # line the root logger stays at WARNING and every progress line below --
    # including which run directory the operator has to look in -- is invisible.
    init_logging()

    # Steps 3-4. draccus owns --config_path and the --dataset.repo_id override;
    # we add no argparse of our own. source_config_path re-scans argv only
    # because draccus does not expose the file it loaded, and the verbatim
    # snapshot in step 9 needs that path.
    cfg = config.parse_config(argv)
    source_path = config.source_config_path(argv)

    # Steps 5-8. Everything that must kill the process before the robot is
    # energised and before the first checkpoint byte is downloaded. A
    # PreflightError is an operator error, so it gets one actionable line and
    # exit code 2 -- a traceback here would only bury the sentence that says
    # what to fix.
    chain_params = None
    chain_policy_config = None
    extra_action_names: tuple[str, ...] = ()
    try:
        lerobot_version = preflight.check_lerobot_version()
        preflight.check_config_version(cfg)

        # Step 5b (chain only). The stage list is EXPANDED from the `chain:`
        # block before anything validates it, because `stages:` is what every
        # later step reads -- preflight, the dataset features, the bundles, the
        # runner. Writing the 22 stages by hand would be 22 chances to put the
        # wrong one-hot index on the wrong stage, which is SILENT: the model
        # runs, it just runs the wrong stage's conditioning, and offline §89
        # showed the one-hot is weak enough that nothing downstream would look
        # obviously wrong.
        #
        # The expansion lands in config.resolved.yaml (step 9), so the run
        # directory records every stage that actually ran, not just the block it
        # came from.
        if cfg.chain.enabled:
            if cfg.stages:
                raise PreflightError(
                    "`chain.enabled: true` expands the stage list itself, but the "
                    f"YAML also lists {len(cfg.stages)} stage(s). Remove "
                    "`stages:` -- keeping both means the file shows one chain and "
                    "the run executes another."
                )
            try:
                chain_params = load_chain_params(
                    cfg.chain.params_path, expected_fps=cfg.dataset.fps
                )
            except ChainParamsError as error:
                # One actionable line and exit 2, like every other operator
                # error. The loader already collected every problem in the file.
                raise PreflightError(str(error)) from error
            cfg.stages = config.expand_chain(cfg, chain_params)

        preflight.check_stage_definitions(cfg)

        # Step 6. Constructed, NOT connected: observation_features and
        # action_features are config-derived and readable already, which is what
        # lets the dimension check run before anything is powered.
        robot = record_adapter.make_robot(cfg.robot)

        # Step 7. Fetches config.json only. It has to happen before
        # make_policy, which overwrites output_features from ds_meta and
        # destroys the checkpoint's own recorded dimensions.
        if cfg.chain.enabled and cfg.chain.model.policy_path.startswith(
            policies.MOCK_POLICY_SCHEME
        ):
            # The hardware-free chain: no config.json exists anywhere, so the
            # dimension gate has nothing to compare and the action width cannot
            # be read from a checkpoint. `chain.model.has_progress` therefore
            # becomes REQUIRED rather than optional -- defaulting it would make a
            # mock chain silently 16-D, and the progress mock would degrade to
            # `hold` with nothing in the log pointing at the config.
            policy_configs = {}
            if cfg.chain.model.has_progress is None:
                raise PreflightError(
                    f"`chain.model.policy_path: {cfg.chain.model.policy_path}` is a "
                    "mock, so there is no checkpoint to read the action width "
                    "from. Set `chain.model.has_progress` explicitly (true for a "
                    "17-D chain, false for a 16-D one)."
                )
            preflight.check_chain_definitions(cfg, chain_params)
        elif cfg.chain.enabled:
            # ONE config object for the one checkpoint, shared by every policy
            # stage. load_policy_configs deliberately gives two stages sharing a
            # path two OBJECTS, because make_policy mutates in place -- but that
            # guards against two DIFFERENT checkpoints. Here there is one, and
            # make_policy is called on it exactly once (in load_chain_bundles,
            # after this gate), so sharing is correct and saves eleven identical
            # config.json fetches.
            chain_policy_config = policies.load_policy_config(
                cfg.chain.model.policy_path
            )
            policy_configs = {
                stage.id: chain_policy_config
                for stage in cfg.stages
                if stage.policy_path
            }
            preflight.check_chain_definitions(cfg, chain_params)
        else:
            policy_configs = policies.load_policy_configs(cfg.stages)

        # Step 8. Dimensions, manual-terminator reachability, dataset name.
        preflight.run_preflight(cfg, robot, policy_configs)

        if cfg.chain.enabled and chain_policy_config is None:
            # Mock chain: the YAML is the only source, and it was required above.
            extra_action_names = (
                (preflight.PROGRESS_ACTION_NAME,)
                if cfg.chain.model.has_progress
                else ()
            )
        elif cfg.chain.enabled:
            # STRICTLY BEFORE make_policy: it overwrites output_features from
            # the dataset, after which the checkpoint's own action width is
            # unrecoverable. The answer decides whether the recording dataset
            # declares a 17th action feature, which is what makes that very
            # overwrite land on 17 and the `tph` weights load at all.
            extra_action_names = preflight.extra_action_names(
                preflight.checkpoint_action_dimension(chain_policy_config),
                preflight.robot_action_dimension(robot),
            )
            declared = cfg.chain.model.has_progress
            if declared is not None and declared != bool(extra_action_names):
                raise PreflightError(
                    f"`chain.model.has_progress: {declared}` but the checkpoint's "
                    f"action width says {bool(extra_action_names)}. The width is "
                    "the fact; set the key to null to take it from the checkpoint, "
                    "or point at the other checkpoint."
                )
            logger.info(
                f"chain: checkpoint action is "
                f"{'17-D (progress)' if extra_action_names else '16-D (no progress)'}"
                f"; completion is gated by "
                f"{'the progress output' if extra_action_names else 'stall + end-pose NN'}"
            )
    except PreflightError as error:
        logger.error(f"preflight failed: {error}")
        return EXIT_PREFLIGHT

    logger.info(f"stage_runner {__version__} on lerobot {lerobot_version}")

    # Step 9. mkdir with exist_ok=False: an existing run directory means a
    # colliding run_id, and appending to it would interleave two runs in one
    # events.jsonl with no way to separate them afterwards.
    run_id = config.resolve_run_id(cfg)
    try:
        # BOTH calls, not just the mkdir: resolve_run_directory does its own
        # `if run_directory.exists(): raise FileExistsError` (config.py), so it is
        # the one that fires on a colliding run_id and the mkdir only ever sees
        # the race. Wrapping the mkdir alone left the ordinary collision --
        # an explicit `output.run_id` in the YAML, or a rerun inside the same
        # second -- exiting with a traceback instead of the one-line exit 2 this
        # block exists to produce. snapshot_configs stays outside: a failure
        # there means something else is wrong.
        run_directory = config.resolve_run_directory(cfg, run_id)
        run_directory.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        # Built from output.root and run_id rather than from the exception,
        # which for the mkdir half is a bare "[Errno 17] File exists".
        logger.error(
            f"preflight failed: run directory {cfg.output.root}/{run_id} already "
            "exists. Set a different `output.run_id` in the YAML, or leave it "
            "null to get a timestamped one."
        )
        return EXIT_PREFLIGHT
    config.snapshot_configs(cfg, run_directory, source_path)
    logger.info(f"run directory: {run_directory}")

    # The default stands only for the paths that never reach run_trial (a failure
    # in step 11). base_is_stopped=True is right for those: nothing has moved.
    outcome = TrialOutcome(results=[])

    # Step 10. Used as a context manager so the log is closed even on a path
    # that never reaches the finally below (a failure inside step 11).
    with EventLog(run_directory / "events.jsonl", run_id) as log:
        # Step 11, in this order because the dependencies are real.
        # a. Three identity pipelines, exactly as record() builds them.
        processors = record_adapter.make_processors()
        # b. Features are derived from those pipelines; LeRobotDataset.create
        #    mkdirs with exist_ok=False, which is why repo_id is the one CLI
        #    override and every trial needs a fresh one.
        dataset = record_adapter.create_dataset(
            cfg, robot, processors, extra_action_names
        )
        # c. After the dataset because make_policy needs ds_meta, and before
        #    connect so that a cold HF cache stalls with the arms unpowered
        #    rather than inside a control loop.
        onehot_step = None
        if cfg.chain.enabled:
            bundles, onehot_step = policies.load_chain_bundles(
                cfg.stages,
                chain_policy_config,
                dataset.meta,
                onehot_k=cfg.chain.model.onehot_k,
                initial_stage=cfg.chain.from_stage,
                n_action_steps=cfg.chain.model.n_action_steps,
            )
        else:
            bundles = policies.load_bundles(cfg.stages, policy_configs, dataset.meta)
        # d. Core already owns the right arrow, the left arrow and Esc; we add
        #    no listener of our own, so the terminator classifier reads the
        #    flags core leaves behind.
        listener, keyboard_events = record_adapter.make_keyboard_events()

        # e (chain only). The monitor needs the keyboard events dict, which is
        #   created HERE and not with the processors in (a) -- that ordering is
        #   upstream record()'s and is kept, so the monitor is built and
        #   installed after both exist. Installing it into robot_action (not
        #   teleop_action) matters: teleop_action is the pipeline the dataset's
        #   ACTION features were derived from in (b).
        chain_runtime = None
        if cfg.chain.enabled:
            from stage_runner.completion import CompletionMonitorStep

            monitor = CompletionMonitorStep(
                keyboard_events,
                settings=cfg.chain.completion.to_settings(),
                arm_joint_names=chain_params.arm_joint_order,
            )
            record_adapter.install_action_monitor(processors, monitor)
            chain_runtime = ChainRuntime(
                params=chain_params,
                monitor=monitor,
                completion=cfg.chain.completion.to_settings(),
                reset=cfg.chain.reset.to_settings(),
                has_progress=bool(extra_action_names),
                action_names=tuple(dataset.meta.features[ACTION]["names"]),
                state_names=tuple(dataset.meta.features[OBS_STATE]["names"]),
                onehot=onehot_step,
                onehot_k=cfg.chain.model.onehot_k,
                allow_manual_complete=cfg.chain.completion.allow_manual_complete,
            )

        # Upstream record() calls this before its record_loop; we do not go
        # through record(), so without it `display_data: true` reaches
        # record_loop and log_rerun_data writes every frame into a recording
        # stream that was never created -- no viewer, no error, and nothing in
        # the log saying why the operator is looking at a blank screen.
        if cfg.display_data:
            init_rerun(session_name="stage_runner")

        try:
            # Step 12. The first statement that can move the robot. It is inside
            # the try so that a connect that fails half way still reaches the
            # teardown below -- and note that "reaches the teardown" is only true
            # because the teardown no longer gates on robot.is_connected:
            # MobileAIRobot.connect() torques the arms, inits the base and
            # enables base motor torque BEFORE it connects the cameras
            # (mobileai.py:347-362), while is_connected requires every camera
            # (:341-345). A RealSense that fails to enumerate therefore leaves a
            # robot that is energised and NOT is_connected.
            robot.connect()

            # Step 13.
            context = StageContext(
                config=cfg,
                robot=robot,
                dataset=dataset,
                processors=processors,
                events=keyboard_events,
                bundles=bundles,
                log=log,
                run_directory=run_directory,
                chain=chain_runtime,
            )

            # Steps 14-18. VideoEncodingManager.__exit__ flushes the encoders,
            # calls dataset.finalize() and cleans an interrupted episode's image
            # directories, so we do NOT call finalize ourselves (upstream
            # record() calls it twice) and we never call push_to_hub at all --
            # upstream's unguarded push in its finally masks earlier exceptions.
            with record_adapter.video_encoding_manager(dataset):
                outcome = runner.run_trial(context)
        finally:
            # Step 19. This block exists because we call record_loop directly
            # instead of lerobot's record(): none of record()'s own finally is
            # ours for free. Without it an exception unwinding out of a stage
            # leaves the base holding its last velocity command until somebody
            # reaches the e-stop. It is also the LAST software layer -- a hard
            # kill skips it entirely (ACCEPTED RISK 2 at the top of this file;
            # the layer that would cover that case is base firmware, see the
            # same note in transitions.py).
            #
            # HARDWARE FIRST, BOOKKEEPING AFTER, and each in its own try so one
            # failure cannot skip the others. The order used to be the reverse,
            # justified as "every event is on disk before anything touches the
            # hardware" -- but EventLog.emit already flushes per line
            # (events.py:_write), so close() adds nothing a crash would
            # otherwise lose, while a hang in log.close() or in pynput's
            # listener.stop() (an X connection that has gone away) would delay
            # or skip the call that stops the base.
            #
            # UNCONDITIONAL, deliberately deviating from the frozen step 19's
            # `if robot.is_connected:`. That guard cannot fire for a
            # half-connected MobileAIRobot -- is_connected is `arms.is_connected
            # and all(cam.is_connected ...)` (mobileai.py:341-345) and connect()
            # brings the cameras up LAST (:347-362) -- so one RealSense that
            # fails to enumerate would skip disconnect() entirely and leave the
            # arms torqued and the base under motor torque, which is the exact
            # case the README's eval section already documents. disconnect() is
            # safe to call anyway: it sends base.set_cmd_vel(0.0, 0.0) FIRST
            # (mobileai.py:568-569), before it touches the arms or the cameras,
            # and it has no not-connected guard of its own. Anything it raises
            # afterwards (a camera teardown on a camera that never opened) is
            # swallowed here, because it arrives strictly after the base is
            # zeroed.
            try:
                robot.disconnect()
            except Exception:
                logger.exception(
                    "robot.disconnect() failed -- CHECK THE BASE IS STOPPED "
                    "before leaving the robot"
                )
            if listener is not None and not is_headless():
                try:
                    listener.stop()
                except Exception:
                    logger.exception("keyboard listener.stop() failed")
            # Idempotent, and the `with` above would do it anyway; it is here so
            # the ordering is explicit.
            log.close()

    # Step 20. An operator abort, a re-record request or a stage error are all
    # "this trial is not a clean measurement", which the caller (and any batch
    # script around it) has to be able to see without parsing the JSONL.
    #
    # The base stop is checked FIRST and outranks the abort, because the two say
    # different things to whoever reads the code: 1 means the trial is unusable,
    # 3 means the ROBOT may still be moving. A trial that aborted AND left the
    # base driving must report the second -- reporting it as 1 puts it in the
    # bucket a batch script retries.
    if not outcome.base_is_stopped:
        return EXIT_BASE_STOP_FAILED
    if any(result.terminated_by in ABORTING_TERMINATORS for result in outcome.results):
        return EXIT_ABORTED
    if outcome.chain_failed:
        # After the abort check and not before it: the two are mutually
        # exclusive in practice (the runner breaks on an abort without setting
        # chain_failed), and if they ever both held, "a human stopped this" is
        # the more urgent instruction.
        logger.error(
            f"the chain broke at stage {outcome.chain_failed_stage_id!r}; "
            f"{len(outcome.results)} stage(s) ran. The episode IS saved -- a "
            "partial chain is data -- with trial_end.completed=false."
        )
        return EXIT_CHAIN_FAILED
    return EXIT_OK
