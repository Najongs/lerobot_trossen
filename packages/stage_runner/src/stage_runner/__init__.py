"""Public re-exports, resolved lazily on first attribute access (PEP 562).

LAZY ON PURPOSE, and the reason is aggregate.py. That module is stdlib-only so an
old events.jsonl can be re-aggregated years later on whatever machine is free,
but an EAGER re-export list here defeats it: importing ``stage_runner.aggregate``
executes this file first, and the eager form pulled in config (draccus),
executors -> record_adapter -> lerobot -> torch, mock_robot -> lerobot.robots and
policies -> torch. MEASURED before this change: on an interpreter without the
robot stack ``python -m stage_runner.aggregate`` died at ``ModuleNotFoundError:
No module named 'draccus'``, so the guarantee aggregate.py's docstring states was
false exactly where it mattered.

The MockRobot re-export stays LOAD-BEARING and still works through
``__getattr__``. lerobot resolves a device class by name:
make_device_from_device_class (utils/import_utils.py) strips "Config" off
MockRobotConfig and searches [parent_module, parent_module + "." +
device_class_name.lower()] -- i.e. ["stage_runner", "stage_runner.mockrobot"].
"MockRobot".lower() is "mockrobot" with NO underscore, so the second candidate
never matches the module file mock_robot.py and the ONLY way the factory finds
the class is this package-level name. It reaches it with
``hasattr(module, "MockRobot")`` and then ``getattr``, and BOTH go through a
module-level ``__getattr__``, so deferring the import does not break resolution.
Do not remove the entry and do not rename the symbol.

What the lazy form no longer does is run mock_robot's
``@RobotConfig.register_subclass`` as a side effect of importing this package.
That registration is what makes ``type: stage_runner_mock_robot`` resolvable when
draccus parses the YAML, so config.parse_config now imports the module itself --
see the comment there.
"""

import importlib
from typing import Any

__version__: str = "0.1.0"

# name -> the submodule that defines it. Each entry costs one import on first
# access and nothing at all if the caller never touches it.
_LAZY_ATTRIBUTES: dict[str, str] = {
    "LATEST_CONFIG_VERSION": "config",
    "DatasetConfig": "config",
    "DefaultsConfig": "config",
    "OutputConfig": "config",
    "StageConfig": "config",
    "StageRunnerConfig": "config",
    "TerminatorConfig": "config",
    "StageContext": "context",
    "EventLog": "events",
    "EXECUTORS": "executors",
    "MockRobot": "mock_robot",
    "MockRobotConfig": "mock_robot",
    "PolicyBundle": "policies",
    "StageResult": "results",
    "TrialOutcome": "results",
    "run_trial": "runner",
}


def __getattr__(name: str) -> Any:
    """Import the submodule that owns ``name`` on first access, then cache it.

    Raising AttributeError for an unknown name is required, not merely tidy:
    ``from stage_runner import cli`` falls back to importing the SUBMODULE only
    after the attribute lookup fails, so answering every name here would break
    every ``from stage_runner import <module>`` in the package.

    An AttributeError raised WHILE IMPORTING the submodule is converted to an
    ImportError, and that conversion is the point of the try. lerobot resolves a
    device class with ``hasattr(module, "MockRobot")``
    (utils/import_utils.make_device_from_device_class), and hasattr swallows any
    AttributeError -- including one raised from inside mock_robot's module body,
    whose very first statement is ``@RobotConfig.register_subclass(...)``, the
    exact shape a lerobot 0.5/0.6 API change produces. The real cause would be
    erased and the factory would report ``Could not locate device class
    'MockRobot' ... Tried modules: ['stage_runner', 'stage_runner.mockrobot']``,
    pointing the reader at a module-search problem that does not exist. An
    ImportError is not swallowed by hasattr, and ``from error`` keeps the
    original traceback.
    """
    module_name = _LAZY_ATTRIBUTES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = importlib.import_module(f".{module_name}", __name__)
    except AttributeError as error:
        raise ImportError(
            f"importing stage_runner.{module_name} for {name!r} failed"
        ) from error
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_ATTRIBUTES))
