"""``scripts/eval_chain.sh`` env-var validation, run as the shell actually runs it.

ONE RULE UNDER TEST: ``DRY_RUN`` and ``RESET_ONLY`` are read as booleans, and a
value that is neither empty, ``0``, ``1`` nor one of ``true|yes|on`` is REFUSED
with exit 2 instead of being read as "off". The old script compared
``[[ "$DRY_RUN" == 1 ]]`` and nothing else, so ``DRY_RUN=true`` -- which is what a
person types -- silently meant "not a dry run" and the script went on to drive the
robot.

``bash -n`` CANNOT TEST THIS. It parses and does not run, so it is blind to every
value-handling bug; the first version of this script had a `grep` reading the
wrong `fps:` out of a YAML and `bash -n` was happy with it (see the script's own
comment). These cases therefore EXECUTE the script, with two things standing in
the way of it reaching the robot:

* a ``uv`` shim first on ``PATH`` that touches a marker file and exits 97. The
  script's first contact with anything heavy is ``POLICY=$(uv run python -c ...)``,
  so the marker is a precise answer to "did validation let it through", and 97 is
  a precise answer to "and then it stopped at the shim". NOTHING in this test ever
  runs the real ``uv``, imports ``lerobot_robot_trossen`` or opens a port.
* ``HOME`` pointed at a temporary directory, because the script does
  ``mkdir -p ~/eval_logs`` before it gets to ``uv`` and must not write into the
  operator's real log directory from a test.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "eval_chain.sh"

# The same two modules every chain test refuses to let in. This file never
# imports stage_runner at all, but the shim exists precisely so that running the
# script cannot pull them in through `uv run`, and asserting it here is what makes
# that claim checkable rather than a comment.
FORBIDDEN_MODULES = ("lerobot_robot_trossen", "trossen_slate")

EXIT_REFUSED = 2
EXIT_SHIM = 97


class EvalChainFlagTest(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue(SCRIPT.is_file(), f"{SCRIPT} is missing")
        self.tmp = Path(tempfile.mkdtemp(prefix="eval_chain_shell_"))
        self.home = self.tmp / "home"
        self.bin = self.tmp / "bin"
        self.marker = self.tmp / "uv_was_called"
        self.home.mkdir()
        self.bin.mkdir()
        shim = self.bin / "uv"
        shim.write_text(
            "#!/bin/sh\n"
            '# Stands in for `uv`. Touches the marker so the test can tell "the\n'
            '# script got this far" from "the script refused", then exits with a\n'
            "# status no other failure in the script produces.\n"
            'touch "$MARKER"\n'
            f"exit {EXIT_SHIM}\n",
            encoding="utf-8",
        )
        shim.chmod(0o755)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)
        leaked = [name for name in FORBIDDEN_MODULES if name in sys.modules]
        assert not leaked, (
            f"robot SDK module(s) {leaked} were imported by this test. The shell "
            "test must never reach a real `uv run`."
        )

    def _run(self, model: str = "M1", **env_overrides: str):
        self.marker.unlink(missing_ok=True)
        env = dict(os.environ)
        env.update(
            PATH=f"{self.bin}{os.pathsep}{env.get('PATH', '')}",
            HOME=str(self.home),
            MARKER=str(self.marker),
        )
        # Not inherited from the caller: a stray DRY_RUN in the developer's shell
        # would change which branch every case below takes.
        for name in ("DRY_RUN", "RESET_ONLY", "FROM_STAGE", "TO_STAGE", "MANUAL"):
            env.pop(name, None)
        env.update(env_overrides)
        completed = subprocess.run(
            ["bash", str(SCRIPT), model],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        return completed, self.marker.exists()

    # ------------------------------------------------------------------ syntax

    def test_the_script_parses(self) -> None:
        """`bash -n` is necessary and not sufficient; the rest of this file is why."""
        completed = subprocess.run(
            ["bash", "-n", str(SCRIPT)], capture_output=True, text=True, timeout=60
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    # ------------------------------------------------- the refusals (exit 2)

    def test_dry_run_true_is_accepted_as_on_and_never_reaches_the_robot(self) -> None:
        """The headline case: `DRY_RUN=true` must not mean "not a dry run"."""
        completed, called = self._run(DRY_RUN="true")
        self.assertTrue(
            called,
            "a valid value must pass validation and reach the checkpoint step; "
            f"stderr: {completed.stderr}",
        )
        self.assertEqual(
            completed.returncode,
            EXIT_SHIM,
            "it got as far as `uv run` and stopped at the shim, which is as far "
            "as any test is allowed to take this script",
        )

    def test_a_value_that_is_neither_zero_nor_one_is_refused(self) -> None:
        for value in ("2", "bogus", "TRUE2", "y", "-1", "1.0", "false", "off"):
            with self.subTest(DRY_RUN=value):
                completed, called = self._run(DRY_RUN=value)
                self.assertEqual(
                    completed.returncode,
                    EXIT_REFUSED,
                    f"DRY_RUN={value!r} must exit 2, not be read as off; "
                    f"stdout={completed.stdout!r} stderr={completed.stderr!r}",
                )
                self.assertFalse(
                    called,
                    "the refusal must happen BEFORE anything heavy runs -- no "
                    "checkpoint download, no robot package import",
                )
                self.assertIn("DRY_RUN", completed.stderr)

    def test_reset_only_is_validated_the_same_way(self) -> None:
        completed, called = self._run(RESET_ONLY="2")
        self.assertEqual(completed.returncode, EXIT_REFUSED, completed.stderr)
        self.assertFalse(called)
        self.assertIn("RESET_ONLY", completed.stderr)

    def test_the_accepted_spellings(self) -> None:
        """Empty, 0, 1, and the three affirmatives in any case."""
        for value in ("", "0", "1", "true", "TRUE", "yes", "On"):
            with self.subTest(DRY_RUN=value):
                completed, called = self._run(DRY_RUN=value)
                self.assertNotEqual(
                    completed.returncode,
                    EXIT_REFUSED,
                    f"DRY_RUN={value!r} must be accepted; stderr: {completed.stderr}",
                )
                self.assertTrue(called)

    # ------------------------------------------------------- stage range

    def test_a_stage_outside_one_to_eleven_is_refused(self) -> None:
        for name, value in (
            ("FROM_STAGE", "0"),
            ("FROM_STAGE", "12"),
            ("TO_STAGE", "99"),
            ("FROM_STAGE", "x"),
            ("TO_STAGE", "1 11"),
        ):
            with self.subTest(**{name: value}):
                completed, called = self._run(**{name: value})
                self.assertEqual(completed.returncode, EXIT_REFUSED, completed.stderr)
                self.assertFalse(called)
                self.assertIn(name, completed.stderr)

    def test_from_stage_after_to_stage_is_refused(self) -> None:
        completed, called = self._run(FROM_STAGE="5", TO_STAGE="3")
        self.assertEqual(completed.returncode, EXIT_REFUSED, completed.stderr)
        self.assertFalse(called)
        self.assertIn("FROM_STAGE", completed.stderr)

    def test_an_empty_stage_variable_takes_the_default(self) -> None:
        """``${TO_STAGE:-11}`` treats empty as unset, which is 1..11.

        Asserted rather than assumed: it is the one spelling that does NOT go
        through the 1..11 regex, and the same convention the two boolean
        variables follow (empty means "not set", i.e. off).
        """
        completed, called = self._run(FROM_STAGE="", TO_STAGE="", DRY_RUN="1")
        self.assertNotEqual(completed.returncode, EXIT_REFUSED, completed.stderr)
        self.assertTrue(called)
        self.assertIn("단계 1~11", completed.stdout)

    def test_a_valid_sub_range_is_accepted(self) -> None:
        completed, called = self._run(FROM_STAGE="3", TO_STAGE="3", DRY_RUN="1")
        self.assertNotEqual(completed.returncode, EXIT_REFUSED, completed.stderr)
        self.assertTrue(called)

    # ------------------------------------------------------------- the model

    def test_an_unknown_model_is_refused_before_anything_runs(self) -> None:
        completed, called = self._run(model="not-a-model")
        self.assertEqual(completed.returncode, EXIT_REFUSED, completed.stderr)
        self.assertFalse(called)


if __name__ == "__main__":
    unittest.main()
