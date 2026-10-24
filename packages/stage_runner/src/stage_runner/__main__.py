"""``python -m stage_runner`` entry point; the package declares no console script."""

import sys

from stage_runner.cli import main

if __name__ == "__main__":
    sys.exit(main())
