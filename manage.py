"""Entry point: ``python manage.py <command> [key=value ...]``. Run without arguments for the list of commands."""
import sys

from pulsegate.cli import main

if __name__ == "__main__":
    sys.exit(main())
