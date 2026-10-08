"""The canonical ``lm3`` command.

The consensus plan speaks of ``lm3 serve`` (§3.1's settings contract, §4's Step 5 exit gate) and
anticipates ``lm3 connect``. Shipping a separate ``lm3-serve`` console script instead would have
added exactly the kind of redundant public control surface this refactor exists to remove, and would
have had to be deprecated the moment a second subcommand arrived.

``machine3`` and ``lm3-setup`` deliberately remain their own entry points: ``machine3`` is the
long-standing pipeline command, and §2.13 requires ``lm3-setup`` to be spawnable as its own
subprocess so the server can Stop a tuning tree without stopping itself. Neither is aliased here --
two spellings of one command is the redundancy, not the cure.

Dispatch is lazy on purpose: ``lm3 --help`` must work in an install that has no ``server`` extra, so
nothing imports FastAPI until ``serve`` is actually requested.
"""
from __future__ import annotations

import argparse
import sys
from typing import Callable, Sequence

#: subcommand -> (help text, importer). The importer is called only when that subcommand runs.
_COMMANDS: dict[str, tuple[str, Callable[[], Callable[[list[str]], int]]]] = {}


def _serve() -> Callable[[list[str]], int]:
    from leafmachine3.server.app import main as serve_main  # noqa: PLC0415 - lazy by design

    return serve_main


_COMMANDS["serve"] = ("run the local LeafMachine3 server", _serve)


def _models() -> Callable[[list[str]], int]:
    from leafmachine3.modelhub.cli import main as models_main  # noqa: PLC0415 - lazy by design

    return models_main


_COMMANDS["models"] = ("install / check the default models from the Hugging Face Hub", _models)


def _doctor() -> Callable[[list[str]], int]:
    from leafmachine3.doctor import main as doctor_main  # noqa: PLC0415 - lazy by design

    return doctor_main


_COMMANDS["doctor"] = ("check that this install will run (Python, packages, GPU driver, accelerator)", _doctor)


def _version() -> Callable[[list[str]], int]:
    from leafmachine3.version_info import main as version_main  # noqa: PLC0415 - lazy by design

    return version_main


_COMMANDS["version"] = ("the LM3 version and the dependency / model locks it pins", _version)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lm3",
        description="LeafMachine3. Run `lm3 <command> --help` for a command's own options.",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    for name, (help_text, _) in sorted(_COMMANDS.items()):
        sub.add_parser(name, help=help_text, add_help=False)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()

    if not args or args[0] in {"-h", "--help"}:
        parser.print_help()
        return 0 if args else 1
    if args[0] in {"-V", "--version"}:
        from leafmachine3 import __version__  # noqa: PLC0415

        print(__version__)
        return 0

    command, rest = args[0], args[1:]
    entry = _COMMANDS.get(command)
    if entry is None:
        parser.print_usage(sys.stderr)
        print(f"lm3: unknown command {command!r}. Choose from: {', '.join(sorted(_COMMANDS))}",
              file=sys.stderr)
        return 2
    # The subcommand owns everything after its own name, so `lm3 serve --host 0.0.0.0` reaches the
    # server's parser unchanged and its --help is the server's, not a rewritten copy.
    return entry[1]()(rest)


if __name__ == "__main__":
    raise SystemExit(main())
