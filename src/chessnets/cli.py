"""The ``chessnets`` command line.

Subcommands are dispatched by name and imported only when they run, so
``chessnets --help`` stays fast and never loads PyTorch.
"""

from __future__ import annotations

import dataclasses
import importlib
import sys
import typing as tp


@dataclasses.dataclass(frozen=True)
class Command:
    module: str
    help: str

    def run(self, argv: list[str], prog: str) -> int:
        main = importlib.import_module(self.module).main
        return main(argv, prog=prog) or 0


CommandTable = dict[str, tp.Union[Command, "CommandGroup"]]


@dataclasses.dataclass(frozen=True)
class CommandGroup:
    help: str
    commands: CommandTable


COMMANDS: CommandTable = {
    "uci": Command(
        "chessnets.play.uci",
        "run a pipeline as a UCI chess engine (also installed as chessnets-uci)",
    ),
    "lichess-bot": Command(
        "chessnets.play.lichess_bot.cli",
        "run a pipeline as a Lichess Bot account",
    ),
    "convert": CommandGroup(
        "convert checkpoints to ChessNets safetensors repositories",
        {
            "lc0": Command(
                "chessnets.loaders.convert_lc0",
                "convert an Lc0/Maia protobuf checkpoint "
                "(requires the chessnets-lc0-protobuf adapter)",
            ),
        },
    ),
}


def _usage(prog: str, commands: CommandTable) -> str:
    width = max(len(name) for name in commands)
    lines = [f"usage: {prog} <command> [<args>]", "", "commands:"]
    lines += [
        f"  {name.ljust(width)}  {entry.help}" for name, entry in commands.items()
    ]
    lines += ["", f"Run '{prog} <command> --help' for a command's options."]
    return "\n".join(lines)


def _dispatch(commands: CommandTable, argv: list[str], prog: str) -> int:
    if not argv:
        print(_usage(prog, commands), file=sys.stderr)
        return 2
    name, rest = argv[0], argv[1:]
    if name in {"-h", "--help"}:
        print(_usage(prog, commands))
        return 0
    if name == "--version" and commands is COMMANDS:
        from chessnets import __version__

        print(f"{prog} {__version__}")
        return 0
    entry = commands.get(name)
    if entry is None:
        print(
            f"{prog}: unknown command {name!r}\n\n{_usage(prog, commands)}",
            file=sys.stderr,
        )
        return 2
    if isinstance(entry, CommandGroup):
        return _dispatch(entry.commands, rest, f"{prog} {name}")
    return entry.run(rest, f"{prog} {name}")


def main(argv: list[str] | None = None) -> int:
    return _dispatch(
        COMMANDS, list(sys.argv[1:] if argv is None else argv), "chessnets"
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["COMMANDS", "Command", "CommandGroup", "main"]
