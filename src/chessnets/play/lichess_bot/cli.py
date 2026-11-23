"""Command-line entry point for the chessnets Lichess runner."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import pathlib
import sys

from ..config import LichessBotConfig
from .runner import LichessBotRunner


def build_parser(prog: str = "chessnets lichess-bot") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Run a chessnets pipeline as a Lichess Bot account.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run the Lichess bot")
    _add_common_options(run)
    run.add_argument(
        "--token-env",
        default="LICHESS_TOKEN",
        help="environment variable containing the Lichess API token",
    )
    run.add_argument(
        "--log-level", default="INFO", help="logging level (default: %(default)s)"
    )
    validate = subparsers.add_parser(
        "validate", help="validate configuration without loading a model"
    )
    _add_common_options(validate)
    return parser


def main(argv: list[str] | None = None, *, prog: str = "chessnets lichess-bot") -> int:
    args = build_parser(prog).parse_args(argv)
    config = _apply_overrides(LichessBotConfig.from_yaml(args.config), args)
    if args.command == "validate":
        print(f"Configuration is valid for model {config.model_id!r}.")
        return 0
    token = os.environ.get(args.token_env)
    if not token:
        raise SystemExit(
            f"Missing Lichess token. Set {args.token_env} or pass a token in Python."
        )
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    runner = LichessBotRunner.from_config(config, token=token)
    runner.run()
    return 0


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", required=True, type=pathlib.Path, help="YAML bot configuration"
    )
    parser.add_argument("--model-id", help="override the configured model ID")
    parser.add_argument("--device", help="override the configured device")
    parser.add_argument(
        "--metadata-mode",
        choices=("frozen", "game"),
        help="use fixed metadata values or derive them from each Lichess game",
    )
    parser.add_argument("--bot-elo", type=int, help="override the bot's Elo rating")
    parser.add_argument(
        "--opponent-elo", type=int, help="override the opponent's Elo rating"
    )
    parser.add_argument("--time-control", help="override the time control, e.g. 300+3")
    parser.add_argument("--prompt", help="override the text prompt")


def _apply_overrides(
    config: LichessBotConfig, args: argparse.Namespace
) -> LichessBotConfig:
    values = dict(config.metadata.values)
    if args.bot_elo is not None:
        values["bot_elo"] = args.bot_elo
    if args.opponent_elo is not None:
        values["opponent_elo"] = args.opponent_elo
    if args.time_control is not None:
        values["time_control"] = args.time_control
    if args.prompt is not None:
        values["prompt"] = args.prompt
    metadata = dataclasses.replace(
        config.metadata,
        mode=args.metadata_mode or config.metadata.mode,
        values=values,
    )
    return dataclasses.replace(
        config,
        model_id=args.model_id or config.model_id,
        device=args.device or config.device,
        metadata=metadata,
    )


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["build_parser", "main"]
