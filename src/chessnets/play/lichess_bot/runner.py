"""Async Lichess bot runner."""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
import typing as tp

import chess

from ..adapters import PLAY_ADAPTERS, PipelineAdapterError, PipelinePlayAdapter
from ..config import LichessBotConfig
from ..context import LiveGameState
from ..matching import parse_exact_time_control
from ..metadata import MetadataResolutionError, resolver_for_config
from .client import LichessClient, LichessClientProtocol, RateLimitedError

logger = logging.getLogger(__name__)


class LichessBotRunner:
    """Run one loaded chessnets pipeline as a Lichess Bot account."""

    def __init__(
        self,
        pipeline: tp.Any,
        config: LichessBotConfig,
        *,
        token: str | None = None,
        client: LichessClientProtocol | None = None,
        adapter: PipelinePlayAdapter | None = None,
    ):
        self.pipeline = pipeline
        self.config = config
        self._client = client or LichessClient(
            token or os.environ.get("LICHESS_TOKEN", ""),
            request_timeout_seconds=config.runtime.request_timeout_seconds,
        )
        self.adapter = adapter or PLAY_ADAPTERS.get(pipeline)
        self.resolver = resolver_for_config(config.metadata)
        self._stop_event = asyncio.Event()
        self._ready_event = asyncio.Event()
        self._game_tasks: dict[str, asyncio.Task[None]] = {}
        self._bot_username: str | None = None
        self._run_task: asyncio.Task[None] | None = None

    @classmethod
    def from_config(
        cls,
        config: LichessBotConfig,
        *,
        token: str | None = None,
        pipeline: tp.Any = None,
        client: LichessClientProtocol | None = None,
    ) -> LichessBotRunner:
        if pipeline is None:
            from ..pipelines.auto_pipeline import AutoPipeline

            pipeline = AutoPipeline.from_pretrained(
                config.model_id,
                device=config.device,
                **config.model_kwargs,
            )
        return cls(pipeline, config, token=token, client=client)

    def run(self) -> None:
        """Run until interrupted from a normal synchronous Python process."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.run_async())
            return
        raise RuntimeError(
            "LichessBotRunner.run() cannot be called inside an active event loop; "
            "use `await runner.run_async()` in a notebook."
        )

    async def run_async(self) -> None:
        """Run until :meth:`stop` is called or the task is cancelled."""
        self._run_task = asyncio.current_task()
        try:
            account = await self._client.account()
            self._bot_username = str(account.get("username") or account.get("id") or "")
            if not self._bot_username:
                raise RuntimeError(
                    "Lichess account response did not include a username"
                )
            if account.get("title") not in {None, "BOT"}:
                logger.warning("Authenticated account is not marked as a BOT account")
            self._validate_pipeline_configuration()
            self._ready_event.set()
            logger.info("Lichess bot %s is ready", self._bot_username)
            await self._event_loop()
        finally:
            self._stop_event.set()
            await self._cancel_game_tasks()
            await self._client.close()
            self._run_task = None

    async def wait_until_ready(self) -> None:
        await self._ready_event.wait()

    async def stop(self) -> None:
        self._stop_event.set()
        await self._cancel_game_tasks()
        if self._run_task is not None and self._run_task is not asyncio.current_task():
            # Closing the HTTP stream wakes the event loop so notebook callers
            # do not have to wait for another NDJSON event.
            await self._client.close()

    async def _event_loop(self) -> None:
        delay = 1.0
        while not self._stop_event.is_set():
            try:
                async for event in self._client.event_stream():
                    delay = 1.0
                    await self._handle_event(event)
                    if self._stop_event.is_set():
                        return
            except asyncio.CancelledError:
                raise
            except RateLimitedError:
                delay = min(60.0, max(delay, 60.0))
                logger.warning("Lichess rate limit reached; retrying in %.1fs", delay)
            except Exception:
                logger.exception("Lichess event stream disconnected")
            if not self._stop_event.is_set():
                await asyncio.sleep(delay + random.random() * min(delay, 1.0))
                delay = min(self.config.runtime.reconnect_max_seconds, delay * 2)

    async def _handle_event(self, event: dict[str, tp.Any]) -> None:
        event_type = event.get("type")
        if event_type == "challenge":
            await self._handle_challenge(event.get("challenge") or {})
        elif event_type == "gameStart":
            game = event.get("game") or {}
            game_id = game.get("id") or event.get("gameId")
            if game_id:
                await self._start_game(str(game_id))
        elif event_type == "gameFinish":
            game = event.get("game") or {}
            game_id = game.get("id") or event.get("gameId")
            if game_id:
                await self._stop_game(str(game_id))

    async def _handle_challenge(self, challenge: dict[str, tp.Any]) -> None:
        challenge_id = challenge.get("id")
        if not challenge_id:
            logger.warning("Ignoring challenge event without an ID")
            return
        if not self.config.challenges.accept_incoming:
            await self._client.decline_challenge(str(challenge_id), "generic")
            return
        reason = self._challenge_rejection_reason(challenge)
        if reason is not None:
            logger.info("Declining challenge %s: %s", challenge_id, reason)
            await self._client.decline_challenge(str(challenge_id), reason)
            return
        await self._client.accept_challenge(str(challenge_id))
        logger.info("Accepted challenge %s", challenge_id)

    def _challenge_rejection_reason(self, challenge: dict[str, tp.Any]) -> str | None:
        policy = self.config.challenges
        if challenge.get("direction") not in {None, "in"}:
            return "generic"
        variant = (challenge.get("variant") or {}).get("key", challenge.get("variant"))
        if variant and variant not in policy.variants:
            return "variant"
        speed = challenge.get("speed")
        if policy.speeds and speed not in policy.speeds:
            return "timeControl"
        if policy.rated is not None and challenge.get("rated") != policy.rated:
            return "casual" if policy.rated is False else "rated"
        if policy_time_control_mismatch(challenge, self.config):
            return "timeControl"
        challenger = challenge.get("challenger") or {}
        if not policy.allow_bots and challenger.get("title") == "BOT":
            return "generic"
        if len(self._game_tasks) >= policy.max_concurrent_games:
            return "busy"
        return None

    async def _start_game(self, game_id: str) -> None:
        if game_id in self._game_tasks:
            return
        if len(self._game_tasks) >= self.config.challenges.max_concurrent_games:
            logger.warning("Ignoring game %s because the runner is busy", game_id)
            return
        task = asyncio.create_task(
            self._play_game(game_id), name=f"lichess-game-{game_id}"
        )
        self._game_tasks[game_id] = task
        task.add_done_callback(lambda _: self._game_tasks.pop(game_id, None))

    async def _stop_game(self, game_id: str) -> None:
        task = self._game_tasks.get(game_id)
        if task is not None:
            task.cancel()

    async def _play_game(self, game_id: str) -> None:
        assert self._bot_username is not None
        last_position: tuple[str, ...] | None = None
        delay = 1.0
        while not self._stop_event.is_set():
            state: LiveGameState | None = None
            try:
                async for event in self._client.game_stream(game_id):
                    event_type = event.get("type")
                    if event_type == "gameFull":
                        state = LiveGameState.from_game_full(event, self._bot_username)
                    elif event_type == "gameState":
                        if state is None:
                            continue
                        state.apply_game_state(event)
                    else:
                        continue
                    if (
                        state.is_terminal
                        or not state.is_playable
                        or not state.is_bot_turn
                    ):
                        continue
                    position = tuple(state.moves)
                    if position == last_position:
                        continue
                    decision = await self._predict(state)
                    await self._client.make_move(
                        game_id,
                        decision.move_uci,
                        offering_draw=decision.offered_draw,
                    )
                    last_position = position
                    logger.info(
                        "Played %s in %s at ply %s",
                        decision.move_uci,
                        game_id,
                        state.ply,
                    )
                return
            except asyncio.CancelledError:
                raise
            except (PipelineAdapterError, MetadataResolutionError, ValueError):
                logger.exception("Cannot play game %s with this pipeline", game_id)
                return
            except RateLimitedError:
                delay = 60.0
                logger.warning("Rate limited while playing %s", game_id)
            except Exception:
                logger.exception("Game stream/inference failed for %s", game_id)
            if not self._stop_event.is_set():
                await asyncio.sleep(delay + random.random() * min(delay, 1.0))
                delay = min(self.config.runtime.reconnect_max_seconds, delay * 2)

    async def _predict(self, state: LiveGameState):
        remaining_ms = (
            state.wtime_ms if state.bot_color == chess.WHITE else state.btime_ms
        )
        if remaining_ms is not None:
            remaining_seconds = remaining_ms / 1000
            if remaining_seconds <= self.config.runtime.clock_safety_margin_seconds:
                raise PipelineAdapterError(
                    f"Only {remaining_seconds:.2f}s remain; refusing late inference"
                )
        started = time.monotonic()
        supported = _supported_time_controls(self.pipeline)
        metadata = self.resolver.resolve(state, supported_time_controls=supported)
        self.adapter.validate_launch(self.pipeline, metadata)
        decision = await self.adapter.predict(self.pipeline, state, metadata)
        elapsed = time.monotonic() - started
        configured_limit = self.config.runtime.inference_timeout_seconds
        if configured_limit is not None and elapsed > configured_limit:
            logger.warning(
                "Inference for %s took %.2fs, above configured %.2fs limit",
                state.game_id,
                elapsed,
                configured_limit,
            )
        return decision

    def _validate_pipeline_configuration(self) -> None:
        required = self.pipeline.input_config.required_inputs
        values = self.config.metadata.values
        aliases = {
            "prompt": "prompts",
            "prompt_template": "prompts",
            "time_control": "time_controls",
            "white_elo": "white_elos",
            "black_elo": "black_elos",
        }
        available = {aliases.get(name, name) for name in values}
        if {"bot_elo", "opponent_elo"}.issubset(values):
            available.update({"elo_self", "elo_opponent", "white_elos", "black_elos"})
        if self.config.metadata.mode == "game":
            # Ratings and the time control can come from gameFull. Prompts and
            # arbitrary extra inputs still need a launch-time binding.
            available.update(
                {
                    "elo_self",
                    "elo_opponent",
                    "white_elos",
                    "black_elos",
                    "time_controls",
                    "white_clock_times",
                    "black_clock_times",
                    "clock_times",
                    "move_seconds",
                }
            )
        missing = sorted(
            name for name in required - {"pgns", "fens"} if name not in available
        )
        if missing:
            raise PipelineAdapterError(
                "Metadata is missing a launch binding for pipeline input(s): "
                + ", ".join(missing)
            )

    async def _cancel_game_tasks(self) -> None:
        tasks = list(self._game_tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._game_tasks.clear()


def _supported_time_controls(pipeline: tp.Any) -> tuple[str, ...]:
    tokenizer = getattr(pipeline, "tokenizer", None)
    tokens = getattr(tokenizer, "tokens", None)
    if tokens is None:
        return ()
    return tuple(token for token in tokens if isinstance(token, str) and "+" in token)


def policy_time_control_mismatch(
    challenge: dict[str, tp.Any], config: LichessBotConfig
) -> bool:
    """Return whether frozen time-control metadata conflicts with a challenge."""

    if config.metadata.mode != "frozen":
        return False
    configured = config.metadata.values.get(
        "time_control", config.metadata.values.get("time_controls")
    )
    if configured is None:
        return False
    time_control = challenge.get("timeControl") or challenge.get("clock") or {}
    if not isinstance(time_control, dict):
        return False
    limit = time_control.get("limit", time_control.get("initial"))
    increment = time_control.get("increment", 0)
    if limit is None:
        return True
    try:
        return parse_exact_time_control(configured) != parse_exact_time_control(
            f"{int(limit)}+{int(increment)}"
        )
    except (TypeError, ValueError):
        return True


__all__ = ["LichessBotRunner", "policy_time_control_mismatch"]
