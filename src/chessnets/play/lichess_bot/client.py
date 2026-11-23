"""Small async client for the Lichess Bot API.

``httpx`` is imported only when the concrete client is constructed.  This
keeps the rest of chessnets importable without installing the optional bot
extra, and makes the client straightforward to replace with a fake in tests.
"""

from __future__ import annotations

import json
import typing as tp
import urllib.parse

from chessnets._optional import import_optional


class LichessAPIError(RuntimeError):
    def __init__(self, status_code: int, message: str, response: tp.Any = None):
        super().__init__(f"Lichess API error {status_code}: {message}")
        self.status_code = status_code
        self.response = response


class RateLimitedError(LichessAPIError):
    def __init__(self, message: str = "Lichess API rate limit exceeded"):
        super().__init__(429, message)


class LichessClientProtocol(tp.Protocol):
    async def account(self) -> dict[str, tp.Any]: ...

    def event_stream(self) -> tp.AsyncIterator[dict[str, tp.Any]]: ...

    def game_stream(self, game_id: str) -> tp.AsyncIterator[dict[str, tp.Any]]: ...

    async def accept_challenge(self, challenge_id: str) -> None: ...

    async def decline_challenge(
        self, challenge_id: str, reason: str = "generic"
    ) -> None: ...

    async def make_move(
        self, game_id: str, move: str, *, offering_draw: bool = False
    ) -> None: ...

    async def close(self) -> None: ...


class LichessClient:
    """Authenticated async Lichess client using the Bot API endpoints."""

    def __init__(
        self,
        token: str,
        *,
        base_url: str = "https://lichess.org",
        request_timeout_seconds: float = 30.0,
        http_client: tp.Any = None,
    ):
        if not token or any(character.isspace() for character in token):
            raise ValueError("A non-empty Lichess API token is required")
        self.base_url = base_url.rstrip("/")
        self._owns_client = http_client is None
        if http_client is None:
            httpx = import_optional("httpx", "lichess-bot")
            http_client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={"Authorization": f"Bearer {token}"},
                timeout=httpx.Timeout(request_timeout_seconds, read=None),
            )
        self._client = http_client

    async def account(self) -> dict[str, tp.Any]:
        return await self._json_request("GET", "/api/account")

    async def accept_challenge(self, challenge_id: str) -> None:
        await self._empty_request(
            "POST", f"/api/challenge/{_path_part(challenge_id)}/accept"
        )

    async def decline_challenge(
        self, challenge_id: str, reason: str = "generic"
    ) -> None:
        await self._empty_request(
            "POST",
            f"/api/challenge/{_path_part(challenge_id)}/decline",
            data={"reason": reason},
        )

    async def make_move(
        self, game_id: str, move: str, *, offering_draw: bool = False
    ) -> None:
        await self._empty_request(
            "POST",
            f"/api/bot/game/{_path_part(game_id)}/move/{_path_part(move)}",
            params={"offeringDraw": "true" if offering_draw else "false"},
        )

    async def event_stream(self) -> tp.AsyncIterator[dict[str, tp.Any]]:
        async for event in self._stream("/api/stream/event"):
            yield event

    async def game_stream(self, game_id: str) -> tp.AsyncIterator[dict[str, tp.Any]]:
        async for event in self._stream(f"/api/bot/game/stream/{_path_part(game_id)}"):
            yield event

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _json_request(
        self, method: str, path: str, **kwargs: tp.Any
    ) -> dict[str, tp.Any]:
        response = await self._client.request(method, path, **kwargs)
        await _raise_for_status(response)
        try:
            value = response.json()
        except (ValueError, json.JSONDecodeError) as exc:
            raise LichessAPIError(
                response.status_code, "Expected JSON response"
            ) from exc
        if not isinstance(value, dict):
            raise LichessAPIError(response.status_code, "Expected a JSON object")
        return value

    async def _empty_request(self, method: str, path: str, **kwargs: tp.Any) -> None:
        response = await self._client.request(method, path, **kwargs)
        await _raise_for_status(response)

    async def _stream(self, path: str) -> tp.AsyncIterator[dict[str, tp.Any]]:
        try:
            stream_context = self._client.stream(
                "GET",
                path,
                headers={"Accept": "application/x-ndjson"},
            )
            async with stream_context as response:
                await _raise_for_status(response)
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise LichessAPIError(200, "Invalid NDJSON event") from exc
                    if not isinstance(event, dict):
                        raise LichessAPIError(200, "NDJSON event must be an object")
                    yield event
        except LichessAPIError:
            raise


async def _raise_for_status(response: tp.Any) -> None:
    status = int(response.status_code)
    if 200 <= status < 300:
        return
    if status == 429:
        raise RateLimitedError()
    try:
        body = response.json()
    except Exception:
        body = getattr(response, "text", "")
    raise LichessAPIError(status, str(body)[:500], response)


def _path_part(value: str) -> str:
    return urllib.parse.quote(str(value), safe="")


__all__ = [
    "LichessAPIError",
    "LichessClient",
    "LichessClientProtocol",
    "RateLimitedError",
]
