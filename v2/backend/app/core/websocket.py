import asyncio
from collections import defaultdict

from fastapi import WebSocket


class ConnectionManager:
    def __init__(self) -> None:
        self._channels: dict[str, set[WebSocket]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, channel: str) -> None:
        await websocket.accept()
        async with self._lock:
            self._channels[channel].add(websocket)

    async def disconnect(self, websocket: WebSocket, channel: str) -> None:
        async with self._lock:
            self._channels[channel].discard(websocket)
            if not self._channels[channel]:
                self._channels.pop(channel, None)

    async def send(self, channel: str, message: dict) -> None:
        for ws in list(self._channels.get(channel, ())):
            try:
                await ws.send_json(message)
            except Exception:
                await self.disconnect(ws, channel)

    async def close_all(self) -> None:
        async with self._lock:
            sockets = [ws for group in self._channels.values() for ws in group]
            self._channels.clear()
        for ws in sockets:
            try:
                await ws.close()
            except Exception:
                pass

    def count(self, channel: str) -> int:
        return len(self._channels.get(channel, ()))


manager = ConnectionManager()
