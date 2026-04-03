import asyncio
import io
import json
import sys
import threading
import traceback
from typing import Set

import websockets

# Config stdout is UTF-8 编码（解决 Windows GBK 编码问题）
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


class LogServer:
    """
    WebSocket Log Server for real-time log streaming.
    Singleton pattern to ensure only one server runs.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.clients: set = set()
                cls._instance.loop = None
                cls._instance.thread = None
                cls._instance.running = False
        return cls._instance

    def start(self, host="localhost", port=8765):
        """Start the WebSocket server in a background thread."""
        with self._lock:
            if self.running:
                return

            self.running = True
            self.thread = threading.Thread(target=self._run_server, args=(host, port), daemon=True)
            self.thread.start()

    def _run_server(self, host, port):
        """Internal method to run the event loop."""
        try:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

            async def server_routine():
                try:
                    async with websockets.serve(self._handler, host, port):
                        print(f"📡 WebSocket Log Server running on ws://{host}:{port}")
                        await asyncio.Future()  # run forever
                except OSError as e:
                    if e.errno == 10048: # Address already in use (Windows)
                        print(f"⚠️ Port {port} is busy. Assuming server is already running.")
                    else:
                        raise e

            self.loop.run_until_complete(server_routine())

        except Exception as e:
            print(f"❌ WebSocket Server Error: {e}")
            traceback.print_exc()
            self.running = False
        finally:
            if self.loop and self.loop.is_running():
                self.loop.close()

    async def _handler(self, websocket):
        """Handle individual client connections."""
        self.clients.add(websocket)
        try:
            # Send a welcome message
            await websocket.send(json.dumps({
                "type": "system",
                "message": "Connected to Benchmark Log Stream"
            }))
            await websocket.wait_closed()
        except Exception:
            pass
        finally:
            self.clients.remove(websocket)

    def broadcast(self, log_entry: dict):
        """Broadcast a log entry to all connected clients."""
        if not self.running:
            return

        if self.loop and self.loop.is_running():
            try:
                # Ensure log_entry is JSON serializable
                message = json.dumps(log_entry, default=str, ensure_ascii=False)
                asyncio.run_coroutine_threadsafe(self._broadcast_message(message), self.loop)
            except Exception:
                # Silent failure to avoid spamming stdout
                pass

    async def _broadcast_message(self, message):
        """Async method to send message to all clients."""
        if self.clients:
            # Create tasks for all clients
            tasks = [asyncio.create_task(client.send(message)) for client in self.clients]
            if tasks:
                await asyncio.wait(tasks, timeout=1.0)

# Global instance
log_server = LogServer()
