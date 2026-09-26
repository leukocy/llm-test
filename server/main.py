"""ASGI entry point. Configuration failure prevents serving requests."""

from server.api import create_app

app = create_app()
