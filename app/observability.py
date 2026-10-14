import os

from phoenix.otel import register


def init_observability():
    register(
        project_name=os.environ.get("PHOENIX_PROJECT_NAME", "production-agentic-system"),
        endpoint=os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "http://phoenix:4317"),
        protocol="grpc",
        auto_instrument=True,
        batch=True,
    )
