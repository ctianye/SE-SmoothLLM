"""FastAPI application factory."""

from fastapi import FastAPI

from se_smoothllm import __version__


def create_app() -> FastAPI:
    """Create the local SE-SmoothLLM HTTP application."""

    application = FastAPI(
        title="SE-SmoothLLM",
        version=__version__,
        description="Local API for early-stopping SmoothLLM defenses.",
    )

    @application.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    return application


app = create_app()
