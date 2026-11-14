"""Shared pytest fixtures for every test.

The app's startup (app/assistant/settings.py load_settings) runs load_dotenv(ENV_FILE), which
would copy the developer's real .env (ROBOFLOW_*, ANTHROPIC_API_KEY, AUTOLABEL_MCP_TOKEN, ...)
into os.environ and undo variables a test removed. Tests must not depend on that file, so
ENV_FILE points at a path that does not exist, for every test. App behaviour is unchanged.
"""

import pytest


@pytest.fixture(autouse=True)
def no_real_dotenv_at_startup(tmp_path, monkeypatch):
    from app.assistant import settings

    monkeypatch.setattr(settings, "ENV_FILE", tmp_path / "no-such-dir" / ".env")
