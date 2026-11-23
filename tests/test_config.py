import pytest

from ml_diff.config import Settings


def test_env_var_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_LOG_LEVEL", "DEBUG")
    settings = Settings()
    assert settings.log_level == "DEBUG"
