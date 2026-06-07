"""Unit tests for the application lifespan / settings validation hook."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from app.shared import lifespan as lifespan_mod
from app.shared.config import Settings
from app.shared.lifespan import lifespan, validate_settings


def _make_validation_error() -> ValidationError:
    try:
        Settings(OPENAI_TEMPERATURE=99)  # out of bounds -> raises
    except ValidationError as exc:
        return exc
    raise AssertionError("expected a ValidationError")


def test_validate_settings_ok():
    # Default/.env settings are valid; should not raise.
    validate_settings()


def test_validate_settings_wraps_validation_error_as_runtime_error():
    err = _make_validation_error()
    with patch.object(lifespan_mod, "get_settings", side_effect=err):
        with pytest.raises(RuntimeError, match="Invalid configuration"):
            validate_settings()


async def test_lifespan_runs_validation_and_yields():
    with patch.object(lifespan_mod, "get_settings", return_value=MagicMock()) as gs:
        async with lifespan(MagicMock()):
            pass
    gs.assert_called_once()
