from pathlib import Path

import pytest

import reachy_mini_conversation_app.config as config_mod


def test_config_raises_on_external_profile_name_collision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Config should fail fast when external/built-in profile names collide."""
    external_profiles = tmp_path / "external_profiles"
    external_profiles.mkdir(parents=True)
    (external_profiles / "default").mkdir()

    monkeypatch.setattr(config_mod.Config, "PROFILES_DIRECTORY", external_profiles)
    monkeypatch.setattr(config_mod.Config, "TOOLS_DIRECTORY", None)

    with pytest.raises(RuntimeError, match="Ambiguous profile names"):
        config_mod.Config()


def test_config_raises_on_external_profile_name_collision_with_builtin_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Config should treat compact built-in profile names as reserved."""
    external_profiles = tmp_path / "external_profiles"
    external_profiles.mkdir(parents=True)
    (external_profiles / "mad_scientist_assistant").mkdir()

    monkeypatch.setattr(config_mod.Config, "PROFILES_DIRECTORY", external_profiles)
    monkeypatch.setattr(config_mod.Config, "TOOLS_DIRECTORY", None)

    with pytest.raises(RuntimeError, match="Ambiguous profile names"):
        config_mod.Config()


def test_config_raises_on_external_tool_name_collision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Config should fail fast when external/built-in tool names collide."""
    external_tools = tmp_path / "external_tools"
    external_tools.mkdir(parents=True)
    (external_tools / "dance.py").write_text("# collision with built-in dance tool\n", encoding="utf-8")

    monkeypatch.setattr(config_mod.Config, "PROFILES_DIRECTORY", config_mod.DEFAULT_PROFILES_DIRECTORY)
    monkeypatch.setattr(config_mod.Config, "TOOLS_DIRECTORY", external_tools)

    with pytest.raises(RuntimeError, match="Ambiguous tool names"):
        config_mod.Config()


def test_config_raises_when_selected_external_profile_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Config should fail fast when selected profile is absent from external root."""
    external_profiles = tmp_path / "external_profiles"
    external_profiles.mkdir(parents=True)

    monkeypatch.setattr(config_mod.Config, "REACHY_MINI_CUSTOM_PROFILE", "missing_profile")
    monkeypatch.setattr(config_mod.Config, "PROFILES_DIRECTORY", external_profiles)
    monkeypatch.setattr(config_mod.Config, "TOOLS_DIRECTORY", None)

    with pytest.raises(RuntimeError, match="Selected profile 'missing_profile' was not found"):
        config_mod.Config()


def test_backend_provider_defaults_to_hf_when_unset() -> None:
    """Non-Gemini models should default to the Hugging Face backend."""
    assert config_mod._normalize_backend_provider(None, None) == config_mod.HF_BACKEND
    assert config_mod._normalize_backend_provider(None, "gpt-realtime") == config_mod.HF_BACKEND
    assert config_mod._normalize_backend_provider(None, "gemini-3.1-flash-live-preview") == config_mod.GEMINI_BACKEND


def test_huggingface_backend_does_not_resolve_model_name() -> None:
    """Hugging Face should rely on the server's model selection."""
    assert config_mod._resolve_model_name(config_mod.HF_BACKEND, None) == ""
    assert config_mod._resolve_model_name(config_mod.HF_BACKEND, "gpt-realtime") == ""
