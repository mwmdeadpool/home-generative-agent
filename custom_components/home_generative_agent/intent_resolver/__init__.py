"""
Three-tier intent resolver for fast HA command execution.

Tier 1: Vector similarity match against embedded entities (~50-100ms)
Tier 2: Small LLM classifier for compound/ambiguous intents (~200-300ms)
Tier 3: Fall through to full LangGraph agent (2-10s)

Based on: reddit.com/r/homelab/comments/1rhfce5/
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)


class IntentTier(IntEnum):
    """Resolution tier."""

    DIRECT = 1  # Single entity + action, high confidence
    COMPOUND = 2  # Multiple commands or needs splitting
    FULL_LLM = 3  # Conversational, non-HA, or low confidence


@dataclass
class ResolvedAction:
    """A single resolved HA action."""

    domain: str
    service: str
    entity_id: str
    friendly_name: str
    data: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0


@dataclass
class IntentResult:
    """Result of intent resolution."""

    tier: IntentTier
    actions: list[ResolvedAction] = field(default_factory=list)
    non_ha_text: str | None = None  # Text that needs full LLM
    original_text: str = ""
    resolution_ms: float = 0.0


# Action verb mappings for deterministic matching
ACTION_VERBS: dict[str, dict[str, str]] = {
    # Lights
    "turn on": {"domain": "light", "service": "turn_on"},
    "turn off": {"domain": "light", "service": "turn_off"},
    "switch on": {"domain": "light", "service": "turn_on"},
    "switch off": {"domain": "light", "service": "turn_off"},
    "dim": {"domain": "light", "service": "turn_on"},  # + brightness data
    "brighten": {"domain": "light", "service": "turn_on"},  # + brightness data
    # Switches
    "toggle": {"domain": "homeassistant", "service": "toggle"},
    # Locks
    "lock": {"domain": "lock", "service": "lock"},
    "unlock": {"domain": "lock", "service": "unlock"},
    # Covers/garage
    "open": {"domain": "cover", "service": "open_cover"},
    "close": {"domain": "cover", "service": "close_cover"},
    # Climate
    "set temperature": {"domain": "climate", "service": "set_temperature"},
    "heat": {"domain": "climate", "service": "set_hvac_mode"},
    "cool": {"domain": "climate", "service": "set_hvac_mode"},
    # Media
    "play": {"domain": "media_player", "service": "media_play"},
    "pause": {"domain": "media_player", "service": "media_pause"},
    "stop": {"domain": "media_player", "service": "media_stop"},
    "volume up": {"domain": "media_player", "service": "volume_up"},
    "volume down": {"domain": "media_player", "service": "volume_down"},
    "mute": {"domain": "media_player", "service": "volume_mute"},
    # Scenes
    "activate": {"domain": "scene", "service": "turn_on"},
    # Scripts
    "run": {"domain": "script", "service": "turn_on"},
    # Fan
    "fan on": {"domain": "fan", "service": "turn_on"},
    "fan off": {"domain": "fan", "service": "turn_off"},
}

# Phrases that indicate non-HA / conversational intent
NON_HA_INDICATORS = [
    "what is", "what's", "who is", "who's", "why", "how does",
    "tell me about", "explain", "describe", "define",
    "what time", "what day", "weather", "forecast",
    "recipe", "directions", "calculate", "remind me",
    "joke", "story", "sing", "thank", "hello", "hi ",
    "good morning", "good night", "how are you",
]
