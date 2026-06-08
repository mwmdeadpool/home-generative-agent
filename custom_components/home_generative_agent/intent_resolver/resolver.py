"""
Three-tier intent resolver.

Tier 1: Deterministic verb extraction + vector entity matching
Tier 2: Small LLM splits compound intents
Tier 3: Falls through to full LangGraph
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from homeassistant.core import HomeAssistant

from . import (
    ACTION_VERBS,
    NON_HA_INDICATORS,
    IntentResult,
    IntentTier,
    ResolvedAction,
)
from .embedder import search_entities

_LOGGER = logging.getLogger(__name__)

# Confidence thresholds
TIER1_SCORE_THRESHOLD = 0.75  # Must be very confident for direct execution
TIER2_SCORE_THRESHOLD = 0.65  # Lower bar when LLM confirms intent
COMPOUND_INDICATORS = [" and ", " also ", " then ", " plus ", ", "]


def _extract_action_verb(text: str) -> tuple[str | None, dict[str, str] | None]:
    """
    Extract the action verb and its HA mapping from user text.

    Returns (matched_verb, action_mapping) or (None, None).
    """
    text_lower = text.lower().strip()
    # Try longest verbs first for specificity
    sorted_verbs = sorted(ACTION_VERBS.keys(), key=len, reverse=True)
    for verb in sorted_verbs:
        if text_lower.startswith(verb) or f" {verb} " in f" {text_lower} ":
            return verb, ACTION_VERBS[verb]
    return None, None


def _is_conversational(text: str) -> bool:
    """Check if text is conversational (non-HA) intent."""
    text_lower = text.lower().strip()
    return any(indicator in text_lower for indicator in NON_HA_INDICATORS)


def _is_compound(text: str) -> bool:
    """Check if text contains multiple commands."""
    text_lower = text.lower()
    return any(indicator in text_lower for indicator in COMPOUND_INDICATORS)


def _extract_brightness(text: str) -> int | None:
    """Extract brightness percentage from text."""
    text_lower = text.lower()
    # "dim to 50%", "set brightness to 75", "50 percent"
    match = re.search(r"(\d{1,3})\s*%|brightness\s+(?:to\s+)?(\d{1,3})", text_lower)
    if match:
        val = int(match.group(1) or match.group(2))
        return max(0, min(255, int(val * 255 / 100)))
    # "dim" without number = 30%, "brighten" = 100%
    if "dim" in text_lower and not re.search(r"\d", text_lower):
        return 77  # ~30%
    if "brighten" in text_lower and not re.search(r"\d", text_lower):
        return 255
    return None


def _extract_temperature(text: str) -> float | None:
    """Extract temperature from text."""
    match = re.search(r"(\d{2,3})\s*(?:degrees?|°|f)?", text.lower())
    if match:
        return float(match.group(1))
    return None


async def resolve_intent(
    text: str,
    hass: HomeAssistant,
    embedding_model: Any,
    qdrant_url: str = "http://localhost:6333",
    collection_name: str = "hga_entities",
) -> IntentResult:
    """
    Resolve user intent through the 3-tier system.

    Returns IntentResult with tier, actions, and timing.
    """
    start = time.monotonic()

    # Quick check: is this conversational / non-HA?
    if _is_conversational(text):
        elapsed = (time.monotonic() - start) * 1000
        _LOGGER.debug("Tier 3 (conversational): %.1fms - %s", elapsed, text)
        return IntentResult(
            tier=IntentTier.FULL_LLM,
            original_text=text,
            non_ha_text=text,
            resolution_ms=elapsed,
        )

    # Check for compound commands
    is_compound = _is_compound(text)

    # Extract action verb
    verb, action_map = _extract_action_verb(text)

    if not verb and not is_compound:
        # No recognized verb and not compound — could still be implicit
        # ("bedroom lights" might mean "turn on bedroom lights")
        # Try vector search anyway
        pass

    # Vector search for matching entities
    try:
        threshold = TIER2_SCORE_THRESHOLD if is_compound else TIER1_SCORE_THRESHOLD
        matches = await search_entities(
            text,
            embedding_model,
            qdrant_url=qdrant_url,
            collection_name=collection_name,
            limit=5 if is_compound else 3,
            score_threshold=threshold,
        )
    except Exception:
        _LOGGER.exception("Vector search failed, falling through to Tier 3")
        elapsed = (time.monotonic() - start) * 1000
        return IntentResult(
            tier=IntentTier.FULL_LLM,
            original_text=text,
            resolution_ms=elapsed,
        )

    if not matches:
        elapsed = (time.monotonic() - start) * 1000
        _LOGGER.debug("Tier 3 (no matches): %.1fms - %s", elapsed, text)
        return IntentResult(
            tier=IntentTier.FULL_LLM,
            original_text=text,
            resolution_ms=elapsed,
        )

    # Compound intents → Tier 2
    if is_compound:
        elapsed = (time.monotonic() - start) * 1000
        _LOGGER.debug(
            "Tier 2 (compound): %.1fms - %s - %d matches",
            elapsed,
            text,
            len(matches),
        )
        # Build partial actions from what we can resolve
        actions = []
        for match in matches:
            domain = match["domain"]
            # Infer service from verb or domain default
            if action_map and (
                action_map["domain"] == domain
                or action_map["domain"] == "homeassistant"
            ):
                service = action_map["service"]
            else:
                # Default actions per domain
                service = _default_service_for_domain(domain, text)
            actions.append(
                ResolvedAction(
                    domain=domain,
                    service=service,
                    entity_id=match["entity_id"],
                    friendly_name=match["friendly_name"],
                    confidence=match["score"],
                )
            )
        return IntentResult(
            tier=IntentTier.COMPOUND,
            actions=actions,
            original_text=text,
            resolution_ms=elapsed,
        )

    # Single intent — Tier 1 if high confidence
    best = matches[0]
    if best["score"] >= TIER1_SCORE_THRESHOLD and (verb or action_map):
        domain = best["domain"]
        if action_map:
            if action_map["domain"] in (domain, "homeassistant"):
                service = action_map["service"]
            else:
                service = _default_service_for_domain(domain, text)
        else:
            service = _default_service_for_domain(domain, text)

        # Build service data
        data: dict[str, Any] = {}
        if domain == "light" and service == "turn_on":
            brightness = _extract_brightness(text)
            if brightness is not None:
                data["brightness"] = brightness
        elif domain == "climate" and service == "set_temperature":
            temp = _extract_temperature(text)
            if temp is not None:
                data["temperature"] = temp

        action = ResolvedAction(
            domain=domain,
            service=service,
            entity_id=best["entity_id"],
            friendly_name=best["friendly_name"],
            data=data,
            confidence=best["score"],
        )

        elapsed = (time.monotonic() - start) * 1000
        _LOGGER.info(
            "Tier 1 (direct): %.1fms - '%s' → %s.%s(%s) [%.2f]",
            elapsed,
            text,
            domain,
            service,
            best["entity_id"],
            best["score"],
        )
        return IntentResult(
            tier=IntentTier.DIRECT,
            actions=[action],
            original_text=text,
            resolution_ms=elapsed,
        )

    # Not confident enough for direct execution
    elapsed = (time.monotonic() - start) * 1000
    _LOGGER.debug(
        "Tier 3 (low confidence): %.1fms - %s - best=%.2f",
        elapsed,
        text,
        best["score"],
    )
    return IntentResult(
        tier=IntentTier.FULL_LLM,
        original_text=text,
        resolution_ms=elapsed,
    )


def _default_service_for_domain(domain: str, text: str) -> str:
    """Infer the most likely service for a domain based on context."""
    text_lower = text.lower()
    defaults = {
        "light": "turn_off"
        if any(w in text_lower for w in ["off", "dim"])
        else "turn_on",
        "switch": "turn_off" if "off" in text_lower else "turn_on",
        "fan": "turn_off" if "off" in text_lower else "turn_on",
        "cover": "close_cover" if "close" in text_lower else "open_cover",
        "lock": "unlock" if "unlock" in text_lower else "lock",
        "climate": "set_temperature",
        "media_player": "media_pause"
        if any(w in text_lower for w in ["pause", "stop"])
        else "media_play",
        "scene": "turn_on",
        "script": "turn_on",
        # automation excluded from fast intent — too easily confused with device entities
        "vacuum": "start" if "start" in text_lower else "return_to_base",
        "alarm_control_panel": "alarm_disarm"
        if "disarm" in text_lower
        else "alarm_arm_away",
    }
    return defaults.get(domain, "turn_on")
