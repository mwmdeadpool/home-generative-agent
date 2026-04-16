"""
Execute resolved intents directly via HA service calls.

Handles Tier 1 (direct) and Tier 2 (compound) execution
without going through the full LLM pipeline.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from homeassistant.components import conversation
from homeassistant.core import HomeAssistant
from homeassistant.helpers import intent

from . import IntentResult, ResolvedAction

_LOGGER = logging.getLogger(__name__)

# Friendly action descriptions for responses
ACTION_DESCRIPTIONS = {
    "turn_on": "turned on",
    "turn_off": "turned off",
    "toggle": "toggled",
    "lock": "locked",
    "unlock": "unlocked",
    "open_cover": "opened",
    "close_cover": "closed",
    "set_temperature": "set temperature for",
    "media_play": "started playing",
    "media_pause": "paused",
    "media_stop": "stopped",
    "volume_up": "turned up volume on",
    "volume_down": "turned down volume on",
    "volume_mute": "muted",
    "trigger": "triggered",
    "start": "started",
    "return_to_base": "sent home",
    "alarm_arm_away": "armed",
    "alarm_disarm": "disarmed",
}


async def execute_direct(
    hass: HomeAssistant,
    result: IntentResult,
    language: str = "en",
) -> conversation.ConversationResult:
    """Execute a Tier 1 direct intent and return a conversation result."""
    if not result.actions:
        return _error_result("No actions to execute.", language)

    action = result.actions[0]
    try:
        service_data: dict[str, Any] = {"entity_id": action.entity_id}
        service_data.update(action.data)

        await hass.services.async_call(
            action.domain,
            action.service,
            service_data,
            blocking=True,
        )

        desc = ACTION_DESCRIPTIONS.get(action.service, action.service)
        speech = f"Done! I've {desc} {action.friendly_name}."

        if action.data.get("brightness"):
            pct = round(action.data["brightness"] / 255 * 100)
            speech = f"Done! I've set {action.friendly_name} to {pct}% brightness."
        elif action.data.get("temperature"):
            speech = (
                f"Done! I've set {action.friendly_name} "
                f"to {action.data['temperature']}°."
            )

        _LOGGER.info(
            "Tier 1 executed: %s.%s(%s) in %.0fms",
            action.domain, action.service, action.entity_id,
            result.resolution_ms,
        )

    except Exception as err:
        _LOGGER.exception("Tier 1 execution failed for %s", action.entity_id)
        speech = f"Sorry, I couldn't {action.service} {action.friendly_name}: {err}"

    return _speech_result(speech, language)


async def execute_compound(
    hass: HomeAssistant,
    result: IntentResult,
    language: str = "en",
) -> conversation.ConversationResult:
    """Execute Tier 2 compound intents in parallel where possible."""
    if not result.actions:
        return _error_result("No actions resolved.", language)

    successes: list[str] = []
    failures: list[str] = []

    # Group into independent (parallel) and dependent (sequential) actions
    # For now, execute all in parallel with a small debounce
    tasks = []
    for action in result.actions:
        tasks.append(_execute_single(hass, action))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    for action, task_result in zip(result.actions, results):
        if isinstance(task_result, Exception):
            failures.append(f"{action.friendly_name} ({task_result})")
        else:
            desc = ACTION_DESCRIPTIONS.get(action.service, action.service)
            successes.append(f"{desc} {action.friendly_name}")

    # Build response
    parts = []
    if successes:
        parts.append("Done! I've " + ", ".join(successes) + ".")
    if failures:
        parts.append("Failed: " + ", ".join(failures) + ".")

    speech = " ".join(parts) if parts else "No actions were executed."

    _LOGGER.info(
        "Tier 2 executed: %d/%d actions in %.0fms",
        len(successes), len(result.actions), result.resolution_ms,
    )

    return _speech_result(speech, language)


async def _execute_single(
    hass: HomeAssistant,
    action: ResolvedAction,
) -> None:
    """Execute a single HA service call."""
    service_data: dict[str, Any] = {"entity_id": action.entity_id}
    service_data.update(action.data)
    await hass.services.async_call(
        action.domain,
        action.service,
        service_data,
        blocking=True,
    )


def _speech_result(
    speech: str,
    language: str,
) -> conversation.ConversationResult:
    """Create a successful conversation result."""
    intent_response = intent.IntentResponse(language=language)
    intent_response.async_set_speech(speech)
    return conversation.ConversationResult(response=intent_response)


def _error_result(
    message: str,
    language: str,
) -> conversation.ConversationResult:
    """Create an error conversation result."""
    intent_response = intent.IntentResponse(language=language)
    intent_response.async_set_error(
        intent.IntentResponseErrorCode.UNKNOWN, message
    )
    return conversation.ConversationResult(response=intent_response)
