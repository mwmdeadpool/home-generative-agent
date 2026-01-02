"""Langgraph tools for Home Generative Agent."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import re
from collections.abc import Mapping
import calendar
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, cast
from zoneinfo import ZoneInfo

import aiofiles
import async_timeout
import dateparser
import homeassistant.util.dt as dt_util
import voluptuous as vol
import yaml
from homeassistant.components import camera
from homeassistant.components.automation.config import _async_validate_config_item
from homeassistant.components.automation.const import DOMAIN as AUTOMATION_DOMAIN
from homeassistant.components.recorder import history as recorder_history
from homeassistant.components.recorder import statistics as recorder_statistics
from homeassistant.config import AUTOMATION_CONFIG_PATH
from homeassistant.const import (
    ATTR_FRIENDLY_NAME,
    SERVICE_RELOAD,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
)
from homeassistant.core import State
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm
from homeassistant.helpers.recorder import get_instance as get_recorder_instance
from homeassistant.helpers.recorder import session_scope as recorder_session_scope
from homeassistant.helpers.httpx_client import get_async_client  # Added import
from homeassistant.util import ulid
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig  # noqa: TC002
from langchain_core.tools import InjectedToolArg, tool
from langgraph.prebuilt import InjectedStore
from langgraph.store.base import BaseStore  # noqa: TC002
from voluptuous import MultipleInvalid

from ..const import (  # noqa: TID252
    AUTOMATION_TOOL_BLUEPRINT_NAME,
    AUTOMATION_TOOL_EVENT_REGISTERED,
    CONF_CRITICAL_ACTION_PIN_HASH,
    CONF_CRITICAL_ACTION_PIN_SALT,
    CONF_GOOGLE_PLACES_API_KEY,
    CONF_LIGHTRAG_API_KEY,
    CONF_LIGHTRAG_URL,
    CONF_NOTIFY_SERVICE,
    CONF_PLEX_ENABLED,
    CONF_PLEX_SERVER_URL,
    CONF_PLEX_TOKEN,
    CONF_REDDIT_CLIENT_ID,
    CONF_REDDIT_CLIENT_SECRET,
    CONF_REDDIT_USER_AGENT,
    CONF_WIKIPEDIA_ENABLED,
    CRITICAL_PIN_MAX_LEN,
    CRITICAL_PIN_MIN_LEN,
    HISTORY_TOOL_CONTEXT_LIMIT,
    HISTORY_TOOL_PURGE_KEEP_DAYS,
    VLM_IMAGE_HEIGHT,
    VLM_IMAGE_WIDTH,
    VLM_SYSTEM_PROMPT,
    VLM_USER_KW_TEMPLATE,
    VLM_USER_PROMPT,
)
from ..core.utils import extract_final, verify_pin  # noqa: TID252
from .helpers import (
    ConfigurableData,
    maybe_fill_lock_entity,
    normalize_intent_for_alarm,
    normalize_intent_for_lock,
    sanitize_tool_args,
)

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping

    from homeassistant.core import HomeAssistant
    from langchain_core.language_models import BaseMessage, LanguageModelInput
    from langchain_core.runnables.base import RunnableSerializable

LOGGER = logging.getLogger(__name__)


def _map_alarm_service(tool_name: str, requested_state: str) -> str:
    """Map requested alarm state or intent to the HA service."""
    service_map = {
        "arm_home": "alarm_arm_home",
        "home": "alarm_arm_home",
        "armed_home": "alarm_arm_home",
        "arm_away": "alarm_arm_away",
        "away": "alarm_arm_away",
        "armed_away": "alarm_arm_away",
        "arm_night": "alarm_arm_night",
        "night": "alarm_arm_night",
        "armed_night": "alarm_arm_night",
        "vacation": "alarm_arm_vacation",
        "arm_vacation": "alarm_arm_vacation",
        "armed_vacation": "alarm_arm_vacation",
        "custom_bypass": "alarm_arm_custom_bypass",
        "arm_custom_bypass": "alarm_arm_custom_bypass",
        "armed_custom_bypass": "alarm_arm_custom_bypass",
        "disarm": "alarm_disarm",
        "off": "alarm_disarm",
        "disarmed": "alarm_disarm",
    }
    default_service = "alarm_arm_home" if tool_name == "HassTurnOn" else "alarm_disarm"
    return service_map.get(requested_state, default_service)


def _extract_alarm_code(tool_args: dict[str, Any]) -> str:
    """Pull the alarm code from the best available slot."""
    code = str(tool_args.get("code", "")).strip()
    if code:
        return code

    dc_val = tool_args.get("device_class")
    if (
        isinstance(dc_val, list)
        and len(dc_val) == 1
        and str(dc_val[0]).strip().isdigit()
    ):
        code = str(dc_val[0]).strip()
        tool_args["device_class"] = []
    elif isinstance(dc_val, str) and dc_val.strip().isdigit():
        code = dc_val.strip()
        tool_args["device_class"] = []
    elif (floor := tool_args.get("floor")) and str(floor).strip().isdigit():
        code = str(floor).strip()
    elif name := tool_args.get("name"):
        tokens = str(name).strip().split()
        if tokens and tokens[-1].isdigit():
            code = tokens[-1]

    if not code:
        msg = "Alarm code required to arm/disarm."
        raise HomeAssistantError(msg)
    tool_args["code"] = code
    return code


def _resolve_alarm_entity(hass: HomeAssistant, tool_args: dict[str, Any]) -> str:
    """Resolve alarm entity_id from args or the environment."""
    entity_id = tool_args.get("entity_id")
    alarm_entities = hass.states.async_entity_ids("alarm_control_panel")
    if entity_id not in alarm_entities:
        entity_id = None

    if not entity_id and (name := tool_args.get("name")):
        slug = str(name).strip().lower().replace(" ", "_")
        parts = [p for p in slug.split("_") if p]
        if parts and parts[-1].isdigit():
            parts = parts[:-1]
        if parts:
            candidate = f"alarm_control_panel.{'_'.join(parts)}"
            if candidate in alarm_entities:
                entity_id = candidate

    if not entity_id and len(alarm_entities) == 1:
        entity_id = alarm_entities[0]

    if not entity_id:
        msg = "Missing alarm entity_id; cannot arm/disarm."
        raise HomeAssistantError(msg)
    return entity_id


def _infer_alarm_state(state_obj: State | None) -> str:
    """Infer a simplified alarm state from the HA state object."""
    if not state_obj:
        return "unknown"
    current_state = str(state_obj.state).lower()
    if current_state in {
        "armed_home",
        "armed_away",
        "armed_night",
        "armed_vacation",
        "armed_custom_bypass",
    }:
        return "armed"
    if current_state in {
        "disarmed",
        "pending",
        "arming",
        "triggered",
        "disarming",
    }:
        return current_state
    return "unknown"


def _alarm_warning(*, is_arm_request: bool, inferred_status: str) -> str | None:
    """Build a warning message if the alarm state is unexpected."""
    if inferred_status == "unknown":
        return "Alarm panel state is unknown after the request."
    if is_arm_request and inferred_status == "disarmed":
        return """
        Arming requested but the panel still shows disarmed; it may still be updating.
        """
    if not is_arm_request and inferred_status != "disarmed":
        return (
            f"Disarm requested but the panel reports {inferred_status}; "
            "please verify at the panel."
        )
    return None


async def _perform_alarm_control(
    hass: HomeAssistant, tool_name: str, tool_args: dict[str, Any]
) -> dict[str, Any]:
    """Arm or disarm an alarm_control_panel entity and report its state."""
    requested_state = str(tool_args.get("state") or "").lower()
    resolved_service = _map_alarm_service(tool_name, requested_state)
    code = _extract_alarm_code(tool_args)
    entity_id = _resolve_alarm_entity(hass, tool_args)

    data: dict[str, Any] = {"entity_id": entity_id, "code": code}
    await hass.services.async_call(
        "alarm_control_panel",
        resolved_service,
        data,
        blocking=True,
    )

    # Give HA a moment to update state before reading it back.
    await asyncio.sleep(2.0)

    state_obj = hass.states.get(entity_id)
    inferred_status = _infer_alarm_state(state_obj)
    is_arm_request = resolved_service.startswith("alarm_arm")
    warning = _alarm_warning(
        is_arm_request=is_arm_request, inferred_status=inferred_status
    )
    result_text = warning or (
        f"Alarm service {resolved_service} completed; panel state: {inferred_status}."
    )
    expected_states = (
        {"armed", "arming", "pending"} if is_arm_request else {"disarmed", "disarming"}
    )

    return {
        "success": warning is None and inferred_status in expected_states,
        "entity_id": entity_id,
        "service": resolved_service,
        "inferred_state": inferred_status,
        "warning": warning,
        "result_text": result_text,
    }


async def _get_camera_image(hass: HomeAssistant, camera_name: str) -> bytes | None:
    """Get an image from a given camera."""
    camera_entity_id: str = f"camera.{camera_name.lower()}"
    state = hass.states.get(camera_entity_id)
    if state and state.state in (STATE_UNAVAILABLE, STATE_UNKNOWN):
        LOGGER.warning(
            "Camera %s is %s; skipping capture", camera_entity_id, state.state
        )
        return None

    max_attempts = 3
    timeout_sec = 5
    backoff_base = 0.35

    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            await asyncio.sleep(backoff_base * (attempt - 1))

        try:
            async with async_timeout.timeout(timeout_sec):
                image = await camera.async_get_image(
                    hass=hass,
                    entity_id=camera_entity_id,
                    width=VLM_IMAGE_WIDTH,
                    height=VLM_IMAGE_HEIGHT,
                )
        except TimeoutError:
            LOGGER.warning(
                "Timed out (%ss) getting image from %s (attempt %s/%s)",
                timeout_sec,
                camera_entity_id,
                attempt,
                max_attempts,
            )
            continue
        except HomeAssistantError:
            LOGGER.exception(
                "Error getting image from camera %s (attempt %s/%s)",
                camera_entity_id,
                attempt,
                max_attempts,
            )
            continue

        if image is None or image.content is None:
            LOGGER.warning(
                "Camera %s returned empty image (attempt %s/%s)",
                camera_entity_id,
                attempt,
                max_attempts,
            )
            continue

        return image.content

    LOGGER.error(
        "Failed to capture image from camera %s after %s attempts",
        camera_entity_id,
        max_attempts,
    )
    return None


def _prompt_func(data: dict[str, Any]) -> list[AnyMessage]:
    system = data["system"]
    text = data["text"]
    image = data["image"]
    prev_text = data.get("prev_text")

    # Build the user content (text first, then optional previous frame text, then image)
    content_parts: list[str | dict[str, Any]] = []

    # Main instruction text
    content_parts.append({"type": "text", "text": text})

    # OPTIONAL: previous frame's one-line description to aid motion/direction grounding
    if prev_text:
        # Keep it short and explicit that it is text-only context, not metadata
        content_parts.append(
            {"type": "text", "text": f'Previous frame (text only): "{prev_text}"'}
        )

    # Image payload last
    content_parts.append(
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image}"},
        }
    )

    return [SystemMessage(content=system), HumanMessage(content=content_parts)]


async def analyze_image(
    vlm_model: RunnableSerializable[LanguageModelInput, BaseMessage],
    image: bytes,
    detection_keywords: list[str] | None = None,
    prev_text: str | None = None,
) -> str:
    """Analyze an image with the preconfigured VLM model."""
    await asyncio.sleep(0)  # keep the event loop snappy

    image_data = base64.b64encode(image).decode("utf-8")
    chain = _prompt_func | vlm_model

    if detection_keywords is not None:
        prompt = VLM_USER_KW_TEMPLATE.format(key_words=" or ".join(detection_keywords))
    else:
        prompt = VLM_USER_PROMPT

    try:
        resp = await chain.ainvoke(
            {
                "system": VLM_SYSTEM_PROMPT,
                "text": prompt,
                "image": image_data,
                "prev_text": prev_text,
            }
        )
    except HomeAssistantError:
        msg = "Error analyzing image with VLM model."
        LOGGER.exception(msg)
        return msg

    LOGGER.debug("Raw VLM model response: %s", resp)

    return extract_final(getattr(resp, "content", "") or "")


@tool(parse_docstring=True)
async def get_and_analyze_camera_image(  # noqa: D417
    camera_name: str,
    detection_keywords: list[str] | None = None,
    *,
    # Hide these arguments from the model.
    config: Annotated[RunnableConfig, InjectedToolArg()],
) -> str:
    """
    Get a camera image and perform scene analysis on it.

    Args:
        camera_name: Name of the camera for scene analysis.
        detection_keywords: Specific objects to look for in image, if any.
            For example, If user says "check the front porch camera for
            boxes and dogs", detection_keywords would be ["boxes", "dogs"].

    """
    if "configurable" not in config:
        return "Configuration not found. Please check your setup."

    hass = config["configurable"]["hass"]
    vlm_model = config["configurable"]["vlm_model"]
    image = await _get_camera_image(hass, camera_name)
    if image is None:
        return "Error getting image from camera."
    return await analyze_image(vlm_model, image, detection_keywords)


@tool(parse_docstring=True)
async def upsert_memory(  # noqa: D417
    content: str,
    context: str = "",
    *,
    memory_id: str = "",
    # Hide these arguments from the model.
    config: Annotated[RunnableConfig, InjectedToolArg()],
) -> str:
    """
    INSERT or UPDATE a memory about users in the database.

    You MUST use this tool to INSERT or UPDATE memories about users.
    Examples of memories are specific facts or concepts learned from interactions
    with users. If a memory conflicts with an existing one then just UPDATE the
    existing one by passing in "memory_id" and DO NOT create two memories that are
    the same. If the user corrects a memory then UPDATE it.

    Args:
        content: The main content of the memory.
            e.g., "I would like to learn french."
        context: Additional relevant context for the memory, if any.
            e.g., "This was mentioned while discussing career options in Europe."
        memory_id: The memory to overwrite.
            ONLY PROVIDE IF UPDATING AN EXISTING MEMORY.

    """
    if "configurable" not in config:
        return "Configuration not found. Please check your setup."

    mem0_client = config["configurable"].get("mem0_client")
    if mem0_client:
        await mem0_client.tools.save_memory(
            text=f"Content: {content}\nContext: {context}"
        )
        return "Stored memory in mem0."

    # Fallback to postgres store
    store = config["configurable"].get("store")
    if not store:
        return "No memory store configured."

    mem_id = memory_id or ulid.ulid_now()

    user_id = config["configurable"]["user_id"]
    await store.aput(
        namespace=(user_id, "memories"),
        key=str(mem_id),
        value={"content": content, "context": context},
    )
    return f"Stored memory {mem_id}"


@tool(parse_docstring=True)
async def add_automation(  # noqa: D417
    automation_yaml: str = "",
    time_pattern: str = "",
    message: str = "",
    *,
    # Hide these arguments from the model.
    config: Annotated[RunnableConfig, InjectedToolArg()],
) -> str:
    """
    Add an automation to Home Assistant.

    You are provided a Home Assistant blueprint as part of this tool if you need it.
    You MUST ONLY use the blueprint to create automations that involve camera image
    analysis. You MUST generate Home Assistant automation YAML for everything else.
    If using the blueprint you MUST provide the arguments "time_pattern" and "message"
    and DO NOT provide the argument "automation_yaml".

    Args:
        automation_yaml: A Home Assistant automation in valid YAML format.
            ONLY provide if NOT using the camera image analysis blueprint.
        time_pattern: Cron-like time pattern (e.g., /30 for "every 30 mins").
            ONLY provide if using the camera image analysis blueprint.
        message: Image analysis prompt (e.g.,"check the front porch camera for boxes")
            ONLY provide if using the camera image analysis blueprint.

    """
    if "configurable" not in config:
        return "Configuration not found. Please check your setup."

    hass = config["configurable"]["hass"]
    mobile_push_service = config["configurable"]["options"].get(CONF_NOTIFY_SERVICE)

    if time_pattern and message:
        automation_data = {
            "alias": message,
            "description": f"Created with blueprint {AUTOMATION_TOOL_BLUEPRINT_NAME}.",
            "use_blueprint": {
                "path": AUTOMATION_TOOL_BLUEPRINT_NAME,
                "input": {
                    "time_pattern": time_pattern,
                    "message": message,
                    "mobile_push_service": mobile_push_service or "",
                },
            },
        }
        automation_yaml = yaml.dump(automation_data)

    automation_parsed = yaml.safe_load(automation_yaml)
    ha_automation_config: dict[str, Any] = {"id": ulid.ulid_now()}
    if isinstance(automation_parsed, list):
        ha_automation_config.update(automation_parsed[0])
    if isinstance(automation_parsed, dict):
        ha_automation_config.update(automation_parsed)

    try:
        await _async_validate_config_item(
            hass=hass,
            config=ha_automation_config,
            raise_on_errors=True,
            warn_on_errors=False,
        )
    except (HomeAssistantError, MultipleInvalid) as err:
        return f"Invalid automation configuration {err}"

    async with aiofiles.open(
        Path(hass.config.config_dir) / AUTOMATION_CONFIG_PATH, encoding="utf-8"
    ) as f:
        ha_exsiting_automation_configs = await f.read()
        ha_exsiting_automations_yaml = yaml.safe_load(ha_exsiting_automation_configs)

    async with aiofiles.open(
        Path(hass.config.config_dir) / AUTOMATION_CONFIG_PATH,
        "a" if ha_exsiting_automations_yaml else "w",
        encoding="utf-8",
    ) as f:
        ha_automation_config_raw = yaml.dump(
            [ha_automation_config], allow_unicode=True, sort_keys=False
        )
        await f.write("\n" + ha_automation_config_raw)

    await hass.services.async_call(AUTOMATION_DOMAIN, SERVICE_RELOAD)
    hass.bus.async_fire(
        AUTOMATION_TOOL_EVENT_REGISTERED,
        {
            "automation_config": ha_automation_config,
            "raw_config": ha_automation_config_raw,
        },
    )

    return f"Added automation {ha_automation_config['id']}"


@tool(parse_docstring=True)
async def confirm_sensitive_action(  # noqa: D417, PLR0911
    action_id: str,
    pin: str,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg()],
    store: Annotated[BaseStore, InjectedStore()],  # noqa: ARG001
) -> str:
    """
    Confirm and execute a pending sensitive action that requires a PIN.

    Args:
        action_id: The action to confirm (provided by agent when it asked for a PIN).
        pin: The user-provided PIN.

    """
    if "configurable" not in config:
        return "Configuration not found. Please check your setup."

    cfg = cast("ConfigurableData", config.get("configurable", {}))
    opts = cfg.get("options", {})
    pin_hash = opts.get(CONF_CRITICAL_ACTION_PIN_HASH, "")
    salt = opts.get(CONF_CRITICAL_ACTION_PIN_SALT, "")
    pending_actions = cfg.get("pending_actions", {})
    provided_pin = str(pin or "").strip()
    requested_action_id = str(action_id or "").strip()

    resolved_action_id = _resolve_action_id(pending_actions, requested_action_id)
    if resolved_action_id is None:
        return "Pending action not found or expired."

    action, action_err = _load_pending_action(pending_actions, resolved_action_id)
    if action_err or not action:
        return action_err or "Pending action not found."

    if _is_wrong_user(cfg, action):
        return "Pending action belongs to a different user; please re-run the request."

    pin_err = _validate_pin_for_action(
        provided_pin=provided_pin,
        pin_hash=pin_hash,
        salt=salt,
        action=action,
    )
    if pin_err:
        return pin_err

    ha_llm_api, api_err = _ensure_api(cfg)
    if api_err or ha_llm_api is None:
        return api_err or "Home Assistant LLM API unavailable."

    result, exec_err = await _execute_pending_action(
        resolved_action_id, action, ha_llm_api, cfg
    )
    return result or exec_err or "Unable to process the confirmation."


def _resolve_action_id(
    pending_actions: dict[str, dict[str, Any]], requested_action_id: str
) -> str | None:
    """Return a valid action_id or None if it cannot be resolved safely."""
    if requested_action_id and requested_action_id in pending_actions:
        return requested_action_id
    if not requested_action_id and len(pending_actions) == 1:
        return next(iter(pending_actions))
    return None


def _load_pending_action(
    pending_actions: dict[str, dict[str, Any]], resolved_action_id: str
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate and return a pending action."""
    action = pending_actions.get(resolved_action_id)
    if not action:
        return None, "Pending action not found or expired."

    created_at = action.get("created_at")
    if created_at:
        try:
            ts = datetime.fromisoformat(created_at)
        except ValueError:
            pending_actions.pop(resolved_action_id, None)
            return None, "Pending action is invalid; please try again."
        if dt_util.utcnow() - ts > timedelta(minutes=10):
            pending_actions.pop(resolved_action_id, None)
            return None, "Pending action expired; please re-run the request."

    action.setdefault("attempts", 0)
    return action, None


def _validate_pin_for_action(
    *, provided_pin: str, pin_hash: str, salt: str, action: dict[str, Any]
) -> str | None:
    """Validate PIN format and value against stored hash/salt."""
    max_pin_attempts = 5
    if not pin_hash or not salt:
        return "No PIN configured; cannot confirm the action."
    if not provided_pin.isdigit() or not (
        CRITICAL_PIN_MIN_LEN <= len(provided_pin) <= CRITICAL_PIN_MAX_LEN
    ):
        return f"Invalid PIN. Use {CRITICAL_PIN_MIN_LEN}-{CRITICAL_PIN_MAX_LEN} digits."
    attempts = int(action.get("attempts", 0) or 0)
    if attempts >= max_pin_attempts:
        return "Too many incorrect attempts; please re-run the request."
    if not verify_pin(provided_pin, hashed=pin_hash, salt=salt):
        action["attempts"] = attempts + 1
        return "Incorrect PIN. Action not executed."
    return None


def _ensure_api(cfg: Mapping[str, Any]) -> tuple[Any | None, str | None]:
    """Return the HA LLM API or an error message."""
    ha_llm_api = cfg.get("ha_llm_api")
    if ha_llm_api is None:
        return None, "Home Assistant LLM API unavailable."
    return ha_llm_api, None


def _is_wrong_user(cfg: Mapping[str, Any], action: Mapping[str, Any]) -> bool:
    requester_id = cfg.get("user_id")
    action_owner = action.get("user")
    return bool(requester_id and action_owner and requester_id != action_owner)


async def _execute_pending_action(
    resolved_action_id: str,
    action: dict[str, Any],
    ha_llm_api: Any,
    cfg: Mapping[str, Any],
) -> tuple[str | None, str | None]:
    """Normalize args, execute the pending action, and clear it."""
    raw_tool_name = action.get("tool_name")
    if not isinstance(raw_tool_name, str) or not raw_tool_name:
        return None, "Pending action is invalid; missing tool name."
    tool_name = raw_tool_name
    tool_args = action.get("tool_args") or {}
    tool_args = normalize_intent_for_alarm(tool_name, tool_args)
    tool_args = normalize_intent_for_lock(tool_name, tool_args)
    tool_args = maybe_fill_lock_entity(tool_args, cfg.get("hass"))
    tool_args = sanitize_tool_args(tool_args)
    try:
        tool_input = llm.ToolInput(tool_name=tool_name, tool_args=tool_args)
        response = await ha_llm_api.async_call_tool(tool_input)
    except (HomeAssistantError, vol.Invalid) as err:
        return None, f"Failed to execute action: {err!r}"

    pending_actions = cfg.get("pending_actions", {})
    pending_actions.pop(resolved_action_id, None)
    return json.dumps(
        {"status": "completed", "action_id": resolved_action_id, "result": response}
    ), None


@tool(parse_docstring=True)
async def alarm_control(  # noqa: D417, PLR0913
    name: str | None = None,
    entity_id: str | None = None,
    state: str | None = None,
    code: str | None = None,
    *,
    bypass_open_sensors: bool | None = None,
    config: Annotated[RunnableConfig, InjectedToolArg()],
) -> str:
    """
    Arm or disarm an alarm control panel using the alarm system code (not the PIN).

    Args:
        name: Friendly name of the alarm panel (for example, "Home Alarm").
        entity_id: Specific alarm entity_id (if known). If not provided, the tool will
            try to resolve it from name or fall back to the only alarm entity.
        state: Desired target state. Examples: "arm_home", "armed_home", "arm_away",
            "armed_away", "disarm", "disarmed". If omitted, defaults to "arm_home".
        code: The alarm panel code (required by most alarm integrations).
        bypass_open_sensors: If True, request bypass of open sensors (if supported).

    """
    if "configurable" not in config:
        return "Configuration not found. Please check your setup."

    hass: HomeAssistant = config["configurable"]["hass"]
    tool_args: dict[str, Any] = {
        "name": name,
        "entity_id": entity_id,
        "state": state,
        "code": code,
        "bypass_open_sensors": bypass_open_sensors,
    }
    # Drop None/empty values so the alarm helper can apply its own defaults.
    tool_args = {k: v for k, v in tool_args.items() if v not in (None, "")}

    try:
        result = await _perform_alarm_control(hass, "alarm_control", tool_args)
    except HomeAssistantError as err:
        return f"Error controlling alarm: {err}"
    return json.dumps(result)


def _get_state_and_decimate(
    data: list[dict[str, str]],
    keys: list[str] | None = None,
    limit: int = HISTORY_TOOL_CONTEXT_LIMIT,
) -> list[dict[str, str]]:
    if keys is None:
        keys = ["state", "last_changed"]
    # Filter entity data to only state values with datetimes.
    state_values = [d for d in data if all(key in d for key in keys)]
    state_values = [{k: sv[k] for k in keys} for sv in state_values]
    # Decimate to avoid adding unnecessary fine grained data to context.
    length = len(state_values)
    if length > limit:
        LOGGER.debug("Decimating sensor data set.")
        factor = max(1, length // limit)
        state_values = state_values[::factor]
    return state_values


def _gen_dict_extract(key: str, var: dict) -> Generator[str]:
    """Find a key in nested dict."""
    if hasattr(var, "items"):
        for k, v in var.items():
            if k == key:
                yield v
            if isinstance(v, dict):
                yield from _gen_dict_extract(key, v)
            elif isinstance(v, list):
                for d in v:
                    yield from _gen_dict_extract(key, d)


def _filter_data(
    entity_id: str, data: list[dict[str, str]], hass: HomeAssistant
) -> dict[str, Any]:
    state_obj = hass.states.get(entity_id)
    if not state_obj:
        return {}

    state_class = state_obj.attributes.get("state_class")

    if state_class in ("measurement", "total"):
        state_values = _get_state_and_decimate(data)
        units = state_obj.attributes.get("unit_of_measurement")
        return {"values": state_values, "units": units}

    if state_class == "total_increasing":
        # For sensors with state class 'total_increasing', the data contains the
        # accumulated growth of the sensor's value since it was first added.
        # Therefore, return the net change.
        state_values = []
        for x in list(_gen_dict_extract("state", {entity_id: data})):
            try:
                state_values.append(float(x))
            except ValueError:
                LOGGER.warning("Found string that could not be converted to float.")
                continue
        # Check if sensor was reset during the time of interest.
        zero_indices = [i for i, x in enumerate(state_values) if math.isclose(x, 0.0)]
        if zero_indices:
            LOGGER.warning("Sensor was reset during time of interest.")
            state_values = state_values[zero_indices[-1] :]
        state_value_change = max(state_values) - min(state_values)
        units = state_obj.attributes.get("unit_of_measurement")
        return {"value": state_value_change, "units": units}

    return {"values": _get_state_and_decimate(data)}


async def _fetch_data_from_history(
    hass: HomeAssistant, start_time: datetime, end_time: datetime, entity_ids: list[str]
) -> dict[str, list[dict[str, Any]]]:
    filters = None
    include_start_time_state = True
    significant_changes_only = True
    minimal_response = True  # If True filter out duplicate states
    no_attributes = False
    compressed_state_format = False

    with recorder_session_scope(hass=hass, read_only=True) as session:
        result = await get_recorder_instance(hass).async_add_executor_job(
            recorder_history.get_significant_states_with_session,
            hass,
            session,
            start_time,
            end_time,
            entity_ids,
            filters,
            include_start_time_state,
            significant_changes_only,
            minimal_response,
            no_attributes,
            compressed_state_format,
        )

    if not result:
        return {}

    # Convert any State objects to dict.
    return {
        e: [s.as_dict() if isinstance(s, State) else s for s in v]
        for e, v in result.items()
    }


async def _fetch_data_from_long_term_stats(
    hass: HomeAssistant, start_time: datetime, end_time: datetime, entity_ids: list[str]
) -> dict[str, list[dict[str, Any]]]:
    period = "hour"
    units = None

    # Only concerned with two statistic types. The "state" type is associated with
    # sensor entities that have State Class of total or total_increasing.
    # The "mean" type is associated with entities with State Class measurement.
    types = {"state", "mean"}

    result = await get_recorder_instance(hass).async_add_executor_job(
        recorder_statistics.statistics_during_period,
        hass,
        start_time,
        end_time,
        set(entity_ids),
        period,
        units,
        types,
    )

    # Make data format consistent with the History format.
    parsed_result: dict[str, list[dict[str, Any]]] = {}
    for k, v in result.items():
        data: list[dict[str, Any]] = [
            {
                "state": d["state"] if "state" in d else d.get("mean"),
                "last_changed": dt_util.as_local(dt_util.utc_from_timestamp(d["end"]))
                if "end" in d
                else None,
            }
            for d in v
        ]
        parsed_result[k] = data

    return parsed_result


def _as_utc(dattim: str, default: datetime, error_message: str) -> datetime:
    """
    Convert a string representing a datetime into a datetime.datetime.

    Args:
        dattim: String representing a datetime.
        default: datatime.datetime to use as default.
        error_message: Message to raise in case of error.

    Raises:
        Homeassistant error if datetime cannot be parsed.

    Returns:
        A datetime.datetime of the string in UTC.

    """
    if dattim is None:
        return default

    parsed_datetime = dt_util.parse_datetime(dattim)
    if parsed_datetime is None:
        raise HomeAssistantError(error_message)

    return dt_util.as_utc(parsed_datetime)


# Allow domains like "sensor", "binary_sensor", "camera", etc.
_DOMAIN_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
# Valid HA entity_id = <domain>.<object_id>
_ENTITY_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")


async def _get_existing_entity_id(
    name: str | None, hass: HomeAssistant, domain: str | None = "sensor"
) -> str:
    """
    Lookup an existing entity by its friendly name.

    Raises ValueError if not found, ambiguous, or invalid domain/entity_id.
    """
    if not isinstance(name, str) or not name.strip():
        msg = "Name must be a non-empty string"
        raise ValueError(msg)
    if not isinstance(domain, str) or not _DOMAIN_PATTERN.match(domain):
        msg = "Domain invalid; must be a valid Home Assistant domain"
        raise ValueError(msg)

    target = name.strip().lower()
    prefix = f"{domain}."
    candidates: list[str] = []

    for state in hass.states.async_all():
        eid = state.entity_id
        if not eid.startswith(prefix):
            continue
        fn = state.attributes.get(ATTR_FRIENDLY_NAME, "")
        if isinstance(fn, str) and fn.strip().lower() == target:
            candidates.append(eid)

    if not candidates:
        msg = f"No '{domain}' entity found with friendly name '{name}'"
        raise ValueError(msg)
    if len(candidates) > 1:
        msg = f"Multiple '{domain}' entities found for '{name}': {candidates}"
        raise ValueError(msg)

    eid = candidates[0]
    if not _ENTITY_ID_PATTERN.match(eid):
        msg = f"Found entity_id '{eid}' is not valid"
        raise ValueError(msg)

    return eid


@tool(parse_docstring=True)
async def get_entity_history(  # noqa: D417
    friendly_names: list[str],
    domains: list[str],
    local_start_time: str,
    local_end_time: str,
    *,
    # Hide these arguments from the model.
    config: Annotated[RunnableConfig, InjectedToolArg()],
) -> dict[str, dict[str, list[dict[str, str]]]]:
    """
    Get entity state histories from Home Assistant.

    Args:
        friendly_names: List of Home Assistant friendly names to get history for,
            for example, ["Front Door", "Living Room Light"].
        domains: List of Home Assistant domains associated with the friendly names,
            for example, ["binary_sensor", "light"]. These must be in the same order as
            friendly_names.
        local_start_time: Start of local time history period in "%Y-%m-%dT%H:%M:%S%z".
        local_end_time: End of local time history period in "%Y-%m-%dT%H:%M:%S%z".

    Returns:
        Entity histories in local time format, for example:
            {
                "binary_sensor.front_door": {"values": [
                    {"state": "off", "last_changed": "2025-07-24T00:00:00-0700"},
                    {"state": "on", "last_changed": "2025-07-24T04:47:28-0700"},
                    ...]}
            }.

    """
    if "configurable" not in config:
        LOGGER.warning("Configuration not found. Please check your setup.")
        return {}

    hass: HomeAssistant = config["configurable"]["hass"]

    try:
        entity_ids = [
            await _get_existing_entity_id(n, hass, d)
            for n in friendly_names
            for d in domains
        ]
    except ValueError:
        LOGGER.exception("Invalid name %s or domain: %s", friendly_names, domains)
        return {}

    now = dt_util.utcnow()
    one_day = timedelta(days=1)
    try:
        start_time = _as_utc(
            dattim=local_start_time,
            default=now - one_day,
            error_message="start_time not valid",
        )
        end_time = _as_utc(
            dattim=local_end_time,
            default=start_time + one_day,
            error_message="end_time not valid",
        )
    except HomeAssistantError:
        LOGGER.exception("Error parsing start or end time.")
        return {}

    threshold = dt_util.now() - timedelta(days=HISTORY_TOOL_PURGE_KEEP_DAYS)

    data: dict[str, list[dict[str, Any]]]
    if start_time < threshold and end_time >= threshold:
        data = await _fetch_data_from_long_term_stats(
            hass=hass, start_time=start_time, end_time=threshold, entity_ids=entity_ids
        )
        data.update(
            await _fetch_data_from_history(
                hass=hass,
                start_time=threshold,
                end_time=end_time,
                entity_ids=entity_ids,
            )
        )
    elif end_time < threshold:
        data = await _fetch_data_from_long_term_stats(
            hass=hass, start_time=start_time, end_time=end_time, entity_ids=entity_ids
        )
    else:
        data = await _fetch_data_from_history(
            hass=hass, start_time=start_time, end_time=end_time, entity_ids=entity_ids
        )

    if not data:
        return {}

    for lst in data.values():
        for d in lst:
            for k, v in d.items():
                try:
                    dattim = dt_util.parse_datetime(v, raise_on_error=True)
                    dattim_local = dt_util.as_local(dattim)
                    d[k] = dattim_local.strftime("%Y-%m-%dT%H:%M:%S%z")
                except (ValueError, TypeError):
                    pass

    return {k: _filter_data(k, v, hass) for k, v in data.items()}


###
# This tool has been replaced by the HA native tool GetLiveContext.
# It is no longer used. Keeping it here for reference only.
###
@tool(parse_docstring=True)
async def get_current_device_state(  # noqa: D417
    names: list[str],
    *,
    # Hide these arguments from the model.
    config: Annotated[RunnableConfig, InjectedToolArg()],
) -> dict[str, str]:
    """
    Get the current state of one or more Home Assistant devices.

    Args:
        names: List of Home Assistant device names.

    """

    def _parse_input_to_yaml(input_text: str) -> dict[str, Any]:
        split_marker = "An overview of the areas and the devices in this smart home:"
        if split_marker not in input_text:
            msg = "Input text format is invalid. Marker not found."
            raise ValueError(msg)

        instructions_part, devices_part = input_text.split(split_marker, 1)
        instructions = instructions_part.strip()
        devices_yaml = devices_part.strip()
        devices = yaml.safe_load(devices_yaml)
        return {"instructions": instructions, "devices": devices}

    if "configurable" not in config:
        LOGGER.warning("Configuration not found. Please check your setup.")
        return {}
    llm_api = config["configurable"]["ha_llm_api"]
    try:
        overview = _parse_input_to_yaml(llm_api.api_prompt)
    except ValueError:
        LOGGER.exception("There was a problem getting device state.")
        return {}

    devices = overview.get("devices", [])
    state_dict: dict[str, str] = {}
    for device in devices:
        name = device.get("names", "Unnamed Device")
        if name not in names:
            continue
        state = device.get("state", None)
        state_dict[name] = state

    return state_dict


# ----- Time and Date Tools -----


@tool(parse_docstring=True)
def current_time(  # noqa: D417
    timezone: str = "Etc/UTC",
) -> str:
    """
    Get the current time in the specified timezone.

    Args:
        timezone: A valid IANA timezone string, e.g., 'Etc/UTC', 'Asia/Bangkok'.
            Defaults to 'Etc/UTC'.

    """
    try:
        now = datetime.now(ZoneInfo(timezone))
        return now.isoformat()
    except Exception as err:
        return f"Error getting time: {err}"


@tool(parse_docstring=True)
def time_since(  # noqa: D417
    past_date: str,
) -> str:
    """
    Get human-readable time since a given datetime.

    Args:
        past_date: A past datetime in ISO 8601 format, e.g. '2024-01-01T00:00:00Z'.

    """
    try:
        past = datetime.fromisoformat(past_date.replace("Z", "+00:00"))
        # Ensure we compare timezone-aware datetimes
        if past.tzinfo is None:
            past = past.replace(tzinfo=dt_util.UTC)
        
        now = dt_util.utcnow()
        delta = now - past
        return f"{delta.days} days, {delta.seconds // 3600} hours ago"
    except Exception as err:
        return f"Error calculating time since: {err}"


@tool(parse_docstring=True)
def add_days(  # noqa: D417
    days: int,
) -> str:
    """
    Get a future date by adding days to today.

    Args:
        days: Number of days to add to the current date.

    """
    try:
        future = datetime.now().date() + timedelta(days=days)
        return future.isoformat()
    except Exception as err:
        return f"Error adding days: {err}"


@tool(parse_docstring=True)
def subtract_days(  # noqa: D417
    days: int,
) -> str:
    """
    Get a past date by subtracting days from today.

    Args:
        days: Number of days to subtract from the current date.

    """
    try:
        past = datetime.now().date() - timedelta(days=days)
        return past.isoformat()
    except Exception as err:
        return f"Error subtracting days: {err}"


@tool(parse_docstring=True)
def date_diff(  # noqa: D417
    start: str,
    end: str,
) -> str:
    """
    Calculate the number of days between two dates.

    Args:
        start: Start date in ISO format, e.g., '2024-01-01'.
        end: End date in ISO format, e.g., '2025-01-01'.

    """
    try:
        start_date = datetime.fromisoformat(start).date()
        end_date = datetime.fromisoformat(end).date()
        diff = (end_date - start_date).days
        return f"{diff} days"
    except Exception as err:
        return f"Error calculating date diff: {err}"


@tool(parse_docstring=True)
def next_weekday(  # noqa: D417
    weekday: str,
) -> str:
    """
    Get the date of the next given weekday.

    Args:
        weekday: The name of the weekday (e.g., 'Monday', 'Friday').

    """
    weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    w = weekday.lower()
    if w not in weekdays:
        return "Invalid weekday name."
    today = date.today()
    today_idx = today.weekday()
    target_idx = weekdays.index(w)
    days_ahead = (target_idx - today_idx + 7) % 7
    if days_ahead == 0:
        days_ahead = 7
    return (today + timedelta(days=days_ahead)).isoformat()


@tool(parse_docstring=True)
def is_leap_year(  # noqa: D417
    year: int,
) -> bool:
    """
    Check if a year is a leap year.

    Args:
        year: The year to check.

    """
    return calendar.isleap(year)


@tool(parse_docstring=True)
def week_number(  # noqa: D417
    date_str: str,
) -> int | str:
    """
    Get the ISO week number of the given date.

    Args:
        date_str: Date in ISO format (e.g., '2025-05-15').

    """
    try:
        return datetime.fromisoformat(date_str).isocalendar().week
    except Exception as err:
        return f"Error getting week number: {err}"


@tool(parse_docstring=True)
def parse_human_date(  # noqa: D417
    description: str,
) -> str:
    """
    Parse a human-readable date expression.

    Args:
        description: A natural language description of a date, e.g. 'next Friday'.

    """
    try:
        parsed = dateparser.parse(description)
        if not parsed:
            return "Could not parse the date description."
        return parsed.date().isoformat()
    except Exception as err:
        return f"Error parsing date: {err}"


# ----- Math Tools -----


@tool(parse_docstring=True)
def add(  # noqa: D417
    a: float,
    b: float,
) -> float:
    """
    Add two numbers.

    Args:
        a: The first number.
        b: The second number.

    """
    return a + b


@tool(parse_docstring=True)
def subtract(  # noqa: D417
    a: float,
    b: float,
) -> float:
    """
    Subtract b from a.

    Args:
        a: The number to subtract from.
        b: The number to subtract.

    """
    return a - b


@tool(parse_docstring=True)
def multiply(  # noqa: D417
    a: float,
    b: float,
) -> float:
    """
    Multiply two numbers.

    Args:
        a: The first factor.
        b: The second factor.

    """
    return a * b


@tool(parse_docstring=True)
def divide(  # noqa: D417
    a: float,
    b: float,
) -> float | str:
    """
    Divide a by b.

    Args:
        a: The numerator.
        b: The denominator (must not be 0).

    """
    if b == 0:
        return "Error: Cannot divide by zero."
    return a / b


@tool(parse_docstring=True)
def percentage_diff(  # noqa: D417
    original: float,
    new: float,
) -> dict[str, Any] | str:
    """
    Calculate the percentage difference between two values.

    Args:
        original: Original value.
        new: New value.

    """
    if original == 0:
        return "Error: Original value cannot be zero."
    percent = ((new - original) / abs(original)) * 100
    return {
        "percentage_change": round(percent, 2),
        "direction": "increase" if percent > 0 else "decrease" if percent < 0 else "no change"
    }


@tool(parse_docstring=True)
def round_number(  # noqa: D417
    value: float,
    places: int = 0,
) -> dict[str, Any]:
    """
    Round a number to a given number of decimal places.

    Args:
        value: Value to round.
        places: Number of decimal places.

    """
    return {
        "rounded_value": round(value, places),
        "decimal_places": places
    }


# ----- Dictionary Tools -----

API_BASE = "https://api.dictionaryapi.dev/api/v2/entries/en/"


@tool(parse_docstring=True)
async def define(  # noqa: D417
    word: str,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg()],
) -> dict[str, Any]:
    """
    Get the definition(s) of an English word.

    Args:
        word: The word to define.

    """
    if "configurable" not in config:
        return {"error": "Configuration not found"}
    hass = config["configurable"]["hass"]
    client = get_async_client(hass)

    try:
        response = await client.get(f"{API_BASE}{word}")
        data = response.json()
        if isinstance(data, dict) and data.get("title") == "No Definitions Found":
            return {"error": f"No definitions found for '{word}'"}
        
        # data is a list of entries
        if not isinstance(data, list):
             return {"error": "Unexpected API response format"}

        meanings = []
        for entry in data:
            for meaning in entry.get("meanings", []):
                meanings.append({
                    "part_of_speech": meaning.get("partOfSpeech"),
                    "definitions": [d.get("definition") for d in meaning.get("definitions", [])]
                })
        return {"word": word, "meanings": meanings}
    except Exception as err:
        return {"error": f"Error fetching definition: {err}"}


@tool(parse_docstring=True)
async def example_usage(  # noqa: D417
    word: str,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg()],
) -> list[str] | dict[str, str]:
    """
    Get example usage of a word, if available.

    Args:
        word: The word to look up examples for.

    """
    if "configurable" not in config:
        return {"error": "Configuration not found"}
    hass = config["configurable"]["hass"]
    client = get_async_client(hass)
    
    try:
        response = await client.get(f"{API_BASE}{word}")
        data = response.json()
        if isinstance(data, dict) and data.get("title") == "No Definitions Found":
            return []
        
        if not isinstance(data, list):
             return {"error": "Unexpected API response format"}

        examples = []
        for entry in data:
            for meaning in entry.get("meanings", []):
                for definition in meaning.get("definitions", []):
                    ex = definition.get("example")
                    if ex:
                        examples.append(ex)
        return examples
    except Exception as err:
        return {"error": f"Error fetching examples: {err}"}


@tool(parse_docstring=True)
async def synonyms(  # noqa: D417
    word: str,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg()],
) -> list[str] | dict[str, str]:
    """
    Get synonyms for a word, if available.

    Args:
        word: The word to find synonyms for.

    """
    if "configurable" not in config:
        return {"error": "Configuration not found"}
    hass = config["configurable"]["hass"]
    client = get_async_client(hass)

    try:
        response = await client.get(f"{API_BASE}{word}")
        data = response.json()
        if isinstance(data, dict) and data.get("title") == "No Definitions Found":
            return []
        
        if not isinstance(data, list):
             return {"error": "Unexpected API response format"}

        synonyms_set = set()
        for entry in data:
            for meaning in entry.get("meanings", []):
                for definition in meaning.get("definitions", []):
                    for syn in definition.get("synonyms", []):
                        synonyms_set.add(syn)
        return list(synonyms_set)
    except Exception as err:
        return {"error": f"Error fetching synonyms: {err}"}


# ----- Google Places Tool -----

GOOGLE_PLACES_TEXT_SEARCH_URL = "https://maps.googleapis.com/maps/api/place/textsearch/json"


@tool(parse_docstring=True)
async def find_nearby_places(  # noqa: D417
    query: str,
    max_results: int = 5,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg()],
) -> list[dict[str, Any]] | dict[str, Any]:
    """
    Find nearby places using Google Places API.

    Useful for finding locations, addresses, or place details.

    Args:
        query: What to search for (e.g., 'Publix', 'gas station', 'pharmacy', 'CVS').
        max_results: Max results to return (default: 5, max: 20).

    """
    if "configurable" not in config:
        return {"error": "Configuration not found"}
    
    options = config["configurable"].get("options", {})
    api_key = options.get(CONF_GOOGLE_PLACES_API_KEY)
    
    if not api_key:
        return {"error": "Google Places API Key is not configured."}

    hass = config["configurable"]["hass"]
    client = get_async_client(hass)
    
    params = {
        "query": query,
        "key": api_key,
    }

    try:
        response = await client.get(GOOGLE_PLACES_TEXT_SEARCH_URL, params=params)
        data = response.json()
        
        if data.get("status") != "OK":
             error_msg = data.get("error_message", data.get("status"))
             if data.get("status") == "ZERO_RESULTS":
                 return []
             return {"error": f"Google Places API Error: {error_msg}"}

        results = data.get("results", [])
        output = []
        
        limit = min(max_results, 20)
        
        for place in results[:limit]:
            # Simplify output for LLM consumption
            item = {
                "name": place.get("name"),
                "address": place.get("formatted_address"),
                "rating": place.get("rating"),
                "user_ratings_total": place.get("user_ratings_total"),
                "place_id": place.get("place_id"),
                "types": place.get("types", []),
            }
            if place.get("opening_hours") and place["opening_hours"].get("open_now") is not None:
                 item["open_now"] = place["opening_hours"]["open_now"]
            
            output.append(item)
            
        return output

    except Exception as err:
        return {"error": f"Error searching places: {err}"}


# ----- Wikipedia Tools -----

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"


@tool(parse_docstring=True)
async def search_wikipedia(  # noqa: D417
    query: str,
    limit: int = 10,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg()],
) -> list[dict[str, Any]] | dict[str, Any]:
    """
    Search Wikipedia for articles matching a query.

    Args:
        query: The search term.
        limit: Max results (default: 10, max: 20).

    """
    if "configurable" not in config:
        return {"error": "Configuration not found"}
    hass = config["configurable"]["hass"]
    client = get_async_client(hass)

    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "utf8": 1,
        "srsearch": query,
        "srlimit": min(limit, 20),
    }

    try:
        response = await client.get(WIKIPEDIA_API_URL, params=params)
        data = response.json()
        
        if "error" in data:
            return {"error": data["error"].get("info", "Unknown API error")}

        search_results = data.get("query", {}).get("search", [])
        
        output = []
        for item in search_results:
            output.append({
                "title": item.get("title"),
                "snippet": item.get("snippet", "").replace('<span class="searchmatch">', '').replace('</span>', ''),
                "pageid": item.get("pageid"),
            })
            
        return output
    except Exception as err:
        return {"error": f"Error searching Wikipedia: {err}"}


@tool(parse_docstring=True)
async def get_wikipedia_page(  # noqa: D417
    title: str,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg()],
) -> dict[str, Any]:
    """
    Get the summary and content of a Wikipedia article.

    Args:
        title: The title of the article.

    """
    if "configurable" not in config:
        return {"error": "Configuration not found"}
    hass = config["configurable"]["hass"]
    client = get_async_client(hass)

    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts|pageimages|info",
        "exintro": 1,
        "explaintext": 1,
        "inprop": "url",
        "titles": title,
        "pithumbsize": 500
    }

    try:
        response = await client.get(WIKIPEDIA_API_URL, params=params)
        data = response.json()
        
        if "error" in data:
            return {"error": data["error"].get("info", "Unknown API error")}

        pages = data.get("query", {}).get("pages", {})
        if not pages:
             return {"error": "Page not found."}
        
        # 'pages' is a dict keyed by pageid, but we typically just want the first one found
        page = next(iter(pages.values()))
        
        if "missing" in page:
            return {"error": f"Page '{title}' does not exist."}

        return {
            "title": page.get("title"),
            "summary": page.get("extract"),
            "url": page.get("fullurl"),
            "image": page.get("thumbnail", {}).get("source")
        }
    except Exception as err:
        return {"error": f"Error getting Wikipedia page: {err}"}


# ----- LightRAG Tools -----


@tool(parse_docstring=True)
async def query_lightrag(  # noqa: D417
    query: str,
    mode: str = "mix",
    *,
    config: Annotated[RunnableConfig, InjectedToolArg()],
) -> dict[str, Any]:
    """
    Search the LightRAG knowledge base for information.

    Args:
        query: The question or topic to search for.
        mode: Retrieval mode ('mix' is recommended). Options: 'mix', 'hybrid', 'local', 'global', 'naive'.

    """
    if "configurable" not in config:
        return {"error": "Configuration not found"}
    
    options = config["configurable"].get("options", {})
    base_url = options.get(CONF_LIGHTRAG_URL, "http://localhost:9600").rstrip("/")
    api_key = options.get(CONF_LIGHTRAG_API_KEY, "")
    
    hass = config["configurable"]["hass"]
    client = get_async_client(hass)

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    payload = {
        "query": query,
        "mode": mode,
    }

    try:
        response = await client.post(
            f"{base_url}/query", 
            json=payload, 
            headers=headers,
            timeout=60
        )
        
        if response.status_code != 200:
            return {"error": f"LightRAG Error ({response.status_code}): {response.text}"}
            
        data = response.json()
        return {
            "response": data.get("response", "No response generated."),
            "mode": mode
        }
    except Exception as err:
        return {"error": f"Error querying LightRAG: {err}"}


# ----- Reddit Tools -----


def _get_reddit_client(config: RunnableConfig) -> Any:
    """Get a configured PRAW Reddit client."""
    if "configurable" not in config:
        raise ValueError("Configuration not found")
        
    options = config["configurable"].get("options", {})
    client_id = options.get(CONF_REDDIT_CLIENT_ID)
    client_secret = options.get(CONF_REDDIT_CLIENT_SECRET)
    user_agent = options.get(CONF_REDDIT_USER_AGENT, "HomeAssistant/1.0.0")
    
    if not client_id or not client_secret:
        raise ValueError("Reddit credentials not configured")
        
    import praw
    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        check_for_async=False
    )


@tool(parse_docstring=True)
async def get_subreddit_posts(
    subreddit: str,
    sort: str = "hot",
    time_filter: str = "day",
    limit: int = 10,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg()],
) -> dict[str, Any]:
    """
    Get posts from a specific subreddit.

    Args:
        subreddit: Name of the subreddit (without r/).
        sort: Sort method: "hot", "new", "rising", "top". Default is "hot".
        time_filter: "hour", "day", "week", "month", "year", "all". Default is "day".
        limit: Number of posts to fetch (1-100). Default is 10.
    """
    hass = config["configurable"]["hass"]
    
    def _fetch():
        reddit = _get_reddit_client(config)
        sub = reddit.subreddit(subreddit)
        
        if sort == "hot":
            posts = sub.hot(limit=limit)
        elif sort == "new":
            posts = sub.new(limit=limit)
        elif sort == "rising":
            posts = sub.rising(limit=limit)
        elif sort == "top":
            posts = sub.top(time_filter=time_filter, limit=limit)
        else:
            posts = sub.hot(limit=limit)
            
        results = []
        for post in posts:
            results.append({
                "title": post.title,
                "author": str(post.author) if post.author else "[deleted]",
                "score": post.score,
                "url": post.url,
                "id": post.id,
                "is_self": post.is_self,
                "text": post.selftext[:500] + "..." if len(post.selftext) > 500 else post.selftext
            })
        return results

    try:
        return await hass.async_add_executor_job(_fetch)
    except Exception as err:
        return {"error": f"Error fetching posts: {err}"}


@tool(parse_docstring=True)
async def get_post_details(
    post_id: str,
    include_comments: bool = False,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg()],
) -> dict[str, Any]:
    """
    Get detailed information about a specific Reddit post.

    Args:
        post_id: Reddit post ID or full URL.
        include_comments: Whether to include top comments. Default is False.
    """
    hass = config["configurable"]["hass"]

    def _fetch():
        reddit = _get_reddit_client(config)
        if "reddit.com" in post_id:
            submission = reddit.submission(url=post_id)
        else:
            submission = reddit.submission(id=post_id)
            
        data = {
            "title": submission.title,
            "author": str(submission.author) if submission.author else "[deleted]",
            "subreddit": str(submission.subreddit),
            "score": submission.score,
            "created_utc": submission.created_utc,
            "url": submission.url,
            "text": submission.selftext,
        }
        
        if include_comments:
            submission.comments.replace_more(limit=0)
            comments = []
            for comment in submission.comments[:10]:
                 if hasattr(comment, 'body'):
                    comments.append({
                        "author": str(comment.author) if comment.author else "[deleted]",
                        "body": comment.body[:300],
                        "score": comment.score
                    })
            data["comments"] = comments
            
        return data

    try:
        return await hass.async_add_executor_job(_fetch)
    except Exception as err:
        return {"error": f"Error fetching post: {err}"}


@tool(parse_docstring=True)
async def search_reddit(
    query: str,
    subreddit: str | None = None,
    sort: str = "relevance",
    time_filter: str = "all",
    limit: int = 10,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg()],
) -> dict[str, Any]:
    """
    Search Reddit for posts matching a query.

    Args:
        query: Search query.
        subreddit: Limit search to specific subreddit (optional).
        sort: "relevance", "hot", "top", "new", "comments". Default is "relevance".
        time_filter: "hour", "day", "week", "month", "year", "all". Default is "all".
        limit: Number of results (1-100). Default is 10.
    """
    hass = config["configurable"]["hass"]

    def _search():
        reddit = _get_reddit_client(config)
        if subreddit:
            api = reddit.subreddit(subreddit)
        else:
            api = reddit.subreddit("all")
            
        results = []
        for post in api.search(query, sort=sort, time_filter=time_filter, limit=limit):
            results.append({
                "title": post.title,
                "subreddit": str(post.subreddit),
                "score": post.score,
                "url": post.url,
                "id": post.id
            })
        return results

    try:
        return await hass.async_add_executor_job(_search)
    except Exception as err:
        return {"error": f"Error searching Reddit: {err}"}


@tool(parse_docstring=True)
async def get_user_profile(
    username: str,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg()],
) -> dict[str, Any]:
    """
    Get public information about a Reddit user.

    Args:
        username: Reddit username (without u/).
    """
    hass = config["configurable"]["hass"]

    def _fetch():
        reddit = _get_reddit_client(config)
        user = reddit.redditor(username)
        return {
            "name": user.name,
            "comment_karma": user.comment_karma,
            "link_karma": user.link_karma,
            "created_utc": user.created_utc,
            "is_mod": user.is_mod,
            "is_employee": user.is_employee
        }

    try:
        return await hass.async_add_executor_job(_fetch)
    except Exception as err:
        return {"error": f"Error fetching profile: {err}"}


# ----- Plex Tools -----


def _get_plex_server(config: RunnableConfig) -> Any:
    """Get a configured PlexServer instance."""
    if "configurable" not in config:
        raise ValueError("Configuration not found")
        
    options = config["configurable"].get("options", {})
    base_url = options.get(CONF_PLEX_SERVER_URL)
    token = options.get(CONF_PLEX_TOKEN)
    
    if not base_url or not token:
        raise ValueError("Plex configuration missing (URL or Token)")
        
    from plexapi.server import PlexServer
    return PlexServer(base_url, token)


@tool(parse_docstring=True)
async def plex_search_movies(
    title: str | None = None,
    year: int | None = None,
    limit: int = 5,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg()],
) -> dict[str, Any]:
    """
    Search for movies in Plex.

    Args:
        title: Title to match.
        year: Year to filter by.
        limit: Max results (default 5).
    """
    hass = config["configurable"]["hass"]
    
    def _search():
        plex = _get_plex_server(config)
        kwargs = {"libtype": "movie"}
        if title:
            kwargs["title"] = title
        if year:
            kwargs["year"] = year
            
        results = plex.library.search(**kwargs)
        data = []
        for vid in results[:limit]:
             data.append({
                 "title": vid.title,
                 "year": vid.year,
                 "key": vid.ratingKey,
                 "summary": vid.summary[:200]
             })
        return data

    try:
        return await hass.async_add_executor_job(_search)
    except Exception as err:
        return {"error": f"Plex error: {err}"}


@tool(parse_docstring=True)
async def plex_get_movie_details(
    movie_key: str,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg()],
) -> dict[str, Any]:
    """
    Get details for a movie by its ratingKey.

    Args:
        movie_key: The unique key for the movie.
    """
    hass = config["configurable"]["hass"]

    def _fetch():
        plex = _get_plex_server(config)
        try:
            item = plex.library.fetchItem(int(movie_key))
            return {
                "title": item.title,
                "year": item.year,
                "summary": item.summary,
                "duration_min": item.duration // 60000 if item.duration else 0,
                "rating": item.rating,
                "directors": [d.tag for d in item.directors],
                "roles": [r.tag for r in item.roles[:5]],
            }
        except Exception as e:
            return {"error": str(e)}

    try:
        return await hass.async_add_executor_job(_fetch)
    except Exception as err:
        return {"error": f"Plex error: {err}"}


@tool(parse_docstring=True)
async def plex_recent_movies(
    count: int = 5,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg()],
) -> list[dict[str, Any]]:
    """
    Get recently added movies.

    Args:
        count: Number of movies to return.
    """
    hass = config["configurable"]["hass"]

    def _fetch():
        plex = _get_plex_server(config)
        # Assuming 'Movies' section exists, or search global
        # Global search for recent movies:
        results = plex.library.search(libtype="movie", sort="addedAt:desc", limit=count)
        return [{"title": m.title, "year": m.year, "key": m.ratingKey} for m in results]

    try:
        return await hass.async_add_executor_job(_fetch)
    except Exception as err:
        return [{"error": f"Plex error: {err}"}]


@tool(parse_docstring=True)
async def plex_create_playlist(
    name: str,
    movie_keys: list[str],
    *,
    config: Annotated[RunnableConfig, InjectedToolArg()],
) -> dict[str, Any]:
    """
    Create a playlist from movie keys.

    Args:
        name: Playlist name.
        movie_keys: List of movie ratingKeys.
    """
    hass = config["configurable"]["hass"]

    def _create():
        plex = _get_plex_server(config)
        items = []
        for key in movie_keys:
            try:
                items.append(plex.library.fetchItem(int(key)))
            except:
                pass
        
        if not items:
            return {"error": "No valid items found"}
            
        pl = plex.createPlaylist(name, items=items)
        return {"name": pl.title, "key": pl.ratingKey, "count": len(items)}

    try:
        return await hass.async_add_executor_job(_create)
    except Exception as err:
        return {"error": f"Plex error: {err}"}


@tool
async def plex_list_playlists(
    *,
    config: Annotated[RunnableConfig, InjectedToolArg()],
) -> list[dict[str, Any]]:
    """List all playlists."""
    hass = config["configurable"]["hass"]

    def _list():
        plex = _get_plex_server(config)
        return [{"title": pl.title, "key": pl.ratingKey, "count": pl.leafCount} for pl in plex.playlists()]

    try:
        return await hass.async_add_executor_job(_list)
    except Exception as err:
        return [{"error": f"Plex error: {err}"}]


@tool(parse_docstring=True)
async def plex_get_playlist_items(
    playlist_key: str,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg()],
) -> list[dict[str, Any]]:
    """
    Get items from a playlist.

    Args:
        playlist_key: Payload key.
    """
    hass = config["configurable"]["hass"]

    def _fetch():
        plex = _get_plex_server(config)
        try:
            pl = plex.playlist(int(playlist_key)) # fetchItem doesnt always work for playlists? 
            # Or fetchItem(key) works. Let's try to find it in playlists() list if fetch fails or just use fetchItem
            # plex.playlist() might need title? 
            # safe way:
            pl = plex.fetchItem(int(playlist_key))
            return [{"title": i.title, "year": i.year, "key": i.ratingKey} for i in pl.items()]
        except Exception as e:
            return [{"error": str(e)}]

    try:
        return await hass.async_add_executor_job(_fetch)
    except Exception as err:
        return [{"error": f"Plex error: {err}"}]


@tool(parse_docstring=True)
async def plex_delete_playlist(
    playlist_key: str,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg()],
) -> dict[str, Any]:
    """
    Delete a playlist.

    Args:
        playlist_key: Playlist key.
    """
    hass = config["configurable"]["hass"]

    def _delete():
        plex = _get_plex_server(config)
        try:
            pl = plex.fetchItem(int(playlist_key))
            pl.delete()
            return {"success": True, "title": pl.title}
        except Exception as e:
            return {"error": str(e)}

    try:
        return await hass.async_add_executor_job(_delete)
    except Exception as err:
        return {"error": f"Plex error: {err}"}


@tool(parse_docstring=True)
async def plex_add_to_playlist(
    playlist_key: str,
    movie_key: str,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg()],
) -> dict[str, Any]:
    """
    Add a movie to a playlist.

    Args:
        playlist_key: Playlist key.
        movie_key: Movie key.
    """
    hass = config["configurable"]["hass"]

    def _add():
        plex = _get_plex_server(config)
        try:
            pl = plex.fetchItem(int(playlist_key))
            movie = plex.library.fetchItem(int(movie_key))
            pl.addItems([movie])
            return {"success": True, "playlist": pl.title, "added": movie.title}
        except Exception as e:
            return {"error": str(e)}

    try:
        return await hass.async_add_executor_job(_add)
    except Exception as err:
        return {"error": f"Plex error: {err}"}


@tool(parse_docstring=True)
async def plex_get_movie_genres(
    movie_key: str,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg()],
) -> dict[str, Any]:
    """
    Get genres for a movie.

    Args:
        movie_key: Movie key.
    """
    hass = config["configurable"]["hass"]

    def _fetch():
        plex = _get_plex_server(config)
        try:
            m = plex.library.fetchItem(int(movie_key))
            return {"title": m.title, "genres": [g.tag for g in m.genres]}
        except Exception as e:
            return {"error": str(e)}

    try:
        return await hass.async_add_executor_job(_fetch)
    except Exception as err:
        return {"error": f"Plex error: {err}"}
