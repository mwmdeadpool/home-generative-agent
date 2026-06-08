"""
Gemini-side compatibility shims for tool schema conversion.

Background
----------
Some tools bound to ChatGoogleGenerativeAI emit JSON schemas that langchain's
``_format_json_schema_to_gapic`` converter passes through to Gemini's proto
without populating the required ``items`` field on array-typed properties.

The known offender is HA's own ``GetLiveContextTool`` (in
``homeassistant.helpers.llm``), which declares::

    vol.Optional("domain"): vol.Any(cv.string, [cv.string])

voluptuous-openapi turns that into ``anyOf: [{type: string}, {type: array}]``.
The langchain_google_genai converter then emits a Gemini ``Schema`` with
``type=ARRAY`` (or ``any_of`` variants) and *no* ``items`` field — which
Gemini rejects with::

    GenerateContentRequest.tools[0].function_declarations[N]
        .parameters.properties[domain].items: missing field.

Until either HA core or langchain_google_genai fixes this upstream, we
post-process the converter's output and inject a default ``items`` schema
into any array node that lacks one.
"""

from __future__ import annotations

import logging
from typing import Any

LOGGER = logging.getLogger(__name__)

_PATCH_FLAG = "_hga_array_items_injected"


def apply_gemini_tool_schema_fix() -> None:
    """Patch langchain_google_genai's schema converter, idempotent."""
    try:
        from langchain_google_genai import _function_utils as fu  # noqa: PLC0415
    except ImportError:
        LOGGER.debug("langchain_google_genai not available; skipping schema patch.")
        return

    if getattr(fu, _PATCH_FLAG, False):
        return

    original = fu._format_json_schema_to_gapic  # noqa: SLF001

    def wrapped(schema: dict[str, Any]) -> dict[str, Any]:
        result = original(schema)
        _inject_array_items(result)
        return result

    fu._format_json_schema_to_gapic = wrapped  # noqa: SLF001
    setattr(fu, _PATCH_FLAG, True)
    LOGGER.debug("Applied Gemini array-items schema fix.")


def _inject_array_items(node: Any) -> None:
    """
    Recursively walk dict/list and add items={'type':'STRING'} to bare arrays.

    The langchain_google_genai converter labels array types inconsistently:
    inner ``anyOf`` variants use the string key ``type`` with value
    ``"ARRAY"``, while the outer property uses the proto-key ``type_`` with
    a proto enum value (``Type.ARRAY``). Check both shapes.
    """
    if isinstance(node, dict):
        if _is_array_node(node) and "items" not in node:
            # Default to string items — appropriate for the known offender
            # (HA's domain filter accepts string|list[string]). If a future
            # tool legitimately needs array-of-{int,object}, langchain's
            # converter populates items on its own and we don't touch it.
            node["items"] = {"type": "STRING"}
        for value in node.values():
            _inject_array_items(value)
    elif isinstance(node, list):
        for item in node:
            _inject_array_items(item)


def _is_array_node(node: dict[str, Any]) -> bool:
    for key in ("type", "type_"):
        value = node.get(key)
        if value is None:
            continue
        if isinstance(value, str) and value.upper() == "ARRAY":
            return True
        # proto enum (Type.ARRAY) — compare via its `.name` attribute
        name = getattr(value, "name", None)
        if isinstance(name, str) and name.upper() == "ARRAY":
            return True
    return False
