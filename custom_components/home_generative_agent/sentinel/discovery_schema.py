"""Schema for LLM discovery output (advisory only)."""

from __future__ import annotations

import voluptuous as vol

DISCOVERY_SCHEMA_VERSION = 1

DISCOVERY_OUTPUT_SCHEMA = vol.Schema(
    {
        vol.Required("schema_version"): vol.In([DISCOVERY_SCHEMA_VERSION]),
        vol.Required("generated_at"): str,
        vol.Required("model"): str,
        vol.Required("candidates"): [
            vol.Schema(
                {
                    vol.Required("candidate_id"): str,
                    vol.Required("title"): str,
                    vol.Required("summary"): str,
                    vol.Required("evidence_paths"): [str],
                    vol.Optional("pattern", default=""): str,
                    vol.Required("confidence_hint"): vol.All(
                        vol.Coerce(float), vol.Range(min=0.0, max=1.0)
                    ),
                    vol.Optional("suggested_type"): str,
                    vol.Optional("semantic_key"): str,
                    vol.Optional("dedupe_reason"): str,
                },
                extra=vol.ALLOW_EXTRA,
            )
        ],
        vol.Optional("filtered_candidates"): [
            vol.Schema(
                {
                    vol.Required("candidate_id"): str,
                    vol.Optional("semantic_key"): str,
                    vol.Required("dedupe_reason"): str,
                },
                extra=vol.ALLOW_EXTRA,
            )
        ],
    },
    extra=vol.ALLOW_EXTRA,
)