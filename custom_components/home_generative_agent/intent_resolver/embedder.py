"""Entity embedding and vector store for Tier 1 resolution.

Embeds all HA entities (friendly_name + domain + area) into Qdrant
for fast similarity matching against user intents.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import device_registry as dr

_LOGGER = logging.getLogger(__name__)

# Domains we care about for fast intent resolution
ACTIONABLE_DOMAINS = {
    "light", "switch", "fan", "cover", "lock", "climate",
    "media_player", "scene", "script",
    "vacuum", "humidifier", "water_heater", "valve",
    "button", "input_boolean", "input_number", "input_select",
    "number", "select", "siren", "alarm_control_panel",
}

# Domains excluded from fast intent resolution because their friendly names
# often overlap with device entity names (e.g. an automation called
# "Turn off office lights when room is unoccupied" would match a request
# to turn off office lights).  These should only be triggered explicitly
# through the full LLM path.
EXCLUDED_DOMAINS = {"automation", "group"}

# Domain priority for tie-breaking when multiple entities match.
# Higher = preferred.  Direct-control domains (light, switch, fan) outrank
# indirect ones (scene, script) so "turn off office lights" resolves to
# the light entity rather than a similarly-named scene or script.
DOMAIN_PRIORITY: dict[str, float] = {
    "light": 1.0,
    "switch": 1.0,
    "fan": 1.0,
    "cover": 1.0,
    "lock": 1.0,
    "climate": 1.0,
    "media_player": 0.95,
    "vacuum": 0.95,
    "humidifier": 0.95,
    "valve": 0.95,
    "siren": 0.9,
    "alarm_control_panel": 0.9,
    "scene": 0.7,
    "script": 0.7,
    "input_boolean": 0.6,
    "input_number": 0.6,
    "input_select": 0.6,
    "number": 0.6,
    "select": 0.6,
    "button": 0.6,
    "water_heater": 0.8,
}


@classmethod
def _build_embedding_text(
    entity_id: str,
    friendly_name: str,
    domain: str,
    area_name: str | None = None,
    aliases: list[str] | None = None,
) -> str:
    """Build the text string to embed for an entity."""
    parts = [friendly_name, domain.replace("_", " ")]
    if area_name:
        parts.append(area_name)
    if aliases:
        parts.extend(aliases)
    # Add common action phrases for this domain
    domain_actions = {
        "light": ["turn on", "turn off", "dim", "brighten"],
        "switch": ["turn on", "turn off", "toggle"],
        "lock": ["lock", "unlock"],
        "cover": ["open", "close"],
        "climate": ["set temperature", "heat", "cool"],
        "media_player": ["play", "pause", "stop", "volume"],
        "fan": ["turn on", "turn off"],
        "scene": ["activate", "set"],
        "script": ["run", "execute"],
    }
    if domain in domain_actions:
        for action in domain_actions[domain]:
            parts.append(f"{action} {friendly_name}")
    return " | ".join(parts)


def build_embedding_text(
    entity_id: str,
    friendly_name: str,
    domain: str,
    area_name: str | None = None,
    aliases: list[str] | None = None,
) -> str:
    """Build the text string to embed for an entity (module-level function)."""
    parts = [friendly_name, domain.replace("_", " ")]
    if area_name:
        parts.append(area_name)
    if aliases:
        parts.extend(aliases)
    domain_actions = {
        "light": ["turn on", "turn off", "dim", "brighten"],
        "switch": ["turn on", "turn off", "toggle"],
        "lock": ["lock", "unlock"],
        "cover": ["open", "close"],
        "climate": ["set temperature", "heat", "cool"],
        "media_player": ["play", "pause", "stop", "volume"],
        "fan": ["turn on", "turn off"],
        "scene": ["activate", "set"],
        "script": ["run", "execute"],
    }
    if domain in domain_actions:
        for action in domain_actions[domain]:
            parts.append(f"{action} {friendly_name}")
    return " | ".join(parts)


async def collect_entities(hass: HomeAssistant) -> list[dict[str, Any]]:
    """Collect all actionable entities with metadata for embedding."""
    ent_reg = er.async_get(hass)
    area_reg = ar.async_get(hass)
    dev_reg = dr.async_get(hass)

    entities = []
    for entry in ent_reg.entities.values():
        domain = entry.domain
        if domain not in ACTIONABLE_DOMAINS:
            continue
        if entry.disabled_by is not None:
            continue

        # Get friendly name from state or registry
        state = hass.states.get(entry.entity_id)
        friendly_name = (
            entry.name
            or entry.original_name
            or (state.attributes.get("friendly_name") if state else None)
            or entry.entity_id
        )

        # Get area name
        area_name = None
        area_id = entry.area_id
        if not area_id and entry.device_id:
            device = dev_reg.async_get(entry.device_id)
            if device:
                area_id = device.area_id
        if area_id:
            area = area_reg.async_get_area(area_id)
            if area:
                area_name = area.name

        # Get aliases
        aliases = list(entry.aliases) if entry.aliases else []

        embedding_text = build_embedding_text(
            entry.entity_id, friendly_name, domain, area_name, aliases
        )

        entities.append({
            "entity_id": entry.entity_id,
            "domain": domain,
            "friendly_name": friendly_name,
            "area_name": area_name,
            "aliases": aliases,
            "embedding_text": embedding_text,
        })

    _LOGGER.info("Collected %d actionable entities for embedding", len(entities))
    return entities


async def embed_entities(
    hass: HomeAssistant,
    entities: list[dict[str, Any]],
    embedding_model: Any,
    qdrant_url: str = "http://localhost:6333",
    collection_name: str = "hga_entities",
) -> int:
    """Embed entities and store in Qdrant.

    Returns the number of entities embedded.
    """
    import httpx

    if not entities:
        return 0

    start = time.monotonic()

    # Create collection if it doesn't exist
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Check if collection exists
        resp = await client.get(f"{qdrant_url}/collections/{collection_name}")
        if resp.status_code == 200:
            _LOGGER.debug("Qdrant collection '%s' already exists", collection_name)
        else:
            # Create with appropriate dimensions
            test_embedding = await embedding_model.aembed_query("test")
            dims = len(test_embedding)
            create_resp = await client.put(
                f"{qdrant_url}/collections/{collection_name}",
                json={
                    "vectors": {
                        "size": dims,
                        "distance": "Cosine",
                    }
                },
            )
            if create_resp.status_code not in (200, 201):
                _LOGGER.error(
                    "Failed to create Qdrant collection '%s': %s %s",
                    collection_name, create_resp.status_code, create_resp.text
                )
                raise RuntimeError(f"Qdrant collection creation failed: {create_resp.status_code}")
            _LOGGER.info(
                "Created Qdrant collection '%s' with %d dimensions",
                collection_name, dims,
            )

        # Generate embeddings in batches
        batch_size = 50
        all_points = []
        for i in range(0, len(entities), batch_size):
            batch = entities[i : i + batch_size]
            texts = [e["embedding_text"] for e in batch]
            embeddings = await embedding_model.aembed_documents(texts)

            for j, (entity, embedding) in enumerate(zip(batch, embeddings)):
                all_points.append({
                    "id": i + j,
                    "vector": embedding,
                    "payload": {
                        "entity_id": entity["entity_id"],
                        "domain": entity["domain"],
                        "friendly_name": entity["friendly_name"],
                        "area_name": entity["area_name"],
                        "aliases": entity["aliases"],
                        "embedding_text": entity["embedding_text"],
                    },
                })

        # Upsert all points
        total_upserted = 0
        for i in range(0, len(all_points), 100):
            batch = all_points[i : i + 100]
            upsert_resp = await client.put(
                f"{qdrant_url}/collections/{collection_name}/points",
                json={"points": batch},
            )
            if upsert_resp.status_code not in (200, 201):
                _LOGGER.error(
                    "Failed to upsert points to '%s': %s %s",
                    collection_name, upsert_resp.status_code, upsert_resp.text
                )
                raise RuntimeError(f"Qdrant upsert failed: {upsert_resp.status_code}")
            total_upserted += len(batch)

    elapsed = (time.monotonic() - start) * 1000
    _LOGGER.info(
        "Embedded %d/%d entities in %.0fms into '%s'",
        total_upserted, len(entities), elapsed, collection_name,
    )
    return total_upserted


async def search_entities(
    query: str,
    embedding_model: Any,
    qdrant_url: str = "http://localhost:6333",
    collection_name: str = "hga_entities",
    limit: int = 5,
    score_threshold: float = 0.65,
) -> list[dict[str, Any]]:
    """Search for entities matching user intent.

    Returns list of matches with scores.
    """
    import httpx

    query_embedding = await embedding_model.aembed_query(query)

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f"{qdrant_url}/collections/{collection_name}/points/search",
            json={
                "vector": query_embedding,
                "limit": limit,
                "score_threshold": score_threshold,
                "with_payload": True,
            },
        )
        if resp.status_code != 200:
            _LOGGER.error("Qdrant search failed: %s", resp.text)
            return []

        results = resp.json().get("result", [])
        matches = []
        for r in results:
            domain = r["payload"]["domain"]
            raw_score = r["score"]
            # Apply domain priority weighting to break ties between
            # entities with similar embeddings (e.g. a light vs an
            # automation whose name contains the same words).
            priority = DOMAIN_PRIORITY.get(domain, 0.5)
            # Weighted score: 80% embedding similarity + 20% domain priority
            weighted_score = (raw_score * 0.8) + (priority * 0.2)
            matches.append({
                "entity_id": r["payload"]["entity_id"],
                "domain": domain,
                "friendly_name": r["payload"]["friendly_name"],
                "area_name": r["payload"].get("area_name"),
                "score": weighted_score,
                "raw_score": raw_score,
            })
        # Re-sort by weighted score (highest first)
        matches.sort(key=lambda m: m["score"], reverse=True)
        return matches
