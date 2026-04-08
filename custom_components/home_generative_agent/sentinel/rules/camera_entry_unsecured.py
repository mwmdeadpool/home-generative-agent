"""Rule: camera activity near unsecured entry."""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import TYPE_CHECKING

from homeassistant.util import dt as dt_util

from custom_components.home_generative_agent.sentinel.models import (
    AnomalyFinding,
    build_anomaly_id,
)

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
        SnapshotEntity,
    )

ENTRY_CLASSES = {"door", "window", "opening"}
ACTIVITY_WINDOW_MIN = 10


class CameraEntryUnsecuredRule:
    """Detect camera activity while nearby entries are unsecured."""

    rule_id = "camera_entry_unsecured"

    def __init__(
        self,
        *,
        camera_entry_links: dict[str, list[str]] | None = None,
    ) -> None:
        self._camera_entry_links: dict[str, list[str]] = camera_entry_links or {}

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:  # noqa: PLR0912, PLR0915
        """Return findings for recent camera activity near unsecured entries."""
        findings: list[AnomalyFinding] = []
        now = dt_util.parse_datetime(snapshot["derived"]["now"]) or dt_util.utcnow()
        window = timedelta(minutes=ACTIVITY_WINDOW_MIN)

        unsecured_by_area: dict[str, list[str]] = {}
        for entity in snapshot["entities"]:
            area = entity.get("area")
            if not area:
                continue
            if entity["domain"] == "lock" and entity["state"] == "unlocked":
                unsecured_by_area.setdefault(area, []).append(entity["entity_id"])
                continue
            if entity["domain"] != "binary_sensor":
                continue
            if entity["attributes"].get("device_class") not in ENTRY_CLASSES:
                continue
            if entity["state"] != "on":
                continue
            unsecured_by_area.setdefault(area, []).append(entity["entity_id"])

        # Reverse map: entity_id -> area, built from *all* snapshot entities so
        # that cross-area linked entries are correctly resolved.
        all_entity_area_map: dict[str, str] = {
            e["entity_id"]: e.get("area", "unknown") for e in snapshot["entities"]
        }

        # Quick lookup by entity_id for linked-entry state checks.
        entity_by_id: dict[str, SnapshotEntity] = {
            e["entity_id"]: e for e in snapshot["entities"]
        }

        # Index entity last_changed by entity_id for VMD/motion fallback lookup.
        last_changed_by_id: dict[str, str] = {
            e["entity_id"]: e["last_changed"] for e in snapshot["entities"]
        }

        for activity in snapshot["camera_activity"]:
            cam = activity["camera_entity_id"]
            area = activity.get("area")
            if not area:
                LOGGER.debug("%s: skipped — no area assigned in snapshot.", cam)
                continue
            last_activity = activity.get("last_activity")
            if not last_activity:
                # Camera has no activity timestamp attribute; use the most
                # recent last_changed of its associated VMD/motion sensors.
                sensor_ids = activity.get("vmd_entities", []) + activity.get(
                    "motion_entities", []
                )
                candidates = [
                    last_changed_by_id[sid]
                    for sid in sensor_ids
                    if sid in last_changed_by_id
                ]
                if not candidates:
                    # No linked sensors in camera_activity (camera doesn't
                    # advertise vmd_entity_id etc.); scan all binary sensors
                    # in the same area as a last resort.  Device-class is not
                    # checked because VMD sensors vary by manufacturer and
                    # often have no device_class; the area constraint is
                    # sufficient.
                    candidates = [
                        e["last_changed"]
                        for e in snapshot["entities"]
                        if e.get("area") == area and e["domain"] == "binary_sensor"
                    ]
                    LOGGER.debug(
                        "%s: area=%s area_binary_sensor_candidates=%s",
                        cam,
                        area,
                        candidates,
                    )
                last_activity = max(candidates) if candidates else None
            if not last_activity:
                LOGGER.debug("%s: skipped — no activity timestamp resolved.", cam)
                continue
            last_dt = dt_util.parse_datetime(last_activity)
            if last_dt is None:
                LOGGER.debug(
                    "%s: skipped — could not parse last_activity=%s.",
                    cam,
                    last_activity,
                )
                continue
            if now - last_dt > window:
                LOGGER.debug(
                    "%s: skipped — last_activity=%s is outside %d-min window.",
                    cam,
                    last_activity,
                    ACTIVITY_WINDOW_MIN,
                )
                continue

            # ---- same-area unsecured entries (original behaviour) ----
            unsecured_same_area = unsecured_by_area.get(area, [])

            # ---- cross-area linked entries ----
            unsecured_linked: list[str] = []
            for linked_eid in self._camera_entry_links.get(cam, []):
                ent = entity_by_id.get(linked_eid)
                if ent is None:
                    continue
                if ent["domain"] == "lock" and ent["state"] == "unlocked":
                    unsecured_linked.append(linked_eid)
                elif (
                    ent["domain"] == "binary_sensor"
                    and ent["attributes"].get("device_class") in ENTRY_CLASSES
                    and ent["state"] == "on"
                ):
                    unsecured_linked.append(linked_eid)

            unsecured_all = sorted(
                set(unsecured_same_area) | set(unsecured_linked)
            )
            if not unsecured_all:
                continue

            evidence = {
                "camera_entity_id": activity["camera_entity_id"],
                "area": area,  # kept for correlator Rule 1 compatibility
                "camera_area": area,  # explicit field for LLM spatial grounding
                "last_activity": last_activity,
                "unsecured_entities": unsecured_all,
                "unsecured_entity_areas": {
                    eid: all_entity_area_map.get(eid, "unknown")
                    for eid in unsecured_all
                },
            }
            # Hash only the same-area entities so that adding cross-area
            # linked entries does not invalidate suppression state for
            # already-notified findings.
            anomaly_id = build_anomaly_id(
                self.rule_id,
                [activity["camera_entity_id"]],
                {
                    "camera_entity_id": activity["camera_entity_id"],
                    "area": area,
                    "last_activity": last_activity,
                    "unsecured_entities": sorted(unsecured_same_area),
                },
            )
            findings.append(
                AnomalyFinding(
                    anomaly_id=anomaly_id,
                    type=self.rule_id,
                    severity="high",
                    confidence=0.6,
                    triggering_entities=[activity["camera_entity_id"]],
                    evidence=evidence,
                    suggested_actions=["check_entry"],
                    is_sensitive=True,
                ),
            )
        return findings
