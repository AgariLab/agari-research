"""Content identities for selecting behavior labels from authenticated traces.

Filesystem locations are provenance, not agent identity. Configuration, model
bytes and implementation bytes remain part of an external agent's identity.
"""

from __future__ import annotations

import hashlib
import re

from agari_kit.mjai import DataError, canonical_json


def _sha256(value) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise DataError("Agent identity requires lowercase SHA-256 content hashes")
    return value


def agent_content_identity(provenance: dict) -> dict:
    """Project a recorded kit agent_provenance onto its runtime content."""
    if not isinstance(provenance, dict):
        raise DataError("Agent identity requires recorded provenance")
    if provenance.get("kind") == "builtin":
        from agari_kit.agent import AGENT_NAMES
        if provenance.get("name") not in AGENT_NAMES:
            raise DataError("Unknown builtin agent identity")
        return {"kind": "builtin", "name": provenance["name"]}
    descriptor, source = provenance.get("descriptor"), provenance.get("source")
    artifacts = provenance.get("artifacts")
    factory = provenance.get("factory")
    if (provenance.get("kind") != "external" or not isinstance(descriptor, dict)
            or not isinstance(source, dict) or not isinstance(artifacts, dict)
            or not isinstance(factory, str) or descriptor.get("factory") != factory
            or not isinstance(descriptor.get("kwargs", {}), dict)
            or not isinstance(descriptor.get("artifacts", {}), dict)
            or set(artifacts) != set(descriptor.get("artifacts", {}))):
        raise DataError("External agent identity has incomplete or inconsistent provenance")
    content = {}
    for name, artifact in artifacts.items():
        declared = descriptor["artifacts"][name]
        if not isinstance(artifact, dict) or not isinstance(declared, dict):
            raise DataError("Invalid external agent artifact provenance")
        digest = _sha256(artifact.get("sha256"))
        if artifact.get("declared_sha256") != digest or declared.get("sha256") != digest:
            raise DataError("External agent artifact hashes disagree")
        content[name] = digest
    if (not isinstance(source.get("package"), str) or not source["package"]
            or not isinstance(source.get("module"), str)
            or source["module"] != factory.split(":", 1)[0]):
        raise DataError("External agent source identity disagrees with its factory")
    return {"kind": "external", "factory": factory, "kwargs": descriptor.get("kwargs", {}),
            "artifacts": content, "source": {"package": source["package"], "module": source["module"],
                                              "source_sha256": _sha256(source.get("source_sha256"))}}


def agent_identity_sha256(provenance: dict) -> str:
    return hashlib.sha256(canonical_json(agent_content_identity(provenance)).encode()).hexdigest()


def normalize_policy_filter(value) -> dict | None:
    """None selects all behavior; an explicit filter must be nonempty."""
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"agent_identity_sha256"}:
        raise DataError("policy_filter requires only agent_identity_sha256")
    values = value["agent_identity_sha256"]
    if not isinstance(values, list) or not values:
        raise DataError("policy_filter requires at least one agent content identity")
    hashes = [_sha256(item) for item in values]
    if len(hashes) != len(set(hashes)):
        raise DataError("policy_filter contains duplicate agent identities")
    return {"agent_identity_sha256": sorted(hashes)}
