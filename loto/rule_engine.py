from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union
import hashlib
import re

import yaml

__all__ = ["RulesError", "RulePack", "load"]


class RulesError(Exception):
    """Exception raised when rule pack validation fails."""

    def __init__(self, code: str, hint: str):
        self.code = code
        self.hint = hint
        super().__init__(f"{code}: {hint}")


@dataclass(frozen=True)
class Domain:
    """Rules for a single domain."""

    min_isolation: str
    verify: List[str]


@dataclass(frozen=True)
class RulePack:
    """Validated rules with deterministic hash."""

    version: str
    domains: Dict[str, Domain]
    _normalized: str

    def hash(self) -> str:
        """Return SHA256 hash of the normalized YAML representation."""
        return hashlib.sha256(self._normalized.encode("utf-8")).hexdigest()


_ALLOWED_DOMAINS = {"steam", "condensate", "instrument_air"}
_ALLOWED_MIN_ISOLATION = {
    "single_block",
    "double_block",
    "double_block_and_bleed",
}


def _raise(path: str, msg: str) -> None:
    raise RulesError(code="RULES/INVALID", hint=f"{path}: {msg}")


def _normalized_yaml(data: Dict[str, object]) -> str:
    """Return canonical YAML string with sorted keys and no comments."""
    return yaml.safe_dump(data, sort_keys=True, default_flow_style=False)


def load(source: Union[str, Path]) -> RulePack:
    """Load and validate a rule pack from YAML text or file path."""

    if isinstance(source, (str, Path)):
        src_str = str(source)
        if "\n" not in src_str:
            p = Path(src_str)
            try:
                if p.exists():
                    text = p.read_text()
                else:
                    text = src_str
            except OSError:
                text = src_str
        else:
            text = src_str
    else:
        text = str(source)

    try:
        raw = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - yaml lib formats message
        _raise("$", f"YAML parse error: {exc}")

    if not isinstance(raw, dict):
        _raise("$", "root must be a mapping")

    allowed_top = {"version", "domains"}
    for key in raw.keys():
        if key not in allowed_top:
            _raise(key, "unexpected field")

    if "version" not in raw:
        _raise("version", "missing")
    version = raw["version"]
    if not isinstance(version, str) or not re.fullmatch(r"\d+\.\d+\.\d+", version):
        _raise("version", "must be semver 'MAJOR.MINOR.PATCH'")

    if "domains" not in raw:
        _raise("domains", "missing")
    domains_raw = raw["domains"]
    if not isinstance(domains_raw, dict):
        _raise("domains", "must be a mapping")

    if set(domains_raw.keys()) != _ALLOWED_DOMAINS:
        missing = _ALLOWED_DOMAINS - set(domains_raw.keys())
        extra = set(domains_raw.keys()) - _ALLOWED_DOMAINS
        if missing:
            _raise("domains", f"missing {', '.join(sorted(missing))}")
        else:
            _raise(f"domains.{next(iter(extra))}", "unexpected domain")

    domains: Dict[str, Domain] = {}
    for name in sorted(_ALLOWED_DOMAINS):
        node = domains_raw.get(name)
        if not isinstance(node, dict):
            _raise(f"domains.{name}", "must be a mapping")
        allowed_keys = {"min_isolation", "verify"}
        for key in node.keys():
            if key not in allowed_keys:
                _raise(f"domains.{name}.{key}", "unexpected field")
        if "min_isolation" not in node:
            _raise(f"domains.{name}.min_isolation", "missing")
        min_iso = node["min_isolation"]
        if min_iso not in _ALLOWED_MIN_ISOLATION:
            _raise(
                f"domains.{name}.min_isolation",
                f"expected one of {sorted(_ALLOWED_MIN_ISOLATION)}, got '{min_iso}'",
            )
        if "verify" not in node:
            _raise(f"domains.{name}.verify", "missing")
        verify = node["verify"]
        if not isinstance(verify, list) or not verify or not all(
            isinstance(v, str) for v in verify
        ):
            _raise(
                f"domains.{name}.verify",
                "must be a non-empty list of strings",
            )
        domains[name] = Domain(min_isolation=min_iso, verify=verify)

    # Build canonical representation and compute hash
    canonical = {
        "version": version,
        "domains": {
            name: {"min_isolation": dom.min_isolation, "verify": dom.verify}
            for name, dom in domains.items()
        },
    }
    normalized = _normalized_yaml(canonical)
    return RulePack(version=version, domains=domains, _normalized=normalized)
