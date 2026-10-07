"""The committed model lock (``models.lock.yaml``): typed access, no I/O beyond reading the file."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Mapping

import yaml

LOCK_PATH = Path(__file__).with_name("models.lock.yaml")
META_FORMAT = "meta"          # tiny json sidecars (metadata, label maps): always installed


@dataclass(frozen=True)
class LockFile:
    src: str                  # path inside the Hub repo
    dest: str                 # path relative to the models root
    format: str               # onnx | torchscript | pytorch | meta
    sha256: str | None
    bytes: int | None
    optional: bool = False    # installed when available, but not required for the action to count as present


@dataclass(frozen=True)
class Unit:
    """One Hub repo contributing files to an action (an ensemble has several)."""
    repo_id: str
    revision: str | None
    model_key: str | None
    files: tuple[LockFile, ...]


@dataclass(frozen=True)
class Action:
    key: str
    required: bool
    placeholder: bool         # listed, but not on the Hub yet: never fetched, reported as pending
    units: tuple[Unit, ...]
    stage: str = ""           # the LM3 module this model serves (== key for defaults)
    model_key: str = ""       # the model's name (Hub repo suffix); set for alternates
    settings: Mapping = field(default_factory=dict)   # alternates: settings lines to point the stage at it

    def files(self, formats: Iterable[str] | None = None) -> Iterator[tuple[Unit, LockFile]]:
        """The (unit, file) pairs to install for ``formats`` (meta files always included)."""
        wanted = set(formats or ())
        for u in self.units:
            for f in u.files:
                if f.format == META_FORMAT or f.format in wanted:
                    yield u, f


@dataclass(frozen=True)
class Lock:
    schema_version: int
    lm3_version: str
    default_formats: tuple[str, ...]
    actions: Mapping[str, Action] = field(default_factory=dict)
    path: str | None = None
    #: non-default models, {stage: {model_key: Action}}; installed only on request, never "missing"
    alternates: Mapping[str, Mapping[str, Action]] = field(default_factory=dict)

    def action(self, key: str) -> Action:
        try:
            return self.actions[key]
        except KeyError:
            raise KeyError(f"{key!r} is not in the model lock (known: {', '.join(self.actions)})") from None

    def alternate(self, stage: str, model_key: str) -> Action:
        try:
            return self.alternates[stage][model_key]
        except KeyError:
            known = [f"{s}={k}" for s, ks in self.alternates.items() for k in ks]
            raise KeyError(f"{stage}={model_key} is not an alternate model in the lock (known: {', '.join(known) or 'none'})") from None

    def all_actions(self) -> Iterator[Action]:
        """Defaults then alternates."""
        yield from self.actions.values()
        for ks in self.alternates.values():
            yield from ks.values()


from typing import Iterable  # noqa: E402  (kept below the dataclasses that reference it in annotations)


def _units(spec: Mapping) -> tuple[Unit, ...]:
    return tuple(
        Unit(repo_id=str(u["repo_id"]), revision=u.get("revision"), model_key=u.get("model_key"),
             files=tuple(LockFile(src=str(f["src"]), dest=str(f["dest"]), format=str(f.get("format", "onnx")),
                                  sha256=f.get("sha256"), bytes=f.get("bytes"), optional=bool(f.get("optional", False)))
                         for f in u.get("files") or ()))
        for u in spec.get("units") or ())


def alternate_record_key(stage: str, model_key: str) -> str:
    """The key an alternate is recorded under in installed.json."""
    return f"{stage}__{model_key}"


def parse_lock(data: Mapping, path: str | None = None) -> Lock:
    actions: dict[str, Action] = {}
    for key, spec in (data.get("actions") or {}).items():
        actions[key] = Action(key=key, required=bool(spec.get("required", True)),
                              placeholder=bool(spec.get("placeholder", False)), units=_units(spec), stage=key,
                              model_key=str(((spec.get("units") or [{}])[0] or {}).get("model_key") or ""))
    alternates: dict[str, dict[str, Action]] = {}
    for stage, models in (data.get("alternates") or {}).items():
        for model_key, spec in (models or {}).items():
            alternates.setdefault(stage, {})[model_key] = Action(
                key=alternate_record_key(stage, model_key), required=False, placeholder=False, units=_units(spec),
                stage=stage, model_key=model_key, settings=dict(spec.get("settings") or {}))
    return Lock(schema_version=int(data.get("schema_version", 1)), lm3_version=str(data.get("lm3_version", "")),
                default_formats=tuple(data.get("default_formats") or ("onnx",)), actions=actions, path=path,
                alternates=alternates)


def load_lock(path: str | os.PathLike[str] | None = None) -> Lock:
    """Read the lock shipped with this LM3 (or ``path`` / ``$LM3_MODELS_LOCK`` for testing)."""
    p = Path(path or os.environ.get("LM3_MODELS_LOCK") or LOCK_PATH)
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return parse_lock(data, path=str(p))
