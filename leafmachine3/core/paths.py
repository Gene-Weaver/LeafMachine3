"""leafmachine3.core.paths -- the ONE canonical path resolver.

Every path LM3 needs at startup is resolved here, by one chain per path, with no step that
falls back to the current working directory. That single rule is what makes "the GUI and the
CLI disagree depending on where you launched them" impossible (plan section 3.1, gate 44).

Three identities are derived here because everything else hangs off them:

``deployment key``   ``LM3_DEPLOYMENT_ID`` canonicalized to a filesystem-safe, JavaScript-
                     reproducible token (section 2.1). Electron derives the SAME key for its
                     single-instance lock, so the algorithm is deliberately mechanical ASCII --
                     see :func:`ascii_slug` -- and the authority is the sha256 of the raw UTF-8
                     bytes, never the cosmetic slug.
``machine key``      identifies the HOST a hardware profile describes (section 3.1). ``node:`` is
                     unconditional so two containerized nodes that share one baked
                     ``/etc/machine-id`` still derive different keys, CPU-only nodes included.
``runtime base``     ``LM3_RUNTIME_DIR`` names the BASE directory; the deployment directory is
                     ``<base>/<canonical deployment key>``. Both readings were plausible, so the
                     base reading is stated normatively: one node-local scratch path hosts several
                     deployments, which the Slurm profile depends on.

Injection, not monkeypatching. Every function that reads the environment takes ``env``; every
function that probes the machine takes a reader callable. That is how the tests drive Windows
drive types, NFS mounts, duplicated container machine IDs, and empty GPU sets from Linux without
touching the developer's real ``~/.config``, ``~/.local/state`` or ``$XDG_RUNTIME_DIR``.

Two interpretations this module had to settle, both recorded here rather than buried in code:

* ``LM3_SETTINGS`` naming a file that does not exist is a HARD ERROR, exactly like an explicit
  ``--config`` that names nothing. Falling through to the deployment default would silently run a
  configuration the user did not choose -- the class of surprise this refactor exists to remove.
  Only the *unset* case falls through to rows 3-5 and to seeding.
* A RELATIVE ``project.output.dir`` is resolved against the SETTINGS FILE's parent. The tree today
  has four incompatible rules for that one field; a settings-relative reading is the only one that
  is stable under "the resolver, not the launcher, decides", because the settings path itself is
  now canonical.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

log = logging.getLogger("leafmachine3.paths")

# -- constants ------------------------------------------------------------------------------- #

DEFAULT_DEPLOYMENT_ID = "default"
DEFAULT_PORT = 8765

MACHINE_KEY_VERSION = "machine-key/v1"
MACHINE_KEY_LENGTH = 16          # hex chars of sha256 kept for <machine-key>
DEPLOYMENT_HASH_LENGTH = 8       # hex chars of sha256 appended to the slug
DEPLOYMENT_SLUG_LENGTH = 32      # slug characters kept before the hash

APP_DIRNAME = "lm3"
SETTINGS_FILENAME = "LM3_settings.yaml"
POSTPROCESS_FILENAME = "postprocessing.yaml"
WORKSPACE_FILENAME = "workspace.json"
WORKSPACE_SCHEMA_VERSION = 1
LEGACY_HARDWARE_FILENAME = "hardware_settings.yaml"

#: The pre-section-3.1 postprocessing settings filename. It lived at the CHECKOUT ROOT and was
#: reached by a bare CWD-relative default in the standalone CLIs; the canonical file is
#: POSTPROCESS_FILENAME inside the deployment config directory.
LEGACY_POSTPROCESS_FILENAME = "postprocessing_settings.yaml"

# Files inside a deployment runtime directory (section 3.1). Named here so core/runtime and the
# server agree on spelling; this module does not create or interpret any of them.
ACTIVITY_LOCK_FILENAME = "activity.lock"
ACTIVE_RECORD_FILENAME = "active.json"
LAST_RECORD_FILENAME = "last.json"
CHILDREN_DIRNAME = "children"
CONNECTION_PRIVATE_FILENAME = "connection.private.json"
CONNECTION_PUBLIC_FILENAME = "connection.public.json"

# Canonical environment variable names (section 3.1). Anything else is legacy or non-canonical.
ENV_SETTINGS = "LM3_SETTINGS"
ENV_HARDWARE = "LM3_HARDWARE"
ENV_POSTPROCESS_SETTINGS = "LM3_POSTPROCESS_SETTINGS"
ENV_SERVER_JOBS = "LM3_SERVER_JOBS"
ENV_RUNS_ROOTS = "LM3_RUNS_ROOTS"
ENV_RUNTIME_DIR = "LM3_RUNTIME_DIR"
ENV_DEPLOYMENT_ID = "LM3_DEPLOYMENT_ID"
ENV_PORT = "LM3_PORT"
ENV_ALLOW_NETWORK_RUNTIME = "LM3_ALLOW_NETWORK_RUNTIME"

# One-release migration (plan section 4, Step 1). Legacy alone -> deprecation warning; legacy and
# canonical agreeing -> warn once; legacy and canonical disagreeing -> startup error.
LEGACY_ENV_ALIASES: dict[str, str] = {
    ENV_SETTINGS: "LM3_SETTINGS_PATH",
    ENV_HARDWARE: "LM3_HARDWARE_SETTINGS",
}

# Filesystem types whose advisory locking cannot be trusted. An unreliable lock is worse than no
# lock because it looks like one, so these are REFUSED rather than documented (section 3.1).
NETWORK_FS_TYPES: frozenset[str] = frozenset({
    "9p", "afs", "beegfs", "ceph", "cifs", "coda", "davfs", "davfs2", "fuse.cephfs",
    "fuse.davfs2", "fuse.glusterfs", "fuse.sshfs", "gfs2", "glusterfs", "gpfs", "lustre",
    "ncpfs", "nfs", "nfs4", "ocfs2", "panfs", "smb", "smb2", "smb3", "smbfs", "sshfs",
    "unc", "remote",
})

_TRUTHY = frozenset({"1", "true", "yes", "on"})

# The package directory, i.e. ``Path(leafmachine3.__file__).parent``. Computed from __file__ so
# importing the package's own __init__ is not required and so dev-checkout detection never has to
# consult the CWD.
_PACKAGE_DIR = Path(__file__).resolve().parent.parent
_SITE_DIR_PARTS = frozenset({"site-packages", "dist-packages"})
_CHECKOUT_MARKERS = ("pyproject.toml", ".git")

# "Warn ONCE" in the legacy-alias migration (plan section 4, Step 1) is a per-PROCESS promise,
# not a per-call one, and it has to hold for every entry point -- the CLI, hardware setup and
# calibrate resolve these variables outside the server, so a server-side memo cannot deliver it.
# The key is the exact (canonical, new, old) triple so a CHANGED environment warns again and the
# latch only ever suppresses a byte-identical repeat.
_LEGACY_WARNED: set[tuple[str, str | None, str | None]] = set()
_LEGACY_WARN_LOCK = threading.Lock()


# -- errors ---------------------------------------------------------------------------------- #

class PathsError(RuntimeError):
    """Base class for every startup-fatal path resolution failure."""


class DeploymentIdentityError(PathsError):
    """``LM3_DEPLOYMENT_ID`` is empty or whitespace-only."""


class DeploymentPortError(PathsError):
    """A named non-default deployment was started without ``LM3_PORT`` (gate 41)."""


class RuntimeDirectoryError(PathsError):
    """No safe runtime base directory could be resolved."""


class NetworkRuntimeRefused(RuntimeDirectoryError):
    """The runtime base directory sits on a network filesystem (gate 42)."""

    def __init__(self, path: Path, fs_type: str) -> None:
        self.path = Path(path)
        self.fs_type = fs_type
        super().__init__(
            f"refusing to place the LM3 runtime directory on a {fs_type} filesystem: {self.path}. "
            f"Advisory locks are unreliable there, and an unreliable lock is worse than no lock. "
            f"Set {ENV_RUNTIME_DIR} to node-local storage, or set "
            f"{ENV_ALLOW_NETWORK_RUNTIME}=1 to override and accept the risk."
        )


class SettingsMissingError(PathsError):
    """An explicitly stated settings path names a file that does not exist."""


class WorkspacePointerError(PathsError):
    """``workspace.json`` is unreadable or does not match its two-key schema."""


class WorkspaceTargetMissingError(WorkspacePointerError):
    """The workspace pointer is valid but the settings file it selects is gone (gate 15)."""

    def __init__(self, settings_path: Path, pointer_path: Path) -> None:
        self.settings_path = Path(settings_path)
        self.pointer_path = Path(pointer_path)
        super().__init__(
            f"selected settings file is missing: {self.settings_path} "
            f"(selected by {self.pointer_path}). Choose a settings file again, or clear the pointer. "
            f"LM3 will not silently fall back to the default settings."
        )


class LegacyEnvConflictError(PathsError):
    """A canonical variable and its legacy alias resolve to different files."""


class PackagedResourceError(PathsError):
    """A resource that must ship with the package is not present in this installation."""


class RelativeEnvPathError(PathsError):
    """A canonical variable holds a relative path.

    Its meaning would then be decided by each process's working directory, so the GUI and the
    CLI would disagree about the same variable depending on where they were launched -- the
    divergence plan section 3.1 (line 1036, "independent of the checkout and CWD") and gate 44
    exist to remove. Worse for ``LM3_RUNTIME_DIR``: two processes would place ``activity.lock``
    on different inodes and BOTH acquire the lease, breaking invariant 1.
    """


# -- small helpers --------------------------------------------------------------------------- #

def _env(env: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if env is None else env


def _env_value(env: Mapping[str, str] | None, name: str) -> str | None:
    """Read ``name``, treating an empty/whitespace-only value as unset (matching today's code)."""
    raw = _env(env).get(name)
    if raw is None:
        return None
    raw = raw.strip()
    return raw or None


def _is_true(value: str | None) -> bool:
    return bool(value) and value.strip().lower() in _TRUTHY


def _expand(env: Mapping[str, str] | None, raw: str) -> Path:
    """Expand ``~`` against the injected HOME, never against the process CWD."""
    path = Path(raw)
    if raw.startswith("~"):
        home = _env_value(env, "HOME") or _env_value(env, "USERPROFILE")
        if home:
            path = Path(raw.replace("~", home, 1))
        else:
            path = path.expanduser()
    return path


def _expand_env_path(
    env: Mapping[str, str] | None,
    raw: str,
    *,
    variable: str,
    error: type[PathsError] = RelativeEnvPathError,
) -> Path:
    """Expand a value read from a CANONICAL ENVIRONMENT VARIABLE, refusing a relative one.

    Deliberately separate from :func:`_expand`, which also serves explicit arguments: a relative
    ``--config`` typed at a shell is one invocation with one unambiguous CWD, while an exported
    variable is read by every process in the deployment. Nothing here absolutizes against the
    CWD -- that would bake the divergence in rather than remove it.
    """
    path = _expand(env, raw)
    if not path.is_absolute():
        raise error(
            f"{variable}={raw!r} is a relative path. LM3 resolves it against no base -- "
            f"a relative value would mean a different file in every process's working "
            f"directory. Set {variable} to an absolute path."
        )
    return path


def _legacy_env_source(env: Mapping[str, str] | None, canonical: str) -> str:
    """Which variable actually supplied the value :func:`resolve_legacy_env` returned.

    A refusal must name the variable the OPERATOR set, not the canonical name they never typed,
    or the message sends them looking for a variable that is not in their environment.
    """
    if _env_value(env, canonical) is not None:
        return canonical
    return LEGACY_ENV_ALIASES.get(canonical) or canonical


def _home(env: Mapping[str, str] | None) -> Path:
    home = _env_value(env, "HOME") or _env_value(env, "USERPROFILE")
    return Path(home) if home else Path.home()


def _platform(platform_name: str | None) -> str:
    return sys.platform if platform_name is None else platform_name


def _mkdir_private(path: Path) -> Path:
    """Create ``path`` (and parents) user-only where the platform supports mode bits."""
    path.mkdir(parents=True, exist_ok=True)
    if os.name == "posix":
        try:
            path.chmod(0o700)
        except OSError:  # pragma: no cover - exotic filesystems refuse chmod
            log.debug("could not tighten permissions on %s", path)
    return path


def _write_atomic(path: Path, text: str, *, mode: int | None = None) -> None:
    """Temp sibling -> fsync -> os.replace -> fsync the directory. The plan's update rule."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    fd = os.open(tmp, flags, 0o600 if mode is None else mode)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        if mode is not None and os.name == "posix":
            os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    _fsync_dir(path.parent)


def _fsync_dir(directory: Path) -> None:
    if os.name != "posix":  # Windows cannot open a directory for fsync; the replace is atomic there
        return
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:  # pragma: no cover
        return
    try:
        os.fsync(fd)
    except OSError:  # pragma: no cover
        pass
    finally:
        os.close(fd)


def _install_if_absent(path: Path, text: str, *, mode: int = 0o600) -> bool:
    """Create ``path`` with ``text`` ONLY if it does not already exist. Atomic. Never clobbers.

    Returns True if this call created the file, False if somebody else already had it.

    ``_write_atomic`` is the wrong tool for an adopt: its ``os.replace`` overwrites unconditionally,
    so a `if not target.exists(): _write_atomic(target, ...)` sequence is a check-then-act race. The
    window is small but it is real and it matters here, because until Step 5b lands there is nothing
    stopping a CLI and the GUI server from performing the same one-release adopt concurrently -- and
    the loser would silently overwrite a file the winner had already published, or worse, a profile
    the user had just saved.

    The primitive is ``os.link``, which fails with ``FileExistsError`` when the destination exists
    and is atomic on POSIX and on Windows NTFS. Content is written and fsynced to a temp sibling
    first, so the link publishes a complete file rather than an empty one. Where hardlinks are
    unsupported this returns False with a warning rather than writing the destination directly --
    see the comment on that branch.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.adopt.tmp")
    try:
        fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            return False                      # another process adopted first; leave its file alone
        except (OSError, AttributeError) as exc:
            # No hardlink support on this filesystem. Do NOT fall back to writing the destination
            # directly: O_CREAT|O_EXCL would settle the create decision safely, but the file becomes
            # visible at creation and a concurrent reader could parse it half-written -- publishing
            # truncated YAML as though it were the deployment's settings. Migration is best-effort
            # and the caller degrades to packaged defaults, so skipping loudly is strictly safer
            # than publishing a partial file.
            log.warning(
                "could not atomically install %s (%s); skipping the one-release migration. "
                "Set the corresponding LM3_* variable explicitly, or copy the file by hand.",
                path, exc)
            return False
        _fsync_dir(path.parent)
        return True
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


# -- deployment identity (section 2.1) ------------------------------------------------------- #

def ascii_slug(raw: str) -> str:
    """The plan's character-by-character slug. Deliberately NOT casefold and NOT NFKC.

    ``str.casefold()`` and NFKC are not reproducible in JavaScript -- JS has ``toLowerCase()``,
    which differs (``ss`` vs ``ss`` for U+00DF), and normalization tables move with engine
    versions. A key that disagrees between Python and Electron produces a directory mismatch
    *before* any golden test could catch it, so the slug is mechanical ASCII and cosmetic; the
    sha256 of the raw UTF-8 bytes is what actually separates namespaces.
    """
    out: list[str] = []
    for ch in raw:
        code = ord(ch)
        if 0x41 <= code <= 0x5A:            # ASCII A-Z -> lowercase by adding 0x20
            out.append(chr(code + 0x20))
        elif 0x61 <= code <= 0x7A or 0x30 <= code <= 0x39:   # ASCII a-z, 0-9 kept
            out.append(ch)
        else:                                # EVERYTHING else, every non-ASCII byte included
            out.append("-")
    collapsed: list[str] = []
    for ch in out:                           # collapse runs of "-"
        if ch == "-" and collapsed and collapsed[-1] == "-":
            continue
        collapsed.append(ch)
    return "".join(collapsed).strip("-")


def canonical_deployment_key(raw: str) -> str:
    """``ascii_slug(raw)[:32] + "-" + sha256(raw.encode("utf-8")).hexdigest()[:8]``, literally.

    Note the literal formula keeps the separator even when the slug is empty, so a raw value made
    entirely of separators canonicalizes to ``-<hash8>``. That is intentional: Electron implements
    the same formula, and a Python-side "tidy up the leading dash" would desynchronize the two.
    """
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:DEPLOYMENT_HASH_LENGTH]
    return f"{ascii_slug(raw)[:DEPLOYMENT_SLUG_LENGTH]}-{digest}"


def raw_deployment_id(env: Mapping[str, str] | None = None) -> str:
    """The raw ``LM3_DEPLOYMENT_ID``. Unset means the literal ``"default"``. Nothing else."""
    raw = _env(env).get(ENV_DEPLOYMENT_ID)
    if raw is None:
        return DEFAULT_DEPLOYMENT_ID
    if not raw.strip():
        raise DeploymentIdentityError(
            f"{ENV_DEPLOYMENT_ID} is empty or whitespace-only. Unset it for the default deployment, "
            f"or give it a non-empty name."
        )
    return raw


def deployment_key(env: Mapping[str, str] | None = None) -> str:
    """The canonical deployment key for this process."""
    return canonical_deployment_key(raw_deployment_id(env))


def is_default_deployment(env: Mapping[str, str] | None = None) -> bool:
    """True when this process runs the default deployment (unset is identical to ``default``)."""
    return raw_deployment_id(env) == DEFAULT_DEPLOYMENT_ID


def resolve_port(env: Mapping[str, str] | None = None) -> int:
    """Server port for this deployment. Gate 41: a named deployment MUST state its port.

    Deliberately NOT called during path resolution: pytest gives every worker a named deployment,
    and a resolver that demanded a port would fail every test at import.
    """
    raw = _env_value(env, ENV_PORT)
    if raw is not None:
        try:
            port = int(raw)
        except ValueError as exc:
            raise DeploymentPortError(f"{ENV_PORT} is not an integer: {raw!r}") from exc
        if not 1 <= port <= 65535:
            raise DeploymentPortError(f"{ENV_PORT} is out of range: {port}")
        return port
    if is_default_deployment(env):
        return DEFAULT_PORT
    raise DeploymentPortError(
        f"deployment {raw_deployment_id(env)!r} is not the default deployment, so it must set "
        f"{ENV_PORT} explicitly. Starting it without one would silently collide with the default "
        f"deployment on port {DEFAULT_PORT}."
    )


# -- machine key (section 3.1) --------------------------------------------------------------- #

@dataclass(frozen=True)
class MachineIdentity:
    """The three inputs to the machine key, and nothing else.

    Driver versions, ORT/provider versions, selected devices, enabled stages and model
    fingerprints stay in the PROFILE's contents as staleness fields. If they entered the key,
    every config change would orphan a profile instead of invalidating it.
    """

    os_machine_id: str | None      # /etc/machine-id, IOPlatformUUID, MachineGuid
    node: str                      # SLURM_NODENAME or hostname -- ALWAYS present
    gpu_uuids: tuple[str, ...] = ()


def read_os_machine_id(platform_name: str | None = None) -> str | None:
    """Best-effort trusted machine identity. Never raises; ``None`` becomes an explicit marker."""
    plat = _platform(platform_name)
    try:
        if plat.startswith("linux"):
            for candidate in (Path("/etc/machine-id"), Path("/var/lib/dbus/machine-id")):
                try:
                    value = candidate.read_text(encoding="utf-8").strip()
                except OSError:
                    continue
                if value:
                    return value
            return None
        if plat == "darwin":
            import subprocess  # noqa: PLC0415 - macOS-only, kept out of the import path

            out = subprocess.run(
                ["/usr/sbin/ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
                capture_output=True, text=True, timeout=10, check=False,
            ).stdout
            for line in out.splitlines():
                if "IOPlatformUUID" in line and '"' in line:
                    return line.rsplit('"', 2)[-2] or None
            return None
        if plat.startswith("win"):
            import winreg  # noqa: PLC0415 - Windows-only; runtime check, never an import-time one

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography") as key:
                value, _ = winreg.QueryValueEx(key, "MachineGuid")
            return str(value).strip() or None
    except Exception:  # noqa: BLE001 - identity probing must never break startup
        log.debug("could not read an OS machine id", exc_info=True)
    return None


def read_node_name(env: Mapping[str, str] | None = None) -> str:
    """``SLURM_NODENAME`` when the scheduler names this node, else the hostname."""
    return _env_value(env, "SLURM_NODENAME") or socket.gethostname()


def read_gpu_uuids() -> tuple[str, ...]:
    """Sorted NVML GPU UUIDs, or an empty tuple. Never raises; CPU-only nodes are normal."""
    try:
        import pynvml  # noqa: PLC0415 - optional dependency, probed at call time

        pynvml.nvmlInit()
        try:
            uuids = []
            for index in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                uuid = pynvml.nvmlDeviceGetUUID(handle)
                if isinstance(uuid, bytes):
                    uuid = uuid.decode("utf-8", "replace")
                if uuid:
                    uuids.append(str(uuid).strip())
        finally:
            pynvml.nvmlShutdown()
        return tuple(sorted(uuids))
    except Exception:  # noqa: BLE001 - no NVML, no driver, no GPUs: all mean "none"
        log.debug("no NVML GPU UUIDs available", exc_info=True)
        return ()


def detect_machine_identity(
    env: Mapping[str, str] | None = None,
    *,
    machine_id_reader: Callable[[], str | None] | None = None,
    node_reader: Callable[[], str] | None = None,
    gpu_uuid_reader: Callable[[], Iterable[str]] | None = None,
    platform_name: str | None = None,
) -> MachineIdentity:
    """Probe this host. Every probe is injectable so tests drive every combination from Linux."""
    machine_id = (machine_id_reader or (lambda: read_os_machine_id(platform_name)))()
    node = (node_reader or (lambda: read_node_name(env)))()
    uuids = tuple(sorted((gpu_uuid_reader or read_gpu_uuids)()))
    machine_id = (machine_id or "").strip() or None
    return MachineIdentity(os_machine_id=machine_id, node=str(node).strip(), gpu_uuids=uuids)


def machine_key_input(identity: MachineIdentity) -> str:
    """The canonical serialization the key hashes. Pinned by golden vectors.

    Every field is ALWAYS present with an explicit ``none`` marker, and ``node:`` is
    unconditional -- never merely a fallback. That is what makes two containerized nodes sharing
    one baked ``/etc/machine-id`` derive different keys, CPU-only nodes included (gate 20).
    """
    machine_id = identity.os_machine_id if identity.os_machine_id else "none"
    gpus = ",".join(sorted(identity.gpu_uuids)) if identity.gpu_uuids else "none"
    return (
        f"{MACHINE_KEY_VERSION}\n"
        f"os-machine-id: {machine_id}\n"
        f"node: {identity.node}\n"
        f"gpu-uuids: {gpus}\n"
    )


#: Memoized default machine key. The probe behind it initializes, enumerates and shuts down NVML;
#: the status stream resolves the hardware profile at 2 Hz, so an uncached probe means an NVML cycle
#: twice a second forever. The identity it hashes -- machine-id, node name, GPU UUIDs -- cannot
#: change inside a process lifetime, so caching it is not merely an optimization, it is correct.
_MACHINE_KEY_CACHE: str | None = None


def machine_key(identity: MachineIdentity | None = None, **detect_kwargs: Any) -> str:
    """sha256 of :func:`machine_key_input`, truncated to 16 hex characters.

    The no-argument call is memoized for the life of the process (see ``_MACHINE_KEY_CACHE``).
    Passing an explicit ``identity`` or any detect kwarg bypasses the cache entirely, so tests can
    still drive every combination. Call :func:`reset_machine_key_cache` between them.
    """
    global _MACHINE_KEY_CACHE
    cacheable = identity is None and not detect_kwargs
    if cacheable and _MACHINE_KEY_CACHE is not None:
        return _MACHINE_KEY_CACHE
    if identity is None:
        identity = detect_machine_identity(**detect_kwargs)
    digest = hashlib.sha256(machine_key_input(identity).encode("utf-8")).hexdigest()
    key = digest[:MACHINE_KEY_LENGTH]
    if cacheable:
        _MACHINE_KEY_CACHE = key
    return key


def reset_machine_key_cache() -> None:
    """Drop the memoized machine key. For tests, and for a process that fakes the identity."""
    global _MACHINE_KEY_CACHE
    _MACHINE_KEY_CACHE = None


# -- filesystem type detection (section 3.1) ------------------------------------------------- #

def _nearest_existing(path: Path) -> Path:
    """Walk up to the closest ancestor that exists -- the base often has not been created yet."""
    candidate = Path(path)
    while True:
        if candidate.exists():
            return candidate
        parent = candidate.parent
        if parent == candidate:
            return candidate
        candidate = parent


def _normalize_for_mount(path: Path) -> Path:
    """Resolve symlinks in the part that exists, then re-attach the part that does not.

    Resolving the whole path would collapse a not-yet-created ``/mnt/share/lm3-runtime`` to ``/``
    and hand back the ROOT filesystem's type -- which is exactly how a refused NFS runtime
    directory would slip through as ``ext4``.
    """
    target = Path(path)
    existing = _nearest_existing(target)
    try:
        resolved = existing.resolve()
    except OSError:  # pragma: no cover
        resolved = existing
    rest = target.parts[len(existing.parts):]
    return resolved.joinpath(*rest) if rest else resolved


def _read_mountinfo() -> str:
    return Path("/proc/self/mountinfo").read_text(encoding="utf-8", errors="replace")


def _linux_fs_type(path: Path, *, mountinfo_reader: Callable[[], str] | None = None) -> str | None:
    """Longest-prefix mount point match from /proc/self/mountinfo -> its filesystem type."""
    try:
        blob = (mountinfo_reader or _read_mountinfo)()
    except OSError:
        return None
    target = _normalize_for_mount(path)
    best: tuple[int, str] | None = None
    for line in blob.splitlines():
        # <id> <parent> <maj:min> <root> <mount point> <opts> [<optional>...] - <fstype> <src> <sopts>
        head, sep, tail = line.partition(" - ")
        if not sep:
            continue
        head_fields = head.split(" ")
        tail_fields = tail.split(" ")
        if len(head_fields) < 5 or not tail_fields:
            continue
        mount_point = head_fields[4].replace("\\040", " ").replace("\\011", "\t")
        fstype = tail_fields[0].strip().lower()
        try:
            mount = Path(mount_point)
        except ValueError:  # pragma: no cover
            continue
        if target == mount or mount in target.parents:
            depth = len(mount.parts)
            if best is None or depth >= best[0]:
                best = (depth, fstype)
    return best[1] if best else None


def _windows_fs_type(path: Path, *, get_drive_type: Callable[[str], int] | None = None) -> str | None:
    """UNC paths and mapped network drives. ``get_drive_type`` is injected by the Linux tests."""
    import ntpath  # noqa: PLC0415 - Windows path grammar, usable from any host for the tests

    raw = str(path)
    if raw.startswith("\\\\") or raw.startswith("//"):
        return "unc"
    drive, _ = ntpath.splitdrive(raw)
    if not drive:
        return None
    if get_drive_type is None:  # pragma: no cover - only reachable on a real Windows host
        import ctypes  # noqa: PLC0415 - Windows-only; behind a runtime platform check

        get_drive_type = ctypes.windll.kernel32.GetDriveTypeW  # type: ignore[attr-defined]
    try:
        kind = int(get_drive_type(drive + "\\"))
    except Exception:  # noqa: BLE001 - a failed probe must not block startup
        return None
    return "remote" if kind == 4 else None   # DRIVE_REMOTE


def filesystem_type(
    path: str | os.PathLike[str],
    *,
    platform_name: str | None = None,
    mountinfo_reader: Callable[[], str] | None = None,
    get_drive_type: Callable[[str], int] | None = None,
) -> str | None:
    """Filesystem type backing ``path``, or ``None`` when this platform cannot say.

    macOS is deliberately ``None`` today: there is no dependency-free ``statfs`` binding, and a
    guessed answer would either refuse a working local disk or bless an NFS mount.
    """
    plat = _platform(platform_name)
    target = Path(path)
    if plat.startswith("linux"):
        return _linux_fs_type(target, mountinfo_reader=mountinfo_reader)
    if plat.startswith("win"):
        return _windows_fs_type(target, get_drive_type=get_drive_type)
    return None


def is_network_filesystem(fs_type: str | None) -> bool:
    """True for a network-backed type. ``fuse.<x>`` is matched by its ``<x>`` too."""
    if not fs_type:
        return False
    name = fs_type.strip().lower()
    if name in NETWORK_FS_TYPES:
        return True
    if name.startswith("fuse."):
        return name.split(".", 1)[1] in NETWORK_FS_TYPES
    return False


def check_runtime_filesystem(
    path: str | os.PathLike[str],
    *,
    env: Mapping[str, str] | None = None,
    fs_type_reader: Callable[[Path], str | None] | None = None,
) -> str | None:
    """Refuse a network-backed runtime directory, unless explicitly overridden (gate 42).

    Applies to the desktop cache fallback exactly as it applies to the cluster path.
    """
    target = Path(path)
    reader = fs_type_reader or (lambda p: filesystem_type(p))
    fs_type = reader(target)
    if not is_network_filesystem(fs_type):
        return fs_type
    if _is_true(_env_value(env, ENV_ALLOW_NETWORK_RUNTIME)):
        log.warning(
            "%s=1: placing the LM3 runtime directory on a %s filesystem (%s). Advisory locks are "
            "UNRELIABLE there -- two LM3 activities may both believe they hold this deployment's "
            "lease and corrupt one another's run. You have accepted that risk explicitly.",
            ENV_ALLOW_NETWORK_RUNTIME, fs_type, target,
        )
        return fs_type
    raise NetworkRuntimeRefused(target, str(fs_type))


# -- platform user directories --------------------------------------------------------------- #

def user_config_dir(env: Mapping[str, str] | None = None, platform_name: str | None = None) -> Path:
    """Platform user-config root. ``<user-config>/lm3/<deployment>/`` hangs off this."""
    plat = _platform(platform_name)
    if plat.startswith("win"):
        appdata = _env_value(env, "APPDATA")
        return Path(appdata) if appdata else _home(env) / "AppData" / "Roaming"
    if plat == "darwin":
        return _home(env) / "Library" / "Application Support"
    xdg = _env_value(env, "XDG_CONFIG_HOME")
    return Path(xdg) if xdg else _home(env) / ".config"


def user_state_dir(env: Mapping[str, str] | None = None, platform_name: str | None = None) -> Path:
    """Platform user-state root. The server jobs root hangs off this."""
    plat = _platform(platform_name)
    if plat.startswith("win"):
        local = _env_value(env, "LOCALAPPDATA")
        return Path(local) if local else _home(env) / "AppData" / "Local"
    if plat == "darwin":
        return _home(env) / "Library" / "Application Support"
    xdg = _env_value(env, "XDG_STATE_HOME")
    return Path(xdg) if xdg else _home(env) / ".local" / "state"


def user_cache_dir(env: Mapping[str, str] | None = None, platform_name: str | None = None) -> Path:
    """Platform user-cache root -- the documented desktop-only runtime fallback (step 5)."""
    plat = _platform(platform_name)
    if plat.startswith("win"):
        local = _env_value(env, "LOCALAPPDATA")
        return Path(local) if local else _home(env) / "AppData" / "Local"
    if plat == "darwin":
        return _home(env) / "Library" / "Caches"
    xdg = _env_value(env, "XDG_CACHE_HOME")
    return Path(xdg) if xdg else _home(env) / ".cache"


def user_data_dir(env: Mapping[str, str] | None = None, platform_name: str | None = None) -> Path:
    """Platform user-DATA root -- where ``output.dir: auto`` puts results on an installed LM3.

    Deliberately distinct from :func:`user_config_dir`. Run outputs are bulk artifacts, not
    configuration, and plan section 3.5 forbids them landing in a hidden configuration directory.
    """
    plat = _platform(platform_name)
    if plat.startswith("win"):
        local = _env_value(env, "LOCALAPPDATA")
        return Path(local) if local else _home(env) / "AppData" / "Local"
    if plat == "darwin":
        return _home(env) / "Library" / "Application Support"
    xdg = _env_value(env, "XDG_DATA_HOME")
    return Path(xdg) if xdg else _home(env) / ".local" / "share"


def deployment_config_dir(
    env: Mapping[str, str] | None = None,
    *,
    deployment: str | None = None,
    platform_name: str | None = None,
) -> Path:
    """``<user-config>/lm3/<canonical deployment key>`` -- settings, hardware, postprocessing."""
    key = deployment or deployment_key(env)
    return user_config_dir(env, platform_name) / APP_DIRNAME / key


def deployment_state_dir(
    env: Mapping[str, str] | None = None,
    *,
    deployment: str | None = None,
    platform_name: str | None = None,
) -> Path:
    """``<user-state>/lm3/<canonical deployment key>`` -- server jobs and other durable state."""
    key = deployment or deployment_key(env)
    return user_state_dir(env, platform_name) / APP_DIRNAME / key


def dev_checkout_root(package_file: str | os.PathLike[str] | None = None) -> Path | None:
    """The repo root when running from a development checkout, else ``None``.

    Detected from ``leafmachine3.__file__``, NOT the CWD: the package's parent must contain a repo
    marker and must not be under ``site-packages``/``dist-packages``. That keeps
    ``cd LM3 && python -m leafmachine3.server`` ergonomic without reintroducing CWD dependence.
    """
    package_dir = Path(package_file).resolve().parent if package_file else _PACKAGE_DIR
    if _SITE_DIR_PARTS & set(package_dir.parts):
        return None
    root = package_dir.parent
    if any((root / marker).exists() for marker in _CHECKOUT_MARKERS):
        return root
    return None


# -- runtime directory (section 3.1, five-step order) ---------------------------------------- #

def _scheduler_job_id(env: Mapping[str, str] | None) -> str | None:
    for name in ("SLURM_JOB_ID", "SLURM_JOBID", "PBS_JOBID", "LSB_JOBID"):
        value = _env_value(env, name)
        if value:
            return value
    return None


def _node_local_scratch(env: Mapping[str, str] | None) -> Path | None:
    """A private directory below node-local job scratch, for a scheduler job (step 3)."""
    for name in ("SLURM_TMPDIR", "JOB_SCRATCH_DIR", "TMPDIR"):
        value = _env_value(env, name)
        if not value:
            continue
        candidate = _expand(env, value)
        if candidate.is_dir() and os.access(candidate, os.W_OK):
            return candidate / "lm3-runtime"
    return None


def _platform_user_runtime(env: Mapping[str, str] | None, platform_name: str | None) -> Path | None:
    """Platform user-runtime location for a desktop/local deployment (step 4)."""
    plat = _platform(platform_name)
    if plat.startswith("linux"):
        xdg = _env_value(env, "XDG_RUNTIME_DIR")
        if xdg:
            base = Path(xdg)
            if base.is_dir():
                return base / APP_DIRNAME
        return None
    if plat == "darwin":
        # macOS gives each user a private per-user temporary directory; that is its user-runtime.
        tmp = _env_value(env, "TMPDIR")
        if tmp and Path(tmp).is_dir():
            return Path(tmp) / APP_DIRNAME
        return None
    if plat.startswith("win"):
        local = _env_value(env, "LOCALAPPDATA")
        if local:
            return Path(local) / APP_DIRNAME / "runtime"
        return None
    return None


def runtime_base_dir(
    runtime_dir: str | os.PathLike[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    platform_name: str | None = None,
    create: bool = False,
    fs_type_reader: Callable[[Path], str | None] | None = None,
    check_filesystem: bool = True,
) -> Path:
    """Resolve the runtime BASE directory. ``<base>/<deployment key>`` is the deployment dir.

    Five-step order, exactly: explicit argument, ``LM3_RUNTIME_DIR``, node-local job scratch for a
    scheduler job, the platform user-runtime location, and finally a documented per-user cache
    fallback -- desktop only. Independent of the checkout and of the CWD at every step.

    ``create=False`` keeps this a PURE query, which is what lets a test assert that the default
    runtime directory is never touched without creating it in the act of asking.
    """
    base: Path | None = None
    if runtime_dir is not None:
        base = _expand(env, str(runtime_dir))
    if base is None:
        value = _env_value(env, ENV_RUNTIME_DIR)
        if value:
            # RuntimeDirectoryError, not the generic refusal, so section 3.1's "runtime registry
            # base ... on miss: startup error" contract and its existing handling still apply.
            base = _expand_env_path(env, value, variable=ENV_RUNTIME_DIR, error=RuntimeDirectoryError)
    scheduled = _scheduler_job_id(env)
    if base is None and scheduled:
        base = _node_local_scratch(env)
        if base is None:
            # Never silently fall back to shared home storage under a scheduler: an allocation
            # writing its lease onto a networked home is the exact failure this plan refuses.
            raise RuntimeDirectoryError(
                f"scheduler job {scheduled} is present but no safe node-local runtime directory "
                f"could be resolved (looked at SLURM_TMPDIR, JOB_SCRATCH_DIR, TMPDIR). Set "
                f"{ENV_RUNTIME_DIR} to node-local scratch; LM3 will not place its lease on shared "
                f"home storage."
            )
    if base is None:
        base = _platform_user_runtime(env, platform_name)
    if base is None:
        base = user_cache_dir(env, platform_name) / APP_DIRNAME / "runtime"
    base = Path(base)
    if check_filesystem:
        check_runtime_filesystem(base, env=env, fs_type_reader=fs_type_reader)
    if create:
        try:
            _mkdir_private(base)
        except OSError as exc:
            raise RuntimeDirectoryError(f"could not create the LM3 runtime directory {base}: {exc}") from exc
    return base


def deployment_runtime_dir(
    runtime_dir: str | os.PathLike[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    deployment: str | None = None,
    platform_name: str | None = None,
    create: bool = False,
    fs_type_reader: Callable[[Path], str | None] | None = None,
    check_filesystem: bool = True,
) -> Path:
    """``<runtime base>/<canonical deployment key>``. The lock, records and children live here."""
    key = deployment or deployment_key(env)
    base = runtime_base_dir(
        runtime_dir, env=env, platform_name=platform_name, create=create,
        fs_type_reader=fs_type_reader, check_filesystem=check_filesystem,
    )
    path = base / key
    if create:
        try:
            _mkdir_private(path)
        except OSError as exc:
            raise RuntimeDirectoryError(f"could not create the deployment runtime directory {path}: {exc}") from exc
    return path


# -- workspace pointer (section 3.1) --------------------------------------------------------- #

def workspace_pointer_path(
    env: Mapping[str, str] | None = None,
    *,
    deployment: str | None = None,
    platform_name: str | None = None,
) -> Path:
    """``<user-config>/lm3/<canonical-deployment>/workspace.json``."""
    return deployment_config_dir(env, deployment=deployment, platform_name=platform_name) / WORKSPACE_FILENAME


def read_workspace_pointer(
    path: str | os.PathLike[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    deployment: str | None = None,
    platform_name: str | None = None,
    require_target: bool = True,
) -> Path | None:
    """The settings file this deployment's GUI selected, or ``None`` when no pointer exists.

    Schema is exactly ``{"schema_version": 1, "settings_path": "/abs/path"}`` -- nothing else, so
    an unknown key is an error rather than a shrug. When the pointer is valid but its target is
    gone this raises :class:`WorkspaceTargetMissingError` (gate 15): silently falling through to
    the default settings would run a configuration the user did not choose.
    """
    pointer = Path(path) if path is not None else workspace_pointer_path(
        env, deployment=deployment, platform_name=platform_name)
    try:
        blob = pointer.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise WorkspacePointerError(f"could not read the workspace pointer {pointer}: {exc}") from exc
    try:
        data = json.loads(blob)
    except ValueError as exc:
        raise WorkspacePointerError(f"workspace pointer {pointer} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise WorkspacePointerError(f"workspace pointer {pointer} must be a JSON object")
    unknown = set(data) - {"schema_version", "settings_path"}
    if unknown:
        raise WorkspacePointerError(
            f"workspace pointer {pointer} carries unknown keys {sorted(unknown)}; the schema is "
            f"exactly schema_version and settings_path"
        )
    version = data.get("schema_version")
    if version != WORKSPACE_SCHEMA_VERSION:
        raise WorkspacePointerError(
            f"workspace pointer {pointer} has schema_version {version!r}; this LM3 understands "
            f"{WORKSPACE_SCHEMA_VERSION}"
        )
    raw = data.get("settings_path")
    if not isinstance(raw, str) or not raw.strip():
        raise WorkspacePointerError(f"workspace pointer {pointer} has no settings_path")
    target = Path(raw)
    if not target.is_absolute():
        raise WorkspacePointerError(f"workspace pointer {pointer} settings_path is not absolute: {raw}")
    if require_target and not target.is_file():
        raise WorkspaceTargetMissingError(target, pointer)
    return target


def write_workspace_pointer(
    settings_path: str | os.PathLike[str],
    path: str | os.PathLike[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    deployment: str | None = None,
    platform_name: str | None = None,
) -> Path:
    """Sole writer: the server's Settings API, on the explicit "choose settings file" action.

    Temp sibling + fsync + ``os.replace``, so a reader sees the old pointer or the new one.
    """
    target = Path(settings_path)
    if not target.is_absolute():
        raise WorkspacePointerError(f"workspace pointer settings_path must be absolute: {target}")
    pointer = Path(path) if path is not None else workspace_pointer_path(
        env, deployment=deployment, platform_name=platform_name)
    _mkdir_private(pointer.parent)
    payload = {"schema_version": WORKSPACE_SCHEMA_VERSION, "settings_path": str(target)}
    _write_atomic(pointer, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return pointer


def clear_workspace_pointer(
    path: str | os.PathLike[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    deployment: str | None = None,
    platform_name: str | None = None,
) -> bool:
    """Remove the pointer so resolution falls back to the deployment default. Returns True if removed."""
    pointer = Path(path) if path is not None else workspace_pointer_path(
        env, deployment=deployment, platform_name=platform_name)
    try:
        pointer.unlink()
        return True
    except FileNotFoundError:
        return False


# -- legacy environment migration (one release) ---------------------------------------------- #

def _warn_legacy_once(key: tuple[str, str | None, str | None], msg: str, *args: object) -> None:
    """Emit ``msg`` the first time this exact (canonical, new, old) triple is seen in-process."""
    with _LEGACY_WARN_LOCK:
        if key in _LEGACY_WARNED:
            return
        _LEGACY_WARNED.add(key)
    # Logged OUTSIDE the lock: a slow handler must not serialize every resolver in the process.
    log.warning(msg, *args)


def reset_legacy_warnings() -> None:
    """Forget which deprecations have been warned about, restoring fresh-process state.

    For test fixtures only: the latch is per process by design, and a test that asserts "warns
    once" needs the process to look new.
    """
    with _LEGACY_WARN_LOCK:
        _LEGACY_WARNED.clear()


def resolve_legacy_env(
    env: Mapping[str, str] | None,
    canonical: str,
    legacy: str | None = None,
) -> str | None:
    """Read ``canonical``, honoring its legacy alias for one release.

    Legacy alone -> deprecation warning. Both naming the same file -> use it and warn once. Both
    naming different files -> :class:`LegacyEnvConflictError`, because guessing which one the user
    meant is exactly how the settings split survived this long.
    """
    legacy = legacy or LEGACY_ENV_ALIASES.get(canonical)
    new = _env_value(env, canonical)
    old = _env_value(env, legacy) if legacy else None
    if old is None:
        return new
    # Latch the EMISSION, never the resolution: the branch structure and the return values below
    # are exactly what they were, so a caller resolving on every status frame still gets the same
    # answer -- it just stops re-printing the same deprecation line.
    key = (canonical, new, old)
    if new is None:
        _warn_legacy_once(
            key, "%s is deprecated; use %s. Honoring it for this release.", legacy, canonical)
        return old
    if os.path.normpath(os.path.expanduser(new)) == os.path.normpath(os.path.expanduser(old)):
        _warn_legacy_once(key, "%s is deprecated and duplicates %s; drop it.", legacy, canonical)
        return new
    # The conflict is NOT latched: it raises on every call, because a split brain that stops
    # being fatal after the first resolution is a split brain that ships.
    raise LegacyEnvConflictError(
        f"{canonical}={new!r} and the deprecated {legacy}={old!r} name different files. Remove "
        f"{legacy}, or make the two agree; LM3 will not guess which settings you meant."
    )


# -- packaged resources ---------------------------------------------------------------------- #

def packaged_resource_dir(package: str, *, required: bool = True) -> Path | None:
    """Locate a packaged data directory through ``importlib.resources``, never a relative path."""
    try:
        from importlib import resources  # noqa: PLC0415 - stdlib, kept beside its only use

        traversable = resources.files(package)
        candidate = Path(str(traversable))
    except (ModuleNotFoundError, TypeError, ValueError, OSError):
        candidate = None
    if candidate is not None and candidate.is_dir():
        return candidate
    if required:
        raise PackagedResourceError(f"packaged resource {package!r} is missing from this installation")
    return None


def calibration_images_dir(*, package: str = "leafmachine3.setup.calibration_images") -> Path:
    """Calibration images, as a packaged resource -- never ``examples/images`` off the CWD.

    A source checkout is still honored as an explicit, checkout-detected second step (the same
    exception the settings chain makes), so calibration works both from a wheel and from this
    repo. Anything else is a packaging error, not a silent CWD read.
    """
    packaged = packaged_resource_dir(package, required=False)
    if packaged is not None:
        return packaged
    checkout = dev_checkout_root()
    if checkout is not None:
        legacy = checkout / "examples" / "images"
        if legacy.is_dir():
            return legacy
    raise PackagedResourceError(
        f"calibration images are missing: {package!r} is not packaged in this installation and no "
        f"development checkout was detected. Reinstall LM3 with its data files."
    )


def packaged_settings_template() -> tuple[str, str]:
    """``(yaml_text, origin)`` used to seed a first-run settings file.

    Order: a packaged template resource, then the development checkout's own
    ``LM3_settings.yaml``, then the built-in defaults serialized. The built-in defaults are the
    package's real template today -- ``leafmachine3.core.config.builtin_defaults()`` -- so this
    never depends on a file that may not ship.
    """
    packaged = packaged_resource_dir("leafmachine3.data", required=False)
    if packaged is not None:
        candidate = packaged / SETTINGS_FILENAME
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8"), str(candidate)
    checkout = dev_checkout_root()
    if checkout is not None:
        candidate = checkout / SETTINGS_FILENAME
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8"), str(candidate)
    import yaml  # noqa: PLC0415 - only needed on the seeding path

    from leafmachine3.core.config import builtin_defaults  # noqa: PLC0415 - avoids an import cycle

    return yaml.safe_dump(builtin_defaults(), sort_keys=False), "leafmachine3.core.config.builtin_defaults()"


# -- the normative precedence table (section 3.1) --------------------------------------------- #

@dataclass(frozen=True)
class SettingsResolution:
    """Where the next-run settings came from, so callers can log it instead of guessing."""

    path: Path
    source: str            # explicit | env | workspace | deployment | checkout | seeded
    seeded: bool = False
    template_origin: str | None = None


def resolve_settings(
    explicit: str | os.PathLike[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    deployment: str | None = None,
    platform_name: str | None = None,
    seed: bool = True,
) -> SettingsResolution:
    """Row 1 of the precedence table: next-run settings.

    1. explicit argument / ``--config`` · 2. ``LM3_SETTINGS`` · 3. the deployment workspace
    pointer · 4. ``<user-config>/lm3/<deployment>/LM3_settings.yaml`` · 5. dev checkout only:
    ``<checkout root>/LM3_settings.yaml``.

    On a complete miss, path 4 is SEEDED from the packaged template, the path is logged loudly,
    and resolution continues -- so a first-run install works. An explicit path (argument or
    ``LM3_SETTINGS``) naming a missing file is always a hard error: the user stated an intent that
    cannot be satisfied, and quietly using a different file is the defect this table replaces.
    """
    if explicit is not None:
        path = _expand(env, str(explicit))
        if not path.is_file():
            raise SettingsMissingError(f"settings file not found: {path} (given explicitly)")
        return SettingsResolution(path, "explicit")

    value = resolve_legacy_env(env, ENV_SETTINGS)
    if value:
        path = _expand_env_path(env, value, variable=_legacy_env_source(env, ENV_SETTINGS))
        if not path.is_file():
            raise SettingsMissingError(f"settings file not found: {path} (from {ENV_SETTINGS})")
        return SettingsResolution(path, "env")

    pointer = read_workspace_pointer(env=env, deployment=deployment, platform_name=platform_name)
    if pointer is not None:
        return SettingsResolution(pointer, "workspace")

    default = deployment_config_dir(env, deployment=deployment, platform_name=platform_name) / SETTINGS_FILENAME
    if default.is_file():
        return SettingsResolution(default, "deployment")

    checkout = dev_checkout_root()
    if checkout is not None:
        candidate = checkout / SETTINGS_FILENAME
        if candidate.is_file():
            return SettingsResolution(candidate, "checkout")

    if not seed:
        return SettingsResolution(default, "deployment")

    text, origin = packaged_settings_template()
    _mkdir_private(default.parent)
    _write_atomic(default, text, mode=0o600)
    log.warning("no LM3 settings file found; seeded a new one at %s from %s", default, origin)
    return SettingsResolution(default, "seeded", seeded=True, template_origin=origin)


def settings_path(
    explicit: str | os.PathLike[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    deployment: str | None = None,
    platform_name: str | None = None,
    seed: bool = True,
) -> Path:
    """The canonical next-run settings path. Thin wrapper over :func:`resolve_settings`."""
    return resolve_settings(
        explicit, env=env, deployment=deployment, platform_name=platform_name, seed=seed).path


def hardware_profile_path(
    *,
    env: Mapping[str, str] | None = None,
    deployment: str | None = None,
    platform_name: str | None = None,
    machine: str | None = None,
) -> Path:
    """Row 2: the hardware profile, DEPLOYMENT-scoped and machine-keyed.

    ``<user-config>/lm3/<canonical-deployment>/hardware_settings.<machine-key>.yaml``.

    Deployment-scoped because ``run_setup`` sizes GPU stages against the SELECTED
    ``compute.devices`` subset and fingerprints config-derived model hashes: two named deployments
    hold separate leases and would otherwise carry each other's measurements forward through one
    shared file. Machine-keyed because one networked ``<user-config>`` is routinely shared by many
    cluster nodes.

    ``--config`` does NOT move this path. Only ``LM3_HARDWARE`` does.

    **This function is PURE and must stay pure.** It reads nothing it does not have to and writes
    NOTHING. Resolving a path is something status snapshots, health probes and Results polling do
    many times a second; making it copy a file was how two legacy profiles ended up in a real
    developer's ``~/.config/lm3`` during a test run.

    There is deliberately **no** ``adopt_legacy`` or ``settings_file`` parameter. The one-release
    legacy adopt lives exclusively in :func:`migrate_legacy_hardware_profile`. Keeping a disabled-by-
    default adopt here would leave the check-then-write race reachable through this function's public
    API and contradict the paragraph above, which is worse than either having one migration path or
    having none.
    """
    value = resolve_legacy_env(env, ENV_HARDWARE)
    if value:
        return _expand_env_path(env, value, variable=_legacy_env_source(env, ENV_HARDWARE))
    key = machine or machine_key()
    target = (deployment_config_dir(env, deployment=deployment, platform_name=platform_name)
              / f"hardware_settings.{key}.yaml")
    return target


def migrate_legacy_hardware_profile(
    *,
    env: Mapping[str, str] | None = None,
    settings_file: str | os.PathLike[str] | None = None,
    deployment: str | None = None,
    platform_name: str | None = None,
    machine: str | None = None,
) -> Path | None:
    """Perform the one-release legacy hardware-profile adopt, ONCE, deliberately.

    Split out of :func:`hardware_profile_path` so that resolving a path can never mutate the
    filesystem. Call this exactly once from a controlled startup path -- server ``create_app`` or
    hardware setup -- and never from a read endpoint.

    Returns the deployment-scoped target if a legacy profile was adopted on this call, else ``None``
    (already present, nothing to adopt, or no settings file to look beside).
    """
    if settings_file is None:
        return None
    target = hardware_profile_path(
        env=env, deployment=deployment, platform_name=platform_name, machine=machine,
    )
    if target.is_file():
        return None
    legacy = Path(settings_file).parent / LEGACY_HARDWARE_FILENAME
    if not legacy.is_file():
        return None
    try:
        _mkdir_private(target.parent)
        if not _install_if_absent(target, legacy.read_text(encoding="utf-8")):
            return None               # another process adopted between the check and the write
    except OSError as exc:
        log.warning("could not adopt the legacy hardware profile %s: %s", legacy, exc)
        return None
    log.warning(
        "adopted the legacy hardware profile %s into %s. That legacy location is deprecated and "
        "will stop being consulted in a future release.", legacy, target,
    )
    return target


def postprocessing_settings_path(
    *,
    env: Mapping[str, str] | None = None,
    deployment: str | None = None,
    platform_name: str | None = None,
) -> Path:
    """Row 3: ``LM3_POSTPROCESS_SETTINGS``, else ``<user-config>/lm3/<deployment>/postprocessing.yaml``.

    On miss the caller uses packaged defaults; the path is still returned so it can be created.
    Unmoved by ``--config``.
    """
    value = _env_value(env, ENV_POSTPROCESS_SETTINGS)
    if value:
        return _expand_env_path(env, value, variable=ENV_POSTPROCESS_SETTINGS)
    return deployment_config_dir(env, deployment=deployment, platform_name=platform_name) / POSTPROCESS_FILENAME


def migrate_legacy_postprocessing_settings(
    *,
    env: Mapping[str, str] | None = None,
    source: str | os.PathLike[str] | None = None,
    deployment: str | None = None,
    platform_name: str | None = None,
) -> Path | None:
    """Adopt a checkout-level ``postprocessing_settings.yaml`` into the canonical row-3 path, once.

    The same shape as :func:`migrate_legacy_hardware_profile`, and separate from
    :func:`postprocessing_settings_path` for the same reason: resolving a path must never write.

    ``source`` defaults to ``<dev checkout root>/postprocessing_settings.yaml``, located from
    ``leafmachine3.__file__`` and NEVER from the CWD -- the CWD default is precisely the defect this
    migration exists to retire. An installed package has no checkout root, so there is nothing to
    adopt and this returns ``None``.

    COPY, never move: an older LM3 launched from that checkout keeps working through the one-release
    window, exactly as the hardware profile does.
    """
    if source is None:
        checkout = dev_checkout_root()
        if checkout is None:
            return None
        source = checkout / LEGACY_POSTPROCESS_FILENAME
    legacy = Path(source)
    if not legacy.is_file():
        return None
    target = postprocessing_settings_path(env=env, deployment=deployment, platform_name=platform_name)
    if target.exists():
        return None
    try:
        _mkdir_private(target.parent)
        if not _install_if_absent(target, legacy.read_text(encoding="utf-8")):
            return None               # another process adopted between the check and the write
    except OSError as exc:
        log.warning("could not adopt the legacy postprocessing settings %s: %s", legacy, exc)
        return None
    log.warning(
        "adopted the legacy postprocessing settings %s into %s. That checkout-level location is "
        "deprecated and will stop being consulted in a future release.", legacy, target,
    )
    return target


def server_jobs_root(
    *,
    env: Mapping[str, str] | None = None,
    deployment: str | None = None,
    platform_name: str | None = None,
    create: bool = False,
) -> Path:
    """Row 4: ``LM3_SERVER_JOBS``, else ``<user-state>/lm3/<deployment>/jobs``. On miss: create.

    Resolved at CALL time on purpose. Today's module-level constant freezes the answer at import,
    before a spawned child has read ``LM3_DEPLOYMENT_ID``.
    """
    value = _env_value(env, ENV_SERVER_JOBS)
    root = _expand_env_path(env, value, variable=ENV_SERVER_JOBS) if value else (
        deployment_state_dir(env, deployment=deployment, platform_name=platform_name) / "jobs")
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


#: The built-in ``project.output.dir`` value. Plan section 3.5: a DEFAULT is not a user statement,
#: so it must not follow the settings-relative rule into a hidden configuration directory.
AUTO_OUTPUT_DIR = "auto"


def default_output_dir(
    env: Mapping[str, str] | None = None,
    *,
    platform_name: str | None = None,
    deployment: str | None = None,
) -> Path:
    """Where ``project.output.dir: auto`` lands (plan section 3.5).

    A development checkout keeps ``<checkout>/runs`` so no existing checkout moves; an installed
    package gets ``<user-data>/lm3/<deployment>/runs``. Never the user-config directory.
    """
    checkout = dev_checkout_root()
    if checkout is not None:
        return checkout / "runs"
    key = deployment or deployment_key(env)
    return user_data_dir(env, platform_name) / APP_DIRNAME / key / "runs"


def resolve_project_output_dir(
    settings_file: str | os.PathLike[str] | None,
    output_dir: str | os.PathLike[str] | None,
    *,
    env: Mapping[str, str] | None = None,
    platform_name: str | None = None,
) -> Path | None:
    """Absolutize ``project.output.dir`` against the SETTINGS FILE's parent, never the CWD.

    The tree has four incompatible rules for this one field today. Settings-relative is the only
    one that stays stable now the settings path itself is canonical: the same YAML resolves to the
    same output directory whether the CLI, the server, or a spawned child reads it.
    """
    if output_dir is None:
        return None
    raw = str(output_dir).strip()
    if not raw:
        return None
    if raw.lower() == AUTO_OUTPUT_DIR:
        return default_output_dir(env=env, platform_name=platform_name)
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    if settings_file is None:
        raise PathsError(
            f"cannot resolve the relative project.output.dir {raw!r} without a settings file to "
            f"resolve it against; LM3 never joins it onto the current working directory"
        )
    # normpath for the same reason Config.resolve_path uses it: collapse ".." lexically without
    # following symlinks, so the published run directory is a clean absolute path.
    return Path(os.path.normpath(str(Path(settings_file).resolve().parent / path)))


def runs_roots(
    *,
    env: Mapping[str, str] | None = None,
    runtime_roots: Sequence[str | os.PathLike[str]] = (),
    settings_file: str | os.PathLike[str] | None = None,
    settings_output_dir: str | os.PathLike[str] | None = None,
) -> list[Path]:
    """Row 5: run-history roots, in order, de-duplicated. On miss: an EMPTY list.

    1. ``LM3_RUNS_ROOTS`` (``os.pathsep``-separated) · 2. active/last runtime output roots, passed
    in by the caller that read the runtime records · 3. ``project.output.dir`` from the resolved
    settings. No CWD, no ``runs``/``examples_out`` guesses -- an empty Results tab is honest, a
    tab full of whatever happened to be beside the launcher is not.
    """
    roots: list[Path] = []

    def add(candidate: str | os.PathLike[str] | None, *, variable: str | None = None) -> None:
        if candidate is None:
            return
        raw = str(candidate).strip()
        if not raw:
            return
        # Per-ENTRY refusal, and only for entries that came from the environment: a relative
        # entry anywhere in LM3_RUNS_ROOTS would name a different history directory in every
        # process. Runtime roots and project.output.dir arrive already resolved by their own
        # rules (resolve_project_output_dir is settings-relative), so they keep them.
        path = _expand_env_path(env, raw, variable=variable) if variable else _expand(env, raw)
        if path not in roots:
            roots.append(path)

    value = _env(env).get(ENV_RUNS_ROOTS) or ""
    for entry in value.split(os.pathsep):
        add(entry, variable=ENV_RUNS_ROOTS)
    for entry in runtime_roots:
        add(entry)
    add(resolve_project_output_dir(settings_file, settings_output_dir))
    return roots


# -- the bounded early resolver (section 2.7) ------------------------------------------------ #

@dataclass(frozen=True)
class EarlyRunPaths:
    """Only what is deterministic from config BEFORE any directory exists.

    Deliberately carries no ``tmp_dir``: ``_ensure_tmp`` (``dirs.py:67-76``) falls back to
    ``<root>/_tmp_original`` on ``OSError``, so a configured scratch path is not knowable in
    advance and must not appear in a ``starting`` record.
    """

    run_name: str
    run_dir: Path              # <output.dir>/<run_name>
    artifact_dir: Path         # section 2.10 role: persistent project storage
    active_state_dir: Path     # section 2.10 role: node-local scratch on a cluster
    active_db_path: Path       # section 2.10 role
    archive_pointer_path: Path | None   # section 2.10 role; None in in-place mode
    log_path: Path
    archive_mode: str          # "in-place" when artifact_dir == active_state_dir, else "staged"


def early_run_paths(
    *,
    output_dir: str | os.PathLike[str],
    run_name: str,
    active_state_dir: str | os.PathLike[str] | None = None,
) -> EarlyRunPaths:
    """Pure. Creates nothing, probes nothing, and never publishes ``tmp_dir``.

    ``output_dir`` must already be absolute -- resolve it through
    :func:`resolve_project_output_dir` first, so this function cannot reintroduce a CWD join.
    """
    run = str(run_name)
    artifact_dir = Path(output_dir) / run
    if not artifact_dir.is_absolute():
        raise PathsError(
            f"early_run_paths needs an absolute output dir; got {output_dir!r}. Resolve it with "
            f"resolve_project_output_dir() rather than joining the current working directory."
        )
    state_dir = Path(active_state_dir) / run if active_state_dir is not None else artifact_dir
    in_place = state_dir == artifact_dir
    return EarlyRunPaths(
        run_name=run,
        run_dir=artifact_dir,
        artifact_dir=artifact_dir,
        active_state_dir=state_dir,
        active_db_path=state_dir / f"{run}.sqlite",
        archive_pointer_path=None if in_place else artifact_dir / "archive.current.json",
        log_path=artifact_dir / "logs" / "lm3.log",
        archive_mode="in-place" if in_place else "staged",
    )


# -- diagnostics ----------------------------------------------------------------------------- #

def describe_resolved_paths(
    *,
    env: Mapping[str, str] | None = None,
    explicit_settings: str | os.PathLike[str] | None = None,
    seed: bool = False,
) -> dict[str, str]:
    """A flat, log-safe summary for startup logs and ``/healthz`` diagnostics.

    Never raises: a diagnostics view that dies on a bad environment tells the user nothing about
    why. Failures are reported as their message text against the key that failed.
    """
    out: dict[str, str] = {}

    def record(key: str, fn: Callable[[], Any]) -> None:
        try:
            out[key] = str(fn())
        except Exception as exc:  # noqa: BLE001 - diagnostics must survive every failure
            out[key] = f"<error: {exc}>"

    record("deployment_id", lambda: raw_deployment_id(env))
    record("deployment_key", lambda: deployment_key(env))
    record("machine_key", machine_key)
    record("runtime_base", lambda: runtime_base_dir(env=env))
    record("runtime_dir", lambda: deployment_runtime_dir(env=env))
    record("settings", lambda: settings_path(explicit_settings, env=env, seed=seed))
    record("hardware_profile", lambda: hardware_profile_path(env=env))
    record("postprocess_settings", lambda: postprocessing_settings_path(env=env))
    record("server_jobs_root", lambda: server_jobs_root(env=env))
    record("dev_checkout", lambda: dev_checkout_root() or "none")
    return out


__all__ = [
    # constants
    "ACTIVE_RECORD_FILENAME", "ACTIVITY_LOCK_FILENAME", "APP_DIRNAME", "CHILDREN_DIRNAME",
    "CONNECTION_PRIVATE_FILENAME", "CONNECTION_PUBLIC_FILENAME", "DEFAULT_DEPLOYMENT_ID",
    "DEFAULT_PORT", "DEPLOYMENT_HASH_LENGTH", "DEPLOYMENT_SLUG_LENGTH", "ENV_ALLOW_NETWORK_RUNTIME",
    "ENV_DEPLOYMENT_ID", "ENV_HARDWARE", "ENV_PORT", "ENV_POSTPROCESS_SETTINGS", "ENV_RUNS_ROOTS",
    "ENV_RUNTIME_DIR", "ENV_SERVER_JOBS", "ENV_SETTINGS", "LAST_RECORD_FILENAME",
    "LEGACY_ENV_ALIASES", "LEGACY_HARDWARE_FILENAME", "LEGACY_POSTPROCESS_FILENAME", "MACHINE_KEY_LENGTH", "MACHINE_KEY_VERSION",
    "NETWORK_FS_TYPES", "POSTPROCESS_FILENAME", "SETTINGS_FILENAME", "WORKSPACE_FILENAME",
    "WORKSPACE_SCHEMA_VERSION",
    # errors
    "DeploymentIdentityError", "DeploymentPortError", "LegacyEnvConflictError",
    "NetworkRuntimeRefused", "PackagedResourceError", "PathsError", "RelativeEnvPathError",
    "RuntimeDirectoryError", "SettingsMissingError", "WorkspacePointerError",
    "WorkspaceTargetMissingError",
    # deployment identity
    "ascii_slug", "canonical_deployment_key", "deployment_key", "is_default_deployment",
    "raw_deployment_id", "resolve_port",
    # machine key
    "MachineIdentity", "detect_machine_identity", "machine_key", "machine_key_input",
    "read_gpu_uuids", "read_node_name", "read_os_machine_id", "reset_machine_key_cache",
    "migrate_legacy_hardware_profile", "migrate_legacy_postprocessing_settings",
    # filesystem
    "check_runtime_filesystem", "filesystem_type", "is_network_filesystem",
    # directories
    "AUTO_OUTPUT_DIR", "default_output_dir", "user_data_dir", "dev_checkout_root", "deployment_config_dir", "deployment_runtime_dir", "deployment_state_dir",
    "runtime_base_dir", "user_cache_dir", "user_config_dir", "user_state_dir",
    # precedence table
    "SettingsResolution", "calibration_images_dir", "hardware_profile_path",
    "packaged_resource_dir", "packaged_settings_template", "postprocessing_settings_path",
    "reset_legacy_warnings", "resolve_legacy_env", "resolve_project_output_dir",
    "resolve_settings", "runs_roots",
    "server_jobs_root", "settings_path",
    # workspace pointer
    "clear_workspace_pointer", "read_workspace_pointer", "workspace_pointer_path",
    "write_workspace_pointer",
    # early resolver + diagnostics
    "EarlyRunPaths", "describe_resolved_paths", "early_run_paths",
]
