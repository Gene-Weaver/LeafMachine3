"""Tests for leafmachine3.core.paths -- the canonical resolver.

Two rules govern this file:

* Nothing here may touch the user's real ``~/.config``, ``~/.local/state``, ``$XDG_RUNTIME_DIR`` or
  ``~/.cache``. Every resolver takes an injected ``env`` mapping, so each test builds its own
  world under ``tmp_path`` and passes it. No ``os.environ`` mutation, no real home.
* Nothing here may depend on the process CWD. Several tests deliberately chdir to prove that.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
from pathlib import Path

import pytest

from leafmachine3.core import paths as P

GOLDEN = Path(__file__).resolve().parent / "golden"


# --------------------------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------------------------- #

def make_env(tmp_path: Path, **extra: str) -> dict[str, str]:
    """A hermetic environment: a fake HOME plus fake XDG roots, and nothing inherited."""
    home = tmp_path / "home"
    (home / ".config").mkdir(parents=True, exist_ok=True)
    (home / ".local" / "state").mkdir(parents=True, exist_ok=True)
    (home / ".cache").mkdir(parents=True, exist_ok=True)
    env = {
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "XDG_STATE_HOME": str(tmp_path / "state"),
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
    }
    env.update(extra)
    return env


@contextlib.contextmanager
def capture_warnings(logger_name: str = "leafmachine3.paths"):
    """Collect this logger's warnings directly.

    Not ``caplog``: ``logging_setup.start_logging`` sets ``propagate = False`` on the
    ``leafmachine3`` logger, so once any pipeline test has run, records never reach the root
    handler caplog installs. Attaching to the logger itself is immune to that ordering.
    """
    records: list[str] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record.getMessage())

    logger = logging.getLogger(logger_name)
    handler = _Collector(level=logging.WARNING)
    previous_level, previous_propagate = logger.level, logger.propagate
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate


def no_network(_path: Path) -> str | None:
    """A filesystem-type reader that always reports ordinary local storage."""
    return "ext4"


DEFAULT_KEY = P.canonical_deployment_key("default")


# --------------------------------------------------------------------------------------------- #
# 1. deployment key -- section 2.1, gates 39/40
# --------------------------------------------------------------------------------------------- #

def test_ascii_slug_is_mechanical_ascii_only():
    # A-Z lowercased by adding 0x20; a-z and 0-9 kept; everything else "-", collapsed, stripped.
    assert P.ascii_slug("GPU0") == "gpu0"
    assert P.ascii_slug("Lab Bench 2") == "lab-bench-2"
    assert P.ascii_slug("--gpu0--") == "gpu0"
    assert P.ascii_slug("a///b") == "a-b"
    assert P.ascii_slug("") == ""


def test_ascii_slug_never_casefolds_or_normalizes():
    # 'ß'.casefold() == 'ss' but 'ß'.lower() == 'ß'; JS has only the latter, so neither may be used.
    assert P.ascii_slug("ß") == ""
    assert P.ascii_slug("Straße") == "stra-e"
    assert P.ascii_slug("实验室") == ""
    assert P.ascii_slug("gpu\U0001f600") == "gpu"
    # A Cyrillic lookalike must not collapse into the ASCII namespace.
    assert P.canonical_deployment_key("gpua") != P.canonical_deployment_key("gpu\u0430")


def test_canonical_key_shape_and_truncation():
    key = P.canonical_deployment_key("a" * 40)
    slug, _, digest = key.rpartition("-")
    assert slug == "a" * 32
    assert len(digest) == 8
    # Truncating the slug must not merge two distinct raw values: the hash still separates them.
    assert P.canonical_deployment_key("a" * 40) != P.canonical_deployment_key("a" * 41)


def test_canonical_key_neutralizes_path_separators_and_traversal():
    for raw in ("a/b", "a\\b", "..", "../../etc/passwd", "/", "."):
        key = P.canonical_deployment_key(raw)
        assert "/" not in key and "\\" not in key
        assert key not in (".", "..")
        # The key is one path component: joining it can never escape its base.
        base = Path("/base")
        assert Path(os.path.normpath(base / key)).parent == base


def test_deployment_id_unset_is_identical_to_default():
    assert P.raw_deployment_id({}) == "default"
    assert P.deployment_key({}) == P.deployment_key({"LM3_DEPLOYMENT_ID": "default"})
    assert P.is_default_deployment({}) is True
    assert P.is_default_deployment({"LM3_DEPLOYMENT_ID": "default"}) is True


def test_empty_or_whitespace_deployment_id_is_rejected_not_defaulted():
    for bad in ("", "   ", "\t\n"):
        with pytest.raises(P.DeploymentIdentityError):
            P.raw_deployment_id({"LM3_DEPLOYMENT_ID": bad})


def test_distinct_deployments_do_not_collide():
    assert P.deployment_key({"LM3_DEPLOYMENT_ID": "gpu0"}) != P.deployment_key({"LM3_DEPLOYMENT_ID": "gpu1"})
    # Case differs in the hash even though the cosmetic slug matches.
    assert P.canonical_deployment_key("GPU0") != P.canonical_deployment_key("gpu0")
    assert P.ascii_slug("GPU0") == P.ascii_slug("gpu0")


def test_deployment_key_golden_vectors():
    """The contract artifact Electron is tested against. Any drift here is a lock mismatch."""
    blob = json.loads((GOLDEN / "deployment_key_vectors.json").read_text(encoding="utf-8"))
    assert blob["schema_version"] == 1
    names = {v["name"] for v in blob["vectors"]}
    # The coverage the plan requires of this file.
    for required in ("unset", "default", "uppercase", "path_separator_posix", "dot_dot",
                     "unicode_sharp_s", "unicode_cjk", "unicode_emoji", "over_length", "empty",
                     "whitespace_only", "slug_empty"):
        assert required in names, f"golden vectors lost coverage of {required}"
    for vector in blob["vectors"]:
        raw = vector["raw"]
        effective = P.DEFAULT_DEPLOYMENT_ID if raw is None else raw
        assert effective == vector["effective_raw"]
        assert P.ascii_slug(effective) == vector["slug"], vector["name"]
        assert P.canonical_deployment_key(effective) == vector["canonical"], vector["name"]
        if vector["rejected_by_env"]:
            with pytest.raises(P.DeploymentIdentityError):
                P.raw_deployment_id({"LM3_DEPLOYMENT_ID": raw})
        elif raw is not None:
            assert P.deployment_key({"LM3_DEPLOYMENT_ID": raw}) == vector["canonical"]
        else:
            assert P.deployment_key({}) == vector["canonical"]


# --------------------------------------------------------------------------------------------- #
# 2. the port rule -- gate 41
# --------------------------------------------------------------------------------------------- #

def test_default_deployment_owns_8765_without_saying_so():
    assert P.resolve_port({}) == P.DEFAULT_PORT == 8765
    assert P.resolve_port({"LM3_DEPLOYMENT_ID": "default"}) == 8765


def test_named_deployment_without_port_is_a_startup_error():
    with pytest.raises(P.DeploymentPortError) as excinfo:
        P.resolve_port({"LM3_DEPLOYMENT_ID": "gpu1"})
    assert "LM3_PORT" in str(excinfo.value)
    assert "8765" in str(excinfo.value)   # names the collision it is refusing


def test_named_deployment_with_port_is_fine():
    assert P.resolve_port({"LM3_DEPLOYMENT_ID": "gpu1", "LM3_PORT": "8766"}) == 8766


@pytest.mark.parametrize("bad", ["not-a-number", "0", "70000", "-1"])
def test_bad_port_values_are_rejected(bad):
    with pytest.raises(P.DeploymentPortError):
        P.resolve_port({"LM3_PORT": bad})


# --------------------------------------------------------------------------------------------- #
# 3. machine key -- section 3.1, gates 12 and 20
# --------------------------------------------------------------------------------------------- #

def test_machine_key_serialization_is_exact():
    ident = P.MachineIdentity(os_machine_id="abc", node="node0042", gpu_uuids=("GPU-b", "GPU-a"))
    assert P.machine_key_input(ident) == (
        "machine-key/v1\n"
        "os-machine-id: abc\n"
        "node: node0042\n"
        "gpu-uuids: GPU-a,GPU-b\n"
    )
    assert len(P.machine_key(ident)) == 16
    assert all(c in "0123456789abcdef" for c in P.machine_key(ident))


def test_missing_fields_become_explicit_none_markers():
    ident = P.MachineIdentity(os_machine_id=None, node="n", gpu_uuids=())
    text = P.machine_key_input(ident)
    assert "os-machine-id: none\n" in text
    assert "gpu-uuids: none\n" in text
    # The line is never omitted -- an omitted line would let two different hosts serialize alike.
    assert text.count("\n") == 4


def test_gate_20_two_container_nodes_share_a_machine_id_and_have_no_gpus():
    """One baked /etc/machine-id across an allocation, CPU-only, must still split by node."""
    baked = "f" * 32
    a = P.MachineIdentity(os_machine_id=baked, node="node0042", gpu_uuids=())
    b = P.MachineIdentity(os_machine_id=baked, node="node0043", gpu_uuids=())
    assert P.machine_key(a) != P.machine_key(b)


def test_gate_12_identical_hardware_different_host_identity():
    gpus = ("GPU-11111111-2222-3333-4444-555555555555",)
    a = P.MachineIdentity(os_machine_id=None, node="gl1234", gpu_uuids=gpus)
    b = P.MachineIdentity(os_machine_id=None, node="gl1235", gpu_uuids=gpus)
    assert P.machine_key(a) != P.machine_key(b)


def test_gpu_uuid_order_does_not_move_the_key():
    a = P.MachineIdentity("m", "n", ("GPU-b", "GPU-a"))
    b = P.MachineIdentity("m", "n", ("GPU-a", "GPU-b"))
    assert P.machine_key(a) == P.machine_key(b)


def test_machine_identity_probes_are_injectable():
    ident = P.detect_machine_identity(
        {"SLURM_NODENAME": "gl9999"},
        machine_id_reader=lambda: "  baked-id  ",
        node_reader=lambda: "gl9999",
        gpu_uuid_reader=lambda: ["GPU-z", "GPU-a"],
    )
    assert ident == P.MachineIdentity("baked-id", "gl9999", ("GPU-a", "GPU-z"))
    # An empty machine id string collapses to the None marker rather than hashing whitespace.
    blank = P.detect_machine_identity({}, machine_id_reader=lambda: "   ",
                                      node_reader=lambda: "n", gpu_uuid_reader=lambda: [])
    assert blank.os_machine_id is None


def test_slurm_nodename_wins_over_hostname():
    assert P.read_node_name({"SLURM_NODENAME": "gl1234"}) == "gl1234"
    assert P.read_node_name({}) == P.read_node_name({})   # hostname, stable within the process


def test_machine_key_input_carries_only_the_three_declared_fields():
    """Driver/ORT versions, devices, stages and model fingerprints must never enter the key."""
    assert set(P.MachineIdentity.__dataclass_fields__) == {"os_machine_id", "node", "gpu_uuids"}
    text = P.machine_key_input(P.MachineIdentity("m", "n", ("GPU-a",)))
    for forbidden in ("driver", "onnx", "provider", "devices", "stage", "model"):
        assert forbidden not in text.lower()


def test_machine_key_golden_vectors():
    blob = json.loads((GOLDEN / "machine_key_vectors.json").read_text(encoding="utf-8"))
    assert blob["schema_version"] == 1
    for vector in blob["vectors"]:
        ident = P.MachineIdentity(
            os_machine_id=vector["os_machine_id"],
            node=vector["node"],
            gpu_uuids=tuple(vector["gpu_uuids"]),
        )
        assert P.machine_key_input(ident) == vector["serialized"], vector["name"]
        assert P.machine_key(ident) == vector["machine_key"], vector["name"]
    keys = {v["machine_key"] for v in blob["vectors"] if v["name"].startswith("container_image_node")}
    assert len(keys) == 2, "the duplicated-image machine-id vectors must not collide"


def test_probe_helpers_never_raise():
    # They run on whatever host CI happens to be; "no answer" is a normal answer.
    assert P.read_os_machine_id() is None or isinstance(P.read_os_machine_id(), str)
    assert isinstance(P.read_gpu_uuids(), tuple)


@pytest.mark.skipif(sys.platform != "win32", reason="reads the Windows registry")
def test_windows_machine_guid_is_readable():   # pragma: no cover - Windows CI only
    value = P.read_os_machine_id()
    assert value is None or isinstance(value, str)


# --------------------------------------------------------------------------------------------- #
# 4. network filesystem refusal -- gate 42
# --------------------------------------------------------------------------------------------- #

MOUNTINFO = (
    "23 28 0:21 / /proc rw,relatime shared:12 - proc proc rw\n"
    "25 28 0:5 / / rw,relatime shared:1 - ext4 /dev/sda1 rw\n"
    "60 25 0:52 / /mnt/share rw,relatime shared:33 - nfs4 fileserver:/export rw\n"
    "61 25 0:53 / /mnt/win rw,relatime shared:34 - cifs //server/share rw\n"
    "62 25 0:54 / /mnt/deep\\040space rw - fuse.sshfs user@host:/ rw\n"
)


def _mountinfo() -> str:
    return MOUNTINFO


def fake_fs_reader(path) -> str | None:
    """Filesystem types as the fake mount table above sees them."""
    return P.filesystem_type(path, platform_name="linux", mountinfo_reader=_mountinfo)


@pytest.mark.parametrize("path,expected", [
    ("/", "ext4"),
    ("/var/tmp", "ext4"),
    ("/mnt/share", "nfs4"),
    ("/mnt/share/lm3/runtime", "nfs4"),   # longest-prefix match, and the dir need not exist
    ("/mnt/win/x", "cifs"),
    ("/mnt/deep space/x", "fuse.sshfs"),
])
def test_linux_filesystem_type_is_longest_prefix(path, expected):
    assert P.filesystem_type(path, platform_name="linux", mountinfo_reader=_mountinfo) == expected


@pytest.mark.parametrize("fs_type", [
    "nfs", "nfs4", "cifs", "smb", "smb3", "fuse.sshfs", "lustre", "gpfs", "afs", "9p", "ceph",
    "glusterfs", "fuse.glusterfs", "NFS4",
])
def test_network_types_are_recognized(fs_type):
    assert P.is_network_filesystem(fs_type) is True


@pytest.mark.parametrize("fs_type", ["ext4", "xfs", "btrfs", "tmpfs", "overlay", "zfs", None, ""])
def test_local_types_are_not_network(fs_type):
    assert P.is_network_filesystem(fs_type) is False


def test_runtime_on_a_network_filesystem_is_refused(tmp_path):
    env = make_env(tmp_path, LM3_RUNTIME_DIR="/mnt/share/lm3-runtime")
    with pytest.raises(P.NetworkRuntimeRefused) as excinfo:
        P.runtime_base_dir(env=env, fs_type_reader=fake_fs_reader)
    message = str(excinfo.value)
    assert "nfs4" in message                       # names the detected type
    assert "/mnt/share/lm3-runtime" in message     # and the path
    assert excinfo.value.fs_type == "nfs4"


def test_network_refusal_has_exactly_one_override_and_it_warns(tmp_path):
    env = make_env(tmp_path, LM3_RUNTIME_DIR="/mnt/share/lm3-runtime",
                   LM3_ALLOW_NETWORK_RUNTIME="1")
    with capture_warnings() as warnings:
        base = P.runtime_base_dir(env=env, fs_type_reader=fake_fs_reader)
    assert base == Path("/mnt/share/lm3-runtime")
    warning = "\n".join(warnings)
    assert "nfs4" in warning and "UNRELIABLE" in warning
    # And it is genuinely the ONLY override: a neighboring truthy variable must not open it.
    with pytest.raises(P.NetworkRuntimeRefused):
        P.runtime_base_dir(env=make_env(tmp_path, LM3_RUNTIME_DIR="/mnt/share/lm3-runtime",
                                        LM3_ALLOW_NETWORK="1"), fs_type_reader=fake_fs_reader)


def test_the_desktop_cache_fallback_is_checked_too(tmp_path):
    """The refusal applies to the desktop fallback, not only the cluster path."""
    env = make_env(tmp_path)                       # no LM3_RUNTIME_DIR, no XDG_RUNTIME_DIR
    with pytest.raises(P.NetworkRuntimeRefused):
        P.runtime_base_dir(env=env, fs_type_reader=lambda _p: "nfs")


def test_windows_network_detection_uses_an_injected_win32_surface():
    # UNC needs no syscall at all.
    assert P.filesystem_type(r"\\server\share\lm3", platform_name="win32") == "unc"
    drive_types = {"Z:\\": 4, "C:\\": 3}           # DRIVE_REMOTE == 4, DRIVE_FIXED == 3
    fake = drive_types.__getitem__
    assert P.filesystem_type(r"Z:\lm3", platform_name="win32", get_drive_type=fake) == "remote"
    assert P.filesystem_type(r"C:\lm3", platform_name="win32", get_drive_type=fake) is None
    assert P.is_network_filesystem("unc") and P.is_network_filesystem("remote")


def test_unknown_platform_reports_nothing_rather_than_guessing():
    assert P.filesystem_type("/anywhere", platform_name="darwin") is None
    assert P.check_runtime_filesystem("/anywhere", env={}, fs_type_reader=lambda _p: None) is None


# --------------------------------------------------------------------------------------------- #
# 5. runtime directory -- base vs final, the five-step order
# --------------------------------------------------------------------------------------------- #

def test_lm3_runtime_dir_names_the_BASE_not_the_deployment_dir(tmp_path):
    base = tmp_path / "scratch" / "lm3-runtime"
    env = make_env(tmp_path, LM3_RUNTIME_DIR=str(base), LM3_DEPLOYMENT_ID="gpu0")
    assert P.runtime_base_dir(env=env, fs_type_reader=no_network) == base
    assert P.deployment_runtime_dir(env=env, fs_type_reader=no_network) == base / P.canonical_deployment_key("gpu0")


def test_one_base_hosts_several_deployments(tmp_path):
    base = tmp_path / "scratch" / "lm3-runtime"
    a = P.deployment_runtime_dir(env=make_env(tmp_path, LM3_RUNTIME_DIR=str(base),
                                              LM3_DEPLOYMENT_ID="slurm-1"), fs_type_reader=no_network)
    b = P.deployment_runtime_dir(env=make_env(tmp_path, LM3_RUNTIME_DIR=str(base),
                                              LM3_DEPLOYMENT_ID="slurm-2"), fs_type_reader=no_network)
    assert a.parent == b.parent == base
    assert a != b


def test_explicit_argument_outranks_the_environment(tmp_path):
    env = make_env(tmp_path, LM3_RUNTIME_DIR=str(tmp_path / "from-env"))
    got = P.runtime_base_dir(tmp_path / "explicit", env=env, fs_type_reader=no_network)
    assert got == tmp_path / "explicit"


def test_scheduler_job_uses_node_local_scratch(tmp_path):
    scratch = tmp_path / "slurm-scratch"
    scratch.mkdir()
    env = make_env(tmp_path, SLURM_JOB_ID="4242", SLURM_TMPDIR=str(scratch),
                   LM3_DEPLOYMENT_ID="slurm-4242")
    base = P.runtime_base_dir(env=env, fs_type_reader=no_network)
    assert base == scratch / "lm3-runtime"


def test_scheduler_without_node_local_scratch_fails_before_running(tmp_path):
    """Never silently place the lease on shared home storage inside an allocation."""
    env = make_env(tmp_path, SLURM_JOB_ID="4242")
    with pytest.raises(P.RuntimeDirectoryError) as excinfo:
        P.runtime_base_dir(env=env, fs_type_reader=no_network)
    assert "4242" in str(excinfo.value)
    assert "SLURM_TMPDIR" in str(excinfo.value)


def test_desktop_uses_the_platform_user_runtime_then_the_cache(tmp_path):
    xdg_runtime = tmp_path / "run-user"
    xdg_runtime.mkdir()
    env = make_env(tmp_path, XDG_RUNTIME_DIR=str(xdg_runtime))
    assert P.runtime_base_dir(env=env, platform_name="linux", fs_type_reader=no_network) == xdg_runtime / "lm3"
    env_no_runtime = make_env(tmp_path)
    assert P.runtime_base_dir(env=env_no_runtime, platform_name="linux",
                              fs_type_reader=no_network) == tmp_path / "cache" / "lm3" / "runtime"


def test_resolution_is_pure_until_asked_to_create(tmp_path):
    """The 'default runtime dir is never touched' test needs a non-creating query."""
    env = make_env(tmp_path, LM3_RUNTIME_DIR=str(tmp_path / "never"), LM3_DEPLOYMENT_ID="gpu0")
    resolved = P.deployment_runtime_dir(env=env, fs_type_reader=no_network)
    assert not (tmp_path / "never").exists()
    created = P.deployment_runtime_dir(env=env, create=True, fs_type_reader=no_network)
    assert created == resolved and created.is_dir()
    if os.name == "posix":
        assert oct(created.stat().st_mode & 0o777) == "0o700"
        assert oct(created.parent.stat().st_mode & 0o777) == "0o700"


def test_windows_runtime_base_uses_localappdata(tmp_path):
    local = tmp_path / "AppData" / "Local"
    env = {"HOME": str(tmp_path), "USERPROFILE": str(tmp_path), "LOCALAPPDATA": str(local)}
    base = P.runtime_base_dir(env=env, platform_name="win32", fs_type_reader=no_network)
    assert base == local / "lm3" / "runtime"


# --------------------------------------------------------------------------------------------- #
# 6. dev-checkout detection from __file__, never the CWD
# --------------------------------------------------------------------------------------------- #

def test_dev_checkout_is_this_repo_regardless_of_cwd(tmp_path, monkeypatch):
    expected = Path(__file__).resolve().parent.parent
    monkeypatch.chdir(tmp_path)
    assert P.dev_checkout_root() == expected
    monkeypatch.chdir("/")
    assert P.dev_checkout_root() == expected


def test_installed_package_is_not_a_dev_checkout(tmp_path):
    fake = tmp_path / "venv" / "lib" / "python3.10" / "site-packages" / "leafmachine3" / "__init__.py"
    fake.parent.mkdir(parents=True)
    fake.write_text("")
    (fake.parent.parent / "pyproject.toml").write_text("")   # even with a marker present
    assert P.dev_checkout_root(fake) is None


def test_a_directory_without_a_repo_marker_is_not_a_checkout(tmp_path):
    fake = tmp_path / "somewhere" / "leafmachine3" / "__init__.py"
    fake.parent.mkdir(parents=True)
    fake.write_text("")
    assert P.dev_checkout_root(fake) is None


# --------------------------------------------------------------------------------------------- #
# 7. precedence row 1 -- next-run settings
# --------------------------------------------------------------------------------------------- #

def write_yaml(path: Path, marker: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {marker}\n", encoding="utf-8")
    return path


def test_explicit_config_wins_and_a_missing_one_is_a_hard_error(tmp_path):
    env = make_env(tmp_path)
    cfg = write_yaml(tmp_path / "explicit" / "LM3_settings.yaml", "explicit")
    assert P.resolve_settings(cfg, env=env).source == "explicit"
    assert P.settings_path(cfg, env=env) == cfg
    with pytest.raises(P.SettingsMissingError) as excinfo:
        P.resolve_settings(tmp_path / "nope.yaml", env=env)
    assert "nope.yaml" in str(excinfo.value)


def test_lm3_settings_env_is_row_two_and_is_also_a_stated_intent(tmp_path):
    cfg = write_yaml(tmp_path / "env" / "LM3_settings.yaml", "env")
    env = make_env(tmp_path, LM3_SETTINGS=str(cfg))
    assert P.resolve_settings(env=env) == P.SettingsResolution(cfg, "env")
    env_missing = make_env(tmp_path, LM3_SETTINGS=str(tmp_path / "gone.yaml"))
    with pytest.raises(P.SettingsMissingError):
        P.resolve_settings(env=env_missing)


def test_workspace_pointer_is_row_three(tmp_path):
    env = make_env(tmp_path, LM3_DEPLOYMENT_ID="gpu0")
    chosen = write_yaml(tmp_path / "chosen" / "LM3_settings.yaml", "chosen")
    default = write_yaml(P.deployment_config_dir(env) / "LM3_settings.yaml", "deployment")
    P.write_workspace_pointer(chosen, env=env)
    resolution = P.resolve_settings(env=env)
    assert resolution.path == chosen and resolution.source == "workspace"
    assert default.is_file()   # row 4 exists and is deliberately outranked


def test_deployment_default_is_row_four(tmp_path):
    env = make_env(tmp_path, LM3_DEPLOYMENT_ID="gpu0")
    default = write_yaml(P.deployment_config_dir(env) / "LM3_settings.yaml", "deployment")
    assert P.resolve_settings(env=env) == P.SettingsResolution(default, "deployment")


def test_dev_checkout_is_row_five_only(tmp_path):
    env = make_env(tmp_path)
    resolution = P.resolve_settings(env=env, seed=False)
    checkout = P.dev_checkout_root()
    assert checkout is not None and (checkout / "LM3_settings.yaml").is_file()
    assert resolution.source == "checkout"
    assert resolution.path == checkout / "LM3_settings.yaml"


def test_a_complete_miss_seeds_the_deployment_default_and_says_so(tmp_path, monkeypatch):
    env = make_env(tmp_path, LM3_DEPLOYMENT_ID="gpu0")
    # Force a complete miss by hiding the development checkout.
    monkeypatch.setattr(P, "dev_checkout_root", lambda *a, **k: None)
    with capture_warnings() as warnings:
        resolution = P.resolve_settings(env=env)
    expected = P.deployment_config_dir(env) / "LM3_settings.yaml"
    assert resolution.path == expected and resolution.seeded is True
    assert expected.is_file() and expected.read_text(encoding="utf-8").strip()
    assert str(expected) in "\n".join(warnings)
    # Seeding is idempotent: the second call finds the file it wrote.
    assert P.resolve_settings(env=env).source == "deployment"


def test_seeding_can_be_declined_for_a_pure_query(tmp_path, monkeypatch):
    env = make_env(tmp_path, LM3_DEPLOYMENT_ID="gpu0")
    monkeypatch.setattr(P, "dev_checkout_root", lambda *a, **k: None)
    resolution = P.resolve_settings(env=env, seed=False)
    assert resolution.seeded is False
    assert not resolution.path.exists()


# --------------------------------------------------------------------------------------------- #
# 8. precedence rows 2-5
# --------------------------------------------------------------------------------------------- #

def test_hardware_profile_is_deployment_scoped_and_machine_keyed(tmp_path):
    env = make_env(tmp_path, LM3_DEPLOYMENT_ID="gpu0")
    path = P.hardware_profile_path(env=env, machine="0123456789abcdef")
    assert path == P.deployment_config_dir(env) / "hardware_settings.0123456789abcdef.yaml"
    other = P.hardware_profile_path(env=make_env(tmp_path, LM3_DEPLOYMENT_ID="gpu1"),
                                    machine="0123456789abcdef")
    assert other != path                       # gate 11: two deployments never share one profile
    on_another_node = P.hardware_profile_path(env=env, machine="fedcba9876543210")
    assert on_another_node != path             # gate 12: one networked config dir, many nodes


def test_lm3_hardware_overrides_the_deployment_profile(tmp_path):
    explicit = tmp_path / "hw.yaml"
    env = make_env(tmp_path, LM3_HARDWARE=str(explicit))
    assert P.hardware_profile_path(env=env, machine="abc") == explicit


def test_legacy_hardware_profile_is_COPIED_not_moved(tmp_path):
    env = make_env(tmp_path, LM3_DEPLOYMENT_ID="gpu0")
    settings = write_yaml(tmp_path / "checkout" / "LM3_settings.yaml", "settings")
    legacy = tmp_path / "checkout" / "hardware_settings.yaml"
    legacy.write_text("tmp_dir: /scratch\n", encoding="utf-8")
    # Resolution is PURE: asking for the path must not copy anything (plan section 3.5 / Phase 5).
    # The resolver takes no settings_file / adopt_legacy at all any more: there is exactly one
    # migration path, and it is not reachable through this function.
    import inspect
    params = inspect.signature(P.hardware_profile_path).parameters
    assert "adopt_legacy" not in params and "settings_file" not in params, (
        "hardware_profile_path must expose no adoption parameter; migration belongs to "
        "migrate_legacy_hardware_profile alone")
    pure = P.hardware_profile_path(env=env, machine="mk1")
    assert not pure.exists(), "resolving a path must never write a file"

    target = P.migrate_legacy_hardware_profile(env=env, machine="mk1", settings_file=settings)
    assert target is not None and target == pure
    assert target.read_text(encoding="utf-8") == "tmp_dir: /scratch\n"
    assert legacy.is_file(), "adoption copies; an older LM3 must keep working"
    assert P.migrate_legacy_hardware_profile(env=env, machine="mk1", settings_file=settings) is None


def test_adoption_never_overwrites_an_existing_deployment_profile(tmp_path):
    env = make_env(tmp_path, LM3_DEPLOYMENT_ID="gpu0")
    settings = write_yaml(tmp_path / "checkout" / "LM3_settings.yaml", "settings")
    (tmp_path / "checkout" / "hardware_settings.yaml").write_text("legacy: true\n", encoding="utf-8")
    target = P.deployment_config_dir(env) / "hardware_settings.mk1.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("mine: true\n", encoding="utf-8")
    P.migrate_legacy_hardware_profile(env=env, machine="mk1", settings_file=settings)
    assert target.read_text(encoding="utf-8") == "mine: true\n"


def test_config_moves_only_the_settings_path(tmp_path):
    """--config affects the settings path for that invocation and nothing else."""
    env = make_env(tmp_path, LM3_DEPLOYMENT_ID="gpu0")
    elsewhere = write_yaml(tmp_path / "elsewhere" / "LM3_settings.yaml", "elsewhere")
    hardware_before = P.hardware_profile_path(env=env, machine="mk1")
    postprocess_before = P.postprocessing_settings_path(env=env)
    assert P.settings_path(elsewhere, env=env) == elsewhere
    assert P.hardware_profile_path(env=env, machine="mk1") == hardware_before
    assert P.postprocessing_settings_path(env=env) == postprocess_before
    assert hardware_before.parent == postprocess_before.parent == P.deployment_config_dir(env)


def test_postprocessing_settings_precedence(tmp_path):
    env = make_env(tmp_path, LM3_DEPLOYMENT_ID="gpu0")
    assert P.postprocessing_settings_path(env=env) == P.deployment_config_dir(env) / "postprocessing.yaml"
    explicit = tmp_path / "pp.yaml"
    env2 = make_env(tmp_path, LM3_POSTPROCESS_SETTINGS=str(explicit))
    assert P.postprocessing_settings_path(env=env2) == explicit


def test_server_jobs_root_precedence_and_creation(tmp_path):
    env = make_env(tmp_path, LM3_DEPLOYMENT_ID="gpu0")
    default = P.server_jobs_root(env=env)
    assert default == P.deployment_state_dir(env) / "jobs"
    assert not default.exists()
    assert P.server_jobs_root(env=env, create=True).is_dir()
    override = tmp_path / "jobs-elsewhere"
    assert P.server_jobs_root(env=make_env(tmp_path, LM3_SERVER_JOBS=str(override))) == override


def test_runs_roots_order_dedup_and_empty_on_miss(tmp_path):
    env = make_env(tmp_path)
    assert P.runs_roots(env=env) == []
    settings = write_yaml(tmp_path / "cfg" / "LM3_settings.yaml", "cfg")
    env2 = make_env(tmp_path, LM3_RUNS_ROOTS=os.pathsep.join([str(tmp_path / "a"), str(tmp_path / "b")]))
    roots = P.runs_roots(env=env2, runtime_roots=[tmp_path / "b", tmp_path / "live"],
                         settings_file=settings, settings_output_dir="runs")
    assert roots == [tmp_path / "a", tmp_path / "b", tmp_path / "live", tmp_path / "cfg" / "runs"]


def test_relative_output_dir_resolves_against_the_settings_file(tmp_path):
    settings = write_yaml(tmp_path / "cfg" / "LM3_settings.yaml", "cfg")
    assert P.resolve_project_output_dir(settings, "runs") == tmp_path / "cfg" / "runs"
    assert P.resolve_project_output_dir(settings, "/abs/out") == Path("/abs/out")
    with pytest.raises(P.PathsError):
        P.resolve_project_output_dir(None, "runs")


def test_calibration_images_come_from_a_resource_or_the_checkout_never_the_cwd(tmp_path, monkeypatch):
    """Row 7: "packaged resource via ``importlib.resources``", and the DEFAULT must satisfy it.

    Asserting only absolute/is_dir/CWD-stability would pass on the dev-checkout ``examples/images``
    step alone -- which is the very directory section 3.1's calibration bullet replaces, and which
    does not exist in a wheel. So this pins the default INSIDE the installed package.
    """
    import leafmachine3

    images = P.calibration_images_dir()
    assert images.is_absolute() and images.is_dir()
    package_root = Path(leafmachine3.__file__).resolve().parent
    assert package_root in images.resolve().parents
    assert images.name == "calibration_images"
    checkout = P.dev_checkout_root()
    if checkout is not None:
        assert images.resolve() != (checkout / "examples" / "images").resolve()
    monkeypatch.chdir(tmp_path)
    assert P.calibration_images_dir() == images
    monkeypatch.setattr(P, "dev_checkout_root", lambda *a, **k: None)
    with pytest.raises(P.PackagedResourceError):
        P.calibration_images_dir(package="leafmachine3.data.definitely_not_packaged")


# --------------------------------------------------------------------------------------------- #
# 9. gate 44 -- no resolved path falls back to the CWD
# --------------------------------------------------------------------------------------------- #

def test_paths_module_contains_no_cwd_read():
    source = Path(P.__file__).read_text(encoding="utf-8")
    for forbidden in ("Path.cwd(", "os.getcwd(", "os.curdir"):
        assert forbidden not in source, f"{forbidden} reintroduces the CWD dependence gate 44 removes"


def test_every_resolved_path_is_identical_from_several_cwds(tmp_path, monkeypatch):
    env = make_env(tmp_path, LM3_DEPLOYMENT_ID="gpu0",
                   LM3_RUNTIME_DIR=str(tmp_path / "rt"))
    settings = write_yaml(P.deployment_config_dir(env) / "LM3_settings.yaml", "deployment")

    def snapshot() -> dict[str, str]:
        return {
            "settings": str(P.settings_path(env=env)),
            "hardware": str(P.hardware_profile_path(env=env, machine="mk1")),
            "postprocess": str(P.postprocessing_settings_path(env=env)),
            "jobs": str(P.server_jobs_root(env=env)),
            "runtime": str(P.deployment_runtime_dir(env=env, fs_type_reader=no_network)),
            "runs": str(P.runs_roots(env=env, settings_file=settings, settings_output_dir="runs")),
            "calibration": str(P.calibration_images_dir()),
            "checkout": str(P.dev_checkout_root()),
        }

    # Each CWD is booby-trapped with the exact files the old code used to pick up.
    decoys = []
    for name in ("cwd_a", "cwd_b"):
        decoy = tmp_path / name
        (decoy / "runs").mkdir(parents=True)
        (decoy / "examples_out").mkdir()
        write_yaml(decoy / "LM3_settings.yaml", "DECOY")
        (decoy / "hardware_settings.yaml").write_text("decoy: true\n", encoding="utf-8")
        (decoy / "postprocessing_settings.yaml").write_text("decoy: true\n", encoding="utf-8")
        decoys.append(decoy)

    results = []
    for cwd in [*decoys, Path("/"), Path(__file__).resolve().parent]:
        monkeypatch.chdir(cwd)
        snap = snapshot()
        for decoy in decoys:
            assert str(decoy) not in json.dumps(snap), f"a CWD decoy was picked up from {cwd}"
        results.append(snap)
    assert all(r == results[0] for r in results)


# The six canonical variables that name a path, each with a relative value and the resolver that
# reads it. A relative value here is the divergence gate 44 removes: its meaning would be decided
# by each process's CWD, so the GUI and the CLI would disagree about the same exported variable.
RELATIVE_ENV_CASES = [
    ("LM3_SETTINGS", "rel.yaml", lambda env: P.settings_path(env=env, seed=False)),
    ("LM3_RUNTIME_DIR", "rt", lambda env: P.deployment_runtime_dir(env=env, fs_type_reader=no_network)),
    ("LM3_HARDWARE", "hw.yaml", lambda env: P.hardware_profile_path(env=env, machine="mk1")),
    ("LM3_POSTPROCESS_SETTINGS", "pp.yaml", lambda env: P.postprocessing_settings_path(env=env)),
    ("LM3_SERVER_JOBS", "jobs", lambda env: P.server_jobs_root(env=env)),
    ("LM3_RUNS_ROOTS", "runs", lambda env: P.runs_roots(env=env)),
]


@pytest.mark.parametrize(("variable", "relative", "resolve"), RELATIVE_ENV_CASES,
                         ids=[case[0] for case in RELATIVE_ENV_CASES])
def test_relative_canonical_env_values_are_refused_from_every_cwd(
    variable, relative, resolve, tmp_path, monkeypatch,
):
    """A relative canonical variable is refused, not silently joined onto the CWD (gate 44).

    Absolutizing it would be worse than refusing: an eagerly absolutized ``rt`` still means
    whatever the LAUNCHING directory said, so two processes would still place ``activity.lock`` on
    two inodes and both acquire the lease (section 1, invariant 1).
    """
    env = make_env(
        tmp_path,
        LM3_DEPLOYMENT_ID="gpu0",
        LM3_RUNTIME_DIR=str(tmp_path / "rt"),
        LM3_SETTINGS=str(write_yaml(tmp_path / "abs" / "LM3_settings.yaml", "abs")),
        LM3_HARDWARE=str(write_yaml(tmp_path / "abs" / "hardware_settings.yaml", "abs")),
        LM3_POSTPROCESS_SETTINGS=str(write_yaml(tmp_path / "abs" / "postprocessing.yaml", "abs")),
        LM3_SERVER_JOBS=str(tmp_path / "abs" / "jobs"),
        LM3_RUNS_ROOTS=str(tmp_path / "abs" / "runs"),
    )
    env[variable] = relative

    # Two booby-trapped working directories, each holding its OWN file or directory of that name:
    # without the refusal the two CWDs would report the same string while naming different inodes.
    decoys = []
    for marker in ("cwd_a", "cwd_b"):
        decoy = tmp_path / f"decoy_{marker}"
        decoy.mkdir()
        if relative.endswith(".yaml"):
            (decoy / relative).write_text(f"# {marker}\n", encoding="utf-8")
        else:
            (decoy / relative).mkdir()
            (decoy / relative / "marker.txt").write_text(marker, encoding="utf-8")
        decoys.append(decoy)

    for decoy in decoys:
        monkeypatch.chdir(decoy)
        with pytest.raises(P.PathsError) as excinfo:
            resolve(env)
        assert variable in str(excinfo.value)
        if variable == "LM3_RUNTIME_DIR":
            # Section 3.1's runtime-registry row says "on miss: startup error", so the refusal has
            # to stay inside the error class that contract's callers already handle.
            assert isinstance(excinfo.value, P.RuntimeDirectoryError)


def test_runs_roots_refuses_a_relative_entry_among_absolute_ones(tmp_path, monkeypatch):
    """The check is per ENTRY, not first-entry-only: one relative entry poisons the whole list."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "runs").mkdir()
    env = make_env(tmp_path, LM3_RUNS_ROOTS=os.pathsep.join([str(tmp_path / "abs"), "runs"]))
    with pytest.raises(P.PathsError) as excinfo:
        P.runs_roots(env=env)
    message = str(excinfo.value)
    assert "LM3_RUNS_ROOTS" in message and "runs" in message


def test_a_relative_legacy_alias_is_refused_by_the_name_the_operator_set(tmp_path):
    """The message must name the variable that is actually in the environment, not its canonical."""
    env = make_env(tmp_path, LM3_SETTINGS_PATH="rel.yaml")
    with pytest.raises(P.PathsError) as excinfo:
        P.settings_path(env=env, seed=False)
    assert "LM3_SETTINGS_PATH" in str(excinfo.value)


def test_an_explicit_relative_argument_is_still_accepted(tmp_path, monkeypatch):
    """Deliberately NOT refused: a relative ``--config`` is one invocation with one CWD.

    Section 3.1 states exactly one hard-error rule for explicit input -- "An explicit ``--config``
    naming a missing file is always a hard error" -- and says nothing about relativity, so refusing
    it here would be an unrequested change to CLI ergonomics.
    """
    write_yaml(tmp_path / "LM3_settings.yaml", "explicit")
    monkeypatch.chdir(tmp_path)
    assert P.settings_path("LM3_settings.yaml", env=make_env(tmp_path)) == Path("LM3_settings.yaml")


def test_runs_roots_never_invents_cwd_relative_history(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "runs").mkdir()
    (tmp_path / "examples_out").mkdir()
    assert P.runs_roots(env=make_env(tmp_path)) == []


# --------------------------------------------------------------------------------------------- #
# 10. workspace pointer -- schema, atomicity, gate 15
# --------------------------------------------------------------------------------------------- #

def test_workspace_pointer_roundtrip_and_exact_schema(tmp_path):
    env = make_env(tmp_path, LM3_DEPLOYMENT_ID="gpu0")
    target = write_yaml(tmp_path / "chosen" / "LM3_settings.yaml", "chosen")
    pointer = P.write_workspace_pointer(target, env=env)
    assert pointer == P.workspace_pointer_path(env=env)
    assert pointer == P.deployment_config_dir(env) / "workspace.json"
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    assert payload == {"schema_version": 1, "settings_path": str(target)}
    assert P.read_workspace_pointer(env=env) == target


def test_absent_pointer_is_none_not_an_error(tmp_path):
    assert P.read_workspace_pointer(env=make_env(tmp_path)) is None


def test_pointer_with_a_missing_target_fails_visibly(tmp_path):
    env = make_env(tmp_path, LM3_DEPLOYMENT_ID="gpu0")
    target = write_yaml(tmp_path / "chosen" / "LM3_settings.yaml", "chosen")
    P.write_workspace_pointer(target, env=env)
    target.unlink()
    with pytest.raises(P.WorkspaceTargetMissingError) as excinfo:
        P.read_workspace_pointer(env=env)
    message = str(excinfo.value)
    assert "selected settings file is missing" in message
    assert str(target) in message
    # And resolution refuses to quietly use anything else.
    with pytest.raises(P.WorkspaceTargetMissingError):
        P.resolve_settings(env=env)


@pytest.mark.parametrize("payload", [
    {"schema_version": 1},                                             # no settings_path
    {"settings_path": "/abs/LM3_settings.yaml"},                       # no schema_version
    {"schema_version": 2, "settings_path": "/abs/LM3_settings.yaml"},  # future schema
    {"schema_version": 1, "settings_path": "relative.yaml"},           # not absolute
    {"schema_version": 1, "settings_path": ""},                        # empty
    {"schema_version": 1, "settings_path": "/abs/x.yaml", "extra": 1},  # nothing else allowed
    ["not", "an", "object"],
])
def test_pointer_schema_violations_are_errors(tmp_path, payload):
    pointer = tmp_path / "workspace.json"
    pointer.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(P.WorkspacePointerError):
        P.read_workspace_pointer(pointer)


def test_malformed_pointer_json_is_an_error(tmp_path):
    pointer = tmp_path / "workspace.json"
    pointer.write_text("{not json", encoding="utf-8")
    with pytest.raises(P.WorkspacePointerError):
        P.read_workspace_pointer(pointer)


def test_pointer_update_is_atomic_and_leaves_no_debris(tmp_path):
    env = make_env(tmp_path, LM3_DEPLOYMENT_ID="gpu0")
    first = write_yaml(tmp_path / "a" / "LM3_settings.yaml", "a")
    second = write_yaml(tmp_path / "b" / "LM3_settings.yaml", "b")
    pointer = P.write_workspace_pointer(first, env=env)
    P.write_workspace_pointer(second, env=env)
    assert P.read_workspace_pointer(env=env) == second
    siblings = sorted(p.name for p in pointer.parent.iterdir())
    assert siblings == ["workspace.json"], f"temp sibling left behind: {siblings}"


def test_pointer_can_be_cleared(tmp_path):
    env = make_env(tmp_path, LM3_DEPLOYMENT_ID="gpu0")
    target = write_yaml(tmp_path / "chosen" / "LM3_settings.yaml", "chosen")
    P.write_workspace_pointer(target, env=env)
    assert P.clear_workspace_pointer(env=env) is True
    assert P.clear_workspace_pointer(env=env) is False
    assert P.read_workspace_pointer(env=env) is None


def test_pointer_target_must_be_absolute_on_write(tmp_path):
    with pytest.raises(P.WorkspacePointerError):
        P.write_workspace_pointer("relative/LM3_settings.yaml", tmp_path / "workspace.json")


# --------------------------------------------------------------------------------------------- #
# 11. legacy environment migration
# --------------------------------------------------------------------------------------------- #

def test_legacy_variable_alone_is_honored_with_a_warning(tmp_path):
    cfg = write_yaml(tmp_path / "legacy" / "LM3_settings.yaml", "legacy")
    env = make_env(tmp_path, LM3_SETTINGS_PATH=str(cfg))
    with capture_warnings() as warnings:
        resolution = P.resolve_settings(env=env)
    assert resolution.path == cfg
    assert "LM3_SETTINGS_PATH" in "\n".join(warnings)


def test_agreeing_legacy_and_canonical_variables_warn_but_work(tmp_path):
    cfg = write_yaml(tmp_path / "both" / "LM3_settings.yaml", "both")
    env = make_env(tmp_path, LM3_SETTINGS=str(cfg), LM3_SETTINGS_PATH=str(cfg))
    assert P.resolve_settings(env=env).path == cfg


def test_agreeing_variables_warn_exactly_once_per_process_in_the_resolver(tmp_path):
    """Section 4 Step 1: "use it and warn once". Once per PROCESS, not once per resolution.

    The latch has to live in ``resolve_legacy_env`` itself, because hardware setup, ``machine3``
    and ``calibrate`` resolve these variables without ever going through the server's memo.
    """
    P.reset_legacy_warnings()
    cfg = write_yaml(tmp_path / "both" / "LM3_settings.yaml", "both")
    env = make_env(tmp_path, LM3_SETTINGS=str(cfg), LM3_SETTINGS_PATH=str(cfg))
    with capture_warnings() as warnings:
        for _ in range(20):
            assert P.resolve_legacy_env(env, "LM3_SETTINGS") == str(cfg)
    assert len([w for w in warnings if "LM3_SETTINGS_PATH" in w]) == 1, warnings

    # A CHANGED environment is a different fact and warns again; the latch only ever suppresses a
    # byte-identical repeat.
    other = write_yaml(tmp_path / "other" / "LM3_settings.yaml", "other")
    env2 = make_env(tmp_path, LM3_SETTINGS=str(other), LM3_SETTINGS_PATH=str(other))
    with capture_warnings() as warnings:
        P.resolve_legacy_env(env2, "LM3_SETTINGS")
    assert len([w for w in warnings if "LM3_SETTINGS_PATH" in w]) == 1, warnings


def test_reset_legacy_warnings_restores_fresh_process_state(tmp_path):
    cfg = write_yaml(tmp_path / "reset" / "LM3_settings.yaml", "reset")
    env = make_env(tmp_path, LM3_SETTINGS_PATH=str(cfg))
    P.reset_legacy_warnings()
    with capture_warnings() as first:
        P.resolve_legacy_env(env, "LM3_SETTINGS")
    with capture_warnings() as second:
        P.resolve_legacy_env(env, "LM3_SETTINGS")
    P.reset_legacy_warnings()
    with capture_warnings() as third:
        P.resolve_legacy_env(env, "LM3_SETTINGS")
    assert len(first) == 1 and second == [] and len(third) == 1


def test_a_conflict_raises_on_every_call_and_is_never_latched(tmp_path):
    """The warn-once latch must not turn a split brain into a one-time complaint."""
    a = write_yaml(tmp_path / "a" / "LM3_settings.yaml", "a")
    b = write_yaml(tmp_path / "b" / "LM3_settings.yaml", "b")
    env = make_env(tmp_path, LM3_SETTINGS=str(a), LM3_SETTINGS_PATH=str(b))
    for _ in range(3):
        with pytest.raises(P.LegacyEnvConflictError):
            P.resolve_legacy_env(env, "LM3_SETTINGS")


def test_conflicting_legacy_and_canonical_variables_are_fatal(tmp_path):
    a = write_yaml(tmp_path / "a" / "LM3_settings.yaml", "a")
    b = write_yaml(tmp_path / "b" / "LM3_settings.yaml", "b")
    env = make_env(tmp_path, LM3_SETTINGS=str(a), LM3_SETTINGS_PATH=str(b))
    with pytest.raises(P.LegacyEnvConflictError) as excinfo:
        P.resolve_settings(env=env)
    assert "LM3_SETTINGS_PATH" in str(excinfo.value)
    hw_env = make_env(tmp_path, LM3_HARDWARE=str(a), LM3_HARDWARE_SETTINGS=str(b))
    with pytest.raises(P.LegacyEnvConflictError):
        P.hardware_profile_path(env=hw_env, machine="mk1")


# --------------------------------------------------------------------------------------------- #
# 12. the bounded early resolver -- section 2.7
# --------------------------------------------------------------------------------------------- #

def test_early_run_paths_are_deterministic_and_create_nothing(tmp_path):
    out = tmp_path / "runs"
    early = P.early_run_paths(output_dir=out, run_name="demo")
    assert early.run_dir == out / "demo"
    assert early.active_db_path == out / "demo" / "demo.sqlite"
    assert early.log_path == out / "demo" / "logs" / "lm3.log"
    assert not out.exists()


def test_early_run_paths_never_publishes_tmp_dir():
    """_ensure_tmp falls back to <root>/_tmp_original on OSError, so tmp is not knowable early."""
    fields = set(P.EarlyRunPaths.__dataclass_fields__)
    assert not any("tmp" in name for name in fields), fields
    assert fields == {
        "run_name", "run_dir", "artifact_dir", "active_state_dir", "active_db_path",
        "archive_pointer_path", "log_path", "archive_mode",
    }


def test_desktop_run_is_in_place_with_a_null_pointer(tmp_path):
    early = P.early_run_paths(output_dir=tmp_path / "runs", run_name="demo")
    assert early.archive_mode == "in-place"
    assert early.archive_pointer_path is None
    assert early.active_state_dir == early.artifact_dir


def test_cluster_run_is_staged_with_a_mandatory_pointer(tmp_path):
    early = P.early_run_paths(output_dir=tmp_path / "project", run_name="demo",
                              active_state_dir=tmp_path / "nodelocal")
    assert early.archive_mode == "staged"
    assert early.active_state_dir == tmp_path / "nodelocal" / "demo"
    assert early.active_db_path == tmp_path / "nodelocal" / "demo" / "demo.sqlite"
    assert early.archive_pointer_path == tmp_path / "project" / "demo" / "archive.current.json"


def test_early_run_paths_refuses_a_relative_output_dir():
    with pytest.raises(P.PathsError):
        P.early_run_paths(output_dir="runs", run_name="demo")


# --------------------------------------------------------------------------------------------- #
# 13. diagnostics
# --------------------------------------------------------------------------------------------- #

def test_describe_resolved_paths_never_raises(tmp_path):
    env = make_env(tmp_path, LM3_DEPLOYMENT_ID="gpu0", LM3_RUNTIME_DIR=str(tmp_path / "rt"))
    summary = P.describe_resolved_paths(env=env)
    assert summary["deployment_id"] == "gpu0"
    assert summary["deployment_key"] == P.canonical_deployment_key("gpu0")
    assert set(summary) >= {"settings", "hardware_profile", "runtime_dir", "server_jobs_root"}
    broken = P.describe_resolved_paths(env={"LM3_DEPLOYMENT_ID": "   "})
    assert broken["deployment_id"].startswith("<error:")


def test_diagnostics_carry_no_secrets(tmp_path):
    summary = P.describe_resolved_paths(env=make_env(tmp_path, LM3_SERVER_TOKEN="s3cret"))
    assert "s3cret" not in json.dumps(summary)
