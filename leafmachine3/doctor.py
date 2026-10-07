"""``lm3 doctor`` -- the install-time gate (PACKAGING_PLAN.md section 3).

Exit 0 means this environment will run the pipeline on the hardware it was installed for. Anything
else names ONE cause, in one sentence, with the command that fixes it. Checks run in order and stop
at the first failure, because a later check's result is meaningless on top of an earlier failure
(a contract mismatch on a box with no driver says nothing useful).

====  =====================  ===================================================================
 #    check                  passes when
====  =====================  ===================================================================
 1    interpreter            Python is the release's 3.11 and runs inside a virtual environment
 2    hardware variant       exactly one onnxruntime distribution is installed (gpu|cpu|macos)
 3    environment contract   every package in ``_env_contract.json`` for that variant is installed
                             at exactly the locked version AND every file it installed still exists
 4    platform               this OS / CPU architecture is supported for that variant
 5    NVIDIA driver          (gpu) NVML sees a GPU and the driver meets the CUDA 12.4 minimum
 6    accelerator            a real ONNX Runtime session binds the accelerator AND runs, in a child
                             process that goes through machine3's own CUDA library-path step
 7    models                 (``--models``) every required default model is installed and current
 8    write access           the runtime state dir and the model folder are writable
====  =====================  ===================================================================

Check 6 is the one that catches the silent CPU fallback. ``get_available_providers()`` lists
``CUDAExecutionProvider`` whenever the GPU build is installed; it only fails when a session is
created, and then onnxruntime quietly binds the CPU. On 2026-10-06 three full pipeline runs went to
the CPU that way after a stray ``pip install onnxruntime`` overwrote the GPU build's binary. Check 2
catches that particular cause directly (both distributions installed); check 6 catches every other.

Checks 1-4 are cheap (no subprocess, no GPU): :func:`startup_gate` runs exactly those and is what
``machine3`` / ``lm3 serve`` should call at startup.

A DEVELOPMENT environment (the ``full`` dependency group: torch, ultralytics) is reported, not
failed, unless ``--production`` is given. Production installs never contain those packages.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from importlib import metadata, resources
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

OK, WARN, FAIL, SKIP = "ok", "warn", "FAIL", "skip"

#: (sys.platform, platform.machine()) pairs each hardware variant supports.
SUPPORTED = {
    "gpu": {("linux", "x86_64"), ("win32", "AMD64")},
    "cpu": {("linux", "x86_64"), ("win32", "AMD64")},
    "macos": {("darwin", "arm64")},
}
#: The provider check 6 must see bound for each variant.
ACCELERATOR = {"gpu": "CUDAExecutionProvider", "cpu": "CPUExecutionProvider", "macos": "CoreMLExecutionProvider"}
#: Max |accelerator - CPU| on the probe model. Conv+ReLU on [0, 1) inputs; fp32 kernels agree to ~1e-6.
PROBE_TOLERANCE = 1e-3
SYNC_HINT = "cd into your LeafMachine3 folder and run: uv sync --frozen --extra {extra}"


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


# --------------------------------------------------------------------------------------------------
# What the checks look at. Built once from the live process; tests build it by hand.
# --------------------------------------------------------------------------------------------------
@dataclass
class Env:
    python_version: str                      # "3.11.17"
    executable: str
    prefix: str
    base_prefix: str
    sys_platform: str                        # sys.platform
    machine: str                             # platform.machine()
    installed: dict[str, str]                # normalized dist name -> version
    environ: Mapping[str, str]
    contract: dict
    #: normalized dist name -> files its RECORD lists that are no longer on disk
    missing_files: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def current(cls) -> "Env":
        installed: dict[str, str] = {}
        missing: dict[str, list[str]] = {}
        for d in metadata.distributions():
            name = d.metadata.get("Name")
            if name and _norm(name) not in installed:
                installed[_norm(name)] = d.version
                gone = missing_record_files(d)
                if gone:
                    missing[_norm(name)] = gone
        return cls(
            python_version=platform.python_version(), executable=sys.executable, prefix=sys.prefix,
            base_prefix=getattr(sys, "base_prefix", sys.prefix), sys_platform=sys.platform,
            machine=platform.machine(), installed=installed, environ=dict(os.environ),
            contract=load_contract(), missing_files=missing,
        )


#: Files a package may list yet legitimately lose. ``nvidia/__init__.py`` is the namespace marker that
#: several nvidia-* wheels each claim; uninstalling any one deletes it for all. LM3 finds the CUDA
#: libraries through ``nvidia.__path__`` (core/cuda_libs.py), so its absence is harmless.
HARMLESS_MISSING = frozenset({"nvidia/__init__.py"})


def missing_record_files(dist: metadata.Distribution) -> list[str]:
    """Files ``dist`` installed (per its RECORD) that are gone.

    Two packages that install into the same directory corrupt each other on uninstall: removing one
    deletes files the other still needs, while the survivor's metadata still claims a clean install.
    Seen on 2026-10-07 with opencv-python / opencv-python-headless (cv2/ deleted) and the nvidia-*
    namespace. Version metadata cannot see this; only the files can. Stat only, no hashing (~10k
    files, well under a second).
    """
    gone = []
    for f in dist.files or ():
        rel = str(f).replace("\\", "/")
        if rel.startswith("../") or "__pycache__/" in rel or rel.endswith(".pyc") or rel in HARMLESS_MISSING:
            continue
        if not Path(dist.locate_file(f)).exists():
            gone.append(rel)
    return gone


def load_contract() -> dict:
    return json.loads(resources.files("leafmachine3").joinpath("_env_contract.json").read_text())


@dataclass
class Check:
    num: int
    name: str
    status: str
    detail: str = ""
    fix: str = ""


@dataclass
class Report:
    lm3_version: str
    variant: Optional[str]
    checks: list[Check] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return all(c.status != FAIL for c in self.checks)

    @property
    def first_failure(self) -> Optional[Check]:
        return next((c for c in self.checks if c.status == FAIL), None)

    def to_json(self) -> dict:
        d = asdict(self)
        d["ready"] = self.ready
        return d


# --------------------------------------------------------------------------------------------------
# Checks 1-4: no subprocess, no GPU
# --------------------------------------------------------------------------------------------------
def detect_variant(env: Env) -> tuple[Optional[str], str]:
    """Which hardware extra is installed, judged by the onnxruntime distribution(s) present."""
    gpu, cpu = "onnxruntime-gpu" in env.installed, "onnxruntime" in env.installed
    if gpu and cpu:
        return None, "both"
    if gpu:
        return "gpu", ""
    if cpu:
        return ("macos" if env.sys_platform == "darwin" else "cpu"), ""
    return None, "none"


def check_interpreter(env: Env) -> Check:
    want = env.contract["python"]
    want_minor = ".".join(want.split(".")[:2])
    have_minor = ".".join(env.python_version.split(".")[:2])
    if have_minor != want_minor:
        return Check(1, "interpreter", FAIL,
                     f"Python {env.python_version} at {env.executable}; LeafMachine3 requires Python {want_minor}",
                     "install with uv, which downloads the right Python itself: uv sync --frozen --extra <gpu|cpu|macos>")
    if env.prefix == env.base_prefix:
        return Check(1, "interpreter", FAIL,
                     f"{env.executable} is not inside a virtual environment",
                     "run LeafMachine3 through uv from its folder (uv run lm3 ...), never with a system Python")
    managed = "/uv/python/" in env.base_prefix.replace("\\", "/")
    if env.python_version != want or not managed:
        why = []
        if env.python_version != want:
            why.append(f"this release was tested on Python {want}")
        if not managed:
            why.append(f"the interpreter at {env.base_prefix} was not installed by uv")
        return Check(1, "interpreter", WARN, f"Python {env.python_version}; " + " and ".join(why),
                     "uv sync --frozen --extra <gpu|cpu|macos> rebuilds .venv on the release interpreter")
    return Check(1, "interpreter", OK, f"Python {env.python_version} (uv-managed) in {env.prefix}")


def check_variant(env: Env) -> tuple[Check, Optional[str]]:
    variant, problem = detect_variant(env)
    expected = env.environ.get("LM3_EXTRA") or None
    if problem == "both":
        return Check(2, "hardware variant", FAIL,
                     f"both onnxruntime {env.installed['onnxruntime']} and onnxruntime-gpu "
                     f"{env.installed['onnxruntime-gpu']} are installed; they overwrite each other's files "
                     "and the GPU build silently runs on the CPU",
                     "uv sync --frozen --extra gpu --reinstall-package onnxruntime-gpu   (removes the stray package and restores the GPU build's files)"), None
    if problem == "none":
        return Check(2, "hardware variant", FAIL, "no onnxruntime is installed",
                     "uv sync --frozen --extra gpu   (or --extra cpu, or --extra macos on a Mac)"), None
    if expected and expected != variant:
        return Check(2, "hardware variant", FAIL,
                     f"this environment holds the {variant} variant but LM3_EXTRA={expected}",
                     SYNC_HINT.format(extra=expected)), variant
    ort = "onnxruntime-gpu" if variant == "gpu" else "onnxruntime"
    return Check(2, "hardware variant", OK, f"{variant} ({ort} {env.installed[ort]})"), variant


def _marker_applies(marker: str, env: Env) -> bool:
    if not marker:
        return True
    try:
        from packaging.markers import Marker
    except ImportError:          # packaging ships with onnxruntime; absent only in a broken env
        return True
    return Marker(marker).evaluate({"sys_platform": env.sys_platform, "platform_machine": env.machine,
                                    "python_version": ".".join(env.python_version.split(".")[:2]),
                                    "python_full_version": env.python_version})


def check_contract(env: Env, variant: str) -> Check:
    rows = [r for r in env.contract["extras"][variant] if _marker_applies(r["marker"], env)]
    missing = [r for r in rows if r["name"] not in env.installed]
    wrong = [(r, env.installed[r["name"]]) for r in rows
             if r["name"] in env.installed and env.installed[r["name"]] != r["version"]]
    if missing or wrong:
        parts = [f"{r['name']} is missing (expected {r['version']})" for r in missing[:3]]
        parts += [f"{r['name']} is {have}, expected {r['version']}" for r, have in wrong[:3]]
        more = len(missing) + len(wrong) - len(parts)
        return Check(3, "environment contract", FAIL,
                     "; ".join(parts) + (f"; and {more} more" if more > 0 else "")
                     + " -- this is not the environment this release was tested with",
                     SYNC_HINT.format(extra=variant))
    damaged = {r["name"]: env.missing_files[r["name"]] for r in rows if env.missing_files.get(r["name"])}
    if damaged:
        desc = "; ".join(f"{name} is missing {len(files)} of its files (e.g. {files[0]})"
                         for name, files in list(damaged.items())[:3])
        more = len(damaged) - 3
        return Check(3, "environment contract", FAIL,
                     desc + (f"; and {more} more packages" if more > 0 else "")
                     + " -- another package's uninstall deleted them; the version metadata still looks right",
                     f"uv sync --frozen --extra {variant} " + " ".join(f"--reinstall-package {n}" for n in damaged))
    return Check(3, "environment contract", OK, f"{len(rows)} packages at their locked versions, files intact")


def check_platform(env: Env, variant: str) -> Check:
    here = (env.sys_platform, env.machine)
    if here not in SUPPORTED[variant]:
        ok = ", ".join(f"{p} {m}" for p, m in sorted(SUPPORTED[variant]))
        return Check(4, "platform", FAIL, f"{env.sys_platform} {env.machine} is not supported for the "
                     f"{variant} variant (supported: {ok})",
                     "see INSTALL.md, Supported platforms")
    return Check(4, "platform", OK, f"{env.sys_platform} {env.machine}")


# --------------------------------------------------------------------------------------------------
# Check 5: NVIDIA driver
# --------------------------------------------------------------------------------------------------
def _version_tuple(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", v))


def nvml_probe() -> tuple[str, list[str]]:
    """(driver version, GPU names) from NVML. Raises on any NVML failure."""
    import pynvml

    pynvml.nvmlInit()
    try:
        drv = pynvml.nvmlSystemGetDriverVersion()
        drv = drv.decode() if isinstance(drv, bytes) else str(drv)
        names = []
        for i in range(int(pynvml.nvmlDeviceGetCount())):
            n = pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(i))
            names.append(n.decode() if isinstance(n, bytes) else str(n))
        return drv, names
    finally:
        pynvml.nvmlShutdown()


def check_driver(env: Env, nvml: Callable[[], tuple[str, list[str]]] = nvml_probe) -> Check:
    cvd = env.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd.strip() in {"", "-1", "none", "NoDevFiles"}:
        return Check(5, "NVIDIA driver", FAIL,
                     f"CUDA_VISIBLE_DEVICES={cvd!r} hides every GPU from LeafMachine3",
                     "unset CUDA_VISIBLE_DEVICES, or set it to the GPU index to use (e.g. 0)")
    try:
        driver, gpus = nvml()
    except Exception as exc:  # noqa: BLE001 - every NVML failure means "no usable driver"
        return Check(5, "NVIDIA driver", FAIL,
                     f"no NVIDIA driver found (NVML could not start: {type(exc).__name__}: {exc})",
                     "install the NVIDIA driver, or reinstall with the CPU variant: uv sync --frozen --extra cpu")
    if not gpus:
        return Check(5, "NVIDIA driver", FAIL, f"driver {driver} is installed but reports no GPU",
                     "check nvidia-smi, or reinstall with the CPU variant: uv sync --frozen --extra cpu")
    minimum = env.contract["cuda_min_driver"].get(env.sys_platform)
    if minimum and _version_tuple(driver) < _version_tuple(minimum):
        return Check(5, "NVIDIA driver", FAIL,
                     f"driver {driver} is below {minimum}, the minimum for the CUDA 12.4 libraries LeafMachine3 ships",
                     f"update the NVIDIA driver to {minimum} or newer, or reinstall with: uv sync --frozen --extra cpu")
    return Check(5, "NVIDIA driver", OK, f"driver {driver} (>= {minimum}); " + ", ".join(gpus))


# --------------------------------------------------------------------------------------------------
# Check 6: the accelerator, in a child process
# --------------------------------------------------------------------------------------------------
def run_probe_subprocess(provider: str, environ: Mapping[str, str], timeout: float = 180.0) -> dict:
    """Run :mod:`leafmachine3.doctor_probe` the way a pipeline worker would start, return its JSON."""
    cmd = [sys.executable, "-m", "leafmachine3.doctor_probe", provider]
    try:
        cp = subprocess.run(cmd, env=dict(environ), capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"requested": provider, "bound": None, "error": f"the probe did not finish in {timeout:.0f} s"}
    last = next((ln for ln in reversed(cp.stdout.splitlines()) if ln.startswith("{")), None)
    out = json.loads(last) if last else {"requested": provider, "bound": None,
                                         "error": f"the probe exited {cp.returncode} without a result"}
    m = re.search(r"Failed to load library (\S+) with error: ([^\n\[]+)", cp.stderr)
    if m:
        out["load_error"] = f"{m.group(1)}: {m.group(2).strip()}"
    if not last and cp.stderr.strip():
        out["stderr_tail"] = cp.stderr.strip().splitlines()[-1][:300]
    return out


def check_accelerator(env: Env, variant: str,
                      probe: Callable[[str, Mapping[str, str]], dict] = run_probe_subprocess) -> Check:
    want = ACCELERATOR[variant]
    res = probe(want, env.environ)
    if res.get("error"):
        return Check(6, "accelerator", FAIL, f"{want}: {res['error']}" + _hints(res, env),
                     _accelerator_fix(variant))
    if res.get("bound") != want:
        why = res.get("load_error") or "onnxruntime gave no reason"
        return Check(6, "accelerator", FAIL,
                     f"asked for {want} but onnxruntime bound {res.get('bound')} -- every model would run on "
                     f"the CPU ({why})" + _hints(res, env), _accelerator_fix(variant))
    diff = res.get("max_abs_diff")
    if diff is None or diff > PROBE_TOLERANCE:
        return Check(6, "accelerator", FAIL,
                     f"{want} ran but returned wrong numbers (max |diff| vs CPU = {diff})",
                     "report this with the output of `lm3 doctor --json`")
    return Check(6, "accelerator", OK, f"{want} bound and ran the probe model (max |diff| vs CPU {diff:.1e})")


def _hints(res: dict, env: Env) -> str:
    hints = []
    if env.environ.get("LM3_CUDA_LIBPATH_SET"):
        hints.append("LM3_CUDA_LIBPATH_SET is set in your environment, which tells LeafMachine3 the CUDA "
                     "library path was already prepared and skips that step -- unset it")
    if res.get("stderr_tail"):
        hints.append(res["stderr_tail"])
    return "".join(f"; {h}" for h in hints)


def _accelerator_fix(variant: str) -> str:
    if variant == "gpu":
        return ("uv sync --frozen --extra gpu --reinstall-package onnxruntime-gpu; if it still fails, "
                "unset LD_LIBRARY_PATH / LM3_CUDA_LIBPATH_SET and run lm3 doctor again")
    return SYNC_HINT.format(extra=variant)


# --------------------------------------------------------------------------------------------------
# Checks 7-8
# --------------------------------------------------------------------------------------------------
def cpu_fallback_ops(model_path: str, environ: Mapping[str, str], timeout: float = 180.0) -> list[str]:
    """Ops in ``model_path`` that onnxruntime's CUDA provider has no kernel for (they run on the CPU).

    A child creates the session through machine3's CUDA library-path step at verbose logging, and
    this parses its "CUDA kernel not found in registries for Op type: X" lines. Shape arithmetic
    onnxruntime keeps on the CPU on purpose never produces that line, so it is not reported.
    """
    try:
        cp = subprocess.run([sys.executable, "-m", "leafmachine3.doctor_probe", "--placement", model_path],
                            env=dict(environ), capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return ["(probe timed out)"]
    return sorted(set(re.findall(r"CUDA kernel not found in registries for Op type: (\w+)", cp.stderr)))


def check_models(settings: Optional[str] = None, variant: Optional[str] = None,
                 environ: Optional[Mapping[str, str]] = None,
                 fallback: Callable[[str, Mapping[str, str]], list[str]] = cpu_fallback_ops) -> Check:
    try:
        from leafmachine3.modelhub import installer
    except ImportError:
        return Check(7, "models", SKIP, "this build has no model installer")
    try:
        st = installer.status(installer.models_root(settings) if settings else None)
    except Exception as exc:  # noqa: BLE001
        return Check(7, "models", FAIL, f"could not read the model folder: {type(exc).__name__}: {exc}",
                     "lm3 models status")
    s = st["summary"]
    if s["missing"] or s["outdated"]:
        what = ([f"missing: {', '.join(s['missing'])}"] if s["missing"] else []) + \
               ([f"outdated: {', '.join(s['outdated'])}"] if s["outdated"] else [])
        return Check(7, "models", FAIL, f"{'; '.join(what)} in {st['root']}", "lm3 models install")
    if s["unavailable"]:
        return Check(7, "models", WARN, f"not published yet and no local copy: {', '.join(s['unavailable'])}",
                     "these stages cannot run until the models are published")
    if variant == "gpu":
        onnx_files = [f["dest"] for a in st["actions"].values() for f in a["files"]
                      if f.get("format") == "onnx" and f.get("present")]
        on_cpu = {dest: ops for dest in onnx_files
                  if (ops := fallback(str(Path(st["root"]) / dest), environ or os.environ))}
        if on_cpu:
            listed = "; ".join(f"{d} ({', '.join(o)})" for d, o in sorted(on_cpu.items()))
            return Check(7, "models", WARN,
                         f"{len(on_cpu)} model(s) run some ops on the CPU, copying data between GPU and host in "
                         f"every forward pass: {listed}",
                         "slower, not wrong. For opset-19 YOLO exports (Resize) the fix is an opset-18 "
                         "re-export: tools/modelhub/fix_resize_opset.py, then `lm3 models install` once published")
        return Check(7, "models", OK, f"all required models current in {st['root']}; "
                     f"all {len(onnx_files)} ONNX models run entirely on the GPU")
    return Check(7, "models", OK, f"all required models current in {st['root']}")


def _nearest_existing(p: Path) -> Path:
    p = p.expanduser()
    while not p.exists() and p.parent != p:
        p = p.parent
    return p


def check_write_access(settings: Optional[str] = None) -> Check:
    targets: list[tuple[str, Path]] = []
    try:
        from leafmachine3.core import paths

        targets.append(("runtime state", paths.deployment_state_dir()))
    except Exception:  # noqa: BLE001 - an unresolvable state dir is itself reported below
        pass
    try:
        from leafmachine3.modelhub import installer

        targets.append(("models", installer.models_root(settings) if settings else installer.models_root()))
    except Exception:  # noqa: BLE001
        pass
    bad = [f"{label} {p} (nearest existing: {_nearest_existing(Path(p))})" for label, p in targets
           if not os.access(_nearest_existing(Path(p)), os.W_OK)]
    if bad:
        return Check(8, "write access", FAIL, "not writable: " + "; ".join(bad),
                     "fix the folder permissions, or point LM3 elsewhere (LM3_MODELS_DIR, LM3 settings)")
    return Check(8, "write access", OK, "; ".join(f"{label} {p}" for label, p in targets) or "nothing to check")


# --------------------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------------------
def dev_note(env: Env) -> Optional[str]:
    dev = {n: env.installed[n] for n in env.contract.get("dev_only", []) if n in env.installed}
    if not dev:
        return None
    return ("development environment: " + ", ".join(f"{n} {v}" for n, v in dev.items())
            + " installed (the `full` group); production installs never contain these")


def diagnose(env: Optional[Env] = None, *, quick: bool = False, models: bool = False,
             production: bool = False, settings: Optional[str] = None,
             nvml: Callable[[], tuple[str, list[str]]] = nvml_probe,
             probe: Callable[[str, Mapping[str, str]], dict] = run_probe_subprocess,
             models_fallback: Callable[[str, Mapping[str, str]], list[str]] = cpu_fallback_ops) -> Report:
    env = env or Env.current()
    rep = Report(lm3_version=env.contract.get("lm3_version", "?"), variant=None)
    stopped = False

    def add(check: Check) -> None:
        nonlocal stopped
        rep.checks.append(check)
        if check.status == FAIL:
            stopped = True                   # later checks are not run: their answer would be noise

    add(check_interpreter(env))
    if stopped:
        return _finish(rep, env, production)
    c2, variant = check_variant(env)
    add(c2)
    rep.variant = variant
    if stopped:
        return _finish(rep, env, production)
    add(check_contract(env, variant))
    if not stopped:
        add(check_platform(env, variant))
    if quick or stopped:
        return _finish(rep, env, production)
    if variant == "gpu":
        add(check_driver(env, nvml))
    if not stopped:
        add(check_accelerator(env, variant, probe))
    if not stopped:
        add(check_models(settings, variant, env.environ, models_fallback) if models
            else Check(7, "models", SKIP, "not checked (add --models)"))
    if not stopped:
        add(check_write_access(settings))
    return _finish(rep, env, production)


def _finish(rep: Report, env: Env, production: bool) -> Report:
    note = dev_note(env)
    if note and production:
        rep.checks.append(Check(0, "production", FAIL, note,
                                "rebuild without the development group: uv sync --frozen --extra "
                                f"{rep.variant or '<gpu|cpu|macos>'}"))
    elif note:
        rep.notes.append(note)
    return rep


class DoctorError(RuntimeError):
    """Raised by :func:`startup_gate` when this environment cannot run LeafMachine3."""

    def __init__(self, report: Report):
        self.report = report
        c = report.first_failure
        super().__init__(f"lm3 doctor check {c.num} ({c.name}) failed: {c.detail}. Fix: {c.fix}" if c else "")


def startup_gate(env: Optional[Env] = None) -> Report:
    """Checks 1-4 only (no subprocess, no GPU; milliseconds). For ``machine3`` / ``lm3 serve`` startup.

    Raises :class:`DoctorError` naming the first failed check; returns the report otherwise so the
    caller can log warnings and the development-environment note.
    """
    rep = diagnose(env, quick=True)
    if not rep.ready:
        raise DoctorError(rep)
    return rep


#: Exit status when the startup gate refuses this environment: sysexits.h EX_CONFIG. Distinct from 75
#: (EXIT_CODE_BUSY: the deployment is busy, retry later) and 2 (a usage error).
EXIT_CODE_ENVIRONMENT = 78
#: ``LM3_STARTUP_GATE=0`` skips the gate (an escape hatch, and what the test suite pins, so pipeline
#: tests do not depend on which environment runs them).
GATE_ENV = "LM3_STARTUP_GATE"


def run_startup_gate(prog: str) -> Optional[int]:
    """Checks 1-4 before ``machine3`` / ``lm3 serve`` start. ``None`` = go on; else the exit code.

    Prints the failed check and its fix to stderr. Warnings (an interpreter uv did not install, a
    different 3.11 patch) are printed and do not stop anything. If the gate itself breaks, it says so
    and lets the program run: a bug here must never be what keeps LM3 from starting.
    """
    if os.environ.get(GATE_ENV, "1").strip().lower() in {"0", "false", "no", "off"}:
        return None
    try:
        rep = startup_gate()
    except DoctorError as exc:
        print(f"{prog}: {exc}", file=sys.stderr)
        print(f"{prog}: run `lm3 doctor` for the full report (or set {GATE_ENV}=0 to bypass at your own risk)",
              file=sys.stderr)
        return EXIT_CODE_ENVIRONMENT
    except Exception as exc:  # noqa: BLE001 - the gate must never be the thing that breaks LM3
        print(f"{prog}: warning: the environment check could not run ({type(exc).__name__}: {exc})",
              file=sys.stderr)
        return None
    for c in rep.checks:
        if c.status == WARN:
            print(f"{prog}: warning: {c.name}: {c.detail}", file=sys.stderr)
    return None


def render(rep: Report, env: Env) -> str:
    mark = {OK: "ok  ", WARN: "warn", FAIL: "FAIL", SKIP: "skip"}
    lines = [f"LeafMachine3 {rep.lm3_version}   variant={rep.variant or '?'}   python={env.python_version}"]
    for c in rep.checks:
        label = f"{c.num} {c.name}" if c.num else c.name
        lines.append(f"  [{mark[c.status]}] {label:24s} {c.detail}")
        if c.status in (FAIL, WARN) and c.fix:
            lines.append(f"  {'':6s} {'':24s} fix: {c.fix}")
    for n in rep.notes:
        lines.append(f"  note: {n}")
    first = rep.first_failure
    if rep.ready:
        lines.append("Result: READY")
    else:
        which = f"check {first.num} ({first.name})" if first.num else f"the {first.name} check"
        lines.append(f"Result: NOT READY -- fix {which} first")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="lm3 doctor", description="Check that this LeafMachine3 install will run.")
    ap.add_argument("--json", action="store_true", help="print the report as one JSON object")
    ap.add_argument("--quick", action="store_true", help="checks 1-4 only (no GPU, no subprocess)")
    ap.add_argument("--models", action="store_true", help="also check the default models (check 7)")
    ap.add_argument("--production", action="store_true",
                    help="fail if development packages (torch, ultralytics) are installed")
    ap.add_argument("--config", default=None, help="settings file whose model folder to check")
    args = ap.parse_args(list(argv) if argv is not None else None)
    env = Env.current()
    rep = diagnose(env, quick=args.quick, models=args.models, production=args.production, settings=args.config)
    print(json.dumps(rep.to_json(), indent=1) if args.json else render(rep, env))
    return 0 if rep.ready else 1


if __name__ == "__main__":
    sys.exit(main())
