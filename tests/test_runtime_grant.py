"""Tests for :mod:`leafmachine3.core.runtime.grant` -- plan section 2.2, gates 8, 16 and 22.

What these exercise, in the plan's terms:

* gate 8  -- "a grant is single-use by rename and expires; a replayed or forged capability fails
  closed", including a *real* concurrent race (threads and, on POSIX, processes) proving exactly
  one winner;
* gate 16 -- "an invalid child cannot consume a valid child's grant";
* gate 22 -- "a child that fails descriptor validation never renames the grant, leaving it
  claimable by the legitimate child".

The ordering assertions are the point of most of this file. Section 2.2 makes the *order* of the
four child steps normative -- no step may forward-reference a later one -- so the fake lease below
records what the filesystem looked like at each callback, which is how a test can tell "validated
before claiming" from "claimed and then validated".
"""
from __future__ import annotations

import json
import os
import stat
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from leafmachine3.core.runtime import grant as g
from leafmachine3.core.runtime._types import (
    GRANT_CONSUMED_SUFFIX,
    LEASE_ENV_VARS,
    SCHEMA_VERSION,
    Activity,
    GrantAlreadyConsumedError,
    GrantExpiredError,
    GrantInvalidError,
    GrantRecord,
    LeaseInheritanceError,
)

DEPLOYMENT = "default"
PARENT = "11111111-1111-4111-8111-111111111111"
CHILD = "22222222-2222-4222-8222-222222222222"
OTHER_CHILD = "33333333-3333-4333-8333-333333333333"
# The only subactivity left after plan revision 14 removed the batch (see section 2.3).
PURPOSE = Activity.CALIBRATION_PIPELINE


# --------------------------------------------------------------------------------------------- #
# Bridge to records.py while Step 2's modules land in parallel
# --------------------------------------------------------------------------------------------- #
# ``grant`` deliberately does not own atomic JSON IO -- ``records`` does, and a second
# implementation would give the registry two answers about what "atomically written" means. While
# both modules are being written at once, ``records`` may not exist yet; when it does, this bridge
# never activates and can be deleted. The stand-in matches the published contract exactly:
# ``atomic_write_json(path, payload, *, mode=0o600)`` and
# ``read_json_file(path) -> (payload | None, error | None)``, the latter never raising.

try:  # pragma: no cover - one branch or the other, depending on build order
    from leafmachine3.core.runtime import records as _real_records

    _RECORDS_AVAILABLE = True
except Exception:  # pragma: no cover
    _real_records = None
    _RECORDS_AVAILABLE = False


class _StandInRecords:
    """Minimal, contract-shaped substitute for ``records`` (see the note above)."""

    @staticmethod
    def atomic_write_json(path, payload, *, mode: int = 0o600) -> None:
        path = Path(path)
        tmp = path.with_name(path.name + ".tmp")
        with open(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode), "w",
                  encoding="utf-8") as handle:
            json.dump(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)

    @staticmethod
    def read_json_file(path):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle), None
        except OSError as exc:
            return None, f"{type(exc).__name__}: {exc}"
        except ValueError as exc:
            return None, f"malformed JSON: {exc}"


@pytest.fixture(autouse=True)
def _records_bridge(monkeypatch):
    if _RECORDS_AVAILABLE:
        return
    monkeypatch.setattr(g, "_records", lambda: _StandInRecords)


# --------------------------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------------------------- #

class FakeLease:
    """A stand-in for ``RuntimeLease`` that records WHEN it was called and what disk looked like.

    ``lease.py`` is another agent's file and its platform mechanics are tested there; what belongs
    here is the *ordering contract* ``grant.redeem_grant`` owes it.
    """

    def __init__(self, deployment_dir: Path, child_run_id: str, *, valid: bool = True) -> None:
        self.deployment_dir = deployment_dir
        self.child_run_id = child_run_id
        self.valid = valid
        self.calls: list[str] = []
        # snapshots of the grant files at each callback, so a test can prove step ordering
        self.at_validate: tuple[bool, bool] | None = None
        self.at_disarm: tuple[bool, bool] | None = None
        self.consumed_flag_at_disarm: bool | None = None

    def _snapshot(self) -> tuple[bool, bool]:
        return (
            g.grant_path(self.deployment_dir, self.child_run_id).exists(),
            g.consumed_grant_path(self.deployment_dir, self.child_run_id).exists(),
        )

    def validate_inherited(self) -> None:
        self.calls.append("validate_inherited")
        self.at_validate = self._snapshot()
        if not self.valid:
            raise LeaseInheritanceError("inherited descriptor is not the deployment lock")

    def disarm_and_clear(self, env=None) -> None:
        self.calls.append("disarm_and_clear")
        self.at_disarm = self._snapshot()
        claimed = g.consumed_grant_path(self.deployment_dir, self.child_run_id)
        if claimed.exists():
            self.consumed_flag_at_disarm = json.loads(claimed.read_text())["consumed"]
        if env is not None:
            for name in LEASE_ENV_VARS:
                env.pop(name, None)


# --------------------------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------------------------- #

def _issue(deployment_dir: Path, *, child_run_id: str = CHILD, parent_run_id: str = PARENT,
           deployment_id: str = DEPLOYMENT, purpose: Activity = PURPOSE,
           ttl_s: float = 60.0, now: float | None = None) -> str:
    """Issue and publish a grant the way a root does; return the raw capability."""
    clock = (lambda: now) if now is not None else None
    grant, raw = g.issue_grant(
        deployment_dir,
        child_run_id=child_run_id,
        parent_run_id=parent_run_id,
        deployment_id=deployment_id,
        purpose=purpose,
        ttl_s=ttl_s,
        clock=clock,
    )
    g.write_grant(deployment_dir, grant)
    return raw


def _child_env(raw: str) -> dict[str, str]:
    return {"LM3_LEASE_CAPABILITY": raw, "LM3_LEASE_FD": "9", "LM3_LEASE_EVENT_HANDLE": "0x40",
            "PATH": "/usr/bin"}


@pytest.fixture()
def deployment_dir(tmp_path: Path) -> Path:
    """A throwaway deployment runtime directory. Never the developer's real one."""
    target = tmp_path / "runtime" / "default"
    (target / "children").mkdir(parents=True)
    return target


# --------------------------------------------------------------------------------------------- #
# Capability values and serialization
# --------------------------------------------------------------------------------------------- #

def test_mint_capability_is_32_random_bytes_of_hex_and_never_repeats():
    values = {g.mint_capability() for _ in range(64)}
    assert len(values) == 64
    for value in values:
        assert len(value) == 2 * g.CAPABILITY_BYTES
        int(value, 16)  # pure hex


def test_mint_capability_rng_is_injectable():
    assert g.mint_capability(rng=lambda n: b"\xab" * n) == "ab" * g.CAPABILITY_BYTES


def test_mint_capability_rejects_a_short_rng():
    # A truncated RNG would silently narrow the capability space; fail closed instead.
    with pytest.raises(GrantInvalidError):
        g.mint_capability(rng=lambda n: b"\x00")


def test_capability_digest_is_sha256_of_utf8():
    import hashlib

    raw = "cafe" * 16
    assert g.capability_digest(raw) == hashlib.sha256(raw.encode("utf-8")).hexdigest()


def test_grant_dict_roundtrip_preserves_every_field():
    grant = GrantRecord(
        child_run_id=CHILD, parent_run_id=PARENT, deployment_id=DEPLOYMENT, purpose=PURPOSE,
        capability_sha256="a" * 64, issued_at=1787932800.25, expires_at=1787932860.25,
    )
    payload = g.grant_to_dict(grant)
    assert payload == {
        "schema_version": SCHEMA_VERSION,
        "child_run_id": CHILD,
        "parent_run_id": PARENT,
        "deployment_id": DEPLOYMENT,
        "purpose": "calibration_pipeline",
        "capability_sha256": "a" * 64,
        "issued_at": 1787932800.25,
        "expires_at": 1787932860.25,
        "consumed": False,
    }
    assert g.grant_from_dict(json.loads(json.dumps(payload))) == grant


@pytest.mark.parametrize(
    "mutation",
    [
        {"schema_version": SCHEMA_VERSION + 1},   # a newer grant authorizes nothing
        {"schema_version": "1"},
        {"purpose": "pipeline"},                  # a ROOT activity is not a subactivity
        {"purpose": "not_an_activity"},
        {"child_run_id": ""},
        {"child_run_id": "../../escape"},         # would write outside children/
        {"parent_run_id": None},
        {"deployment_id": 7},
        {"capability_sha256": ""},
        {"issued_at": "soon"},
        {"expires_at": True},
        {"expires_at": 1787932800.25},            # not after issued_at
        {"consumed": "yes"},
    ],
)
def test_grant_from_dict_fails_closed_on_every_fault(mutation):
    payload = {
        "schema_version": SCHEMA_VERSION, "child_run_id": CHILD, "parent_run_id": PARENT,
        "deployment_id": DEPLOYMENT, "purpose": "calibration_pipeline",
        "capability_sha256": "a" * 64, "issued_at": 1787932800.25, "expires_at": 1787932860.25,
        "consumed": False,
    }
    payload.update(mutation)
    with pytest.raises(GrantInvalidError):
        g.grant_from_dict(payload)


@pytest.mark.parametrize("bad", ["", "..", "../evil", "a/b", "a\\b", ".hidden", "x" * 129])
def test_grant_paths_refuse_ids_that_could_escape_children(deployment_dir, bad):
    with pytest.raises(GrantInvalidError):
        g.grant_path(deployment_dir, bad)
    with pytest.raises(GrantInvalidError):
        g.consumed_grant_path(deployment_dir, bad)


# --------------------------------------------------------------------------------------------- #
# The root side
# --------------------------------------------------------------------------------------------- #

def test_issue_grant_returns_the_raw_value_and_writes_only_its_digest(deployment_dir):
    grant, raw = g.issue_grant(
        deployment_dir, child_run_id=CHILD, parent_run_id=PARENT, deployment_id=DEPLOYMENT,
        purpose=PURPOSE, ttl_s=60.0, clock=lambda: 1787932800.25,
    )
    assert grant.issued_at == 1787932800.25 and grant.expires_at == 1787932860.25
    assert grant.capability_sha256 == g.capability_digest(raw)
    assert grant.consumed is False

    path = g.write_grant(deployment_dir, grant)
    assert path == deployment_dir / "children" / f"{CHILD}.grant.json"
    text = path.read_text()
    assert raw not in text, "the raw capability must never touch disk"
    assert grant.capability_sha256 in text
    # 0600: the grant is in the user's runtime directory, and section 3.2's rule about secrets
    # applies to everything the registry writes.
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.parametrize("purpose", [Activity.PIPELINE, Activity.HARDWARE_SETUP])
def test_issue_grant_refuses_a_root_activity(deployment_dir, purpose):
    # Only CHILD_ACTIVITIES may run under a parent's inherited lease (section 2.2).
    with pytest.raises(GrantInvalidError):
        g.issue_grant(deployment_dir, child_run_id=CHILD, parent_run_id=PARENT,
                      deployment_id=DEPLOYMENT, purpose=purpose)


def test_issue_grant_refuses_a_nonpositive_ttl(deployment_dir):
    with pytest.raises(GrantInvalidError):
        g.issue_grant(deployment_dir, child_run_id=CHILD, parent_run_id=PARENT,
                      deployment_id=DEPLOYMENT, purpose=PURPOSE, ttl_s=0)


def test_write_grant_refuses_to_rearm_a_consumed_run_id(deployment_dir):
    raw = _issue(deployment_dir)
    g.claim_grant(deployment_dir, child_run_id=CHILD)
    grant, _ = g.issue_grant(deployment_dir, child_run_id=CHILD, parent_run_id=PARENT,
                             deployment_id=DEPLOYMENT, purpose=PURPOSE)
    with pytest.raises(GrantInvalidError):
        g.write_grant(deployment_dir, grant)
    assert raw  # the first capability is spent; nothing re-armed it


def test_write_grant_creates_the_children_directory(tmp_path):
    bare = tmp_path / "runtime" / "default"
    bare.mkdir(parents=True)
    grant, _ = g.issue_grant(bare, child_run_id=CHILD, parent_run_id=PARENT,
                             deployment_id=DEPLOYMENT, purpose=PURPOSE)
    assert g.write_grant(bare, grant).exists()


# --------------------------------------------------------------------------------------------- #
# validate_grant -- forgery and expiry, with no filesystem effect at all
# --------------------------------------------------------------------------------------------- #

def _valid_grant(raw: str) -> GrantRecord:
    return GrantRecord(
        child_run_id=CHILD, parent_run_id=PARENT, deployment_id=DEPLOYMENT, purpose=PURPOSE,
        capability_sha256=g.capability_digest(raw), issued_at=1000.0, expires_at=1060.0,
    )


def test_validate_grant_accepts_the_matching_child():
    raw = g.mint_capability()
    g.validate_grant(_valid_grant(raw), capability=raw, child_run_id=CHILD, parent_run_id=PARENT,
                     deployment_id=DEPLOYMENT, purpose=PURPOSE, now=1000.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"capability": "0" * 64}, id="wrong-sha"),
        pytest.param({"deployment_id": "other-deadbeef"}, id="wrong-deployment"),
        pytest.param({"parent_run_id": OTHER_CHILD}, id="wrong-parent"),
        # With the batch gone there is no OTHER subactivity to be wrong, so a wrong purpose is
        # necessarily a ROOT activity claiming to be a child. Both roots are covered.
        pytest.param({"child_run_id": OTHER_CHILD}, id="wrong-child"),
        pytest.param({"purpose": Activity.PIPELINE}, id="root-purpose-pipeline"),
        pytest.param({"purpose": Activity.HARDWARE_SETUP}, id="root-purpose-hardware-setup"),
    ],
)
def test_validate_grant_rejects_forgeries(kwargs):
    raw = g.mint_capability()
    call = {"capability": raw, "child_run_id": CHILD, "parent_run_id": PARENT,
            "deployment_id": DEPLOYMENT, "purpose": PURPOSE, "now": 1000.0}
    call.update(kwargs)
    with pytest.raises(GrantInvalidError):
        g.validate_grant(_valid_grant(raw), **call)


def test_validate_grant_expires_at_the_boundary():
    raw = g.mint_capability()
    grant = _valid_grant(raw)
    common = {"capability": raw, "child_run_id": CHILD, "parent_run_id": PARENT,
              "deployment_id": DEPLOYMENT, "purpose": PURPOSE}
    g.validate_grant(grant, now=1059.999, **common)               # still live
    with pytest.raises(GrantExpiredError):
        g.validate_grant(grant, now=1060.0, **common)             # expires_at is exclusive
    with pytest.raises(GrantExpiredError):
        g.validate_grant(grant, now=99999.0, **common)


# --------------------------------------------------------------------------------------------- #
# Gate 8 -- single use by rename, replay fails closed
# --------------------------------------------------------------------------------------------- #

def test_claim_renames_and_a_replay_finds_nothing(deployment_dir):
    _issue(deployment_dir)
    source = g.grant_path(deployment_dir, CHILD)
    claimed = g.claim_grant(deployment_dir, child_run_id=CHILD)

    assert claimed == g.consumed_grant_path(deployment_dir, CHILD)
    assert claimed.name.endswith(GRANT_CONSUMED_SUFFIX)
    assert claimed.exists() and not source.exists()

    with pytest.raises(GrantAlreadyConsumedError):
        g.claim_grant(deployment_dir, child_run_id=CHILD)


def test_claim_of_a_grant_that_was_never_issued_fails_closed(deployment_dir):
    with pytest.raises(GrantAlreadyConsumedError):
        g.claim_grant(deployment_dir, child_run_id=CHILD)


def test_concurrent_threads_produce_exactly_one_winner(deployment_dir):
    # A real race, repeated, rather than a sequential simulation of one: every thread is released
    # from the same barrier, so the renames genuinely overlap.
    workers = 16
    for round_index in range(25):
        child = f"{CHILD[:-2]}{round_index:02d}"
        _issue(deployment_dir, child_run_id=child)
        barrier = threading.Barrier(workers)
        outcomes: list[str] = []
        lock = threading.Lock()

        def attempt() -> None:
            barrier.wait()
            try:
                g.claim_grant(deployment_dir, child_run_id=child)
                result = "won"
            except GrantAlreadyConsumedError:
                result = "lost"
            with lock:
                outcomes.append(result)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(lambda _: attempt(), range(workers)))

        assert outcomes.count("won") == 1, f"round {round_index}: {outcomes}"
        assert outcomes.count("lost") == workers - 1


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork-based race is POSIX-only")
def test_concurrent_processes_produce_exactly_one_winner(deployment_dir):
    # Threads share one interpreter; separate processes are the shape the plan actually describes,
    # so the rename is proven to be the single-winner mechanism at the OS level too.
    _issue(deployment_dir)
    contenders = 8
    gate_r, gate_w = os.pipe()
    pids = []
    for _ in range(contenders):
        pid = os.fork()
        if pid == 0:  # pragma: no cover - child process, never measured by coverage
            try:
                os.close(gate_w)
                os.read(gate_r, 1)  # released simultaneously with every sibling
                g.claim_grant(deployment_dir, child_run_id=CHILD)
                os._exit(0)
            except BaseException:
                os._exit(1)
        pids.append(pid)

    os.close(gate_r)
    os.write(gate_w, b"\0" * contenders)
    os.close(gate_w)

    winners = 0
    for pid in pids:
        _, status = os.waitpid(pid, 0)
        if os.waitstatus_to_exitcode(status) == 0:
            winners += 1
    assert winners == 1
    assert g.consumed_grant_path(deployment_dir, CHILD).exists()
    assert not g.grant_path(deployment_dir, CHILD).exists()


# --------------------------------------------------------------------------------------------- #
# Step 3 -- revalidation after the claim, and the encoded ordering
# --------------------------------------------------------------------------------------------- #

def test_revalidate_marks_the_claimed_file_consumed(deployment_dir):
    raw = _issue(deployment_dir)
    claimed = g.claim_grant(deployment_dir, child_run_id=CHILD)
    redeemed = g.revalidate_claimed_grant(
        claimed, capability=raw, child_run_id=CHILD, parent_run_id=PARENT,
        deployment_id=DEPLOYMENT, purpose=PURPOSE,
    )
    assert redeemed.consumed is True
    assert json.loads(claimed.read_text())["consumed"] is True


def test_revalidate_refuses_a_path_that_was_never_claimed(deployment_dir):
    # The ordering is structural, not just documented: step 3 only accepts step 2's output, so it
    # cannot be run before the rename.
    raw = _issue(deployment_dir)
    with pytest.raises(GrantInvalidError):
        g.revalidate_claimed_grant(
            g.grant_path(deployment_dir, CHILD), capability=raw, child_run_id=CHILD,
            parent_run_id=PARENT, deployment_id=DEPLOYMENT, purpose=PURPOSE,
        )
    assert g.grant_path(deployment_dir, CHILD).exists()


def test_revalidate_rejects_a_grant_that_expired_after_the_claim(deployment_dir):
    # "Re-read the claimed file and revalidate expiry and identity, since the claim itself can be
    # delayed" (section 2.2 step 3).
    raw = _issue(deployment_dir, now=1000.0, ttl_s=60.0)
    claimed = g.claim_grant(deployment_dir, child_run_id=CHILD)
    with pytest.raises(GrantExpiredError):
        g.revalidate_claimed_grant(
            claimed, capability=raw, child_run_id=CHILD, parent_run_id=PARENT,
            deployment_id=DEPLOYMENT, purpose=PURPOSE, clock=lambda: 5000.0,
        )
    # The claim stands (it is spent either way) but it was never marked consumed, so no reader can
    # mistake the aborted child for a redeemed one.
    assert json.loads(claimed.read_text())["consumed"] is False


def test_revalidate_refuses_an_already_marked_file(deployment_dir):
    raw = _issue(deployment_dir)
    claimed = g.claim_grant(deployment_dir, child_run_id=CHILD)
    g.revalidate_claimed_grant(claimed, capability=raw, child_run_id=CHILD, parent_run_id=PARENT,
                               deployment_id=DEPLOYMENT, purpose=PURPOSE)
    with pytest.raises(GrantAlreadyConsumedError):
        g.revalidate_claimed_grant(claimed, capability=raw, child_run_id=CHILD,
                                   parent_run_id=PARENT, deployment_id=DEPLOYMENT, purpose=PURPOSE)


# --------------------------------------------------------------------------------------------- #
# redeem_grant -- the whole ordered procedure
# --------------------------------------------------------------------------------------------- #

def test_redeem_runs_the_four_steps_in_order_and_clears_the_environment(deployment_dir):
    raw = _issue(deployment_dir)
    env = _child_env(raw)
    lease = FakeLease(deployment_dir, CHILD)

    redeemed = g.redeem_grant(
        deployment_dir, child_run_id=CHILD, parent_run_id=PARENT, deployment_id=DEPLOYMENT,
        purpose=PURPOSE, capability=raw, lease=lease, env=env,
    )

    assert redeemed.consumed is True
    assert lease.calls == ["validate_inherited", "disarm_and_clear"]
    # step 1 ran while the grant was still unclaimed ...
    assert lease.at_validate == (True, False)
    # ... and step 4 ran only after step 3 had marked the claimed file.
    assert lease.at_disarm == (False, True)
    assert lease.consumed_flag_at_disarm is True

    # The raw capability and both lease references are gone before any expensive work begins, so a
    # later fork cannot carry them (invariant 3, section 2.2 step 4).
    for name in LEASE_ENV_VARS:
        assert name not in env
    assert env == {"PATH": "/usr/bin"}


def test_redeem_is_single_use_and_a_replay_fails_closed(deployment_dir):
    raw = _issue(deployment_dir)
    first_env, second_env = _child_env(raw), _child_env(raw)
    g.redeem_grant(deployment_dir, child_run_id=CHILD, parent_run_id=PARENT,
                   deployment_id=DEPLOYMENT, purpose=PURPOSE, capability=raw,
                   lease=FakeLease(deployment_dir, CHILD), env=first_env)

    replay = FakeLease(deployment_dir, CHILD)
    with pytest.raises(GrantAlreadyConsumedError):
        g.redeem_grant(deployment_dir, child_run_id=CHILD, parent_run_id=PARENT,
                       deployment_id=DEPLOYMENT, purpose=PURPOSE, capability=raw,
                       lease=replay, env=second_env)
    # A replay never reaches step 4, so it never disarms anything on the winner's behalf.
    assert replay.calls == ["validate_inherited"]
    assert second_env["LM3_LEASE_CAPABILITY"] == raw


def test_redeem_rejects_an_expired_grant_without_renaming_it(deployment_dir):
    raw = _issue(deployment_dir, now=1000.0, ttl_s=60.0)
    env = _child_env(raw)
    lease = FakeLease(deployment_dir, CHILD)
    with pytest.raises(GrantExpiredError):
        g.redeem_grant(deployment_dir, child_run_id=CHILD, parent_run_id=PARENT,
                       deployment_id=DEPLOYMENT, purpose=PURPOSE, capability=raw,
                       lease=lease, clock=lambda: 5000.0, env=env)
    assert g.grant_path(deployment_dir, CHILD).exists()
    assert not g.consumed_grant_path(deployment_dir, CHILD).exists()
    assert "disarm_and_clear" not in lease.calls
    assert env["LM3_LEASE_FD"] == "9"


@pytest.mark.parametrize(
    "override",
    [
        pytest.param({"capability": "f" * 64}, id="forged-capability"),
        pytest.param({"deployment_id": "other-deadbeef"}, id="wrong-deployment"),
        pytest.param({"parent_run_id": OTHER_CHILD}, id="wrong-parent"),
        pytest.param({"purpose": Activity.HARDWARE_SETUP}, id="root-purpose"),
    ],
)
def test_redeem_forgeries_fail_closed_and_leave_the_grant_claimable(deployment_dir, override):
    raw = _issue(deployment_dir)
    call = {"child_run_id": CHILD, "parent_run_id": PARENT, "deployment_id": DEPLOYMENT,
            "purpose": PURPOSE, "capability": raw}
    call.update(override)
    with pytest.raises(GrantInvalidError):
        g.redeem_grant(deployment_dir, lease=FakeLease(deployment_dir, CHILD),
                       env=_child_env(raw), **call)
    assert g.grant_path(deployment_dir, CHILD).exists()
    assert not g.consumed_grant_path(deployment_dir, CHILD).exists()

    # ... and the legitimate child still gets the run it was entitled to.
    redeemed = g.redeem_grant(deployment_dir, child_run_id=CHILD, parent_run_id=PARENT,
                              deployment_id=DEPLOYMENT, purpose=PURPOSE, capability=raw,
                              lease=FakeLease(deployment_dir, CHILD), env=_child_env(raw))
    assert redeemed.consumed is True


def test_redeem_rejects_a_malformed_grant_file(deployment_dir):
    _issue(deployment_dir)
    g.grant_path(deployment_dir, CHILD).write_text("{not json")
    with pytest.raises(GrantInvalidError):
        g.redeem_grant(deployment_dir, child_run_id=CHILD, parent_run_id=PARENT,
                       deployment_id=DEPLOYMENT, purpose=PURPOSE, capability="x" * 64,
                       lease=FakeLease(deployment_dir, CHILD), env={})
    assert not g.consumed_grant_path(deployment_dir, CHILD).exists()


# --------------------------------------------------------------------------------------------- #
# Gate 16 -- an invalid child cannot consume a valid child's grant
# --------------------------------------------------------------------------------------------- #

def test_an_invalid_child_cannot_consume_a_valid_childs_grant(deployment_dir):
    valid_raw = _issue(deployment_dir, child_run_id=CHILD)
    intruder_raw = _issue(deployment_dir, child_run_id=OTHER_CHILD)

    # The intruder holds a perfectly good capability -- for a DIFFERENT child. Pointing it at the
    # valid child's grant must fail before the rename, or the valid child is locked out of a run it
    # was entitled to.
    intruder = FakeLease(deployment_dir, CHILD)
    with pytest.raises(GrantInvalidError):
        g.redeem_grant(deployment_dir, child_run_id=CHILD, parent_run_id=PARENT,
                       deployment_id=DEPLOYMENT, purpose=PURPOSE, capability=intruder_raw,
                       lease=intruder, env=_child_env(intruder_raw))

    assert g.grant_path(deployment_dir, CHILD).exists()
    assert not g.consumed_grant_path(deployment_dir, CHILD).exists()
    assert "disarm_and_clear" not in intruder.calls

    redeemed = g.redeem_grant(deployment_dir, child_run_id=CHILD, parent_run_id=PARENT,
                              deployment_id=DEPLOYMENT, purpose=PURPOSE, capability=valid_raw,
                              lease=FakeLease(deployment_dir, CHILD), env=_child_env(valid_raw))
    assert redeemed.child_run_id == CHILD
    # The intruder's own grant is untouched by any of this.
    assert g.grant_path(deployment_dir, OTHER_CHILD).exists()


# --------------------------------------------------------------------------------------------- #
# Gate 22 -- failed lease-reference validation never renames the grant
# --------------------------------------------------------------------------------------------- #

def test_a_child_that_fails_lease_validation_never_renames_the_grant(deployment_dir):
    raw = _issue(deployment_dir)
    bogus = FakeLease(deployment_dir, CHILD, valid=False)
    env = _child_env(raw)

    with pytest.raises(LeaseInheritanceError):
        g.redeem_grant(deployment_dir, child_run_id=CHILD, parent_run_id=PARENT,
                       deployment_id=DEPLOYMENT, purpose=PURPOSE, capability=raw,
                       lease=bogus, env=env)

    # Gate 22, literally: the grant is exactly where the root left it, still claimable.
    source = g.grant_path(deployment_dir, CHILD)
    assert source.exists()
    assert not g.consumed_grant_path(deployment_dir, CHILD).exists()
    assert json.loads(source.read_text())["consumed"] is False
    assert bogus.calls == ["validate_inherited"]
    assert env["LM3_LEASE_CAPABILITY"] == raw  # nothing was disarmed on the rejected path

    legitimate = FakeLease(deployment_dir, CHILD)
    redeemed = g.redeem_grant(deployment_dir, child_run_id=CHILD, parent_run_id=PARENT,
                              deployment_id=DEPLOYMENT, purpose=PURPOSE, capability=raw,
                              lease=legitimate, env=_child_env(raw))
    assert redeemed.consumed is True
    assert legitimate.calls == ["validate_inherited", "disarm_and_clear"]


def test_lease_validation_precedes_even_reading_the_grant(deployment_dir):
    # No grant on disk at all: the lease-reference failure still wins, which proves the descriptor
    # check is not conditional on the grant being present.
    bogus = FakeLease(deployment_dir, CHILD, valid=False)
    with pytest.raises(LeaseInheritanceError):
        g.redeem_grant(deployment_dir, child_run_id=CHILD, parent_run_id=PARENT,
                       deployment_id=DEPLOYMENT, purpose=PURPOSE, capability="x" * 64,
                       lease=bogus, env={})


# --------------------------------------------------------------------------------------------- #
# Retention
# --------------------------------------------------------------------------------------------- #

def test_prune_removes_expired_grants_and_keeps_live_ones(deployment_dir):
    _issue(deployment_dir, child_run_id=CHILD, now=1000.0, ttl_s=60.0)          # expired at 5000
    _issue(deployment_dir, child_run_id=OTHER_CHILD, now=4990.0, ttl_s=60.0)    # still live
    assert g.prune_grants(deployment_dir, clock=lambda: 5000.0) == 1
    assert not g.grant_path(deployment_dir, CHILD).exists()
    assert g.grant_path(deployment_dir, OTHER_CHILD).exists()


def test_prune_removes_an_unusable_grant_file(deployment_dir):
    _issue(deployment_dir)
    g.grant_path(deployment_dir, CHILD).write_text("{truncated")
    assert g.prune_grants(deployment_dir, clock=lambda: 1000.0) == 1


def test_prune_keeps_the_newest_consumed_markers_up_to_retention(deployment_dir):
    for index in range(6):
        child = f"{OTHER_CHILD[:-2]}{index:02d}"
        _issue(deployment_dir, child_run_id=child, now=9000.0, ttl_s=600.0)
        g.claim_grant(deployment_dir, child_run_id=child)
        os.utime(g.consumed_grant_path(deployment_dir, child), (1000 + index, 1000 + index))

    assert g.prune_grants(deployment_dir, retention=2, clock=lambda: 9100.0) == 4
    survivors = sorted(p.name for p in (deployment_dir / "children").glob(f"*{GRANT_CONSUMED_SUFFIX}"))
    assert survivors == sorted(
        f"{OTHER_CHILD[:-2]}{index:02d}{GRANT_CONSUMED_SUFFIX}" for index in (4, 5)
    )


def test_prune_on_a_deployment_with_no_children_directory_is_zero(tmp_path):
    bare = tmp_path / "empty"
    bare.mkdir()
    assert g.prune_grants(bare) == 0
