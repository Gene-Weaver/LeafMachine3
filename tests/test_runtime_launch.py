"""Step 3 entry task 3: one composer for every child launch.

The lease handoff (section 2.2) and the status pipe (section 2.4) both inject into the SAME two
``Popen`` keywords, and both of those keywords are EXHAUSTIVE allowlists -- ``pass_fds`` on POSIX,
``lpAttributeList["handle_list"]`` on Windows. Supplied independently the second replaces the first
silently, and the child starts without a reference it believes it holds.
"""
from __future__ import annotations

import pytest

from leafmachine3.core.runtime import _win32
from leafmachine3.core.runtime.launch import (
    ComposedLaunch, LaunchComposeError, LaunchContribution, compose_launch, composed_launch,
)


def lease_posix(fd: int = 7) -> LaunchContribution:
    return LaunchContribution(name="lease", env={"LM3_LEASE_FD": str(fd)},
                              popen_kwargs={"pass_fds": (fd,)})


def status_posix(fd: int = 9) -> LaunchContribution:
    return LaunchContribution(name="status-pipe", env={"LM3_STATUS_FD": str(fd)},
                              popen_kwargs={"pass_fds": (fd,)})


# --- the defect this exists for ---------------------------------------------------------------- #

def test_two_pass_fds_contributions_are_merged_not_overwritten() -> None:
    composed = compose_launch([lease_posix(7), status_posix(9)])
    assert composed.popen_kwargs["pass_fds"] == (7, 9), (
        "one contribution's descriptor was dropped -- the child would start without it")
    assert composed.env["LM3_LEASE_FD"] == "7"
    assert composed.env["LM3_STATUS_FD"] == "9"
    assert composed.popen_kwargs["close_fds"] is True


def test_naively_passing_both_would_have_lost_one() -> None:
    """The counterfactual, stated as a test so the reason the composer exists cannot be forgotten."""
    naive: dict = {}
    naive.update(lease_posix(7).popen_kwargs)
    naive.update(status_posix(9).popen_kwargs)
    assert naive["pass_fds"] == (9,), "dict.update is exactly the silent drop"
    assert compose_launch([lease_posix(7), status_posix(9)]).popen_kwargs["pass_fds"] == (7, 9)


def test_two_windows_handle_lists_are_merged() -> None:
    lease = LaunchContribution(
        name="lease", env={"LM3_LEASE_EVENT_HANDLE": "111"},
        popen_kwargs=_win32.handle_list_popen_kwargs([111], platform_name="linux"))
    status = LaunchContribution(
        name="status-pipe", env={"LM3_STATUS_HANDLE": "222"},
        popen_kwargs=_win32.handle_list_popen_kwargs([222], platform_name="linux"))

    composed = compose_launch([lease, status], platform_name="linux")
    handles = composed.popen_kwargs["startupinfo"].lpAttributeList["handle_list"]
    assert handles == [111, 222]
    assert composed.popen_kwargs["close_fds"] is True
    assert composed.inherited == (111, 222)


# --- refusing rather than picking a winner ------------------------------------------------------ #

def test_conflicting_environment_values_are_refused() -> None:
    a = LaunchContribution(name="a", env={"LM3_LEASE_CAPABILITY": "aaa"})
    b = LaunchContribution(name="b", env={"LM3_LEASE_CAPABILITY": "bbb"})
    with pytest.raises(LaunchComposeError, match="refusing to pick a winner"):
        compose_launch([a, b])


def test_identical_environment_values_are_fine() -> None:
    a = LaunchContribution(name="a", env={"LM3_DEPLOYMENT_ID": "gpu0"})
    b = LaunchContribution(name="b", env={"LM3_DEPLOYMENT_ID": "gpu0"})
    assert compose_launch([a, b]).env["LM3_DEPLOYMENT_ID"] == "gpu0"


def test_conflicting_ordinary_keywords_are_refused() -> None:
    a = LaunchContribution(name="a", popen_kwargs={"cwd": "/one"})
    b = LaunchContribution(name="b", popen_kwargs={"cwd": "/two"})
    with pytest.raises(LaunchComposeError, match="refusing to pick a winner"):
        compose_launch([a, b])


def test_close_fds_false_is_refused() -> None:
    """It would hand the child every open descriptor, which is invariant 3 inverted."""
    with pytest.raises(LaunchComposeError, match="close_fds=False"):
        compose_launch([lease_posix(), LaunchContribution(name="x", popen_kwargs={"close_fds": False})])


def test_mixing_descriptors_and_handles_is_refused() -> None:
    windows = LaunchContribution(
        name="win", popen_kwargs=_win32.handle_list_popen_kwargs([5], platform_name="linux"))
    with pytest.raises(LaunchComposeError, match="never both"):
        compose_launch([lease_posix(7), windows])


def test_duplicate_descriptors_appear_once() -> None:
    assert compose_launch([lease_posix(7), lease_posix(7)]).popen_kwargs["pass_fds"] == (7,)


# --- close runs on every path ------------------------------------------------------------------ #

def test_every_close_runs_even_when_one_raises() -> None:
    """Gate 34: a failed spawn must not leak a lease reference, so no closer may be skipped."""
    ran: list[str] = []

    def boom() -> None:
        ran.append("first")
        raise OSError("close failed")

    composed = compose_launch([
        LaunchContribution(name="first", close=boom),
        LaunchContribution(name="second", close=lambda: ran.append("second")),
    ])
    with pytest.raises(OSError):
        composed.close()
    assert ran == ["first", "second"], "a closer was skipped because an earlier one raised"


def test_the_context_manager_closes_on_an_exception() -> None:
    closed: list[str] = []
    with pytest.raises(RuntimeError):
        with composed_launch([LaunchContribution(name="lease", close=lambda: closed.append("lease"))]):
            raise RuntimeError("spawn failed")
    assert closed == ["lease"], "a failed spawn leaked the lease reference"


def test_a_launch_with_no_allowlist_leaves_close_fds_alone() -> None:
    """An ordinary worker spawn contributes nothing and must not acquire an allowlist by accident."""
    composed = compose_launch([LaunchContribution(name="plain", env={"A": "1"})])
    assert "pass_fds" not in composed.popen_kwargs
    assert "startupinfo" not in composed.popen_kwargs
    assert "close_fds" not in composed.popen_kwargs
    assert isinstance(composed, ComposedLaunch)


def test_base_kwargs_and_env_are_carried_through() -> None:
    composed = compose_launch([lease_posix(7)], base_env={"PATH": "/bin"},
                              base_kwargs={"cwd": "/work", "pass_fds": (3,)})
    assert composed.env["PATH"] == "/bin"
    assert composed.popen_kwargs["cwd"] == "/work"
    assert composed.popen_kwargs["pass_fds"] == (3, 7)
