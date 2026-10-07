"""Compose the launch keywords for ONE child process from several contributors.

Plan section 2.2's lease handoff and section 2.4's status pipe both inject arguments into the same
``Popen`` / ``CreateProcess`` call, and both do it through the SAME two keywords:

* POSIX -- ``pass_fds``, a tuple that is exhaustive. Whatever is not in it is closed in the child.
* Windows -- ``startupinfo.lpAttributeList["handle_list"]``, also exhaustive. Whatever is not in it
  is not inherited.

So supplying them independently is not "two settings that happen to coexist": the second value
REPLACES the first, silently, and the child starts without a reference it believes it holds. On the
lease side that is a child that validates a descriptor which is not there; on the status-pipe side
it is a handshake that can never complete and a server that waits out its timeout on a run that did
in fact acquire.

Hence one composer, and it is the only thing allowed to build these keywords. It merges what must
merge, and REFUSES anything it cannot merge rather than picking a winner -- a launcher silently
losing an argument is the failure this module exists to prevent, so it must never be this module's
own behavior either.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Mapping, Sequence

from . import _win32
from ._types import RuntimeRegistryError

__all__ = [
    "LaunchComposeError",
    "LaunchContribution",
    "ComposedLaunch",
    "compose_launch",
]

#: Keywords whose values are exhaustive inheritance allowlists, and which therefore MUST be merged
#: rather than overwritten. Everything else is an ordinary keyword and a conflict is an error.
_MERGED_KEYWORDS = frozenset({"pass_fds", "startupinfo", "close_fds"})


class LaunchComposeError(RuntimeRegistryError):
    """Two contributions cannot be combined without losing one of them."""


def _noop() -> None:
    return None


@dataclass(frozen=True)
class LaunchContribution:
    """What one participant needs added to a child launch.

    ``close`` MUST run on every path including every failure path -- on Windows a failed spawn
    otherwise leaks an inheritable duplicate that nothing will ever release (gate 34).
    """

    name: str
    env: Mapping[str, str] = field(default_factory=dict)
    popen_kwargs: Mapping[str, Any] = field(default_factory=dict)
    close: Callable[[], None] = _noop

    @classmethod
    def from_handoff(cls, handoff: Any, *, name: str = "lease") -> "LaunchContribution":
        """Adapt a :class:`~leafmachine3.core.runtime._types.ChildHandoff`."""
        return cls(name=name, env=dict(handoff.env), popen_kwargs=dict(handoff.popen_kwargs),
                   close=handoff.close)


@dataclass(frozen=True)
class ComposedLaunch:
    """The merged result. Pass ``env`` and ``**popen_kwargs`` to exactly one ``Popen``."""

    env: dict[str, str]
    popen_kwargs: dict[str, Any]
    close: Callable[[], None]
    #: Every descriptor/handle the child will inherit, for assertions and diagnostics.
    inherited: tuple[int, ...] = ()


def _handle_list(startupinfo: Any) -> list[int]:
    attributes = getattr(startupinfo, "lpAttributeList", None) or {}
    return list(attributes.get("handle_list", ()))


def compose_launch(
    contributions: Sequence[LaunchContribution],
    *,
    base_env: Mapping[str, str] | None = None,
    base_kwargs: Mapping[str, Any] | None = None,
    platform_name: str | None = None,
) -> ComposedLaunch:
    """Merge ``contributions`` into ONE set of launch keywords.

    Merged: ``pass_fds`` (union, order-stable), the Windows ``handle_list`` (union, order-stable),
    and ``close_fds`` (required true whenever an allowlist is present). Everything else must agree:
    two contributions setting the same ordinary keyword, or the same environment variable, to
    DIFFERENT values raises rather than picking one.
    """
    env: dict[str, str] = dict(base_env or {})
    kwargs: dict[str, Any] = dict(base_kwargs or {})
    origin: dict[str, str] = {}

    fds: list[int] = list(kwargs.pop("pass_fds", ()) or ())
    handles: list[int] = _handle_list(kwargs.pop("startupinfo", None))
    allowlisted = bool(fds or handles)

    for contribution in contributions:
        for key, value in contribution.env.items():
            if key in env and env[key] != value:
                raise LaunchComposeError(
                    f"{contribution.name!r} sets {key}={value!r} but {origin.get(key, 'the base')!r} "
                    f"already set it to {env[key]!r}; refusing to pick a winner"
                )
            env[key] = value
            origin[key] = contribution.name

        for key, value in contribution.popen_kwargs.items():
            if key == "pass_fds":
                fds.extend(int(fd) for fd in value)
                allowlisted = True
            elif key == "startupinfo":
                handles.extend(_handle_list(value))
                allowlisted = True
            elif key == "close_fds":
                if value is False:
                    raise LaunchComposeError(
                        f"{contribution.name!r} asks for close_fds=False, which would hand the "
                        f"child every open descriptor and break invariant 3"
                    )
            elif key in kwargs and kwargs[key] != value:
                raise LaunchComposeError(
                    f"{contribution.name!r} sets Popen {key}={value!r} but "
                    f"{origin.get(key, 'the base')!r} already set it to {kwargs[key]!r}; "
                    f"refusing to pick a winner"
                )
            else:
                kwargs[key] = value
                origin[key] = contribution.name

    ordered_fds = _dedupe(fds)
    ordered_handles = _dedupe(handles)
    if ordered_fds and ordered_handles:
        raise LaunchComposeError(
            "contributions mix POSIX descriptors with Windows handles in one launch; a child "
            "inherits one or the other, never both"
        )
    if ordered_fds:
        kwargs["pass_fds"] = tuple(ordered_fds)
    if ordered_handles:
        kwargs.update(_win32.handle_list_popen_kwargs(ordered_handles, platform_name=platform_name))
    if allowlisted:
        # subprocess REQUIRES close_fds=True alongside a non-empty handle_list, and it is also the
        # property invariant 3 leans on: an ordinary worker spawned without an allowlist inherits
        # nothing.
        kwargs["close_fds"] = True

    closers = [c.close for c in contributions]

    def close_all() -> None:
        """Run every close, even if one raises. The first failure is re-raised afterwards."""
        first: BaseException | None = None
        for closer in closers:
            try:
                closer()
            except BaseException as exc:                       # noqa: BLE001 - all must still run
                first = first or exc
        if first is not None:
            raise first

    return ComposedLaunch(env=env, popen_kwargs=kwargs, close=close_all,
                          inherited=tuple(ordered_fds or ordered_handles))


def _dedupe(values: Sequence[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


@contextlib.contextmanager
def composed_launch(
    contributions: Sequence[LaunchContribution],
    *,
    base_env: Mapping[str, str] | None = None,
    base_kwargs: Mapping[str, Any] | None = None,
    platform_name: str | None = None,
) -> Iterator[ComposedLaunch]:
    """:func:`compose_launch` with the close guaranteed, which is how it should normally be used."""
    composed = compose_launch(contributions, base_env=base_env, base_kwargs=base_kwargs,
                              platform_name=platform_name)
    try:
        yield composed
    finally:
        composed.close()
