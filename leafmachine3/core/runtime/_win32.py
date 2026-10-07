"""The real Win32 binding behind :class:`~leafmachine3.core.runtime._types.Win32Surface`.

Why this file exists at all: the Windows lease is a named ``Global\\`` kernel event (plan section
2.2), and every call it needs -- ``CreateEventW``, ``OpenEventW``, ``DuplicateHandle``,
``CompareObjectHandles``, ``SetHandleInformation``, ``CloseHandle``, plus the current user's SID and
a current-user-only ``SECURITY_ATTRIBUTES`` -- is a ``ctypes`` call that cannot exist on Linux. LM3
develops and tests on Linux, so the adapter never calls ``ctypes`` directly: it calls the injected
:class:`Win32Surface`, of which this module builds the real one.

**Nothing here touches ``ctypes.WinDLL`` or ``ctypes.wintypes`` at import time.** ``import ctypes``
is portable; ``ctypes.WinDLL`` and ``ctypes.wintypes`` are not, so both are reached only from inside
a function that has already checked :func:`is_windows`. That is what keeps ``lease.py`` -- and every
test that imports it -- clean on Linux while the Windows *logic* stays unit-tested against a fake.

The one piece of Windows plumbing that is NOT a Win32 call lives here too:
:func:`handle_list_popen_kwargs`, the ``STARTUPINFOEX`` handle allowlist (plan section 2.2, step 3).
On Windows it builds a real :class:`subprocess.STARTUPINFO`; off Windows it builds
:class:`HandleAllowlist`, which exposes the same ``lpAttributeList`` mapping so a Linux test asserts
on exactly the field the real launch would use.
"""
from __future__ import annotations

import ctypes
import subprocess
import sys
import threading
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ._types import EVENT_ALL_ACCESS, LeaseError, Win32Surface

__all__ = [
    "DUPLICATE_SAME_ACCESS",
    "HandleAllowlist",
    "Win32ApiSurface",
    "build_real_surface",
    "handle_list_popen_kwargs",
    "is_windows",
]

# -- Win32 numeric constants this module needs beyond the ones _types already publishes -------- #
DUPLICATE_SAME_ACCESS = 0x00000002
SDDL_REVISION_1 = 1
TOKEN_QUERY = 0x0008
TOKEN_USER_CLASS = 1                 # TOKEN_INFORMATION_CLASS.TokenUser
ERROR_INSUFFICIENT_BUFFER = 122


def is_windows(platform_name: str | None = None) -> bool:
    """Platform test with the same injection discipline as the rest of the runtime package."""
    return (platform_name or sys.platform).startswith("win")


# --------------------------------------------------------------------------------------------- #
# STARTUPINFOEX handle allowlist
# --------------------------------------------------------------------------------------------- #

@dataclass(frozen=True)
class HandleAllowlist:
    """Off-Windows stand-in for :class:`subprocess.STARTUPINFO` with a ``handle_list``.

    It deliberately mirrors the single attribute the adapter sets, so the assertion a Linux test
    writes (``startupinfo.lpAttributeList["handle_list"] == [dup]``) is the assertion that would
    hold against the real object on Windows. It is inert: nothing can spawn a process with it, and
    ``handle_list_popen_kwargs`` returns the real ``STARTUPINFO`` whenever it runs on Windows.
    """

    lpAttributeList: Mapping[str, Any]   # noqa: N815 - mirrors the Win32/subprocess spelling exactly


def handle_list_popen_kwargs(handles: Sequence[int], *, platform_name: str | None = None) -> dict[str, Any]:
    """The ``Popen`` keywords that pass EXACTLY ``handles`` to the child and nothing else.

    ``close_fds=True`` is not optional: ``subprocess`` requires it whenever ``handle_list`` is
    non-empty, and it is also the property invariant 3 depends on -- an ordinary executor worker
    spawned without this allowlist inherits nothing.
    """
    handle_list = list(handles)
    if is_windows(platform_name):
        startupinfo: Any = subprocess.STARTUPINFO(lpAttributeList={"handle_list": handle_list})
    else:
        startupinfo = HandleAllowlist(lpAttributeList={"handle_list": handle_list})
    return {"startupinfo": startupinfo, "close_fds": True}


# --------------------------------------------------------------------------------------------- #
# The real ctypes surface
# --------------------------------------------------------------------------------------------- #

class Win32ApiSurface:
    """The real :class:`Win32Surface`. Constructed only on Windows, only by :func:`build_real_surface`.

    ``GetLastError`` is captured into thread-local state by every wrapper immediately after its call
    rather than read later from :func:`ctypes.get_last_error`: the adapter asks for the error of the
    *immediately preceding* call, and an intervening call from another thread of the same process
    would otherwise be able to answer for it.
    """

    def __init__(self) -> None:
        if not is_windows():
            # Defense in depth: build_real_surface already refuses, but a direct construction must
            # not reach ctypes.WinDLL on a platform that has no such attribute.
            raise LeaseError("Win32ApiSurface is Windows-only; the lease adapter is chosen by platform")
        self._local = threading.local()
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)      # type: ignore[attr-defined]
        self._advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)      # type: ignore[attr-defined]
        # CompareObjectHandles is exported by KernelBase.dll (Windows 10 / Server 2016, LM3's
        # normative Windows floor); kernel32 does not forward it on every build.
        try:
            self._kernelbase = ctypes.WinDLL("kernelbase", use_last_error=True)   # type: ignore[attr-defined]
        except OSError:                                              # pragma: no cover - Windows only
            self._kernelbase = self._kernel32
        self._sid: str | None = None
        self._security_attributes: Any = None
        self._configure_signatures()

    def _configure_signatures(self) -> None:
        """Declare pointer-sized arguments and results before any native call.

        ctypes otherwise treats Python integers and return values as C ints. In
        particular a 64-bit SID pointer cannot be passed to ConvertSidToStringSidW
        without truncation/overflow, preventing every real Windows lease acquisition.
        """
        from ctypes import wintypes                                  # noqa: PLC0415 - Windows-only import

        pointer = ctypes.c_void_p
        handle_pointer = ctypes.POINTER(wintypes.HANDLE)
        dword_pointer = ctypes.POINTER(wintypes.DWORD)
        signatures = (
            (self._kernel32, "GetCurrentProcess", [], wintypes.HANDLE),
            (self._kernel32, "CreateEventW", [pointer, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR], wintypes.HANDLE),
            (self._kernel32, "OpenEventW", [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR], wintypes.HANDLE),
            (self._kernel32, "CloseHandle", [wintypes.HANDLE], wintypes.BOOL),
            (self._kernel32, "DuplicateHandle", [wintypes.HANDLE, wintypes.HANDLE, wintypes.HANDLE,
                                               handle_pointer, wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.BOOL),
            (self._kernel32, "SetHandleInformation", [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD], wintypes.BOOL),
            (self._kernel32, "LocalFree", [pointer], pointer),
            (self._advapi32, "OpenProcessToken", [wintypes.HANDLE, wintypes.DWORD, handle_pointer], wintypes.BOOL),
            (self._advapi32, "GetTokenInformation", [wintypes.HANDLE, ctypes.c_int, pointer,
                                                    wintypes.DWORD, dword_pointer], wintypes.BOOL),
            (self._advapi32, "ConvertSidToStringSidW", [pointer, ctypes.POINTER(wintypes.LPWSTR)], wintypes.BOOL),
            (self._advapi32, "ConvertStringSecurityDescriptorToSecurityDescriptorW",
             [wintypes.LPCWSTR, wintypes.DWORD, ctypes.POINTER(pointer), dword_pointer], wintypes.BOOL),
        )
        for library, name, arguments, result in signatures:
            function = getattr(library, name)
            function.argtypes = arguments
            function.restype = result
        compare = getattr(self._kernelbase, "CompareObjectHandles", None)
        if compare is not None:
            compare.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
            compare.restype = wintypes.BOOL

    # -- error plumbing ------------------------------------------------------------------------ #

    def _stash(self, code: int) -> None:
        self._local.last_error = int(code)

    def get_last_error(self) -> int:
        return int(getattr(self._local, "last_error", 0))

    # -- events -------------------------------------------------------------------------------- #

    def create_event(
        self,
        name: str,
        *,
        security_attributes: Any,
        manual_reset: bool = False,
        initial_state: bool = False,
        inheritable: bool = False,
    ) -> int:
        from ctypes import wintypes                                  # noqa: PLC0415 - Windows-only import

        fn = self._kernel32.CreateEventW
        fn.restype = wintypes.HANDLE
        sa = security_attributes
        if sa is not None and inheritable:
            # The owner handle is never inheritable (plan section 2.2, "the whole ballgame"), so this
            # branch exists only for completeness; bInheritHandle lives in SECURITY_ATTRIBUTES.
            sa.bInheritHandle = True
        handle = fn(
            ctypes.byref(sa) if sa is not None else None,
            wintypes.BOOL(manual_reset),
            wintypes.BOOL(initial_state),
            ctypes.c_wchar_p(name),
        )
        self._stash(ctypes.get_last_error())
        if not handle:
            raise OSError(self.get_last_error(), f"CreateEventW failed for {name!r}")
        return int(handle)

    def open_event(self, name: str, *, desired_access: int = EVENT_ALL_ACCESS,
                   inheritable: bool = False) -> int:
        from ctypes import wintypes                                  # noqa: PLC0415 - Windows-only import

        fn = self._kernel32.OpenEventW
        fn.restype = wintypes.HANDLE
        handle = fn(wintypes.DWORD(desired_access), wintypes.BOOL(inheritable), ctypes.c_wchar_p(name))
        self._stash(ctypes.get_last_error())
        if not handle:
            raise OSError(self.get_last_error(), f"OpenEventW failed for {name!r}")
        return int(handle)

    # -- handles ------------------------------------------------------------------------------- #

    def close_handle(self, handle: int) -> None:
        from ctypes import wintypes                                  # noqa: PLC0415 - Windows-only import

        ok = self._kernel32.CloseHandle(wintypes.HANDLE(handle))
        self._stash(ctypes.get_last_error())
        if not ok:
            raise OSError(self.get_last_error(), f"CloseHandle failed for handle {handle}")

    def duplicate_handle(self, handle: int, *, inheritable: bool) -> int:
        from ctypes import wintypes                                  # noqa: PLC0415 - Windows-only import

        target = wintypes.HANDLE()
        current = self._kernel32.GetCurrentProcess()
        ok = self._kernel32.DuplicateHandle(
            wintypes.HANDLE(current), wintypes.HANDLE(handle), wintypes.HANDLE(current),
            ctypes.byref(target), wintypes.DWORD(0), wintypes.BOOL(inheritable),
            wintypes.DWORD(DUPLICATE_SAME_ACCESS),
        )
        self._stash(ctypes.get_last_error())
        if not ok:
            raise OSError(self.get_last_error(), f"DuplicateHandle failed for handle {handle}")
        return int(target.value or 0)

    def compare_object_handles(self, first: int, second: int) -> bool:
        from ctypes import wintypes                                  # noqa: PLC0415 - Windows-only import

        fn = getattr(self._kernelbase, "CompareObjectHandles", None)
        if fn is None:                                               # pragma: no cover - Windows only
            raise LeaseError(
                "CompareObjectHandles is unavailable. Windows 10 / Server 2016 is LM3's normative "
                "floor for the runtime lease; there is no weaker validation to fall back to."
            )
        same = fn(wintypes.HANDLE(first), wintypes.HANDLE(second))
        self._stash(ctypes.get_last_error())
        return bool(same)

    def set_handle_information(self, handle: int, mask: int, flags: int) -> None:
        from ctypes import wintypes                                  # noqa: PLC0415 - Windows-only import

        ok = self._kernel32.SetHandleInformation(
            wintypes.HANDLE(handle), wintypes.DWORD(mask), wintypes.DWORD(flags)
        )
        self._stash(ctypes.get_last_error())
        if not ok:
            raise OSError(self.get_last_error(), f"SetHandleInformation failed for handle {handle}")

    # -- identity ------------------------------------------------------------------------------ #

    def current_user_sid(self) -> str:
        """The process token's user SID in canonical ``ConvertSidToStringSidW`` form."""
        if self._sid is not None:
            return self._sid
        from ctypes import wintypes                                  # noqa: PLC0415 - Windows-only import

        class _SidAndAttributes(ctypes.Structure):
            _fields_ = [("Sid", ctypes.c_void_p), ("Attributes", wintypes.DWORD)]

        class _TokenUser(ctypes.Structure):
            _fields_ = [("User", _SidAndAttributes)]

        token = wintypes.HANDLE()
        if not self._advapi32.OpenProcessToken(
            wintypes.HANDLE(self._kernel32.GetCurrentProcess()), wintypes.DWORD(TOKEN_QUERY),
            ctypes.byref(token),
        ):
            raise OSError(ctypes.get_last_error(), "OpenProcessToken failed")
        try:
            size = wintypes.DWORD(0)
            self._advapi32.GetTokenInformation(token, TOKEN_USER_CLASS, None, 0, ctypes.byref(size))
            buffer = ctypes.create_string_buffer(size.value)
            if not self._advapi32.GetTokenInformation(
                token, TOKEN_USER_CLASS, buffer, size, ctypes.byref(size)
            ):
                raise OSError(ctypes.get_last_error(), "GetTokenInformation(TokenUser) failed")
            user = ctypes.cast(buffer, ctypes.POINTER(_TokenUser)).contents
            text = ctypes.c_wchar_p()
            if not self._advapi32.ConvertSidToStringSidW(user.User.Sid, ctypes.byref(text)):
                raise OSError(ctypes.get_last_error(), "ConvertSidToStringSidW failed")
            try:
                self._sid = str(text.value)
            finally:
                self._kernel32.LocalFree(text)
        finally:
            self._kernel32.CloseHandle(token)
        return self._sid

    def current_user_security_attributes(self) -> Any:
        """``SECURITY_ATTRIBUTES`` whose DACL grants the current user ``EVENT_ALL_ACCESS`` and nobody else.

        Built from SDDL rather than by hand-assembling an ACL: ``D:P(A;;0x1F0003;;;<sid>)`` is one
        protected, non-inherited allow ACE. ``EVENT_ALL_ACCESS`` and not less, because ``CreateEventW``
        opening an EXISTING named event requests exactly that -- a stingier DACL would lock our own
        next process out of its own lease (plan section 2.2, "ACL detail that will otherwise bite").
        """
        if self._security_attributes is not None:
            return self._security_attributes
        from ctypes import wintypes                                  # noqa: PLC0415 - Windows-only import

        class _SecurityAttributes(ctypes.Structure):
            _fields_ = [
                ("nLength", wintypes.DWORD),
                ("lpSecurityDescriptor", ctypes.c_void_p),
                ("bInheritHandle", wintypes.BOOL),
            ]

        sddl = f"D:P(A;;0x{EVENT_ALL_ACCESS:X};;;{self.current_user_sid()})"
        descriptor = ctypes.c_void_p()
        ok = self._advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW(
            ctypes.c_wchar_p(sddl), wintypes.DWORD(SDDL_REVISION_1), ctypes.byref(descriptor), None
        )
        if not ok:
            raise OSError(ctypes.get_last_error(), f"could not build the lease DACL from {sddl!r}")
        sa = _SecurityAttributes()
        sa.nLength = ctypes.sizeof(_SecurityAttributes)
        sa.lpSecurityDescriptor = descriptor
        sa.bInheritHandle = False       # the OWNER handle is never inheritable
        # Keep the descriptor alive for as long as the SECURITY_ATTRIBUTES that points at it; the
        # surface is process-lived, so it is freed at exit with the process rather than by LocalFree.
        sa._lm3_descriptor = descriptor  # type: ignore[attr-defined]  # noqa: SLF001
        self._security_attributes = sa
        return sa


_REAL_SURFACE_LOCK = threading.Lock()
_REAL_SURFACE: Win32Surface | None = None


def build_real_surface(platform_name: str | None = None) -> Win32Surface:
    """The process-wide real surface. Raises :class:`LeaseError` off Windows -- never a stub.

    A no-op stand-in would let the Windows lease "succeed" on a platform that has no lease at all,
    which is precisely the silent-fallback class of bug the plan refuses for ``Local\\``.
    """
    if not is_windows(platform_name):
        raise LeaseError(
            f"the Win32 lease surface is unavailable on {platform_name or sys.platform!r}. The "
            f"Windows lease adapter must be given an explicit surface (tests inject a fake)."
        )
    global _REAL_SURFACE
    with _REAL_SURFACE_LOCK:
        if _REAL_SURFACE is None:
            _REAL_SURFACE = Win32ApiSurface()
        return _REAL_SURFACE
