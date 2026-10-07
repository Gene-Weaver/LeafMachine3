"""The CUDA library-dir lookup must survive ``nvidia`` losing its ``__init__.py`` (see core/cuda_libs.py).

Reproduced for real on 2026-10-07: switching a lock-built env from the dev variant back to production
deleted the shared ``nvidia/__init__.py``; the old ``nvidia.__file__`` lookup then returned nothing and
onnxruntime bound the CPU. These tests build that exact namespace-package shape on disk.
"""
from __future__ import annotations

import os
import sys
import types

import pytest

from leafmachine3.core import cuda_libs


@pytest.fixture
def namespace_nvidia(tmp_path, monkeypatch):
    """An ``nvidia`` namespace package: two roots, lib dirs inside, NO __init__.py, __file__ None."""
    a, b = tmp_path / "site_a" / "nvidia", tmp_path / "site_b" / "nvidia"
    for d in (a / "cublas" / "lib", a / "cudnn" / "lib", b / "cuda_runtime" / "lib", a / "nvjitlink"):
        d.mkdir(parents=True)
    mod = types.ModuleType("nvidia")
    mod.__file__ = None
    mod.__path__ = [str(a), str(b)]
    monkeypatch.setitem(sys.modules, "nvidia", mod)
    return sorted(str(p) for p in (a / "cublas" / "lib", a / "cudnn" / "lib", b / "cuda_runtime" / "lib"))


def test_lib_dirs_found_without_init_py(namespace_nvidia):
    assert cuda_libs.nvidia_lib_dirs() == namespace_nvidia


def test_no_nvidia_wheels_means_no_dirs(monkeypatch):
    monkeypatch.setitem(sys.modules, "nvidia", None)          # makes `import nvidia` raise ImportError
    assert cuda_libs.nvidia_lib_dirs() == []


def test_machine3_reexecs_with_the_dirs_when_init_py_is_gone(namespace_nvidia, monkeypatch):
    from leafmachine3 import machine3

    calls = []
    monkeypatch.delenv(machine3._LIBPATH_FLAG, raising=False)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/already/here")
    monkeypatch.setattr(os, "execv", lambda exe, argv: calls.append(os.environ["LD_LIBRARY_PATH"]))
    machine3._exec_with_cuda_libpath()
    assert len(calls) == 1
    assert calls[0].split(os.pathsep) == namespace_nvidia + ["/already/here"]


def test_executor_exports_the_dirs_when_init_py_is_gone(namespace_nvidia, monkeypatch):
    from leafmachine3.core.executor import DeviceManager

    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    DeviceManager.ensure_cuda_libpath()
    assert set(namespace_nvidia) <= set(os.environ["LD_LIBRARY_PATH"].split(os.pathsep))
