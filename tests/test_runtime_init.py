"""``leafmachine3.core.runtime.__init__`` -- the published surface of the runtime package.

The plan says "Add ``leafmachine3/core/runtime.py``"; it is a package instead, so that three
implementers could work on disjoint files. That substitution is only safe while the package's
``__init__`` really does re-export everything the submodules own -- otherwise a name that the plan
treats as part of one module (``acquire_root_lease`` for section 3.3's acquisition ordering,
``build_launch_manifest`` for section 3.4's ``run_manifest.json``) is reachable only by importing a
submodule directly, and Step 3's wiring in ``machine3.py`` would have to reach across the package
boundary in a way the docstring promises it will not have to.

These tests are the enforcement of that promise. The set-difference test is the load-bearing one:
it fails automatically whenever a submodule grows an export that nobody added to ``_LAZY``, which
is exactly how the thirteen missing names got missed in the first place.
"""
from __future__ import annotations

import importlib

import pytest

import leafmachine3.core.runtime as runtime
from leafmachine3.core.runtime import config_io, grant, lease, records

#: The four implementation modules of the layout documented in the package docstring. ``_types`` is
#: excluded on purpose: its names are imported EAGERLY at the top of ``__init__`` (it is the shared
#: contract every other module consumes), so it is covered by the resolution test below rather than
#: by the lazy-map test.
_IMPLEMENTATION_MODULES = (lease, records, grant, config_io)


@pytest.mark.parametrize("module", _IMPLEMENTATION_MODULES, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_every_submodule_export_is_re_exported_by_the_package(module):
    """A submodule's ``__all__`` is the source of truth; the package must publish all of it."""
    missing = sorted(set(module.__all__) - set(runtime.__all__))
    assert missing == [], (
        f"{module.__name__} exports {missing} but leafmachine3.core.runtime does not. Add each name "
        f"to _LAZY (mapped to {module.__name__.rsplit('.', 1)[-1]!r}) and to __all__."
    )


@pytest.mark.parametrize("module", _IMPLEMENTATION_MODULES, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_re_exported_names_are_the_same_objects_as_the_submodule_ones(module):
    """Re-export, not re-implementation -- a second definition here would be a second answer."""
    for name in module.__all__:
        assert getattr(runtime, name) is getattr(module, name), f"{name} is not {module.__name__}.{name}"


def test_every_published_name_resolves():
    """``__all__`` is a promise about ``from ... import *``; an unresolvable entry breaks it."""
    unresolved = sorted(name for name in runtime.__all__ if not hasattr(runtime, name))
    assert unresolved == []


def test_published_surface_has_no_duplicates():
    assert len(runtime.__all__) == len(set(runtime.__all__))


def test_lazy_map_and_all_agree():
    """Every lazily mapped name is published, so ``_LAZY`` cannot grow a private-by-accident entry."""
    assert sorted(set(runtime._LAZY) - set(runtime.__all__)) == []


def test_section_3_3_and_3_4_entry_points_import_by_their_documented_path():
    """The names Step 3 calls from ``machine3.py``, spelled the way it will spell them.

    section 3.3 puts the acquisition between ``cfg.validate()`` and ``ensure_hardware_profile(cfg)``
    and gives the cleanup lock the job of classifying abandoned records; section 3.4 requires the
    ``<run>/logs/run_manifest.json`` builder. All of them cross the package boundary.
    """
    from leafmachine3.core.runtime import (  # noqa: F401  -- importing IS the assertion
        LAUNCH_MANIFEST_FILENAME,
        acquire_root_lease,
        build_launch_manifest,
        cleanup_lease,
        launch_manifest_path,
        probe_deployment_occupied,
    )

    assert LAUNCH_MANIFEST_FILENAME == "run_manifest.json"


def test_unknown_attribute_raises_attribute_error_naming_the_package():
    with pytest.raises(AttributeError, match="no attribute 'definitely_not_exported'"):
        runtime.definitely_not_exported


def test_dir_lists_the_lazy_names_so_tab_completion_sees_the_whole_surface():
    listed = set(dir(runtime))
    assert set(runtime._LAZY).issubset(listed)


def test_the_package_still_imports_from_a_cold_module_cache():
    """Guards the PEP 562 path itself: a typo in ``_LAZY``'s module names only shows up on import."""
    reloaded = importlib.reload(runtime)
    assert set(reloaded.__all__) == set(runtime.__all__)
