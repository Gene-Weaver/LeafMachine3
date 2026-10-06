"""Model installer: fetch the pinned default LM3 models from the Hugging Face Hub.

Public surface (used by the ``lm3 models`` CLI, the server's ``/v1/models`` routes and the GUI):

* :func:`registry.load_lock` -- the committed ``models.lock.yaml`` (which repo + commit + sha each
  model artifact comes from). Pinned on purpose: an LM3 release installs exactly these files.
* :func:`installer.models_root` -- where the models live (``$LM3_MODELS_DIR`` else ``<settings>/models``).
* :func:`installer.status` -- per-action state: missing / current / outdated / pending / placeholder.
* :func:`installer.install` -- download what is missing or outdated, with a ``.backup`` of every file
  it replaces, verified hashes, and rollback on any failure. Crash-safe: :func:`installer.repair`
  restores stranded backups the next time anything looks at the folder.
"""
from leafmachine3.modelhub.installer import (  # noqa: F401
    InstallError, install, models_root, repair, status, verify,
)
from leafmachine3.modelhub.registry import Lock, load_lock  # noqa: F401
