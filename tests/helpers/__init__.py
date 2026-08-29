"""Out-of-process helpers for tests that need real processes rather than fakes.

The runtime lease is a KERNEL object on both platforms, so several of its guarantees -- the lock
belongs to the open file description, killing the root does not free it while a child lives, an
executor worker inherits nothing -- are only true if real processes are involved. Anything that can
be proved in-process is proved in-process; the modules here exist for the rest.
"""
