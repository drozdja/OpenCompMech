"""Project command modules.

This marker is intentional.  Several environments ship an unrelated
site-packages module named :mod:`scripts`; without a local package marker,
``from scripts...`` can import that module instead of this repository when a
command is launched as ``python scripts/foo.py``.
"""
