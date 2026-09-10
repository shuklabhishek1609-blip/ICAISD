"""greenbench -- compute-aware evaluation of traffic forecasting models.

Companion code for the ICAISD 2026 submission. Import order matters: call
`greenbench.compat.apply_all(repo_root)` and put the LargeST checkout on
sys.path before importing `runner`, which pulls in `src.*`.
"""

__version__ = '0.1.0'

from . import compat, instrument  # safe: no dependency on src.*

__all__ = ['compat', 'instrument']
