"""Distributed constraint optimization -- Weiss Ch 12.

``model`` holds the constraint network ``N = <X, D, C>`` of section 2.1 and the
agent/variable partition of section 2.2; ``dfstree`` builds the DFS pseudo-tree
ADOPT requires; ``adopt`` is the solver of section 4.1; ``bruteforce`` is the
centralised oracle the tests check it against.
"""

from .model import ConstraintNetwork
from .dfstree import DFSTree, build_dfs_forest
from .adopt import AdoptResult, solve_adopt
from .bruteforce import solve_bruteforce

__all__ = [
    "ConstraintNetwork",
    "DFSTree",
    "build_dfs_forest",
    "AdoptResult",
    "solve_adopt",
    "solve_bruteforce",
]
