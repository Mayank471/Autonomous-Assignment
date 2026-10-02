"""Automated warehouse multi-agent path finding with minimum-change plan repair.

A fleet of robots plans collision-free pickup-and-delivery routes with
prioritized multi-goal Space-Time A* (:mod:`warehouse.planner`).  During
execution, blockages, breakdowns and emergency tasks invalidate some plans; the
affected agents negotiate a repair (:mod:`warehouse.negotiation`) found by
Keep/Release CBS (:mod:`warehouse.repair.cbs`), which changes as few agents'
plans as possible and, among those repairs, minimises total time.

The ``dcop`` subpackage (ADOPT) is course material kept for reference; the
repair system does not use it.
"""

__version__ = "2.0.0"
