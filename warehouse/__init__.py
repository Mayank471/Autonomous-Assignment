"""Automated warehouse multiagent pathfinding with PGP/DCOP plan repair.

Implements the system described in the assignment: a fleet of robots plans
collision-free pickup/delivery routes with prioritized Space-Time A*, then
repairs those plans locally when the environment disrupts them, negotiating
with neighbouring agents rather than replanning globally.

The design follows Weiss, *Multiagent Systems* (2nd ed.):

* Ch 11 section 3   -- coordination prior to local planning (social laws,
                       organizational structuring, contract net)
* Ch 11 section 4   -- local planning prior to coordination (MPCP)
* Ch 11 section 6.3 -- partial global planning (PGP) and its six mechanisms,
                       which structure the repair layer
* Ch 12             -- DCOP formulation and the ADOPT solver, which decides
                       *which* agents change their plans
"""

__version__ = "1.0.0"
