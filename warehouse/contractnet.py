"""The contract-net protocol -- Weiss Ch 11 section 3.3.

Used here for one specific job: when an agent breaks down, its unfinished tasks
have to go somewhere.  Section 6.2 flags exactly this as part of recovery --

    "this does not even account for opportunities for agents to reallocate
    responsibilities among themselves, such as if an agent that has deviated
    from expectations has exhausted a resource ... and it must now fall to
    another agent that has reserve resources"

-- and section 3.3 supplies the protocol.  The textbook's sequence is followed
directly: an announcement carrying an *eligibility specification*, a *task
abstraction* and a *bid specification*; eligible agents decide whether to bid
and what to put in the bid; the announcer awards, or on receiving no acceptable
bids may "give up, try again ..., broaden the eligibility requirements to
increase the pool of potential bidders, or decompose the task differently".

Bids are priced at *marginal* cost -- how much longer the bidder's own route
gets if it takes the task on -- so the award goes to whoever is genuinely
cheapest rather than merely nearest.  The textbook's closing point is worth
keeping in view: "no agent is forced to be part of a contract.  The agents
engage in a rudimentary form of negotiation, and form teams through mutual
selection."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from .grid import WarehouseGrid
from .tasks import AgentTasks, Task


@dataclass(frozen=True)
class Announcement:
    """A task offered to the fleet."""

    task: Task
    announcer: int
    #: Eligibility specification: how far a bidder may be from the pickup.
    max_distance: int
    #: Task abstraction: enough for a bidder to price it without the full plan.
    abstract_cost: int

    def eligible(self, grid: WarehouseGrid, position: int) -> bool:
        return grid.manhattan(position, self.task.pickup) <= self.max_distance


@dataclass(frozen=True)
class Bid:
    """A bidder's offer, priced at the marginal cost of accepting."""

    agent: int
    task_id: int
    marginal_cost: float

    def __lt__(self, other: "Bid") -> bool:
        return (self.marginal_cost, self.agent) < (other.marginal_cost, other.agent)


@dataclass
class ContractNetResult:
    """Who took what, and how much talking it took."""

    awards: dict[int, int] = field(default_factory=dict)  # task_id -> agent
    unawarded: tuple[int, ...] = ()
    messages: int = 0
    rounds: int = 0
    bids_received: int = 0

    @property
    def n_awarded(self) -> int:
        return len(self.awards)


def marginal_cost(
    grid: WarehouseGrid,
    position: int,
    pending: Sequence[Task],
    task: Task,
) -> float:
    """Extra travel the bidder takes on by appending ``task`` to its queue.

    A cheap Manhattan estimate rather than a planned route: the textbook's
    bidders reason from a *task abstraction*, and running Space-Time A* for
    every bid from every agent would cost more than the reallocation saves.
    """
    tail = pending[-1].delivery if pending else position
    return float(
        grid.manhattan(tail, task.pickup) + grid.manhattan(task.pickup, task.delivery)
    )


def run_contract_net(
    grid: WarehouseGrid,
    announcer: int,
    tasks: Iterable[Task],
    bidders: Iterable[int],
    positions: Mapping[int, int],
    agent_tasks: Mapping[int, AgentTasks],
    *,
    max_distance: int | None = None,
    broaden_factor: float = 2.0,
    max_rounds: int = 3,
) -> ContractNetResult:
    """Auction ``tasks`` to ``bidders``, awarding each to its cheapest bidder.

    Args:
        max_distance: initial eligibility radius.  Defaults to a third of the
            floor's diameter.
        broaden_factor: how much to widen eligibility each time a task attracts
            no bids -- the textbook's "broaden the eligibility requirements to
            increase the pool of potential bidders".
        max_rounds: how many times to re-announce before giving up on a task.
    """
    result = ContractNetResult()
    bidder_list = [b for b in bidders if b != announcer]
    outstanding = list(tasks)
    if not bidder_list or not outstanding:
        result.unawarded = tuple(t.task_id for t in outstanding)
        return result

    radius = (
        max_distance
        if max_distance is not None
        else (grid.n_rows + grid.n_cols) // 3
    )
    # Provisional queues, so a bidder that wins one task prices the next from
    # where that one leaves it rather than bidding cheaply on everything.
    claimed: dict[int, list[Task]] = {
        b: list(agent_tasks[b].tasks) if b in agent_tasks else [] for b in bidder_list
    }

    for round_index in range(max_rounds):
        if not outstanding:
            break
        result.rounds += 1
        still_open: list[Task] = []

        for task in outstanding:
            announcement = Announcement(
                task=task,
                announcer=announcer,
                max_distance=radius,
                abstract_cost=grid.manhattan(task.pickup, task.delivery),
            )
            # Announce to every bidder; each decides for itself whether to bid.
            result.messages += len(bidder_list)

            offers: list[Bid] = []
            for bidder in bidder_list:
                position = positions.get(bidder)
                if position is None or not announcement.eligible(grid, position):
                    continue
                offers.append(
                    Bid(
                        agent=bidder,
                        task_id=task.task_id,
                        marginal_cost=marginal_cost(
                            grid, position, claimed[bidder], task
                        ),
                    )
                )

            result.messages += len(offers)
            result.bids_received += len(offers)

            if not offers:
                still_open.append(task)
                continue

            winner = min(offers)
            result.awards[task.task_id] = winner.agent
            claimed[winner.agent].append(task)
            result.messages += 1  # the award

        outstanding = still_open
        radius = int(radius * broaden_factor)

    result.unawarded = tuple(t.task_id for t in outstanding)
    return result


def reallocate_tasks(
    grid: WarehouseGrid,
    broken_agent: int,
    agent_tasks: dict[int, AgentTasks],
    progress: Mapping[int, int],
    positions: Mapping[int, int],
    active_agents: Iterable[int],
) -> ContractNetResult:
    """Auction off a broken agent's unfinished tasks and update the queues in place.

    The broken agent keeps nothing: its queue is truncated at the tasks it had
    already completed.  Winners get their new task appended, so their next
    replan picks it up.
    """
    tasks = agent_tasks[broken_agent]
    done = progress.get(broken_agent, 0)
    unfinished = tasks.tasks[done:]
    if not unfinished:
        return ContractNetResult()

    result = run_contract_net(
        grid,
        announcer=broken_agent,
        tasks=unfinished,
        bidders=active_agents,
        positions=positions,
        agent_tasks=agent_tasks,
    )

    by_id = {t.task_id: t for t in unfinished}
    for task_id, winner in result.awards.items():
        agent_tasks[winner].tasks.append(by_id[task_id])

    # The broken agent retains only what it actually finished.
    tasks.tasks = tasks.tasks[:done]
    return result
