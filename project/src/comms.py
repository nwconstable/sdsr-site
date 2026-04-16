"""
comms.py

Communication constraints and protocol interruption logging for the
wetlands GNN experiment.

Classes
-------
ProtocolInterruptionLogger : Records dropout/blackout events; writes JSON.
CommunicationChannel       : Controls comm schedule, per-round dropout,
                             and gossip pairing.  Attaches a logger automatically.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path


class ProtocolInterruptionLogger:
    """Record dropout and blackout events across training rounds.

    A **dropout** is a drone that fails to participate in a round.
    A **blackout** is a round where the per-round dropout rate exceeds
    ``baseline_p``; every absent drone in such a round is tagged
    ``reason="blackout"`` rather than ``reason="dropout"``.
    """

    def __init__(self, baseline_p: float) -> None:
        self.baseline_p = baseline_p
        self._events: list[dict] = []
        self._logged_rounds: set[int] = set()   # rounds that had at least one event
        self._all_rounds: set[int] = set()      # every round record_round was called for
        self._blackout_rounds = 0               # rounds classified as blackout
        self._total_dropouts = 0                # sum of "dropout" events
        self._total_blackouts = 0               # sum of "blackout" events
        self._per_drone: dict[int, dict[str, int]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(self, round_num: int, drone_id: int, reason: str) -> None:
        """Append a single event.  *reason* must be ``"dropout"`` or ``"blackout"``."""
        if reason not in ("dropout", "blackout"):
            raise ValueError(f"reason must be 'dropout' or 'blackout', got {reason!r}")
        self._events.append({"round": round_num, "drone_id": drone_id, "reason": reason})
        self._logged_rounds.add(round_num)
        entry = self._per_drone.setdefault(drone_id, {"dropout": 0, "blackout": 0})
        entry[reason] += 1
        if reason == "blackout":
            self._total_blackouts += 1
        else:
            self._total_dropouts += 1

    def record_round(
        self,
        round_num: int,
        participants: list[int],
        all_drone_ids: list[int],
    ) -> None:
        """Classify and log every drone that did not participate this round.

        Per-round dropout rate = ``n_dropped / n_all``.  If the rate exceeds
        ``baseline_p`` the round is treated as a blackout and each absent
        drone is logged with ``reason="blackout"``; otherwise ``reason="dropout"``.
        Rounds with no dropouts are still counted in ``total_rounds``.
        """
        self._all_rounds.add(round_num)
        if not all_drone_ids:
            return
        present = set(participants)
        dropped = [d for d in all_drone_ids if d not in present]
        dropout_rate = len(dropped) / len(all_drone_ids)
        reason = "blackout" if dropout_rate > self.baseline_p else "dropout"
        if reason == "blackout" and dropped:
            self._blackout_rounds += 1
        for drone_id in dropped:
            self.log(round_num, drone_id, reason)

    def summary(self) -> dict:
        """Return aggregate statistics.

        Keys
        ----
        total_rounds      : number of rounds ``record_round`` was called for
        total_dropouts    : total ``"dropout"`` events logged
        total_blackouts   : total ``"blackout"`` events logged
        blackout_rounds   : number of rounds classified as blackout
        per_drone         : dict mapping str(drone_id) -> {"dropout": N, "blackout": N}
        """
        return {
            "total_rounds": len(self._all_rounds),
            "total_dropouts": self._total_dropouts,
            "total_blackouts": self._total_blackouts,
            "blackout_rounds": self._blackout_rounds,
            "per_drone": {str(k): dict(v) for k, v in self._per_drone.items()},
        }

    def save(self, path: str | Path) -> None:
        """Write the full event log and summary to a JSON file at *path*."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.payload(), fh, indent=2)

    def payload(self) -> dict:
        """Return the serializable logger payload used for JSON export."""
        return {"summary": self.summary(), "events": list(self._events)}


class CommunicationChannel:
    """Model communication schedule and per-round dropout for K drones.

    Parameters
    ----------
    comm_every  : int    — steps between communication events
    dropout_p   : float  — per-drone independent dropout probability each round
    baseline_p  : float  — expected baseline dropout rate; rounds exceeding this
                           are flagged as blackout by the logger.
                           Defaults to ``dropout_p``.
    seed        : int    — RNG seed for reproducibility (``None`` = random)

    Attributes
    ----------
    logger : ProtocolInterruptionLogger
        Attached automatically.  Accessible for post-experiment analysis and
        JSON export via ``channel.logger.save(path)``.
    """

    def __init__(
        self,
        comm_every: int,
        dropout_p: float,
        baseline_p: float | None = None,
        seed: int | None = None,
    ) -> None:
        self.comm_every = comm_every
        self.dropout_p = dropout_p
        self.baseline_p = baseline_p if baseline_p is not None else dropout_p
        self._rng = random.Random(seed)
        self._round_num = 0
        self.logger = ProtocolInterruptionLogger(self.baseline_p)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_comm_round(self, step: int) -> bool:
        """Return ``True`` iff ``step % comm_every == 0``."""
        return step % self.comm_every == 0

    def sample_participants(self, drone_ids: list[int]) -> list[int]:
        """Return the subset of drones that participate this round.

        Each drone is kept independently with probability ``1 - dropout_p``.
        ``logger.record_round`` is called automatically before returning.
        """
        survivors = [d for d in drone_ids if self._rng.random() >= self.dropout_p]
        self.logger.record_round(self._round_num, survivors, list(drone_ids))
        self._round_num += 1
        return survivors

    def gossip_pairs(
        self,
        drone_ids: list[int],
        allowed_pairs: list[tuple[int, int]] | None = None,
    ) -> list[tuple[int, int]]:
        """Return non-overlapping random pairs of drones for gossip exchange.

        Applies per-drone dropout via ``sample_participants``, shuffles the
        survivors, then pairs adjacent elements.  The last drone is skipped
        when the survivor count is odd.  The logger round counter advances
        by one (same as a ``sample_participants`` call).
        """
        if allowed_pairs is None:
            survivors = self.sample_participants(drone_ids)
            self._rng.shuffle(survivors)
            return [
                (survivors[i], survivors[i + 1])
                for i in range(0, len(survivors) - 1, 2)
            ]

        eligible_nodes = sorted(
            {
                int(node_id)
                for left, right in allowed_pairs
                for node_id in (left, right)
                if node_id in drone_ids
            }
        )
        survivors = set(self.sample_participants(eligible_nodes))
        shuffled_pairs = [
            (int(left), int(right))
            for left, right in allowed_pairs
            if left in survivors and right in survivors
        ]
        self._rng.shuffle(shuffled_pairs)

        matched: set[int] = set()
        selected_pairs: list[tuple[int, int]] = []
        for left, right in shuffled_pairs:
            if left in matched or right in matched:
                continue
            selected_pairs.append((left, right))
            matched.add(left)
            matched.add(right)
        return selected_pairs


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    DRONES = [0, 1, 2, 3]
    ROUNDS = 50
    DROPOUT_P = 0.25
    BASELINE_P = 0.20
    SEED = 42

    # --- FedAvg style: sample_participants for 50 rounds ---
    print(f"FedAvg-style simulation  "
          f"| drones={DRONES}  dropout_p={DROPOUT_P}  baseline_p={BASELINE_P}")
    print()
    fedavg_channel = CommunicationChannel(
        comm_every=5, dropout_p=DROPOUT_P, baseline_p=BASELINE_P, seed=SEED,
    )
    for step in range(ROUNDS):
        participants = fedavg_channel.sample_participants(DRONES)
        comm_tag = "<COMM>" if fedavg_channel.is_comm_round(step) else "      "
        print(f"  Round {step:>2} {comm_tag}  participants={participants}")

    print()
    summary = fedavg_channel.logger.summary()
    print("Summary (FedAvg channel):")
    for k, v in summary.items():
        if k != "per_drone":
            print(f"  {k:<20}: {v}")
    print("  per_drone:")
    for drone_id, counts in summary["per_drone"].items():
        print(f"    drone {drone_id}: {counts}")

    # Verify required summary keys
    required_keys = {"total_rounds", "total_dropouts", "total_blackouts", "per_drone"}
    assert required_keys.issubset(summary.keys()), (
        f"Missing keys: {required_keys - set(summary.keys())}"
    )
    assert summary["total_rounds"] == ROUNDS, (
        f"Expected {ROUNDS} rounds, got {summary['total_rounds']}"
    )

    # --- Gossip pairs demo (separate channel so rounds don't mix) ---
    print()
    print("Gossip-style demo (5 rounds):")
    gossip_channel = CommunicationChannel(
        comm_every=1, dropout_p=DROPOUT_P, baseline_p=BASELINE_P, seed=SEED,
    )
    for step in range(5):
        pairs = gossip_channel.gossip_pairs(DRONES)
        print(f"  Round {step}  pairs={pairs}")

    # --- JSON save/load ---
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name
    fedavg_channel.logger.save(tmp_path)
    with open(tmp_path, encoding="utf-8") as fh:
        payload = json.load(fh)
    os.unlink(tmp_path)

    assert "summary" in payload, "JSON missing 'summary' key"
    assert "events" in payload, "JSON missing 'events' key"
    assert isinstance(payload["events"], list), "'events' is not a list"

    print()
    print(f"JSON verified: {len(payload['events'])} events saved.")
    print()
    print("comms.py smoke test passed.")
    sys.exit(0)
