"""Home Assistant coordination for Samsung Art Mode publishing (M005/S02).

Art Mode publishing coordinates with Home Assistant through a **single-writer
exclusive lease**: one publisher may mutate the Frame at a time. The caller
records the pre-mutation automation state when constructing the coordination
adapter; on a failed publish :meth:`HomeAssistantCoordinationAdapter.restore_prior_state`
replays it so the Frame never sits in a torn state.
"""

from __future__ import annotations

from typing import Any

from curator.dest.base import DestinationError


class SimulatorLeaseManager:
    """Single-writer exclusive lease with a deterministic holder.

    ``acquire`` succeeds when the lease is free or already held by the same
    holder (re-entrant) and fails with ``False`` when another holder owns it —
    the exclusivity guarantee. ``release`` only clears a lease the caller
    actually holds.
    """

    def __init__(self) -> None:
        self._holder: str | None = None

    def is_held(self) -> bool:
        return self._holder is not None

    @property
    def holder(self) -> str | None:
        return self._holder

    def acquire(self, holder: str) -> bool:
        if self._holder is not None and self._holder != holder:
            return False
        self._holder = holder
        return True

    def release(self, holder: str) -> None:
        if self._holder == holder:
            self._holder = None


class HomeAssistantCoordinationAdapter:
    """Coordinates Art Mode publishes with a Home Assistant lease.

    The pre-mutation automation state is recorded at construction
    (:attr:`prior_state`); :meth:`restore_prior_state` replays it after a failed
    publish. :meth:`acquire_lease` raises :class:`DestinationError` when another
    holder already owns the lease.
    """

    def __init__(
        self,
        lease_manager: SimulatorLeaseManager,
        prior_automation_state: dict[str, Any] | None = None,
        holder: str = "samsung-art-mode",
    ) -> None:
        self._lease_manager = lease_manager
        self.prior_state = prior_automation_state
        self.current_state = prior_automation_state
        self.restore_calls = 0
        self.holder = holder

    def acquire_lease(self) -> bool:
        if not self._lease_manager.acquire(self.holder):
            raise DestinationError(
                f"lease held by {self._lease_manager.holder!r};"
                f" cannot acquire for {self.holder!r}"
            )
        return True

    def release_lease(self) -> None:
        self._lease_manager.release(self.holder)

    def restore_prior_state(self) -> dict[str, Any] | None:
        """Restore the recorded pre-mutation automation state."""
        self.restore_calls += 1
        self.current_state = self.prior_state
        return self.current_state
