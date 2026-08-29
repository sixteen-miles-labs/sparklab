import torch


class TableManager:
    def __init__(self, max_running_reqs: int, page_table: torch.Tensor) -> None:
        self._max_running_reqs = max_running_reqs
        self._free_slots = list(range(max_running_reqs))
        self.page_table = page_table
        # NOTE: dummy request also use this pool to get the input ids, so we need to
        # make sure the token pool is initialized with valid values (token_id = 0).
        self.token_pool = torch.zeros_like(page_table, dtype=torch.int32)

    @property
    def available_size(self) -> int:
        return len(self._free_slots)

    def allocate(self) -> int:
        return self._free_slots.pop()

    def free(self, slot: int) -> None:
        self._free_slots.append(slot)

    def rebuild(self, page_table: torch.Tensor) -> None:
        """Re-point the page table, reallocate the token pool, and free all slots.

        Idle-only: all request slots are expected to be free at rebuild time.
        """
        self.page_table = page_table
        self.token_pool = torch.zeros_like(page_table, dtype=torch.int32)
        self._free_slots = list(range(self._max_running_reqs))
