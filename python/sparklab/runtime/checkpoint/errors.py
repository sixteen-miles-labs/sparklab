"""Expected cache admission failures; neither permits unsafe publication."""


class StoreFull(RuntimeError):
    pass


class StoreBusy(RuntimeError):
    pass


class StoreUnavailable(OSError):
    pass
