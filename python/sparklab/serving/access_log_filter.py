"""Hide desktop-app polling traffic from uvicorn's access log.

The SparkLab desktop app polls a handful of read-only endpoints every 1-2s to keep its UI
current: ``GET /health`` (lifecycle), ``GET /v1/stats`` (runtime metrics), ``GET /v1/requests``
(request-log ring, carries a ``?since=&limit=`` query string), ``GET /v1/cache/status``, and a
bare ``GET /v1`` liveness probe. Uvicorn's ``uvicorn.access`` logger logs every one of these at
INFO, which floods both the engine's own stdout and the desktop's "Logs" screen (which tails
that stdout) with lines nobody reads.

These lines are registered as debug-level noise (per spec): suppressed by default, and only
shown once the engine's existing verbosity knob -- the ``LOG_LEVEL`` env var consumed by
:func:`sparklab.utils.init_logger` -- is raised to ``DEBUG`` (e.g. ``LOG_LEVEL=DEBUG``).
Everything else -- non-GET requests, and any other endpoint such as
``POST /v1/chat/completions`` -- is always logged, unaffected by this filter.
"""

from __future__ import annotations

import logging
import os

# Path prefixes considered desktop-polling traffic. Matched against the request path with
# any query string stripped (uvicorn's access record embeds the query string in the logged
# path, e.g. "/v1/requests?since=123&limit=50").
_POLLING_PATH_PREFIXES: tuple[str, ...] = (
    "/health",
    "/v1/stats",
    "/v1/requests",
    "/v1/cache/status",
)

# The bare "/v1" probe is matched *exactly*, never as a prefix -- every path above (and
# other real endpoints like "/v1/models") also starts with "/v1", so prefix-matching it
# would silently swallow everything under /v1.
_BARE_PROBE_PATH = "/v1"


def _debug_logging_enabled() -> bool:
    """True when the engine's existing verbosity switch requests debug output.

    Hooks into ``LOG_LEVEL`` (see ``sparklab.utils.logger.init_logger``) rather than adding
    a second, competing toggle: ``LOG_LEVEL=DEBUG`` already turns on debug-level logging for
    every sparklab logger, so it's the natural gate for these debug-registered access lines
    too.
    """
    return os.getenv("LOG_LEVEL", "").upper() == "DEBUG"


class PollingAccessLogFilter(logging.Filter):
    """``logging.Filter`` for the ``uvicorn.access`` logger.

    Drops access-log records for the desktop app's polling endpoints unless debug logging is
    enabled (:func:`_debug_logging_enabled`). Records that don't look like uvicorn's access
    log shape, or whose path isn't one of the polling paths, always pass through unchanged.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if _debug_logging_enabled():
            return True

        # uvicorn's h11/httptools protocols log access lines as:
        #   access_logger.info('%s - "%s %s HTTP/%s" %d', client_addr, method, full_path,
        #                       http_version, status)
        # i.e. record.args == (client_addr, method, full_path, http_version, status).
        args = record.args
        if not isinstance(args, tuple) or len(args) < 3:
            return True

        full_path = args[2]
        if not isinstance(full_path, str):
            return True

        path = full_path.split("?", 1)[0]

        if path == _BARE_PROBE_PATH:
            return False
        return not any(path.startswith(prefix) for prefix in _POLLING_PATH_PREFIXES)


def install_polling_access_log_filter() -> None:
    """Attach :class:`PollingAccessLogFilter` to uvicorn's access logger, once.

    Idempotent so callers (e.g. ``run_api_server``, which may be reached more than once in a
    process such as in tests) can invoke it unconditionally without piling up duplicate filter
    instances.
    """
    access_logger = logging.getLogger("uvicorn.access")
    if not any(isinstance(f, PollingAccessLogFilter) for f in access_logger.filters):
        access_logger.addFilter(PollingAccessLogFilter())
