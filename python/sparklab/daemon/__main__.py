from __future__ import annotations

from sparklab.daemon import main  # package dispatch: client verb → client, else → server

raise SystemExit(main(prog="python -m sparklab.daemon"))
