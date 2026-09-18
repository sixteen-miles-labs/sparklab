"""Explicit GGUF snapshot selection for native recipes."""

from pathlib import Path

from .base import BackendError


def filenames(deployment):
    opts = deployment.backend_options
    selected = {}
    for key in ("gguf_file", "vision_file"):
        name = opts.get(key)
        if name is None and key == "vision_file":
            continue
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not name.endswith(".gguf")
        ):
            raise BackendError(f"{key} must be a single .gguf filename")
        selected[key] = name
    if len(set(selected.values())) != len(selected):
        raise BackendError("Text GGUF and vision projector must be different files")
    return selected


def validate(path, deployment):
    from sparklab.models.bonsai2.vision import vision_config
    from sparklab.models.bonsai2.weights import parse_config
    from sparklab.models.gguf.config import build_gguf_shim

    selected = filenames(deployment)
    if not path.is_dir():
        raise BackendError("GGUF artifact must be a snapshot directory")
    try:
        for name in selected.values():
            target = path / name
            if not target.is_file() or target.resolve().parent != path.resolve():
                raise ValueError(f"Missing or external GGUF artifact {name}")
        shim = build_gguf_shim(str(path / selected["gguf_file"]))
        if shim.model_type != "qwen35":
            raise ValueError("This recipe requires a Bonsai Qwen GGUF")
        text = parse_config(shim)
        if "vision_file" in selected:
            vc = vision_config(path / selected["vision_file"])
            if vc["out_hidden_size"] != text.hidden_size:
                raise ValueError("Vision projector width differs from text model")
        return {
            "files": selected,
            "total_bytes": sum((path / n).stat().st_size for n in selected.values()),
        }
    except (ValueError, KeyError, OSError) as exc:
        raise BackendError(f"Invalid GGUF snapshot: {exc}") from exc
