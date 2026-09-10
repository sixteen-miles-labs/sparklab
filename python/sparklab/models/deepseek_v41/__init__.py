"""Native DeepSeek V4.1 Flash text research support."""
from .config import parse_config
from .model import DeepseekV41ForCausalLM
from .weight import iter_weights

__all__ = ["parse_config", "DeepseekV41ForCausalLM", "iter_weights"]
