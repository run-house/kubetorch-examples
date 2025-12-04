from .configuration_llada2_moe import LLaDA2MoeConfig
from .modeling_llada2_moe import LLaDA2MoeModelLM, LLaDA2MoeDecoderLayer

# Aliases for compatibility
LLaDA2Config = LLaDA2MoeConfig
LLaDA2ForCausalLM = LLaDA2MoeModelLM
LLaDA2DecoderLayer = LLaDA2MoeDecoderLayer
ModelClass = LLaDA2MoeModelLM
