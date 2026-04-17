from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import yaml


@dataclass
class ModelCfg:
    name: str
    device: str = "auto"
    dtype: str = "auto"
    is_chat_model: bool = True


@dataclass
class DecodingCfg:
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.05
    max_new_tokens: int = 220
    do_sample: bool = True
    num_return_sequences: int = 3
    num_beams: int = 1


@dataclass
class QuantizationCfg:
    mode: str = "none"

    # 4-bit
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"

    llm_int8_enable_fp32_cpu_offload: bool = False
    llm_int8_has_fp16_weight: bool = False
    llm_int8_threshold: float = 6.0
    offload_folder: Optional[str] = None
    max_memory: Optional[Dict[str, str]] = None


@dataclass
class ChatCfg:
    system_prompt: str = "You are a helpful assistant."


@dataclass
class NewsCfg:
    length_min: int = 80
    length_max: int = 140


def load_config(
    path: str = "config.yaml",
) -> tuple[ModelCfg, DecodingCfg, QuantizationCfg, ChatCfg, NewsCfg]:
    print(f"Loading configuration from {path}")
    raw = yaml.safe_load(open(path, "r", encoding="utf-8"))

    model = ModelCfg(**raw["model"])
    decoding = DecodingCfg(**raw.get("decoding", {}))
    quant = QuantizationCfg(**raw.get("quantization", {}))
    chat = ChatCfg(**raw.get("chat", {}))
    news = NewsCfg(**raw.get("news", {}))
    return model, decoding, quant, chat, news
