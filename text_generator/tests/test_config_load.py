import logging
import re

logger = logging.getLogger(__name__)


def test_example():
    logger.info("Este es un log que siempre se mostrará")
    assert True


def test_config_loading(config):
    assert config is not None


def test_config_loading_model_name(config):
    assert config["model_cfg"] is not None
    assert isinstance(config["model_cfg"].name, str)


def test_config_loading_model_device(config):
    device_value = config["model_cfg"].device
    assert device_value in {"auto", "cpu"} or re.match(
        r"^cuda(:\d+)?$", device_value
    ), f"Invalid device: {device_value}"


def test_config_loading_model_dtype(config):
    assert config["model_cfg"].dtype is not None
    assert isinstance(config["model_cfg"].dtype, str)


def test_config_loading_model_chat(config):
    assert config["model_cfg"].is_chat_model is not None
    assert isinstance(config["model_cfg"].is_chat_model, bool)


def test_config_loading_decoding_temperature(config):
    assert config["decoding_cfg"] is not None
    assert config["decoding_cfg"].temperature >= 0.0


def test_config_loading_decoding_top_p(config):
    assert config["decoding_cfg"].top_p is not None
    assert 0.0 <= config["decoding_cfg"].top_p <= 1.0


def test_config_loading_decoding_top_k(config):
    assert config["decoding_cfg"].top_k is not None
    assert config["decoding_cfg"].top_k > 0


def test_config_loading_decoding_repetition_penalty(config):
    assert config["decoding_cfg"].repetition_penalty is not None
    assert config["decoding_cfg"].repetition_penalty >= 1.0


def test_config_loading_decoding_max_new_tokens(config):
    assert config["decoding_cfg"].max_new_tokens is not None
    assert config["decoding_cfg"].max_new_tokens > 0


def test_config_loading_decoding_do_sample(config):
    assert config["decoding_cfg"].do_sample is not None
    assert isinstance(config["decoding_cfg"].do_sample, bool)


def test_config_loading_decoding_num_return_sequences(config):
    assert config["decoding_cfg"].num_return_sequences is not None
    assert config["decoding_cfg"].num_return_sequences > 0


def test_config_loading_decoding_num_beams(config):
    assert config["decoding_cfg"].num_beams is not None
    assert config["decoding_cfg"].num_beams > 0


def test_quant_mode_supported(config):
    q = config["quant_cfg"]
    assert q.mode in {"none", "4bit", "8bit-offload"}, f"Modo no soportado: {q.mode}"


def test_quant_4bit_fields_types_and_values(config):
    q = config["quant_cfg"]
    if q.mode == "4bit":
        assert isinstance(q.bnb_4bit_use_double_quant, bool)

        assert q.bnb_4bit_quant_type in {
            "nf4",
            "fp4",
        }, f"bnb_4bit_quant_type inválido: {q.bnb_4bit_quant_type}"

        assert q.bnb_4bit_compute_dtype in {
            "bfloat16",
            "float16",
        }, f"bnb_4bit_compute_dtype inválido: {q.bnb_4bit_compute_dtype}"


def test_quant_8bit_offload_optional_fields(config):
    q = config["quant_cfg"]
    if q.mode == "8bit-offload":
        assert (
            q.llm_int8_enable_fp32_cpu_offload is True
        ), "llm_int8_enable_fp32_cpu_offload debe ser True en 8bit-offload"

        if q.max_memory is not None:
            assert isinstance(q.max_memory, dict)
            assert (
                any(k.startswith("cuda") for k in q.max_memory.keys())
                or "cuda" in q.max_memory
            )
            assert "cpu" in q.max_memory

        if q.offload_folder is not None:
            assert isinstance(q.offload_folder, str)
            assert len(q.offload_folder) > 0


def test_config_loading_news(config):
    assert config["news_cfg"] is not None
    assert config["news_cfg"].length_min > 0
    assert config["news_cfg"].length_max > 0
    assert config["news_cfg"].length_min < config["news_cfg"].length_max
    assert config["news_cfg"].length_max < config["decoding_cfg"].max_new_tokens
