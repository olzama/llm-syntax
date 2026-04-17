from pathlib import Path

import pytest

from src.text_generator.config_loader import load_config


@pytest.fixture(scope="session")
def config():
    config_path = Path("src/config/config.yaml")
    model_cfg, decoding_cfg, quant_cfg, chat_cfg, news_cfg = load_config(config_path)
    return {
        "model_cfg": model_cfg,
        "decoding_cfg": decoding_cfg,
        "quant_cfg": quant_cfg,
        "chat_cfg": chat_cfg,
        "news_cfg": news_cfg,
    }
