import os
import sys
from pathlib import Path

import pytest
from transformers import AutoConfig

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import Config
from src.tokenizer import NatureLM1BTokenizer


@pytest.mark.skipif(
    not Path("models/base-llm/NLM-1b").exists(),
    reason="Local NLM-1b path is unavailable.",
)
def test_local_nlm1b_tokenizer_and_config_smoke():
    cfg = Config.from_yaml("configs/polytamgen_sft.yaml")
    model_path = cfg.pp2d.llm_model_name_or_path

    tokenizer = NatureLM1BTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    ids = tokenizer.encode("<mol>CCO</mol>", add_special_tokens=False)

    assert len(ids) >= 3
    assert tokenizer.pad_token_id is not None or tokenizer.pad_token is None
    assert model_cfg.vocab_size > 1000
    assert model_cfg.hidden_size > 0
