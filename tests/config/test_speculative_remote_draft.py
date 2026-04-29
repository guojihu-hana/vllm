# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.config.speculative import SpeculativeConfig


def _make_stub_spec_config() -> SpeculativeConfig:
    cfg = SpeculativeConfig.__new__(SpeculativeConfig)
    cfg.tensor_parallel_size = None
    cfg.num_speculative_tokens = 1
    cfg.draft_model_config = None
    cfg.target_model_config = None
    cfg.method = "draft_model"
    cfg.remote_draft_enabled = False
    cfg.remote_draft_endpoint = None
    return cfg


def test_remote_draft_requires_draft_model_method():
    cfg = _make_stub_spec_config()
    cfg.remote_draft_enabled = True
    cfg.method = "ngram"
    cfg.remote_draft_endpoint = "tcp://127.0.0.1:18861"

    with pytest.raises(ValueError, match="only supports method='draft_model'"):
        cfg._verify_args()


def test_remote_draft_requires_endpoint():
    cfg = _make_stub_spec_config()
    cfg.remote_draft_enabled = True
    cfg.method = "draft_model"
    cfg.remote_draft_endpoint = None

    with pytest.raises(ValueError, match="remote_draft_endpoint must be provided"):
        cfg._verify_args()


def test_remote_draft_valid_with_endpoint():
    cfg = _make_stub_spec_config()
    cfg.remote_draft_enabled = True
    cfg.method = "draft_model"
    cfg.remote_draft_endpoint = "tcp://127.0.0.1:18861"

    assert cfg._verify_args() is cfg
