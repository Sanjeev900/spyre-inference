# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MRV2: which Spyre speculator the runner builds, per speculative method.

One ordered (predicate, qualname) entry per supported method, mirroring upstream's
init_speculator dispatch order. supports() answers the platform gate without
importing any speculator class; select() resolves its class lazily so an engine
only pays the imports of the method it runs (e.g. dspark's module chain pulls
vllm.models.deepseek_v4).
"""

import importlib
from collections.abc import Callable
from typing import Any

import torch
from vllm.config import VllmConfig

SPYRE_SPECULATORS: list[tuple[Callable[[Any], bool], str]] = [
    (
        lambda spec: spec.use_dflash(),
        "spyre_inference.v1.worker.spyre.spec_decode:SpyreDFlashSpeculator",
    ),
    (
        lambda spec: spec.use_dspark(),
        "spyre_inference.v1.worker.spyre.dspark:SpyreDSparkSpeculator",
    ),
]


def supports(spec: Any) -> bool:
    """Whether the platform admits this speculative config. Imports nothing."""
    return any(pred(spec) for pred, _ in SPYRE_SPECULATORS)


def select(vllm_config: VllmConfig, device: torch.device, spyre_device: torch.device):
    """Build the Spyre speculator for the configured method.

    The runner discards upstream's instance, so every entry builds a Spyre class.
    """
    spec = vllm_config.speculative_config
    assert spec is not None  # the runner only builds a speculator when one exists
    for pred, qualname in SPYRE_SPECULATORS:
        if pred(spec):
            module_name, class_name = qualname.rsplit(":", 1)
            cls = getattr(importlib.import_module(module_name), class_name)
            return cls(vllm_config, device, spyre_device)
    raise ValueError(f"Spyre has no speculator for speculative method {spec.method!r}")
