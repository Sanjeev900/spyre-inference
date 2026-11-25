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

"""MRV2: DSpark speculator (vllm/v1/worker/gpu/spec_decode/dspark/speculator.py).

Upstream's DSparkSpeculator over the Spyre DFlash overrides. The class lives in
its own module so the dspark import chain (vllm.models.deepseek_v4, via
dspark.utils) stays behind the dispatch table's lazy builder.
"""

from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

from spyre_inference.v1.worker.spyre.spec_decode import SpyreDFlashSpeculator


class SpyreDSparkSpeculator(SpyreDFlashSpeculator, DSparkSpeculator):
    """The Markov loop runs in the model's propose_block; the rest is DFlash's."""

    keep_on_host = ("model.markov_head.", "draft_id_to_target_id")
