# Copyright 2018 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

from .cpu import _lapack
from .cpu._lapack import eig
from .cpu._lapack import schur


EigComputationMode = eig.ComputationMode
SchurComputationMode = schur.ComputationMode
SchurSort = schur.Sort


def registrations() -> dict[str, list[tuple[str, Any, int]]]:
  return {
      "cpu": [
          (name, value, int(name.endswith("_ffi")))
          for name, value in _lapack.registrations().items()
      ]
  }


def batch_partitionable_targets() -> list[str]:
  return [name for name in _lapack.registrations() if name.endswith("_ffi")]
