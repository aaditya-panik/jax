/* Copyright 2025 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef JAXLIB_CPU_LAPACK_KERNELS_DLOPEN_LAPACK64_H_
#define JAXLIB_CPU_LAPACK_KERNELS_DLOPEN_LAPACK64_H_

#include "absl/status/status.h"

namespace jax {

// Attempts to dlopen libblas64.so and load ILP64 BLAS symbols.
absl::Status InitializeIlp64Blas();

// Attempts to dlopen liblapack64.so and load ILP64 LAPACK symbols.
absl::Status InitializeIlp64Lapack();

}  // namespace jax

#endif  // JAXLIB_CPU_LAPACK_KERNELS_DLOPEN_LAPACK64_H_
