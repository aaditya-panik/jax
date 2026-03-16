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

#include "jaxlib/cpu/lapack_kernels_dlopen_lapack64.h"

#include <cstdint>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "jaxlib/cpu/lapack_kernels.h"
#include "xla/tsl/platform/env.h"

namespace ffi = xla::ffi;

namespace jax {

namespace {

// Helper to look up a symbol from a library handle and assign it to the
// int64_t kernel function pointer if found.
template <typename KernelType>
void TryAssign(void* handle, const char* name) {
  void* sym = nullptr;
  if (tsl::Env::Default()->GetSymbolFromLibrary(handle, name, &sym).ok() &&
      sym != nullptr) {
    AssignKernelFn<KernelType>(sym);
  }
}

}  // namespace

absl::Status InitializeIlp64Blas() {
  void* handle = nullptr;
  absl::Status status =
      tsl::Env::Default()->LoadDynamicLibrary("libblas64.so", &handle);
  if (!status.ok()) {
    return absl::NotFoundError(
        absl::StrFormat("Failed to load libblas64.so: %s", status.message()));
  }

  TryAssign<TriMatrixEquationSolver<ffi::DataType::F32, int64_t>>(handle,
                                                                  "strsm_");
  TryAssign<TriMatrixEquationSolver<ffi::DataType::F64, int64_t>>(handle,
                                                                  "dtrsm_");
  TryAssign<TriMatrixEquationSolver<ffi::DataType::C64, int64_t>>(handle,
                                                                  "ctrsm_");
  TryAssign<TriMatrixEquationSolver<ffi::DataType::C128, int64_t>>(handle,
                                                                   "ztrsm_");
  return absl::OkStatus();
}

absl::Status InitializeIlp64Lapack() {
  void* handle = nullptr;
  absl::Status status =
      tsl::Env::Default()->LoadDynamicLibrary("liblapack64.so", &handle);
  if (!status.ok()) {
    return absl::NotFoundError(
        absl::StrFormat("Failed to load liblapack64.so: %s", status.message()));
  }

  // LU Decomposition
  TryAssign<LuDecomposition<ffi::DataType::F32, int64_t>>(handle, "sgetrf_");
  TryAssign<LuDecomposition<ffi::DataType::F64, int64_t>>(handle, "dgetrf_");
  TryAssign<LuDecomposition<ffi::DataType::C64, int64_t>>(handle, "cgetrf_");
  TryAssign<LuDecomposition<ffi::DataType::C128, int64_t>>(handle, "zgetrf_");

  // QR Factorization
  TryAssign<QrFactorization<ffi::DataType::F32, int64_t>>(handle, "sgeqrf_");
  TryAssign<QrFactorization<ffi::DataType::F64, int64_t>>(handle, "dgeqrf_");
  TryAssign<QrFactorization<ffi::DataType::C64, int64_t>>(handle, "cgeqrf_");
  TryAssign<QrFactorization<ffi::DataType::C128, int64_t>>(handle, "zgeqrf_");

  // Pivoting QR Factorization
  TryAssign<PivotingQrFactorization<ffi::DataType::F32, int64_t>>(handle,
                                                                  "sgeqp3_");
  TryAssign<PivotingQrFactorization<ffi::DataType::F64, int64_t>>(handle,
                                                                  "dgeqp3_");
  TryAssign<PivotingQrFactorization<ffi::DataType::C64, int64_t>>(handle,
                                                                  "cgeqp3_");
  TryAssign<PivotingQrFactorization<ffi::DataType::C128, int64_t>>(handle,
                                                                   "zgeqp3_");

  // Orthogonal QR
  TryAssign<OrthogonalQr<ffi::DataType::F32, int64_t>>(handle, "sorgqr_");
  TryAssign<OrthogonalQr<ffi::DataType::F64, int64_t>>(handle, "dorgqr_");
  TryAssign<OrthogonalQr<ffi::DataType::C64, int64_t>>(handle, "cungqr_");
  TryAssign<OrthogonalQr<ffi::DataType::C128, int64_t>>(handle, "zungqr_");

  // Cholesky Factorization
  TryAssign<CholeskyFactorization<ffi::DataType::F32, int64_t>>(handle,
                                                                "spotrf_");
  TryAssign<CholeskyFactorization<ffi::DataType::F64, int64_t>>(handle,
                                                                "dpotrf_");
  TryAssign<CholeskyFactorization<ffi::DataType::C64, int64_t>>(handle,
                                                                "cpotrf_");
  TryAssign<CholeskyFactorization<ffi::DataType::C128, int64_t>>(handle,
                                                                 "zpotrf_");

  // SVD (divide and conquer)
  TryAssign<SingularValueDecomposition<ffi::DataType::F32, int64_t>>(handle,
                                                                     "sgesdd_");
  TryAssign<SingularValueDecomposition<ffi::DataType::F64, int64_t>>(handle,
                                                                     "dgesdd_");
  TryAssign<SingularValueDecompositionComplex<ffi::DataType::C64, int64_t>>(
      handle, "cgesdd_");
  TryAssign<SingularValueDecompositionComplex<ffi::DataType::C128, int64_t>>(
      handle, "zgesdd_");

  // SVD (QR)
  TryAssign<SingularValueDecompositionQR<ffi::DataType::F32, int64_t>>(
      handle, "sgesvd_");
  TryAssign<SingularValueDecompositionQR<ffi::DataType::F64, int64_t>>(
      handle, "dgesvd_");
  TryAssign<SingularValueDecompositionQRComplex<ffi::DataType::C64, int64_t>>(
      handle, "cgesvd_");
  TryAssign<SingularValueDecompositionQRComplex<ffi::DataType::C128, int64_t>>(
      handle, "zgesvd_");

  // Symmetric Eigenvalue Decomposition
  TryAssign<EigenvalueDecompositionSymmetric<ffi::DataType::F32, int64_t>>(
      handle, "ssyevd_");
  TryAssign<EigenvalueDecompositionSymmetric<ffi::DataType::F64, int64_t>>(
      handle, "dsyevd_");
  TryAssign<EigenvalueDecompositionHermitian<ffi::DataType::C64, int64_t>>(
      handle, "cheevd_");
  TryAssign<EigenvalueDecompositionHermitian<ffi::DataType::C128, int64_t>>(
      handle, "zheevd_");

  // General Eigenvalue Decomposition
  TryAssign<EigenvalueDecomposition<ffi::DataType::F32, int64_t>>(handle,
                                                                  "sgeev_");
  TryAssign<EigenvalueDecomposition<ffi::DataType::F64, int64_t>>(handle,
                                                                  "dgeev_");
  TryAssign<EigenvalueDecompositionComplex<ffi::DataType::C64, int64_t>>(
      handle, "cgeev_");
  TryAssign<EigenvalueDecompositionComplex<ffi::DataType::C128, int64_t>>(
      handle, "zgeev_");

  // Schur Decomposition
  TryAssign<SchurDecomposition<ffi::DataType::F32, int64_t>>(handle, "sgees_");
  TryAssign<SchurDecomposition<ffi::DataType::F64, int64_t>>(handle, "dgees_");
  TryAssign<SchurDecompositionComplex<ffi::DataType::C64, int64_t>>(handle,
                                                                    "cgees_");
  TryAssign<SchurDecompositionComplex<ffi::DataType::C128, int64_t>>(handle,
                                                                     "zgees_");

  // Hessenberg Decomposition
  TryAssign<HessenbergDecomposition<ffi::DataType::F32, int64_t>>(handle,
                                                                  "sgehrd_");
  TryAssign<HessenbergDecomposition<ffi::DataType::F64, int64_t>>(handle,
                                                                  "dgehrd_");
  TryAssign<HessenbergDecomposition<ffi::DataType::C64, int64_t>>(handle,
                                                                  "cgehrd_");
  TryAssign<HessenbergDecomposition<ffi::DataType::C128, int64_t>>(handle,
                                                                   "zgehrd_");

  // Tridiagonal Reduction
  TryAssign<TridiagonalReduction<ffi::DataType::F32, int64_t>>(handle,
                                                               "ssytrd_");
  TryAssign<TridiagonalReduction<ffi::DataType::F64, int64_t>>(handle,
                                                               "dsytrd_");
  TryAssign<TridiagonalReduction<ffi::DataType::C64, int64_t>>(handle,
                                                               "chetrd_");
  TryAssign<TridiagonalReduction<ffi::DataType::C128, int64_t>>(handle,
                                                                "zhetrd_");

  // Tridiagonal Solver
  TryAssign<TridiagonalSolver<ffi::DataType::F32, int64_t>>(handle, "sgtsv_");
  TryAssign<TridiagonalSolver<ffi::DataType::F64, int64_t>>(handle, "dgtsv_");
  TryAssign<TridiagonalSolver<ffi::DataType::C64, int64_t>>(handle, "cgtsv_");
  TryAssign<TridiagonalSolver<ffi::DataType::C128, int64_t>>(handle, "zgtsv_");

  return absl::OkStatus();
}

}  // namespace jax
