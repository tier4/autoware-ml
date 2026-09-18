// Copyright (c) OpenMMLab. All rights reserved.
// Copyright 2026 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Adapted from mmcv/ops/csrc/common/cuda/diff_iou_rotated_cuda_kernel.cuh and
// mmcv/ops/csrc/pytorch/cuda/diff_iou_rotated_cuda.cu, which were adapted from
// https://github.com/lilanxiao/Rotated_IoU/cuda_op/sort_vert_kernel.cu

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>

#include <cmath>

/**
 * @file diff_iou_rotated_cuda.cu
 * @brief CUDA kernel sorting the vertices of the intersection polygon of two rotated boxes.
 *
 * The differentiable rotated IoU builds, for every box pair, a padded set of candidate
 * vertices (the 8 corners plus the 16 edge-edge intersections) with a validity mask. This
 * kernel orders the valid vertices counter-clockwise around their mean so the shoelace
 * formula can integrate the polygon area on the Python side.
 */

namespace
{
/// Maximum number of sorted vertex indices per polygon: 8 vertices plus the duplicated first one.
constexpr int kMaxNumVertexIndices = 9;

/// Offset of the first edge-edge intersection candidate in the vertex axis (after the 8 corners).
constexpr int kIntersectionOffset = 8;

/// Tolerance used when comparing vertex coordinates.
constexpr float kEpsilon = 1e-8f;

/// Upper bound on the number of threads per CUDA block.
constexpr int kMaxThreadsPerBlock = 512;

/// Choose the largest power of two thread count that does not exceed the work size.
inline int optimal_num_threads(int work_size)
{
  const int pow_2 = static_cast<int>(std::log(static_cast<double>(work_size)) / std::log(2.0));
  return std::max(std::min(1 << pow_2, kMaxThreadsPerBlock), 1);
}

/**
 * @brief Compare two vertices normalized around the polygon mean.
 *
 * Order: the minimum lies on the positive x axis and values grow counter-clockwise.
 *
 * @return True when vertex 1 sorts before vertex 2; equal vertices return false.
 */
__device__ bool compare_vertices(float x1, float y1, float x2, float y2)
{
  if (fabsf(x1 - x2) < kEpsilon && fabsf(y2 - y1) < kEpsilon) {
    return false;  // equal vertices never sort before each other
  }

  if (y1 > 0 && y2 < 0) return true;
  if (y1 < 0 && y2 > 0) return false;

  const float n1 = x1 * x1 + y1 * y1 + kEpsilon;
  const float n2 = x2 * x2 + y2 * y2 + kEpsilon;
  const float diff = fabsf(x1) * x1 / n1 - fabsf(x2) * x2 / n2;

  if (y1 > 0 && y2 > 0) {
    return diff > kEpsilon;
  }
  if (y1 < 0 && y2 < 0) {
    return diff < kEpsilon;
  }
  return false;
}

/**
 * @brief Sort the valid vertices of every intersection polygon counter-clockwise.
 *
 * One block handles one batch element, one thread one polygon. The output index row has the
 * layout (A, B, C, ..., A, X, X, X): the sorted valid vertices, the first one repeated to close
 * the polygon, then padding with the index of an arbitrary invalid intersection candidate, whose
 * value and gradient are zero.
 *
 * @param b Batch size.
 * @param n Number of box pairs per batch element.
 * @param m Number of candidate vertices per pair (24).
 * @param vertices Normalized candidate vertices of shape (b, n, m, 2).
 * @param mask Validity mask of shape (b, n, m).
 * @param num_valid Number of valid vertices per pair of shape (b, n).
 * @param idx Output sorted indices of shape (b, n, 9).
 */
__global__ void diff_iou_rotated_sort_vertices_forward_cuda_kernel(
  int b, int n, int m, const float * __restrict__ vertices, const bool * __restrict__ mask,
  const int * __restrict__ num_valid, int * __restrict__ idx)
{
  const int batch_idx = blockIdx.x;
  vertices += batch_idx * n * m * 2;
  mask += batch_idx * n * m;
  num_valid += batch_idx * n;
  idx += batch_idx * n * kMaxNumVertexIndices;

  const int index = threadIdx.x;  // index of polygon
  const int stride = blockDim.x;
  for (int i = index; i < n; i += stride) {
    int pad = kIntersectionOffset;  // index of an arbitrary invalid intersection point
    for (int j = kIntersectionOffset; j < m; ++j) {
      if (!mask[i * m + j]) {
        pad = j;
        break;
      }
    }
    // The intersection of two convex quadrilaterals has at most 8 vertices. Float noise on
    // nearly collinear edges can still flag extra intersection candidates as valid, so the
    // count is clamped to the row capacity: without it the loops below would write past this
    // polygon's 9 slots into the next polygon's row.
    const int n_valid = min(num_valid[i], kMaxNumVertexIndices - 1);
    if (n_valid < 3) {
      // not enough vertices for a polygon: take an invalid intersection point (zero padding)
      for (int j = 0; j < kMaxNumVertexIndices; ++j) {
        idx[i * kMaxNumVertexIndices + j] = pad;
      }
    } else {
      // sort the valid vertices; the number of valid vertices is known and at most 8
      for (int j = 0; j < n_valid; ++j) {
        // initialize with a "big" value
        float x_min = 1;
        float y_min = -kEpsilon;
        int i_take = 0;
        int i2 = 0;
        float x2 = 0.0f;
        float y2 = 0.0f;
        if (j != 0) {
          i2 = idx[i * kMaxNumVertexIndices + j - 1];
          x2 = vertices[i * m * 2 + i2 * 2 + 0];
          y2 = vertices[i * m * 2 + i2 * 2 + 1];
        }
        for (int k = 0; k < m; ++k) {
          const float x = vertices[i * m * 2 + k * 2 + 0];
          const float y = vertices[i * m * 2 + k * 2 + 1];
          if (mask[i * m + k] && compare_vertices(x, y, x_min, y_min)) {
            if ((j == 0) || (j != 0 && compare_vertices(x2, y2, x, y))) {
              x_min = x;
              y_min = y;
              i_take = k;
            }
          }
        }
        idx[i * kMaxNumVertexIndices + j] = i_take;
      }
      // duplicate the first index to close the polygon
      idx[i * kMaxNumVertexIndices + n_valid] = idx[i * kMaxNumVertexIndices + 0];

      // pad the remaining slots
      for (int j = n_valid + 1; j < kMaxNumVertexIndices; ++j) {
        idx[i * kMaxNumVertexIndices + j] = pad;
      }

      // Corner case: the two boxes are exactly the same. idx then holds duplicate elements,
      // which breaks the shoelace formula. By construction the duplicates only appear in the
      // first 8 positions (they are "corners in box", not "intersections of edges").
      if (n_valid == 8) {
        int counter = 0;
        for (int j = 0; j < 4; ++j) {
          const int check = idx[i * kMaxNumVertexIndices + j];
          for (int k = 4; k < kIntersectionOffset; ++k) {
            if (idx[i * kMaxNumVertexIndices + k] == check) counter++;
          }
        }
        if (counter == 4) {
          idx[i * kMaxNumVertexIndices + 4] = idx[i * kMaxNumVertexIndices + 0];
          for (int j = 5; j < kMaxNumVertexIndices; ++j) {
            idx[i * kMaxNumVertexIndices + j] = pad;
          }
        }
      }
    }
  }
}
}  // namespace

/**
 * @brief Launch the vertex sorting kernel.
 *
 * @param vertices Normalized candidate vertices of shape (B, N, 24, 2), float32, contiguous.
 * @param mask Validity mask of shape (B, N, 24), bool, contiguous.
 * @param num_valid Number of valid vertices per pair of shape (B, N), int32, contiguous.
 * @return Sorted vertex indices of shape (B, N, 9), int32.
 */
at::Tensor diff_iou_rotated_sort_vertices_forward_cuda(
  const at::Tensor vertices, const at::Tensor mask, const at::Tensor num_valid)
{
  const at::cuda::CUDAGuard device_guard(vertices.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int b = vertices.size(0);
  const int n = vertices.size(1);
  const int m = vertices.size(2);
  at::Tensor idx = torch::zeros(
    {b, n, kMaxNumVertexIndices}, at::device(vertices.device()).dtype(at::ScalarType::Int));
  if (b == 0 || n == 0) {
    return idx;
  }

  diff_iou_rotated_sort_vertices_forward_cuda_kernel<<<b, optimal_num_threads(n), 0, stream>>>(
    b, n, m, vertices.data_ptr<float>(), mask.data_ptr<bool>(), num_valid.data_ptr<int>(),
    idx.data_ptr<int>());
  AT_CUDA_CHECK(cudaGetLastError());

  return idx;
}
