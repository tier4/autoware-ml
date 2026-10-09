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
//
// Differences from mmcv: the output row holds every candidate (m + 1 slots instead of 9), the
// number of valid vertices is counted from the mask, the launch covers all polygons instead of
// one block per batch element, and the identical-box special case is gone because the Python
// side drops duplicate candidates before calling the kernel.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>

/**
 * @file diff_iou_rotated_cuda.cu
 * @brief CUDA kernel sorting the vertices of the intersection polygon of two rotated boxes.
 *
 * The differentiable rotated IoU builds, for every box pair, a padded set of candidate
 * vertices (the 8 corners plus the 16 edge-edge intersections) with a validity mask. This
 * kernel orders the valid vertices counter-clockwise around their mean so the shoelace
 * formula can integrate the polygon area on the Python side.
 *
 * Two convex quadrilaterals share at most 8 polygon vertices, but the candidates are
 * classified with float tolerances, and near-degenerate pairs (nearly identical boxes, a box
 * inscribed in the other) can leave more than 8 of them valid. The output row therefore has
 * room for every candidate, so no valid vertex is ever dropped.
 */

namespace
{
/// Offset of the first edge-edge intersection candidate in the vertex axis (after the 8 corners).
constexpr int kIntersectionOffset = 8;

/// Tolerance used when comparing vertex coordinates.
constexpr float kEpsilon = 1e-8f;

/// Number of threads per CUDA block; one thread sorts one polygon.
constexpr int kThreadsPerBlock = 256;

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
 * One thread handles one polygon; the batch and pair axes are flattened so the grid covers
 * every polygon no matter how the pairs are split across batch elements. The output index row
 * has the layout (A, B, C, ..., A, X, X, X): the sorted valid vertices, the first one repeated to
 * close the polygon, then padding with the index of an arbitrary invalid intersection candidate,
 * whose value and gradient are zero. The row holds m + 1 indices, enough for every candidate plus
 * the closing duplicate, and the number of valid vertices is counted from the mask, so the
 * writes below stay inside the row for any mask.
 *
 * @param num_polygons Number of polygons, i.e. batch size times box pairs per batch element.
 * @param m Number of candidate vertices per pair (24).
 * @param vertices Normalized candidate vertices of shape (num_polygons, m, 2).
 * @param mask Validity mask of shape (num_polygons, m).
 * @param idx Output sorted indices of shape (num_polygons, m + 1).
 */
__global__ void diff_iou_rotated_sort_vertices_forward_cuda_kernel(
  int num_polygons, int m, const float * __restrict__ vertices, const bool * __restrict__ mask,
  int * __restrict__ idx)
{
  const int row = m + 1;                                    // indices per polygon
  const int index = blockIdx.x * blockDim.x + threadIdx.x;  // index of polygon
  const int stride = gridDim.x * blockDim.x;
  for (int i = index; i < num_polygons; i += stride) {
    int pad = kIntersectionOffset;  // index of an arbitrary invalid intersection point
    for (int j = kIntersectionOffset; j < m; ++j) {
      if (!mask[i * m + j]) {
        pad = j;
        break;
      }
    }
    int n_valid = 0;
    for (int j = 0; j < m; ++j) {
      n_valid += mask[i * m + j] ? 1 : 0;
    }
    if (n_valid < 3) {
      // not enough vertices for a polygon: take an invalid intersection point (zero padding)
      for (int j = 0; j < row; ++j) {
        idx[i * row + j] = pad;
      }
    } else {
      // sort the valid vertices
      for (int j = 0; j < n_valid; ++j) {
        // initialize with a "big" value
        float x_min = 1;
        float y_min = -kEpsilon;
        int i_take = 0;
        int i2 = 0;
        float x2 = 0.0f;
        float y2 = 0.0f;
        if (j != 0) {
          i2 = idx[i * row + j - 1];
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
        idx[i * row + j] = i_take;
      }
      // duplicate the first index to close the polygon
      idx[i * row + n_valid] = idx[i * row + 0];

      // pad the remaining slots
      for (int j = n_valid + 1; j < row; ++j) {
        idx[i * row + j] = pad;
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
 * @return Sorted vertex indices of shape (B, N, 25), int32.
 */
at::Tensor diff_iou_rotated_sort_vertices_forward_cuda(
  const at::Tensor vertices, const at::Tensor mask)
{
  const at::cuda::CUDAGuard device_guard(vertices.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int b = vertices.size(0);
  const int n = vertices.size(1);
  const int m = vertices.size(2);
  at::Tensor idx =
    torch::zeros({b, n, m + 1}, at::device(vertices.device()).dtype(at::ScalarType::Int));
  if (b == 0 || n == 0) {
    return idx;
  }

  const int num_polygons = b * n;
  const int num_blocks = (num_polygons + kThreadsPerBlock - 1) / kThreadsPerBlock;
  diff_iou_rotated_sort_vertices_forward_cuda_kernel<<<num_blocks, kThreadsPerBlock, 0, stream>>>(
    num_polygons, m, vertices.data_ptr<float>(), mask.data_ptr<bool>(), idx.data_ptr<int>());
  AT_CUDA_CHECK(cudaGetLastError());

  return idx;
}
