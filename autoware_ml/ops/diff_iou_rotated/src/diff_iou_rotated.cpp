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
// Adapted from mmcv/ops/csrc/pytorch/diff_iou_rotated.cpp.

#include <torch/torch.h>

#ifdef WITH_CUDA
// Implemented in diff_iou_rotated_cuda.cu
at::Tensor diff_iou_rotated_sort_vertices_forward_cuda(
  const at::Tensor vertices, const at::Tensor mask);
#endif

/**
 * @brief Sort the valid vertices of the intersection polygon of rotated box pairs.
 *
 * @param vertices Candidate vertices normalized around their mean, shape (B, N, 24, 2), float32.
 * @param mask Validity mask of the candidates, shape (B, N, 24), bool.
 * @return Sorted vertex indices of shape (B, N, 25), int32. See the CUDA kernel for the layout.
 */
at::Tensor diff_iou_rotated_sort_vertices_forward(const at::Tensor vertices, const at::Tensor mask)
{
  TORCH_CHECK(
    vertices.dim() == 4 && vertices.size(3) == 2, "vertices must have shape (B, N, M, 2)");
  TORCH_CHECK(mask.dim() == 3, "mask must have shape (B, N, M)");
  TORCH_CHECK(
    mask.size(0) == vertices.size(0) && mask.size(1) == vertices.size(1) &&
      mask.size(2) == vertices.size(2),
    "mask must match the leading dimensions of vertices");
  TORCH_CHECK(vertices.scalar_type() == at::ScalarType::Float, "vertices must be float32");
  TORCH_CHECK(mask.scalar_type() == at::ScalarType::Bool, "mask must be bool");
  TORCH_CHECK(vertices.is_contiguous(), "vertices must be contiguous");
  TORCH_CHECK(mask.is_contiguous(), "mask must be contiguous");

  if (vertices.is_cuda()) {
#ifdef WITH_CUDA
    TORCH_CHECK(mask.is_cuda(), "all inputs must be on the same device");
    return diff_iou_rotated_sort_vertices_forward_cuda(vertices, mask);
#else
    TORCH_CHECK(false, "diff_iou_rotated_sort_vertices_forward was compiled without CUDA");
#endif
  }
  TORCH_CHECK(false, "diff_iou_rotated_sort_vertices_forward is only implemented for CUDA tensors");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.doc() = "Vertex sorting kernel for the differentiable rotated IoU";
  m.def(
    "diff_iou_rotated_sort_vertices_forward", &diff_iou_rotated_sort_vertices_forward,
    "Sort intersection polygon vertices counter-clockwise (CUDA)", py::arg("vertices"),
    py::arg("mask"));
}
