// Copyright 2025 TIER IV, Inc.
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

#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>

// Forward declarations for CUDA kernels
void bev_pool(
  int b, int d, int h, int w, int n, int c, int n_intervals, const float * x,
  const int * geom_feats, const int * interval_starts, const int * interval_lengths, float * out);

void bev_pool_grad(
  int b, int d, int h, int w, int n, int c, int n_intervals, const float * out_grad,
  const int * geom_feats, const int * interval_starts, const int * interval_lengths,
  float * x_grad);

/**
 * @brief Perform forward pass of BEV (Bird's Eye View) pooling.
 *
 * Pools scattered point features into a dense BEV grid by summing features
 * that fall into the same grid cell. Uses interval-based grouping for efficiency.
 *
 * @param x Input features tensor of shape (N, C).
 * @param geom_feats Geometric coordinates tensor of shape (N, 4).
 *                   Format: (height_idx, width_idx, depth_idx, batch_idx).
 * @param interval_lengths Number of points in each pooling interval, shape (n_intervals,).
 * @param interval_starts Starting index of each pooling interval, shape (n_intervals,).
 * @param b Batch size.
 * @param d Depth dimension of output grid.
 * @param h Height dimension of output grid.
 * @param w Width dimension of output grid.
 * @return Pooled output tensor of shape (B, D, H, W, C).
 */
at::Tensor bev_pool_forward(
  const at::Tensor x, const at::Tensor geom_feats, const at::Tensor interval_lengths,
  const at::Tensor interval_starts, int b, int d, int h, int w)
{
  const int n = x.size(0);
  const int c = x.size(1);
  const int n_intervals = interval_lengths.size(0);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));

  const float * x_ptr = x.data_ptr<float>();
  const int * geom_feats_ptr = geom_feats.data_ptr<int>();
  const int * interval_lengths_ptr = interval_lengths.data_ptr<int>();
  const int * interval_starts_ptr = interval_starts.data_ptr<int>();

  auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
  at::Tensor out = torch::zeros({b, d, h, w, c}, options);
  float * out_ptr = out.data_ptr<float>();

  bev_pool(
    b, d, h, w, n, c, n_intervals, x_ptr, geom_feats_ptr, interval_starts_ptr, interval_lengths_ptr,
    out_ptr);

  return out;
}

/**
 * @brief Perform backward pass of BEV pooling for gradient computation.
 *
 * Computes gradients of the input features given the gradient of the output.
 * Since forward pass sums features, backward pass broadcasts the gradient
 * to all points that contributed to each grid cell.
 *
 * @param out_grad Gradient of loss w.r.t. output, shape (B, D, H, W, C).
 * @param geom_feats Geometric coordinates tensor of shape (N, 4).
 * @param interval_lengths Number of points in each pooling interval, shape (n_intervals,).
 * @param interval_starts Starting index of each pooling interval, shape (n_intervals,).
 * @param b Batch size.
 * @param d Depth dimension of output grid.
 * @param h Height dimension of output grid.
 * @param w Width dimension of output grid.
 * @return Gradient tensor for input features, shape (N, C).
 */
at::Tensor bev_pool_backward(
  const at::Tensor out_grad, const at::Tensor geom_feats, const at::Tensor interval_lengths,
  const at::Tensor interval_starts, int b, int d, int h, int w)
{
  const int n = geom_feats.size(0);
  const int c = out_grad.size(4);
  const int n_intervals = interval_lengths.size(0);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(out_grad));

  const float * out_grad_ptr = out_grad.data_ptr<float>();
  const int * geom_feats_ptr = geom_feats.data_ptr<int>();
  const int * interval_lengths_ptr = interval_lengths.data_ptr<int>();
  const int * interval_starts_ptr = interval_starts.data_ptr<int>();

  auto options = torch::TensorOptions().dtype(out_grad.dtype()).device(out_grad.device());
  at::Tensor x_grad = torch::zeros({n, c}, options);
  float * x_grad_ptr = x_grad.data_ptr<float>();

  bev_pool_grad(
    b, d, h, w, n, c, n_intervals, out_grad_ptr, geom_feats_ptr, interval_starts_ptr,
    interval_lengths_ptr, x_grad_ptr);

  return x_grad;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.doc() = "BEV pooling CUDA operations for bird's-eye view feature extraction";
  m.def("bev_pool_forward", &bev_pool_forward, "BEV pooling forward pass (CUDA)");
  m.def("bev_pool_backward", &bev_pool_backward, "BEV pooling backward pass (CUDA)");
}
