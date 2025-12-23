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

/**
 * @file bev_pool_cuda.cu
 * @brief CUDA kernels for BEV (Bird's Eye View) pooling operations.
 *
 * This file implements efficient GPU kernels for pooling image features
 * into a bird's-eye view grid representation, commonly used in 3D perception
 * tasks for autonomous driving.
 */

namespace
{
/// Number of threads per CUDA block for kernel launches.
constexpr int kThreadsPerBlock = 256;

/// Number of coordinates per geometric feature (height, width, depth, batch).
constexpr int kGeomFeatureDim = 4;
}  // namespace

/**
 * @brief CUDA kernel for BEV pooling forward pass.
 *
 * Each thread processes one (interval, channel) pair. Features from points
 * in the same interval are summed and written to the corresponding BEV grid cell.
 *
 * @param b Batch size.
 * @param d Depth dimension of output grid.
 * @param h Height dimension of output grid.
 * @param w Width dimension of output grid.
 * @param n Total number of input points (unused but kept for API compatibility).
 * @param c Number of feature channels.
 * @param n_intervals Number of unique pooling intervals.
 * @param x Input features, shape (N, C).
 * @param geom_feats Geometric coordinates, shape (N, 4).
 * @param interval_starts Starting index of each interval.
 * @param interval_lengths Number of points in each interval.
 * @param out Output tensor, shape (B, D, H, W, C).
 */
__global__ void bev_pool_kernel(
  [[maybe_unused]] int b, int d, int h, int w, [[maybe_unused]] int n, int c, int n_intervals,
  const float * __restrict__ x, const int * __restrict__ geom_feats,
  const int * __restrict__ interval_starts, const int * __restrict__ interval_lengths,
  float * __restrict__ out)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int interval_idx = idx / c;
  const int channel_idx = idx % c;

  if (interval_idx >= n_intervals) {
    return;
  }

  const int interval_start = interval_starts[interval_idx];
  const int interval_length = interval_lengths[interval_idx];

  const int * cur_geom_feats = geom_feats + interval_start * kGeomFeatureDim;
  const float * cur_x = x + interval_start * c + channel_idx;

  // Compute output index: geom_feats format is (h_idx, w_idx, d_idx, b_idx)
  const int h_idx = cur_geom_feats[0];
  const int w_idx = cur_geom_feats[1];
  const int d_idx = cur_geom_feats[2];
  const int b_idx = cur_geom_feats[3];

  float * cur_out =
    out + b_idx * d * h * w * c + d_idx * h * w * c + h_idx * w * c + w_idx * c + channel_idx;

  // Sum features from all points in this interval
  float sum = 0.0f;
  for (int i = 0; i < interval_length; ++i) {
    sum += cur_x[i * c];
  }
  *cur_out = sum;
}

/**
 * @brief CUDA kernel for BEV pooling backward pass.
 *
 * Broadcasts the gradient from each BEV grid cell to all points that
 * contributed to that cell during the forward pass.
 *
 * @param b Batch size.
 * @param d Depth dimension of output grid.
 * @param h Height dimension of output grid.
 * @param w Width dimension of output grid.
 * @param n Total number of input points (unused but kept for API compatibility).
 * @param c Number of feature channels.
 * @param n_intervals Number of unique pooling intervals.
 * @param out_grad Gradient of loss w.r.t. output, shape (B, D, H, W, C).
 * @param geom_feats Geometric coordinates, shape (N, 4).
 * @param interval_starts Starting index of each interval.
 * @param interval_lengths Number of points in each interval.
 * @param x_grad Output gradient tensor, shape (N, C).
 */
__global__ void bev_pool_grad_kernel(
  [[maybe_unused]] int b, int d, int h, int w, [[maybe_unused]] int n, int c, int n_intervals,
  const float * __restrict__ out_grad, const int * __restrict__ geom_feats,
  const int * __restrict__ interval_starts, const int * __restrict__ interval_lengths,
  float * __restrict__ x_grad)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int interval_idx = idx / c;
  const int channel_idx = idx % c;

  if (interval_idx >= n_intervals) {
    return;
  }

  const int interval_start = interval_starts[interval_idx];
  const int interval_length = interval_lengths[interval_idx];

  const int * cur_geom_feats = geom_feats + interval_start * kGeomFeatureDim;
  float * cur_x_grad = x_grad + interval_start * c + channel_idx;

  // Compute output gradient index
  const int h_idx = cur_geom_feats[0];
  const int w_idx = cur_geom_feats[1];
  const int d_idx = cur_geom_feats[2];
  const int b_idx = cur_geom_feats[3];

  const float * cur_out_grad =
    out_grad + b_idx * d * h * w * c + d_idx * h * w * c + h_idx * w * c + w_idx * c + channel_idx;

  // Broadcast gradient to all points in the interval
  const float grad_value = *cur_out_grad;
  for (int i = 0; i < interval_length; ++i) {
    cur_x_grad[i * c] = grad_value;
  }
}

/**
 * @brief Launch BEV pooling forward kernel.
 *
 * @param b Batch size.
 * @param d Depth dimension.
 * @param h Height dimension.
 * @param w Width dimension.
 * @param n Number of input points.
 * @param c Number of channels.
 * @param n_intervals Number of pooling intervals.
 * @param x Input features pointer.
 * @param geom_feats Geometric features pointer.
 * @param interval_starts Interval start indices pointer.
 * @param interval_lengths Interval lengths pointer.
 * @param out Output tensor pointer.
 */
void bev_pool(
  int b, int d, int h, int w, int n, int c, int n_intervals, const float * x,
  const int * geom_feats, const int * interval_starts, const int * interval_lengths, float * out)
{
  const int total_threads = n_intervals * c;
  const int num_blocks = (total_threads + kThreadsPerBlock - 1) / kThreadsPerBlock;

  bev_pool_kernel<<<num_blocks, kThreadsPerBlock>>>(
    b, d, h, w, n, c, n_intervals, x, geom_feats, interval_starts, interval_lengths, out);
}

/**
 * @brief Launch BEV pooling backward kernel.
 *
 * @param b Batch size.
 * @param d Depth dimension.
 * @param h Height dimension.
 * @param w Width dimension.
 * @param n Number of input points.
 * @param c Number of channels.
 * @param n_intervals Number of pooling intervals.
 * @param out_grad Output gradient pointer.
 * @param geom_feats Geometric features pointer.
 * @param interval_starts Interval start indices pointer.
 * @param interval_lengths Interval lengths pointer.
 * @param x_grad Input gradient pointer (output).
 */
void bev_pool_grad(
  int b, int d, int h, int w, int n, int c, int n_intervals, const float * out_grad,
  const int * geom_feats, const int * interval_starts, const int * interval_lengths, float * x_grad)
{
  const int total_threads = n_intervals * c;
  const int num_blocks = (total_threads + kThreadsPerBlock - 1) / kThreadsPerBlock;

  bev_pool_grad_kernel<<<num_blocks, kThreadsPerBlock>>>(
    b, d, h, w, n, c, n_intervals, out_grad, geom_feats, interval_starts, interval_lengths, x_grad);
}
