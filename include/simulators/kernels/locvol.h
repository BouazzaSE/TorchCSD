#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <torch/torch.h>
#include "simulators/kernels/common.h"

template<typename T1, typename T2> void locvol_generate_params(T1*, UniformBounds<T2>, UniformBounds<T2>, UniformBounds<T2>, UniformBounds<T2>, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, long, cudaStream_t);
template<typename T1, typename T2, long num_nodes_x, long num_nodes_y> void locvol_generate_payoffs(T1*, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, T2, long, long, cudaStream_t);
