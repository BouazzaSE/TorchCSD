#pragma once
#include <cuda_runtime.h>
#include <torch/torch.h>

template<typename T, bool compute_backward> void csoftplus_forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, cudaStream_t);
template<typename T> void csoftplus_backward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, cudaStream_t);

template<typename T, bool compute_backward> void csigmoid_forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, cudaStream_t);
template<typename T> void csigmoid_backward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, cudaStream_t);
