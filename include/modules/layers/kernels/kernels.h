#pragma once
#include <cuda_runtime.h>
#include <torch/torch.h>

template<typename T, bool compute_backward> void ccall_layer_forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, T, cudaStream_t);
template<typename T> void ccall_layer_backward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, T, cudaStream_t);

template<typename T> void cput_from_call_layer_forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, T, cudaStream_t);
