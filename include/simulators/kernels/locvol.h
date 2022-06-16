/*
Copyright 2022 Bouazza SAADEDDINE

This file is part of TorchCSD.

TorchCSD is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

TorchCSD is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with TorchCSD.  If not, see <https://www.gnu.org/licenses/>.
*/



#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <torch/torch.h>
#include "simulators/kernels/common.h"

template<typename T1, typename T2> void locvol_generate_params(T1*, UniformBounds<T2>, UniformBounds<T2>, UniformBounds<T2>, UniformBounds<T2>, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, long, cudaStream_t);
template<typename T1, typename T2, long num_nodes_x, long num_nodes_y> void locvol_generate_payoffs(T1*, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, T2, long, long, cudaStream_t);
