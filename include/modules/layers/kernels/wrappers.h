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
#include <torch/torch.h>
#include <cuda_runtime.h>
#include "modules/layers/kernels/kernels.h"

namespace csd {
    namespace modules {
        namespace layers{
            namespace kernels {
                namespace ccall_layer{
                    template<typename T, bool compute_backward>
                    void forward(torch::Tensor x, torch::Tensor y, torch::Tensor a, torch::Tensor b, torch::Tensor u, torch::Tensor v, torch::Tensor dudx1, torch::Tensor dvdx1, T S, cudaStream_t stream){
                        ccall_layer_forward<T, compute_backward>(x, y, a, b, u, v, dudx1, dvdx1, S, stream);
                    }
                    template<typename T>
                    void backward(torch::Tensor du, torch::Tensor dv, torch::Tensor dudx1, torch::Tensor dvdx1, torch::Tensor dx, torch::Tensor dy, T S, cudaStream_t stream){
                        ccall_layer_backward<T>(du, dv, dudx1, dvdx1, dx, dy, S, stream);
                    }
                }
                namespace cput_from_call_layer{
                    template<typename T>
                    void forward(torch::Tensor x, torch::Tensor y, torch::Tensor a, torch::Tensor b, torch::Tensor u, torch::Tensor v, T S, cudaStream_t stream){
                        cput_from_call_layer_forward(x, y, a, b, u, v, S, stream);
                    }
                }
            }
        }
    }
}