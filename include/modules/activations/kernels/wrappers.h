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
#include "modules/activations/kernels/kernels.h"

namespace csd {
    namespace modules {
        namespace activations{
            namespace kernels {
                namespace csoftplus{
                    // TODO: automate the boilerplate below
                    template<typename T, bool compute_backward>
                    void forward(torch::Tensor x, torch::Tensor y, torch::Tensor u, torch::Tensor v, torch::Tensor dudx, torch::Tensor dvdx, cudaStream_t stream){
                        csoftplus_forward<T, compute_backward>(x, y, u, v, dudx, dvdx, stream);
                    }
                    template<typename T>
                    void backward(torch::Tensor du, torch::Tensor dv, torch::Tensor dudx, torch::Tensor dvdx, torch::Tensor dx, torch::Tensor dy, cudaStream_t stream){
                        csoftplus_backward<T>(du, dv, dudx, dvdx, dx, dy, stream);
                    }
                }
                namespace csigmoid{
                    // TODO: automate the boilerplate below
                    template<typename T, bool compute_backward>
                    void forward(torch::Tensor x, torch::Tensor y, torch::Tensor u, torch::Tensor v, torch::Tensor dudx, torch::Tensor dvdx, cudaStream_t stream){
                        csigmoid_forward<T, compute_backward>(x, y, u, v, dudx, dvdx, stream);
                    }
                    template<typename T>
                    void backward(torch::Tensor du, torch::Tensor dv, torch::Tensor dudx, torch::Tensor dvdx, torch::Tensor dx, torch::Tensor dy, cudaStream_t stream){
                        csigmoid_backward<T>(du, dv, dudx, dvdx, dx, dy, stream);
                    }
                }
            }
        }
    }
}