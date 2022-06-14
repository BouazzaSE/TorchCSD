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