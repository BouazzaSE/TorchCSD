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