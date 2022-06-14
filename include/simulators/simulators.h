#pragma once
#include <torch/torch.h>
#include "simulators/kernels/common.h"
#include "simulators/kernels/locvol.h"

namespace csd {
    namespace simulators {
            namespace kernels {
                namespace curand {
                    template<typename T>
                    void init_states(T* curand_states, long n, cudaStream_t stream){
                        curand_init_states(curand_states, n, stream);
                    }
                }
                namespace locvol {
                    template<typename T1, typename T2>
                    void generate_params(T1 *curand_states, UniformBounds<T2> r_b, UniformBounds<T2> stk_b, UniformBounds<T2> mat_b, UniformBounds<T2> lv_b, torch::Tensor r, torch::Tensor stk, torch::Tensor mat, torch::Tensor lv, long num_steps_per_year, cudaStream_t stream){
                        locvol_generate_params<T1, T2>(curand_states, r_b, stk_b, mat_b, lv_b, r, stk, mat, lv, num_steps_per_year, stream);
                    }

                    template<typename T1, typename T2, long num_nodes_x, long num_nodes_y>
                    void generate_payoffs(T1 *curand_states, torch::Tensor r, torch::Tensor stk, torch::Tensor mat, torch::Tensor lv, torch::Tensor nodes_x, torch::Tensor nodes_y, torch::Tensor call_payoffs, torch::Tensor put_payoffs, torch::Tensor call_dpayoffs, torch::Tensor put_dpayoffs, T2 S0, long num_steps_per_year, long num_inner_samples, cudaStream_t stream){
                        locvol_generate_payoffs<T1, T2, num_nodes_x, num_nodes_y>(curand_states, r, stk, mat, lv, nodes_x, nodes_y, call_payoffs, put_payoffs, call_dpayoffs, put_dpayoffs, S0, num_steps_per_year, num_inner_samples, stream);
                    }
                }
            }
    }
}