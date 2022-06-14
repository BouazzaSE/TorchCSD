#pragma once
#include <curand_kernel.h>
#include <torch/torch.h>

template<typename T>
struct UniformBounds {
    T min;
    T max;
};

template<typename T> void curand_init_states(T*, long, cudaStream_t);
// template<typename T1, typename T2> void generate_diff_direction(T1*, torch::Tensor, torch::Tensor, torch::Tensor, T2, T2, cudaStream_t);

namespace{
    template <typename T> struct type{}; 

    template<typename T1, typename T2>
    __device__ T2 _gen_uniform(T1 *curand_state, type<T2>);

    template<typename T>
    __device__ float _gen_uniform(T *curand_state, type<float>){
        return curand_uniform(curand_state);
    }

    template<typename T>
    __device__ double _gen_uniform(T *curand_state, type<double>){
        return curand_uniform_double(curand_state);
    }

    template<typename T1, typename T2>
    __device__ T2 _gen_uniform(T1 *curand_state){
        return _gen_uniform(curand_state, type<T2>{});
    }

    template<typename T1, typename T2>
    __device__ T2 _gen_normal(T1 *curand_state, type<T2>);

    template<typename T>
    __device__ float _gen_normal(T *curand_state, type<float>){
        return curand_normal(curand_state);
    }

    template<typename T>
    __device__ double _gen_normal(T *curand_state, type<double>){
        return curand_normal_double(curand_state);
    }

    template<typename T1, typename T2>
    __device__ T2 _gen_normal(T1 *curand_state){
        return _gen_normal(curand_state, type<T2>{});
    }
}