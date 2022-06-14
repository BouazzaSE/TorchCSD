#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <torch/torch.h>
#include "simulators/kernels/common.h"

namespace {
    const int ntpb = 512;
    template<typename T, size_t N>
    using TensorRestrict64 = torch::GenericPackedTensorAccessor<T, N, torch::RestrictPtrTraits, int64_t>;
}

template<typename T>
__global__ void _curand_init_states(T *curand_states, long n){
    const long tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if(tidx < n) curand_init(0, tidx, 0, &curand_states[tidx]);
}

template<typename T>
void curand_init_states(T *curand_states, long n, cudaStream_t stream){
    _curand_init_states<<<(n+ntpb-1)/ntpb, ntpb, 0, stream>>>(curand_states, n);
}

template void curand_init_states<curandStateMRG32k3a_t>(curandStateMRG32k3a_t*, long, cudaStream_t);
template void curand_init_states<curandStatePhilox4_32_10_t>(curandStatePhilox4_32_10_t*, long, cudaStream_t);
