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



#include <cuda_runtime.h>
#include <torch/torch.h>
#include "modules/layers/kernels/kernels.h"

namespace {
    const int ntpb = 512;
    template<typename T, size_t N>
    using TensorRestrict64 = torch::GenericPackedTensorAccessor<T, N, torch::RestrictPtrTraits, int64_t>;
    int64_t zero_arr_2[2] = {0, 0};
}

// The CUDA kernels here basically implement the forward and backward passes through certain layer functions.
// During the forward pass (kernels ending with _forward), we have the possibility of computing 
// additional expressions with the goal of reusing them during the backward phase. 
// This behaviour is parametrized via the compute_backward template parameter.
// During the backward pass, the Cauchy-Riemann equations allow us to write the chain-rule for PyTorch's automatic 
// differentiation using only du/dx and dv/dx.

template<typename T, bool compute_backward>
__global__ void _ccall_layer_forward(const TensorRestrict64<T, 3> x, const TensorRestrict64<T, 3> y, const TensorRestrict64<T, 3> a, const TensorRestrict64<T, 3> b, TensorRestrict64<T, 2> u, TensorRestrict64<T, 2> v, TensorRestrict64<T, 2> dudx1, TensorRestrict64<T, 2> dvdx1, T S) {
    const long tidx = blockDim.x * blockIdx.x + threadIdx.x;
    const long d1 = x.size(0);
    const long d2 = x.size(1);
    const long n = d1 * d2;
    if (tidx < n){
        const long i = tidx / d2;
        const long j = tidx % d2;
        T x_0 = x[i][j][0]; T y_0 = y[i][j][0];
        T x_1 = x[i][j][1]; T y_1 = y[i][j][1];
        T a_0 = a[i][j][0]; T b_0 = b[i][j][0];
        T a_1 = a[i][j][1]; T b_1 = b[i][j][1];
        T a_2 = a[i][j][2]; T b_2 = b[i][j][2];
        T A_0 = a_0*b_2 + a_2*b_0;
        T A_1 = sin(A_0);
        T A_2 = exp(-a_0*a_2 + b_0*b_2);
        T A_3 = A_2*(a_1*y_1 + b_1*x_1);
        T A_4 = cos(A_0);
        T A_5 = A_2*(a_1*x_1 - b_1*y_1);
        u[i][j] = -A_1*A_3 - A_4*A_5 + S*x_0;
        v[i][j] = A_1*A_5 - A_3*A_4 + S*y_0;
        if(compute_backward) {
            dudx1[i][j] = -A_2*(A_1*b_1 + A_4*a_1); // only storing dudx1 since dudx0=S
            dvdx1[i][j] = A_2*(A_1*a_1 - A_4*b_1); // only storing dvdx1 since dvdx0=0
        }
    }
}

template<typename T, bool compute_backward>
void ccall_layer_forward(torch::Tensor x, torch::Tensor y, torch::Tensor a, torch::Tensor b, torch::Tensor u, torch::Tensor v, torch::Tensor dudx1, torch::Tensor dvdx1, T S, cudaStream_t stream){
    const long n = x.size(0)*x.size(1);
    const auto _x = x.packed_accessor64<T, 3, torch::RestrictPtrTraits>();
    const auto _y = y.packed_accessor64<T, 3, torch::RestrictPtrTraits>();
    const auto _a = a.packed_accessor64<T, 3, torch::RestrictPtrTraits>();
    const auto _b = b.packed_accessor64<T, 3, torch::RestrictPtrTraits>();
    auto _u = u.packed_accessor64<T, 2, torch::RestrictPtrTraits>();
    auto _v = v.packed_accessor64<T, 2, torch::RestrictPtrTraits>();
    auto _dudx1 = TensorRestrict64<T, 2>(nullptr, zero_arr_2, zero_arr_2);
    auto _dvdx1 = TensorRestrict64<T, 2>(nullptr, zero_arr_2, zero_arr_2);
    if(compute_backward){
        _dudx1 = dudx1.packed_accessor64<T, 2, torch::RestrictPtrTraits>();
        _dvdx1 = dvdx1.packed_accessor64<T, 2, torch::RestrictPtrTraits>();
    }
    _ccall_layer_forward<T, compute_backward><<<(n+ntpb-1)/ntpb, ntpb, 0, stream>>>(_x, _y, _a, _b, _u, _v, _dudx1, _dvdx1, S);
}

template<typename T>
__global__ void _ccall_layer_backward(const TensorRestrict64<T, 2> du, const TensorRestrict64<T, 2> dv, const TensorRestrict64<T, 2> dudx1, const TensorRestrict64<T, 2> dvdx1, TensorRestrict64<T, 3> dx, TensorRestrict64<T, 3> dy, T S) {
    const long tidx = blockDim.x * blockIdx.x + threadIdx.x;
    const long d1 = du.size(0);
    const long d2 = du.size(1);
    const long n = d1 * d2;
    if (tidx < n){
        const long i = tidx / d2;
        const long j = tidx % d2;
        T _dudx1 = dudx1[i][j];
        T _dvdx1 = dvdx1[i][j];
        T _du = du[i][j];
        T _dv = dv[i][j];
        dx[i][j][0] = _du*S; dx[i][j][1] = _du*_dudx1+_dv*_dvdx1;
        dy[i][j][0] = _dv*S; dy[i][j][1] = -_du*_dvdx1+_dv*_dudx1;
    }
}

template<typename T>
void ccall_layer_backward(torch::Tensor du, torch::Tensor dv, torch::Tensor dudx1, torch::Tensor dvdx1, torch::Tensor dx, torch::Tensor dy, T S, cudaStream_t stream){
    const long n = du.size(0)*du.size(1);
    const auto _du = du.packed_accessor64<T, 2, torch::RestrictPtrTraits>();
    const auto _dv = dv.packed_accessor64<T, 2, torch::RestrictPtrTraits>();
    const auto _dudx1 = dudx1.packed_accessor64<T, 2, torch::RestrictPtrTraits>();
    const auto _dvdx1 = dvdx1.packed_accessor64<T, 2, torch::RestrictPtrTraits>();
    auto _dx = dx.packed_accessor64<T, 3, torch::RestrictPtrTraits>();
    auto _dy = dy.packed_accessor64<T, 3, torch::RestrictPtrTraits>();
    _ccall_layer_backward<T><<<(n+ntpb-1)/ntpb, ntpb, 0, stream>>>(_du, _dv, _dudx1, _dvdx1, _dx, _dy, S);
}

template void ccall_layer_forward<float, true>(torch::Tensor x, torch::Tensor y, torch::Tensor a, torch::Tensor b, torch::Tensor u, torch::Tensor v, torch::Tensor dudx1, torch::Tensor dvdx1, float S, cudaStream_t stream);
template void ccall_layer_forward<float, false>(torch::Tensor x, torch::Tensor y, torch::Tensor a, torch::Tensor b, torch::Tensor u, torch::Tensor v, torch::Tensor dudx1, torch::Tensor dvdx1, float S, cudaStream_t stream);
template void ccall_layer_forward<double, true>(torch::Tensor x, torch::Tensor y, torch::Tensor a, torch::Tensor b, torch::Tensor u, torch::Tensor v, torch::Tensor dudx1, torch::Tensor dvdx1, double S, cudaStream_t stream);
template void ccall_layer_forward<double, false>(torch::Tensor x, torch::Tensor y, torch::Tensor a, torch::Tensor b, torch::Tensor u, torch::Tensor v, torch::Tensor dudx1, torch::Tensor dvdx1, double S, cudaStream_t stream);
template void ccall_layer_backward<float>(torch::Tensor du, torch::Tensor dv, torch::Tensor dudx1, torch::Tensor dvdx1, torch::Tensor dx, torch::Tensor dy, float S, cudaStream_t stream);
template void ccall_layer_backward<double>(torch::Tensor du, torch::Tensor dv, torch::Tensor dudx1, torch::Tensor dvdx1, torch::Tensor dx, torch::Tensor dy, double S, cudaStream_t stream);

template<typename T>
__global__ void _cput_from_call_layer_forward(const TensorRestrict64<T, 1> x, const TensorRestrict64<T, 1> y, const TensorRestrict64<T, 2> a, const TensorRestrict64<T, 2> b, TensorRestrict64<T, 1> u, TensorRestrict64<T, 1> v, T S) {
    const long tidx = blockDim.x * blockIdx.x + threadIdx.x;
    const long n = x.size(0);
    if (tidx < n){
        T _x = x[tidx]; T _y = y[tidx];
        T a_0 = a[tidx][0]; T b_0 = b[tidx][0];
        T a_1 = a[tidx][1]; T b_1 = b[tidx][1];
        T a_2 = a[tidx][2]; T b_2 = b[tidx][2];
        T A_0 = a_0*b_2 + a_2*b_0;
        T A_1 = exp(-a_0*a_2 + b_0*b_2);
        T A_2 = A_1*cos(A_0);
        T A_3 = A_1*sin(A_0);
        u[tidx] = A_2*a_1 + A_3*b_1 - S + _x;
        v[tidx] = A_2*b_1 - A_3*a_1 + _y;
    }
}

template<typename T>
void cput_from_call_layer_forward(torch::Tensor x, torch::Tensor y, torch::Tensor a, torch::Tensor b, torch::Tensor u, torch::Tensor v, T S, cudaStream_t stream){
    const long n = x.size(0);
    const auto _x = x.packed_accessor64<T, 1, torch::RestrictPtrTraits>();
    const auto _y = y.packed_accessor64<T, 1, torch::RestrictPtrTraits>();
    const auto _a = a.packed_accessor64<T, 2, torch::RestrictPtrTraits>();
    const auto _b = b.packed_accessor64<T, 2, torch::RestrictPtrTraits>();
    const auto _u = u.packed_accessor64<T, 1, torch::RestrictPtrTraits>();
    const auto _v = v.packed_accessor64<T, 1, torch::RestrictPtrTraits>();
    _cput_from_call_layer_forward<T><<<(n+ntpb-1)/ntpb, ntpb, 0, stream>>>(_x, _y, _a, _b, _u, _v, S);
}

template void cput_from_call_layer_forward<float>(torch::Tensor x, torch::Tensor y, torch::Tensor a, torch::Tensor b, torch::Tensor u, torch::Tensor v, float S, cudaStream_t stream);
template void cput_from_call_layer_forward<double>(torch::Tensor x, torch::Tensor y, torch::Tensor a, torch::Tensor b, torch::Tensor u, torch::Tensor v, double S, cudaStream_t stream);
