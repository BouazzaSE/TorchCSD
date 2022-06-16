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
#include "modules/activations/kernels/kernels.h"

namespace {
    const int ntpb = 512;
    template<typename T, size_t N>
    using TensorRestrict64 = torch::GenericPackedTensorAccessor<T, N, torch::RestrictPtrTraits, int64_t>;
    int64_t zero_arr_3[3] = {0, 0, 0};
}

// The CUDA kernels here basically implement the forward and backward passes through activation functions.
// During the forward pass (kernels ending with _forward), we have the possibility of computing 
// additional expressions with the goal of reusing them during the backward phase. 
// This behaviour is parametrized via the compute_backward template parameter.
// During the backward pass, the Cauchy-Riemann equations allow us to write the chain-rule for PyTorch's automatic 
// differentiation using only du/dx and dv/dx.

template<typename T, bool compute_backward>
__global__ void _csoftplus_forward(const TensorRestrict64<T, 3> x, const TensorRestrict64<T, 3> y, TensorRestrict64<T, 3> u, TensorRestrict64<T, 3> v, TensorRestrict64<T, 3> dudx, TensorRestrict64<T, 3> dvdx) {
    const long tidx = blockDim.x * blockIdx.x + threadIdx.x;
    const long d1 = x.size(0);
    const long d2 = x.size(1);
    const long d3 = x.size(2);
    const long n = d1 * d2 * d3;
    if (tidx < n){
        const long i = tidx / (d2 * d3);
        const long j = (tidx % (d2 * d3)) / d3;
        const long k = tidx % d3;
        T r, sin, cos, _x, a, m;
        sincos(y[i][j][k], &sin, &cos);
        _x = x[i][j][k];
        if(_x<0){
            r = exp(_x);
            m = 1+r*(r+2*cos);
            a = r+cos;
            u[i][j][k] = 0.5*log(m);
            v[i][j][k] = atan2(r*sin, 1+r*cos);
            if(compute_backward){
                dudx[i][j][k] = r*a/m;
                dvdx[i][j][k] = r*sin/m;
            }
        } else {
            r = exp(-_x);
            m = 1+r*(r+2*cos);
            u[i][j][k] = _x+0.5*log(m);
            v[i][j][k] = atan2(sin, r+cos);
            if(compute_backward){
                dudx[i][j][k] = (1+r*cos)/m;
                dvdx[i][j][k] = r*sin/m;
            }
        }
    }
}

template<typename T, bool compute_backward>
void csoftplus_forward(torch::Tensor x, torch::Tensor y, torch::Tensor u, torch::Tensor v, torch::Tensor dudx, torch::Tensor dvdx, cudaStream_t stream){
    const long n = x.size(0)*x.size(1)*x.size(2);
    const auto _x = x.packed_accessor64<T, 3, torch::RestrictPtrTraits>();
    const auto _y = y.packed_accessor64<T, 3, torch::RestrictPtrTraits>();
    auto _u = u.packed_accessor64<T, 3, torch::RestrictPtrTraits>();
    auto _v = v.packed_accessor64<T, 3, torch::RestrictPtrTraits>();
    auto _dudx = TensorRestrict64<T, 3>(nullptr, zero_arr_3, zero_arr_3);
    auto _dvdx = TensorRestrict64<T, 3>(nullptr, zero_arr_3, zero_arr_3);
    if(compute_backward){
        _dudx = dudx.packed_accessor64<T, 3, torch::RestrictPtrTraits>();
        _dvdx = dvdx.packed_accessor64<T, 3, torch::RestrictPtrTraits>();
    }
    _csoftplus_forward<T, compute_backward><<<(n+ntpb-1)/ntpb, ntpb, 0, stream>>>(_x, _y, _u, _v, _dudx, _dvdx);
}

template<typename T>
__global__ void _csoftplus_backward(const TensorRestrict64<T, 3> du, const TensorRestrict64<T, 3> dv, const TensorRestrict64<T, 3> dudx, const TensorRestrict64<T, 3> dvdx, TensorRestrict64<T, 3> dx, TensorRestrict64<T, 3> dy) {
    const long tidx = blockDim.x * blockIdx.x + threadIdx.x;
    const long d1 = du.size(0);
    const long d2 = du.size(1);
    const long d3 = du.size(2);
    const long n = d1 * d2 * d3;
    if (tidx < n){
        const long i = tidx / (d2 * d3);
        const long j = (tidx % (d2 * d3)) / d3;
        const long k = tidx % d3;
        T _dudx, _dvdx, _du, _dv;
        _dudx = dudx[i][j][k];
        _dvdx = dvdx[i][j][k];
        _du = du[i][j][k];
        _dv = dv[i][j][k];
        dx[i][j][k] = _du*_dudx+_dv*_dvdx;
        dy[i][j][k] = -_du*_dvdx+_dv*_dudx;
    }
}

template<typename T>
void csoftplus_backward(torch::Tensor du, torch::Tensor dv, torch::Tensor dudx, torch::Tensor dvdx, torch::Tensor dx, torch::Tensor dy, cudaStream_t stream){
    const long n = du.size(0)*du.size(1)*du.size(2);
    const auto _du = du.packed_accessor64<T, 3, torch::RestrictPtrTraits>();
    const auto _dv = dv.packed_accessor64<T, 3, torch::RestrictPtrTraits>();
    const auto _dudx = dudx.packed_accessor64<T, 3, torch::RestrictPtrTraits>();
    const auto _dvdx = dvdx.packed_accessor64<T, 3, torch::RestrictPtrTraits>();
    auto _dx = dx.packed_accessor64<T, 3, torch::RestrictPtrTraits>();
    auto _dy = dy.packed_accessor64<T, 3, torch::RestrictPtrTraits>();
    _csoftplus_backward<T><<<(n+ntpb-1)/ntpb, ntpb, 0, stream>>>(_du, _dv, _dudx, _dvdx, _dx, _dy);
}

template void csoftplus_forward<float, true>(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, cudaStream_t);
template void csoftplus_forward<float, false>(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, cudaStream_t);
template void csoftplus_forward<double, true>(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, cudaStream_t);
template void csoftplus_forward<double, false>(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, cudaStream_t);
template void csoftplus_backward<float>(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, cudaStream_t);
template void csoftplus_backward<double>(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, cudaStream_t);


template<typename T, bool compute_backward>
__global__ void _csigmoid_forward(const TensorRestrict64<T, 3> x, const TensorRestrict64<T, 3> y, TensorRestrict64<T, 3> u, TensorRestrict64<T, 3> v, TensorRestrict64<T, 3> dudx, TensorRestrict64<T, 3> dvdx) {
    const long tidx = blockDim.x * blockIdx.x + threadIdx.x;
    const long d1 = x.size(0);
    const long d2 = x.size(1);
    const long d3 = x.size(2);
    const long n = d1 * d2 * d3;
    if (tidx < n){
        const long i = tidx / (d2 * d3);
        const long j = (tidx % (d2 * d3)) / d3;
        const long k = tidx % d3;
        T sin, cos, _x, r, m, a;
        sincos(y[i][j][k], &sin, &cos);
        _x = x[i][j][k];
        if(_x<0){
            r = exp(_x);
            m = 1+r*(r+2*cos);
            a = r*(r+cos)/m;
            u[i][j][k] = a;
            v[i][j][k] = r*sin/m;
            if(compute_backward){
                dudx[i][j][k] = r*(2*r+cos)/m-2*a*a;
                dvdx[i][j][k] = r*sin*(1-2*a)/m;
            }
        } else {
            r = exp(-_x);
            m = 1+r*(r+2*cos);
            a = (1+r*cos)/m;
            u[i][j][k] = a;
            v[i][j][k] = r*sin/m;
            if(compute_backward){
                dudx[i][j][k] = r*(r*(2+r*cos)+cos)/(m*m);
                dvdx[i][j][k] = r*sin*(1-2*a)/m;
            }
        }
    }
}

template<typename T, bool compute_backward>
void csigmoid_forward(torch::Tensor x, torch::Tensor y, torch::Tensor u, torch::Tensor v, torch::Tensor dudx, torch::Tensor dvdx, cudaStream_t stream){
    const long n = x.size(0)*x.size(1)*x.size(2);
    const auto _x = x.packed_accessor64<T, 3, torch::RestrictPtrTraits>();
    const auto _y = y.packed_accessor64<T, 3, torch::RestrictPtrTraits>();
    auto _u = u.packed_accessor64<T, 3, torch::RestrictPtrTraits>();
    auto _v = v.packed_accessor64<T, 3, torch::RestrictPtrTraits>();
    auto _dudx = TensorRestrict64<T, 3>(nullptr, zero_arr_3, zero_arr_3);
    auto _dvdx = TensorRestrict64<T, 3>(nullptr, zero_arr_3, zero_arr_3);
    if(compute_backward){
        _dudx = dudx.packed_accessor64<T, 3, torch::RestrictPtrTraits>();
        _dvdx = dvdx.packed_accessor64<T, 3, torch::RestrictPtrTraits>();
    }
    _csigmoid_forward<T, compute_backward><<<(n+ntpb-1)/ntpb, ntpb, 0, stream>>>(_x, _y, _u, _v, _dudx, _dvdx);
}

template<typename T>
__global__ void _csigmoid_backward(const TensorRestrict64<T, 3> du, const TensorRestrict64<T, 3> dv, const TensorRestrict64<T, 3> dudx, const TensorRestrict64<T, 3> dvdx, TensorRestrict64<T, 3> dx, TensorRestrict64<T, 3> dy) {
    const long tidx = blockDim.x * blockIdx.x + threadIdx.x;
    const long d1 = du.size(0);
    const long d2 = du.size(1);
    const long d3 = du.size(2);
    const long n = d1 * d2 * d3;
    if (tidx < n){
        const long i = tidx / (d2 * d3);
        const long j = (tidx % (d2 * d3)) / d3;
        const long k = tidx % d3;
        T _dudx, _dvdx, _du, _dv;
        _dudx = dudx[i][j][k];
        _dvdx = dvdx[i][j][k];
        _du = du[i][j][k];
        _dv = dv[i][j][k];
        dx[i][j][k] = _du*_dudx+_dv*_dvdx;
        dy[i][j][k] = -_du*_dvdx+_dv*_dudx;
    }
}

template<typename T>
void csigmoid_backward(torch::Tensor du, torch::Tensor dv, torch::Tensor dudx, torch::Tensor dvdx, torch::Tensor dx, torch::Tensor dy, cudaStream_t stream){
    const long n = du.size(0)*du.size(1)*du.size(2);
    const auto _du = du.packed_accessor64<T, 3, torch::RestrictPtrTraits>();
    const auto _dv = dv.packed_accessor64<T, 3, torch::RestrictPtrTraits>();
    const auto _dudx = dudx.packed_accessor64<T, 3, torch::RestrictPtrTraits>();
    const auto _dvdx = dvdx.packed_accessor64<T, 3, torch::RestrictPtrTraits>();
    auto _dx = dx.packed_accessor64<T, 3, torch::RestrictPtrTraits>();
    auto _dy = dy.packed_accessor64<T, 3, torch::RestrictPtrTraits>();
    _csigmoid_backward<T><<<(n+ntpb-1)/ntpb, ntpb, 0, stream>>>(_du, _dv, _dudx, _dvdx, _dx, _dy);
}

template void csigmoid_forward<float, true>(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, cudaStream_t);
template void csigmoid_forward<float, false>(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, cudaStream_t);
template void csigmoid_forward<double, true>(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, cudaStream_t);
template void csigmoid_forward<double, false>(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, cudaStream_t);
template void csigmoid_backward<float>(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, cudaStream_t);
template void csigmoid_backward<double>(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, cudaStream_t);
