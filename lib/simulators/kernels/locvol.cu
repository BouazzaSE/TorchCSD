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



#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <torch/torch.h>
#include "simulators/kernels/common.h"
#include "simulators/kernels/locvol.h"

namespace {
    const int ntpb = 512;
    template<typename T, size_t N>
    using TensorRestrict64 = torch::GenericPackedTensorAccessor<T, N, torch::RestrictPtrTraits, int64_t>;
}

template<typename T>
__device__ long _binary_search(T x, T* arr, long low, long high){
    // Generic binary search of x inside an ordered array arr
    long mid;
    while(true){
        mid = (high+low)/2;
        if(mid == low) return mid;
        if(x < arr[mid]){
            high = mid;
        } else if (x > arr[mid]){
            low = mid;
        } else {
            return mid;
        }
    }
}

template<typename T1, typename T2>
__global__ void _locvol_generate_params(T1 *curand_states, UniformBounds<T2> r_b, UniformBounds<T2> stk_b, UniformBounds<T2> mat_b, UniformBounds<T2> lv_b, TensorRestrict64<T2, 1> r_arr, TensorRestrict64<T2, 1> stk_arr, TensorRestrict64<T2, 1> mat_arr, TensorRestrict64<T2, 3> lv_arr, long num_steps_per_year){
    // CUDA kernel to generate random instances of the model and product parameters inside our fixed grid local volatility model
    // Distributions of the parameters are assumed to be uniform
    // Bounds of the uniforms are:
    // - r_b.min and r_b.max for the risk-free rate;
    // - stk_b.min and stk_b.max for the strike;
    // - mat_b.min and mat_b.max for the maturity;
    // - lv_b.min and lv_b.max for each local volatility node;
    // Maturities are assumed to be multiples of 1/num_steps_per_year
    const long tidx = blockDim.x * blockIdx.x + threadIdx.x;
    const long num_samples = r_arr.size(0);
    if(tidx<num_samples){
        T1 curand_state = curand_states[tidx];
        r_arr[tidx] = _gen_uniform<T1, T2>(&curand_state)*(r_b.max-r_b.min)+r_b.min;
        stk_arr[tidx] = _gen_uniform<T1, T2>(&curand_state)*(stk_b.max-stk_b.min)+stk_b.min;
        T2 mat = _gen_uniform<T1, T2>(&curand_state)*(mat_b.max-mat_b.min)+mat_b.min;
        mat_arr[tidx] = ((T2) ((long) (mat*num_steps_per_year)))/num_steps_per_year;
        const long num_nodes_x = lv_arr.size(1);
        const long num_nodes_y = lv_arr.size(2);
        for(int i=0; i<num_nodes_x; i++)
            for(int j=0; j<num_nodes_y; j++){
                lv_arr[tidx][i][j] = _gen_uniform<T1, T2>(&curand_state)*(lv_b.max-lv_b.min)+lv_b.min;
            }
        curand_states[tidx] = curand_state;
    }
}

template<typename T1, typename T2>
void locvol_generate_params(T1 *curand_states, UniformBounds<T2> r_b, UniformBounds<T2> stk_b, UniformBounds<T2> mat_b, UniformBounds<T2> lv_b, torch::Tensor r, torch::Tensor stk, torch::Tensor mat, torch::Tensor lv, long num_steps_per_year, cudaStream_t stream){
    // Wrapper around CUDA kernel _locvol_generate_params
    const long n = r.size(0);
    auto _r = r.packed_accessor64<T2, 1, torch::RestrictPtrTraits>();
    auto _stk = stk.packed_accessor64<T2, 1, torch::RestrictPtrTraits>();
    auto _mat = mat.packed_accessor64<T2, 1, torch::RestrictPtrTraits>();
    auto _lv = lv.packed_accessor64<T2, 3, torch::RestrictPtrTraits>();
    _locvol_generate_params<T1, T2><<<(n+ntpb-1)/ntpb, ntpb, 0, stream>>>(curand_states, r_b, stk_b, mat_b, lv_b, _r, _stk, _mat, _lv, num_steps_per_year);
}

template<typename T1, typename T2, long num_nodes_x, long num_nodes_y>
__global__ void _locvol_generate_payoffs(T1 *curand_states, const TensorRestrict64<T2, 1> r_arr, const TensorRestrict64<T2, 1> stk_arr, const TensorRestrict64<T2, 1> mat_arr, const TensorRestrict64<T2, 3> lv_arr, const TensorRestrict64<T2, 1> nodes_x, const TensorRestrict64<T2, 1> nodes_y, TensorRestrict64<T2, 1> call_payoffs_arr, TensorRestrict64<T2, 1> put_payoffs_arr, TensorRestrict64<T2, 2> call_dpayoffs_arr, TensorRestrict64<T2, 2> put_dpayoffs_arr, T2 S0, long num_steps_per_year, long num_inner_samples) {
    // CUDA kernel to generate paths of our fixed-grid local volatility model, each path corresponding to a random instance of the model and product parameters
    // A simple variance reduction is performed here where instead of projecting one single payoff, 
    // we project an average of num_inner_samples payoffs where num_inner_samples is small (usually 16, at most 32).
    constexpr long num_inputs = 3+num_nodes_x*num_nodes_y;
    T2 dS[num_inputs-1], dpayoff[num_inputs], lv[num_nodes_x][num_nodes_y], _nodes_x[num_nodes_x], _nodes_y[num_nodes_y];
    const long tidx = blockDim.x * blockIdx.x + threadIdx.x;
    const long num_samples = r_arr.size(0);
    if(tidx < num_samples){
        T1 curand_state = curand_states[tidx];
        T2 dt = 1.0/num_steps_per_year;
        T2 sqrt_dt = sqrt(dt);
        T2 r, stk, mat;
        r = r_arr[tidx];
        stk = stk_arr[tidx];
        mat = mat_arr[tidx];
        for(int i=0; i<num_nodes_x; i++) _nodes_x[i] = nodes_x[i];
        for(int i=0; i<num_nodes_y; i++) _nodes_y[i] = nodes_y[i];
        long num_steps = mat*num_steps_per_year;
        for(int i=0; i<num_nodes_x; i++)
            for(int j=0; j<num_nodes_y; j++){
                lv[i][j] = lv_arr[tidx][i][j];
            }
        call_payoffs_arr[tidx] = 0;
        put_payoffs_arr[tidx] = 0;
        for(int i=0; i<num_inputs; i++){
            call_dpayoffs_arr[tidx][i] = 0;
            put_dpayoffs_arr[tidx][i] = 0;
        }
        T2 df = exp(-r*mat);
        T2 t, S_prev, S, Z, call_payoff, put_payoff;
        long _k, _l;
        for(int n=0; n<num_inner_samples; n++){
            S = log(S0);
            for(int i=0; i<num_inputs-1; i++) dS[i] = 0;
            t = 0;
            _k = 0;
            T2 _lv, dpayoffdT, d1;
            for(int i=0; i<num_steps; i++){
                Z = _gen_normal<T1, T2>(&curand_state);
                _k = _binary_search(t, _nodes_x, _k, num_nodes_x);
                _l = _binary_search(S, _nodes_y, 0, num_nodes_y);
                T2 _delta, _dlv, _dx, _dy;
                // Computing derivatives in a forward way
                if((_nodes_x[0]<=t) && (t<_nodes_x[num_nodes_x-1]) && (_nodes_y[0]<=S) && (S<_nodes_y[num_nodes_y-1])){
                    _delta = lv[_k+1][_l]-lv[_k][_l];
                    _dx = _nodes_x[_k+1]-_nodes_x[_k];
                    _lv = _delta*(t-_nodes_x[_k])/_dx+lv[_k][_l];
                    _delta = lv[_k+1][_l+1]-lv[_k][_l+1]-_delta;
                    _dy = _nodes_y[_l+1]-_nodes_y[_l];
                    _delta *= (t-_nodes_x[_k])/_dx;
                    _delta += lv[_k][_l+1]-lv[_k][_l];
                    _delta /= _dy;
                    _lv += (S-_nodes_y[_l])*_delta;
                    // d/dr
                    _dlv = dS[0]*_delta;
                    dS[0] += dt-_lv*_dlv*dt+_dlv*sqrt_dt*Z;
                    // d/dlv
                    _dlv = dS[2+_k*num_nodes_y+_l]*_delta+(_nodes_y[_l+1]-S)*(_nodes_x[_k+1]-t)/(_dx*_dy);
                    dS[2+_k*num_nodes_y+_l] += -_lv*_dlv*dt+_dlv*sqrt_dt*Z;
                    _dlv = dS[2+_k*num_nodes_y+_l+1]*_delta+(S-_nodes_y[_l])*(_nodes_x[_k+1]-t)/(_dx*_dy);
                    dS[2+_k*num_nodes_y+_l+1] += -_lv*_dlv*dt+_dlv*sqrt_dt*Z;
                    _dlv = dS[2+(_k+1)*num_nodes_y+_l]*_delta+(_nodes_y[_l+1]-S)*(t-_nodes_x[_k])/(_dx*_dy);
                    dS[2+(_k+1)*num_nodes_y+_l] += -_lv*_dlv*dt+_dlv*sqrt_dt*Z;
                    _dlv = dS[2+(_k+1)*num_nodes_y+_l+1]*_delta+(S-_nodes_y[_l])*(t-_nodes_x[_k])/(_dx*_dy);
                    dS[2+(_k+1)*num_nodes_y+_l+1] += -_lv*_dlv*dt+_dlv*sqrt_dt*Z;
                } else if((_nodes_y[0]<=S)&&(S<_nodes_y[num_nodes_y-1])){
                    _dy = _nodes_y[_l+1]-_nodes_y[_l];
                    _delta = (lv[_k][_l+1]-lv[_k][_l])/_dy;
                    _lv = lv[_k][_l]+_delta*(S-_nodes_y[_l]);
                    // d/dr
                    _dlv = dS[0]*_delta;
                    dS[0] += dt-_lv*_dlv*dt+_dlv*sqrt_dt*Z;
                    // d/dlv
                    _dlv = dS[2+_k*num_nodes_y+_l]*_delta+(_nodes_y[_l+1]-S)/_dy;
                    dS[2+_k*num_nodes_y+_l] += -_lv*_dlv*dt+_dlv*sqrt_dt*Z;
                    _dlv = dS[2+_k*num_nodes_y+_l+1]*_delta+(S-_nodes_y[_l])/_dy;
                    dS[2+_k*num_nodes_y+_l+1] += -_lv*_dlv*dt+_dlv*sqrt_dt*Z;
                } else if((_nodes_x[0]<=t)&&(t<_nodes_x[num_nodes_x-1])){
                    _dx = _nodes_x[_k+1]-_nodes_x[_k];
                    _delta = (lv[_k+1][_l]-lv[_k][_l])/_dx;
                    _lv = lv[_k][_l]+_delta*(t-_nodes_x[_k]);
                    // d/dr
                    dS[0] += dt;
                    // d/dlv
                    _dlv = (_nodes_x[_k+1]-t)/_dx;
                    dS[2+_k*num_nodes_y+_l] += -_lv*_dlv*dt+_dlv*sqrt_dt*Z;
                    _dlv = (t-_nodes_x[_k])/_dx;
                    dS[2+(_k+1)*num_nodes_y+_l] += -_lv*_dlv*dt+_dlv*sqrt_dt*Z;
                } else {
                    _lv = lv[_k][_l];
                    // d/dr
                    dS[0] += dt;
                    // d/dlv
                    dS[2+_k*num_nodes_y+_l] += -_lv*dt+sqrt_dt*Z;
                }
                if(i==num_steps-1) S_prev = exp(S);
                S += (r-0.5*_lv*_lv)*dt+_lv*sqrt_dt*Z;
                t += dt;
            }
            S = exp(S);
            for(int i=0; i<num_inputs-1; i++){
                dS[i] *= S;
            }
            call_payoff = S>stk ? df*(S-stk) : 0;
            call_payoffs_arr[tidx] += call_payoff;
            for(int i=0; i<num_inputs; i++){
                if(i==2) continue;
                dpayoff[i] = S>stk ? dS[(i>=2)?(i-1):i]*df : 0;
            }
            dpayoff[0] -= mat*call_payoff;
            dpayoff[1] -= S>stk ? df : 0;
            dpayoffdT = -r*call_payoff;
            if(S>stk) dpayoffdT += df*r*S;
            d1 = (log(stk/S_prev)-(r-0.5*_lv*_lv)*dt)/(_lv*sqrt_dt);
            dpayoffdT += 0.5*df*_lv*_lv*S*S*exp(-0.5*d1*d1)/(sqrt(2*M_PI*dt)*_lv*stk);
            dpayoff[2] = dpayoffdT;
            for(int i=0; i<num_inputs; i++) call_dpayoffs_arr[tidx][i] += dpayoff[i];
            put_payoff = stk>S ? df*(stk-S) : 0;
            put_payoffs_arr[tidx] += put_payoff;
            for(int i=0; i<num_inputs; i++){
                if(i==2) continue;
                dpayoff[i] = stk>S ? -dS[(i>=2)?(i-1):i]*df : 0;
            }
            dpayoff[0] -= mat*put_payoff;
            dpayoff[1] += stk>S ? df : 0;
            dpayoff[2] = dpayoffdT-r*df*stk;
            for(int i=0; i<num_inputs; i++) put_dpayoffs_arr[tidx][i] += dpayoff[i];
        }
        call_payoffs_arr[tidx] /= num_inner_samples;
        put_payoffs_arr[tidx] /= num_inner_samples;
        for(int i=0; i<num_inputs; i++){
            call_dpayoffs_arr[tidx][i] /= num_inner_samples;
            put_dpayoffs_arr[tidx][i] /= num_inner_samples;
        }
        curand_states[tidx] = curand_state;
    }
}

template<typename T1, typename T2, long num_nodes_x, long num_nodes_y>
void locvol_generate_payoffs(T1 *curand_states, torch::Tensor r, torch::Tensor stk, torch::Tensor mat, torch::Tensor lv, torch::Tensor nodes_x, torch::Tensor nodes_y, torch::Tensor call_payoffs, torch::Tensor put_payoffs, torch::Tensor call_dpayoffs, torch::Tensor put_dpayoffs, T2 S0, long num_steps_per_year, long num_inner_samples, cudaStream_t stream){
    // Wrapper around CUDA kernel _locvol_generate_payoffs
    const long n = r.size(0);
    auto _r = r.packed_accessor64<T2, 1, torch::RestrictPtrTraits>();
    auto _stk = stk.packed_accessor64<T2, 1, torch::RestrictPtrTraits>();
    auto _mat = mat.packed_accessor64<T2, 1, torch::RestrictPtrTraits>();
    auto _lv = lv.packed_accessor64<T2, 3, torch::RestrictPtrTraits>();
    auto _nodes_x = nodes_x.packed_accessor64<T2, 1, torch::RestrictPtrTraits>();
    auto _nodes_y = nodes_y.packed_accessor64<T2, 1, torch::RestrictPtrTraits>();
    auto _call_payoffs = call_payoffs.packed_accessor64<T2, 1, torch::RestrictPtrTraits>();
    auto _put_payoffs = put_payoffs.packed_accessor64<T2, 1, torch::RestrictPtrTraits>();
    auto _call_dpayoffs = call_dpayoffs.packed_accessor64<T2, 2, torch::RestrictPtrTraits>();
    auto _put_dpayoffs = put_dpayoffs.packed_accessor64<T2, 2, torch::RestrictPtrTraits>();
    _locvol_generate_payoffs<T1, T2, num_nodes_x, num_nodes_y><<<(n+ntpb-1)/ntpb, ntpb, 0, stream>>>(curand_states, _r, _stk, _mat, _lv, _nodes_x, _nodes_y, _call_payoffs, _put_payoffs, _call_dpayoffs, _put_dpayoffs, S0, num_steps_per_year, num_inner_samples);
}

template void locvol_generate_params<curandStateMRG32k3a_t, float>(curandStateMRG32k3a_t*, UniformBounds<float>, UniformBounds<float>, UniformBounds<float>, UniformBounds<float>, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, long, cudaStream_t);
template void locvol_generate_params<curandStateMRG32k3a_t, double>(curandStateMRG32k3a_t*, UniformBounds<double>, UniformBounds<double>, UniformBounds<double>, UniformBounds<double>, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, long, cudaStream_t);
template void locvol_generate_params<curandStatePhilox4_32_10_t, float>(curandStatePhilox4_32_10_t*, UniformBounds<float>, UniformBounds<float>, UniformBounds<float>, UniformBounds<float>, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, long, cudaStream_t);
template void locvol_generate_params<curandStatePhilox4_32_10_t, double>(curandStatePhilox4_32_10_t*, UniformBounds<double>, UniformBounds<double>, UniformBounds<double>, UniformBounds<double>, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, long, cudaStream_t);

template void locvol_generate_payoffs<curandStateMRG32k3a_t, float, 5, 5>(curandStateMRG32k3a_t*, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, float, long, long, cudaStream_t);
template void locvol_generate_payoffs<curandStateMRG32k3a_t, double, 5, 5>(curandStateMRG32k3a_t*, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, long, long, cudaStream_t);
template void locvol_generate_payoffs<curandStatePhilox4_32_10_t, float, 5, 5>(curandStatePhilox4_32_10_t*, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, float, long, long, cudaStream_t);
template void locvol_generate_payoffs<curandStatePhilox4_32_10_t, double, 5, 5>(curandStatePhilox4_32_10_t*, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, long, long, cudaStream_t);

template void locvol_generate_payoffs<curandStateMRG32k3a_t, float, 6, 6>(curandStateMRG32k3a_t*, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, float, long, long, cudaStream_t);
template void locvol_generate_payoffs<curandStateMRG32k3a_t, double, 6, 6>(curandStateMRG32k3a_t*, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, long, long, cudaStream_t);
template void locvol_generate_payoffs<curandStatePhilox4_32_10_t, float, 6, 6>(curandStatePhilox4_32_10_t*, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, float, long, long, cudaStream_t);
template void locvol_generate_payoffs<curandStatePhilox4_32_10_t, double, 6, 6>(curandStatePhilox4_32_10_t*, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, long, long, cudaStream_t);

template void locvol_generate_payoffs<curandStateMRG32k3a_t, float, 8, 8>(curandStateMRG32k3a_t*, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, float, long, long, cudaStream_t);
template void locvol_generate_payoffs<curandStateMRG32k3a_t, double, 8, 8>(curandStateMRG32k3a_t*, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, long, long, cudaStream_t);
template void locvol_generate_payoffs<curandStatePhilox4_32_10_t, float, 8, 8>(curandStatePhilox4_32_10_t*, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, float, long, long, cudaStream_t);
template void locvol_generate_payoffs<curandStatePhilox4_32_10_t, double, 8, 8>(curandStatePhilox4_32_10_t*, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, long, long, cudaStream_t);

template void locvol_generate_payoffs<curandStateMRG32k3a_t, float, 9, 9>(curandStateMRG32k3a_t*, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, float, long, long, cudaStream_t);
template void locvol_generate_payoffs<curandStateMRG32k3a_t, double, 9, 9>(curandStateMRG32k3a_t*, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, long, long, cudaStream_t);
template void locvol_generate_payoffs<curandStatePhilox4_32_10_t, float, 9, 9>(curandStatePhilox4_32_10_t*, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, float, long, long, cudaStream_t);
template void locvol_generate_payoffs<curandStatePhilox4_32_10_t, double, 9, 9>(curandStatePhilox4_32_10_t*, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, long, long, cudaStream_t);
