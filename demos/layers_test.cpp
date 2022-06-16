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



#include <ATen/Context.h>
#include <ATen/Functions.h>
#include <ATen/core/TensorBody.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#include <iomanip>
#include <iostream>
#include <torch/torch.h>
#include <torch/types.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "modules/activations/activations.h"
#include "modules/layers/layers.h"
#include "modules/modules.h"
#include "nets/vanilla_locvol/vanilla_locvol.h"
#include "simulators/simulators.h"
#include "utils/parse_args.h"

void AffineCSDTest(int device_idx){
    torch::manual_seed(0);
    auto device = torch::Device(torch::kCUDA, device_idx);
    auto affine = csd::modules::layers::ComplexRealAffine(4, 8);
    affine->to(device);
    auto x = torch::randn({16, 4}, torch::dtype(torch::kFloat32).device(device));
    auto y = torch::randn({16, 4}, torch::dtype(torch::kFloat32).device(device));
    auto uv = affine->forward_complex(x, y);
    std::cout << uv.first << std::endl;
    std::cout << uv.second << std::endl;
}

void AffineCSDGradTest(int device_idx){
    torch::manual_seed(0);
    auto device = torch::Device(torch::kCUDA, device_idx);
    auto affine = csd::modules::layers::ComplexRealAffine(4, 8);
    affine->to(device);
    auto x = torch::randn({16, 4}, torch::dtype(torch::kFloat32).device(device)).requires_grad_();
    auto y = torch::randn({16, 4}, torch::dtype(torch::kFloat32).device(device)).requires_grad_();
    auto uv = affine->forward_complex(x, y);
    (2*uv.first+5*uv.second).square().mean().backward();
    std::cout << x.grad() << std::endl;
    std::cout << y.grad() << std::endl;
}

void ComplexSigmoidTest(int device_idx){
    torch::manual_seed(0);
    auto device = torch::Device(torch::kCUDA, device_idx);
    auto sigmoid = csd::modules::activations::ComplexSigmoid();
    auto x = torch::randn({2, 16, 4}, torch::dtype(torch::kFloat32).device(device));
    auto y = torch::randn({2, 16, 4}, torch::dtype(torch::kFloat32).device(device));
    auto uv = sigmoid(x, y);
    std::cout << uv.first << std::endl;
    std::cout << uv.second << std::endl;
}

void ComplexSigmoidGradTest(int device_idx){
    torch::manual_seed(0);
    auto device = torch::Device(torch::kCUDA, device_idx);
    auto sigmoid = csd::modules::activations::ComplexSigmoid();
    auto x = torch::randn({2, 16, 4}, torch::dtype(torch::kFloat32).device(device)).requires_grad_();
    auto y = torch::randn({2, 16, 4}, torch::dtype(torch::kFloat32).device(device)).requires_grad_();
    auto uv = sigmoid(x, y);
    (2*uv.first+5*uv.second).square().mean().backward();
    std::cout << x.grad() << std::endl;
    std::cout << y.grad() << std::endl;
}

void ComplexSoftplusTest(int device_idx){
    torch::manual_seed(0);
    auto device = torch::Device(torch::kCUDA, device_idx);
    auto softplus = csd::modules::activations::ComplexSoftplus();
    auto x = torch::randn({2, 16, 4}, torch::dtype(torch::kFloat32).device(device));
    auto y = torch::randn({2, 16, 4}, torch::dtype(torch::kFloat32).device(device));
    auto uv = softplus(x, y);
    std::cout << uv.first << std::endl;
    std::cout << uv.second << std::endl;
}

void ComplexSoftplusGradTest(int device_idx){
    torch::manual_seed(0);
    auto device = torch::Device(torch::kCUDA, device_idx);
    auto softplus = csd::modules::activations::ComplexSoftplus();
    auto x = torch::randn({2, 16, 4}, torch::dtype(torch::kFloat32).device(device)).requires_grad_();
    auto y = torch::randn({2, 16, 4}, torch::dtype(torch::kFloat32).device(device)).requires_grad_();
    auto uv = softplus(x, y);
    (2*uv.first+5*uv.second).square().mean().backward();
    std::cout << x.grad() << std::endl;
    std::cout << y.grad() << std::endl;
}

void CallLayerTest(int device_idx){
    torch::manual_seed(0);
    auto device = torch::Device(torch::kCUDA, device_idx);
    auto call_layer = csd::modules::layers::ComplexCallLayer();
    auto x = torch::randn({2, 16, 4}, torch::dtype(torch::kFloat32).device(device));
    auto y = torch::randn({2, 16, 4}, torch::dtype(torch::kFloat32).device(device));
    auto u = torch::randn({2, 16, 2}, torch::dtype(torch::kFloat32).device(device));
    auto v = torch::randn({2, 16, 2}, torch::dtype(torch::kFloat32).device(device));
    auto uv = call_layer->forward_complex(u, v, x, y, 1);
    std::cout << uv.first << std::endl;
    std::cout << uv.second << std::endl;
}

void CallLayerGradTest(int device_idx){
    torch::manual_seed(0);
    auto device = torch::Device(torch::kCUDA, device_idx);
    auto call_layer = csd::modules::layers::ComplexCallLayer();
    auto x = torch::randn({2, 16, 4}, torch::dtype(torch::kFloat32).device(device));
    auto y = torch::randn({2, 16, 4}, torch::dtype(torch::kFloat32).device(device));
    auto u = torch::randn({2, 16, 2}, torch::dtype(torch::kFloat32).device(device)).requires_grad_();
    auto v = torch::randn({2, 16, 2}, torch::dtype(torch::kFloat32).device(device)).requires_grad_();
    auto uv = call_layer->forward_complex(u, v, x, y, 1);
    (2*uv.first+5*uv.second).square().mean().backward();
    std::cout << u.grad() << std::endl;
    std::cout << v.grad() << std::endl;
}

void StandardizeTest(int device_idx){
    torch::manual_seed(0);
    auto device = torch::Device(torch::kCUDA, device_idx);
    auto x = torch::randn({16, 4}, torch::dtype(torch::kFloat32).device(device));
    auto x_mean = torch::randn({1, 4}, torch::dtype(torch::kFloat32).device(device));
    auto x_std = torch::randn({1, 4}, torch::dtype(torch::kFloat32).device(device));
    auto a = torch::randn({1, 4}, torch::dtype(torch::kFloat32).device(device));
    auto y = torch::randn({2, 16, 4}, torch::dtype(torch::kFloat32).device(device));
    auto alpha = (x.select(-1, 2).unsqueeze(-1) > a) / x_std;
    auto u = (x - x_mean)*alpha;
    auto v = y*alpha;
    assert(y.dim() > x.dim());
    x = x.expand_as(y);
    u = u.expand_as(y);
    std::cout << u << std::endl;
    std::cout << v << std::endl;
}

void PutFromCallLayerTest(int device_idx){
    torch::manual_seed(0);
    auto device = torch::Device(torch::kCUDA, device_idx);
    auto put_from_call_layer = csd::modules::layers::ComplexPutFromCallLayer();
    auto x = torch::randn({16, 4}, torch::dtype(torch::kFloat32).device(device));
    auto y = torch::randn({16, 4}, torch::dtype(torch::kFloat32).device(device));
    auto u = torch::randn({16, 1}, torch::dtype(torch::kFloat32).device(device));
    auto v = torch::randn({16, 1}, torch::dtype(torch::kFloat32).device(device));
    auto uv = put_from_call_layer->forward_complex(u.select(-1, 0), v.select(-1, 0), x, y, 1);
    std::cout << uv.first << std::endl;
    std::cout << uv.second << std::endl;
}

void PutFromCallLayerGradTest(int device_idx){
    torch::manual_seed(0);
    auto device = torch::Device(torch::kCUDA, device_idx);
    auto put_from_call_layer = csd::modules::layers::ComplexPutFromCallLayer();
    auto x = torch::randn({16, 4}, torch::dtype(torch::kFloat32).device(device));
    auto y = torch::randn({16, 4}, torch::dtype(torch::kFloat32).device(device));
    auto u = torch::randn({16, 1}, torch::dtype(torch::kFloat32).device(device)).requires_grad_();
    auto v = torch::randn({16, 1}, torch::dtype(torch::kFloat32).device(device)).requires_grad_();
    auto uv = put_from_call_layer->forward_complex(u.select(-1, 0), v.select(-1, 0), x, y, 1);
    (2*uv.first+5*uv.second).square().mean().backward();
    std::cout << u.grad() << std::endl;
    std::cout << v.grad() << std::endl;
}

void VanillaLocVolTest(int device_idx){
    torch::manual_seed(0);
    auto device = torch::Device(torch::kCUDA, device_idx);
    auto x = torch::randn({16, 4}, torch::dtype(torch::kFloat32).device(device));
    auto x_mean = torch::randn({1, 4}, torch::dtype(torch::kFloat32).device(device));
    auto x_std = torch::randn({1, 4}, torch::dtype(torch::kFloat32).device(device));
    auto a = torch::randn({1, 4}, torch::dtype(torch::kFloat32).device(device));
    auto y = torch::randn({2, 16, 4}, torch::dtype(torch::kFloat32).device(device));
    auto net = csd::nets::VanillaLocVolSoftplusSigmoid(4, 6, 2*4, 1.0, a, std::numeric_limits<float>::epsilon());
    net->to(device);
    net->x_mean.copy_(x_mean);
    net->x_std.copy_(x_std);
    auto uv = net->forward_complex(x, y);
    std::cout << uv.first << std::endl;
    std::cout << uv.second << std::endl;
}

void ComplexJointLossTest(int device_idx){
    torch::manual_seed(0);
    auto device = torch::Device(torch::kCUDA, device_idx);
    auto u = torch::randn({2, 16, 4}, torch::dtype(torch::kFloat32).device(device));
    auto Y = torch::randn({16, 1}, torch::dtype(torch::kFloat32).device(device));
    auto Y_bis = torch::randn({16, 1}, torch::dtype(torch::kFloat32).device(device));
    auto dY = torch::randn({16, 4}, torch::dtype(torch::kFloat32).device(device));
    auto dY_bis = torch::randn({16, 4}, torch::dtype(torch::kFloat32).device(device));
    auto lb = torch::empty({16, 1}, torch::dtype(torch::kFloat32).device(device)).fill_(0.2);
    auto Y_pred = torch::randn({16, 1}, torch::dtype(torch::kFloat32).device(device));
    auto Y_pred_bis = torch::randn({16, 1}, torch::dtype(torch::kFloat32).device(device));
    auto dY_pred = torch::randn({16, 4}, torch::dtype(torch::kFloat32).device(device));
    auto dY_pred_bis = torch::randn({16, 4}, torch::dtype(torch::kFloat32).device(device));
    double y_std = 0.6;
    double y_bis_std = 0.4;
    double sum_inv_diffs_var = 1.23;
    double sum_inv_put_diffs_var = 0.9;
    assert(dY.size(1)==4);
    auto loss = csd::modules::layers::StochasticDirectionalLoss(1, 1, 1, 1, 1, y_std*y_std, y_bis_std*y_bis_std, sum_inv_diffs_var, sum_inv_put_diffs_var, dY.size(1));
    std::cout << loss(Y_pred, Y_pred_bis, dY_pred, dY_pred_bis, u, Y, Y_bis, dY, dY_bis, lb) << std::endl;
}

void SimLocVolTest(int device_idx){
    auto device = torch::Device(torch::kCUDA, device_idx);
    UniformBounds<float> r_b = {0., 0.05};
    UniformBounds<float> stk_b = {0.05, 2.5};
    UniformBounds<float> mat_b = {0.05, 2.5};
    UniformBounds<float> lv_b = {0.1, 2.};
    long num_nodes_x = 6;
    long num_nodes_y = 6;
    long num_paths = 1<<21;
    long num_inner_samples = 1<<5;
    long num_steps_per_year = 1000;
    curandStateMRG32k3a_t *curand_states;
    C10_CUDA_CHECK(cudaSetDevice(device.index()));
    auto stream = c10::cuda::getCurrentCUDAStream();
    cudaEvent_t evt_start, evt_stop;
    float elapsed_time;
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&evt_start, cudaEventDefault));
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&evt_stop, cudaEventDefault));
    C10_CUDA_CHECK(cudaMalloc((void **)&curand_states, num_paths*sizeof(curandState))); // TODO: we probably don't need that many states
    auto nodes_x = torch::linspace(mat_b.min, mat_b.max, num_nodes_x, torch::dtype(torch::kFloat32).device(device));
    auto nodes_y = torch::linspace(log(stk_b.min), log(stk_b.max), num_nodes_y, torch::dtype(torch::kFloat32).device(device));
    csd::simulators::kernels::curand::init_states(curand_states, num_paths, stream);
    auto x = torch::empty({num_paths, 3+num_nodes_x*num_nodes_y}, torch::dtype(torch::kFloat32).device(device));
    auto r = x.select(-1, 0);
    auto stk = x.select(-1, 1);
    auto mat = x.select(-1, 2);
    auto lv = x.slice(1, 3).view({-1, num_nodes_x, num_nodes_y});
    C10_CUDA_CHECK(cudaEventRecord(evt_start, stream.stream()));
    csd::simulators::kernels::locvol::generate_params(curand_states, r_b, stk_b, mat_b, lv_b, r, stk, mat, lv, num_steps_per_year, stream);
    C10_CUDA_CHECK(cudaEventRecord(evt_stop, stream.stream()));
    C10_CUDA_CHECK(cudaEventSynchronize(evt_stop));
    C10_CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, evt_start, evt_stop));
    std::cout << std::fixed;
    std::cout << std::setprecision(4);
    std::cout << "Generated " << num_paths << " instances of model parameters in " << elapsed_time/1000 << " seconds" << std::endl;
    float S0 = 1.;
    auto call_payoffs = torch::empty({num_paths, 1}, torch::dtype(torch::kFloat32).device(device));
    auto put_payoffs = torch::empty({num_paths, 1}, torch::dtype(torch::kFloat32).device(device));
    auto call_dpayoffs = torch::empty({num_paths, 3+num_nodes_x*num_nodes_y}, torch::dtype(torch::kFloat32).device(device));
    auto put_dpayoffs = torch::empty({num_paths, 3+num_nodes_x*num_nodes_y}, torch::dtype(torch::kFloat32).device(device));
    r.fill_(0.02);
    stk.fill_(1.2);
    mat.fill_(1.5);
    lv.fill_(0.5);
    C10_CUDA_CHECK(cudaEventRecord(evt_start, stream.stream()));
    csd::simulators::kernels::locvol::generate_payoffs<curandStateMRG32k3a_t, float, 6, 6>(curand_states, r, stk, mat, lv, nodes_x, nodes_y, call_payoffs.select(-1, 0), put_payoffs.select(-1, 0), call_dpayoffs, put_dpayoffs, S0, num_steps_per_year, num_inner_samples, stream);
    C10_CUDA_CHECK(cudaEventRecord(evt_stop, stream.stream()));
    C10_CUDA_CHECK(cudaEventSynchronize(evt_stop));
    C10_CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, evt_start, evt_stop));
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "Generated " << num_paths << " labels in " << elapsed_time/1000 << " seconds" << std::endl;
    auto lb = (S0-(-r*mat).exp_().mul_(stk)).relu_().unsqueeze(1);
    std::cout << call_payoffs.mean() << std::endl;
    std::cout << call_payoffs.std() << std::endl;
    std::cout << call_payoffs.quantile(0.1) << std::endl;
    std::cout << call_payoffs.quantile(0.25) << std::endl;
    std::cout << call_payoffs.quantile(0.5) << std::endl;
    std::cout << call_payoffs.quantile(0.75) << std::endl;
    std::cout << call_payoffs.quantile(0.9) << std::endl;
    std::cout << "----" << std::endl;
    std::cout << put_payoffs.mean() << std::endl;
    std::cout << put_payoffs.std() << std::endl;
    std::cout << put_payoffs.quantile(0.1) << std::endl;
    std::cout << put_payoffs.quantile(0.25) << std::endl;
    std::cout << put_payoffs.quantile(0.5) << std::endl;
    std::cout << put_payoffs.quantile(0.75) << std::endl;
    std::cout << put_payoffs.quantile(0.9) << std::endl;
    std::cout << "----" << std::endl;
    std::cout << lb.mean() << std::endl;
    std::cout << lb.std() << std::endl;
    std::cout << lb.quantile(0.1) << std::endl;
    std::cout << lb.quantile(0.25) << std::endl;
    std::cout << lb.quantile(0.5) << std::endl;
    std::cout << lb.quantile(0.75) << std::endl;
    std::cout << lb.quantile(0.9) << std::endl;
    std::cout << "----" << std::endl;
    std::cout << call_dpayoffs.mean(0) << std::endl;
    std::cout << call_dpayoffs.std(0) << std::endl;
    std::cout << call_dpayoffs.quantile(0.1, 0) << std::endl;
    std::cout << call_dpayoffs.quantile(0.25, 0) << std::endl;
    std::cout << call_dpayoffs.quantile(0.5, 0) << std::endl;
    std::cout << call_dpayoffs.quantile(0.75, 0) << std::endl;
    std::cout << call_dpayoffs.quantile(0.9, 0) << std::endl;
    std::cout << "----" << std::endl;
    std::cout << put_dpayoffs.mean(0) << std::endl;
    std::cout << put_dpayoffs.std(0) << std::endl;
    std::cout << put_dpayoffs.quantile(0.1, 0) << std::endl;
    std::cout << put_dpayoffs.quantile(0.25, 0) << std::endl;
    std::cout << put_dpayoffs.quantile(0.5, 0) << std::endl;
    std::cout << put_dpayoffs.quantile(0.75, 0) << std::endl;
    std::cout << put_dpayoffs.quantile(0.9, 0) << std::endl;
}

void VanillaLocVolSensiTest(int device_idx){
    torch::manual_seed(0);
    auto device = torch::Device(torch::kCUDA, device_idx);
    auto x = torch::randn({16, 4}, torch::dtype(torch::kFloat32).device(device));
    auto x_mean = torch::randn({1, 4}, torch::dtype(torch::kFloat32).device(device));
    auto x_std = torch::randn({1, 4}, torch::dtype(torch::kFloat32).device(device));
    auto a = torch::randn({1, 4}, torch::dtype(torch::kFloat32).device(device));
    auto net = csd::nets::VanillaLocVolSoftplusSigmoid(4, 6, 2*4, 1.0, a, std::numeric_limits<float>::epsilon());
    net->to(device);
    net->x_mean.copy_(x_mean);
    net->x_std.copy_(x_std);
    auto uv = net->forward_fwdad_diff(x);
    std::cout << uv.first << std::endl;
    std::cout << uv.second << std::endl;
    std::cout << "---" << std::endl;
    auto dx = torch::empty({16, 4}, torch::dtype(torch::kFloat32).device(device));
    for(auto& param: net->parameters()){
        param.set_requires_grad(false);
    }
    x.requires_grad_();
    for(int i=0; i<16; i++){
        if(i>0) x.grad().zero_();
        net->forward(x).index({i, 0}).backward();
        dx[i].copy_(x.grad()[i]);
    }
    std::cout << dx << std::endl;
    std::cout << "---" << std::endl;
    std::cout << "max abs err: " << (uv.second-dx).abs_().max() << std::endl;
    std::cout << "(float32 machine eps: " << std::numeric_limits<float>::epsilon() << ")" << std::endl;
}

void VanillaLocVolSerializeTest(int device_idx){
    torch::manual_seed(0);
    auto device = torch::Device(torch::kCUDA, device_idx);
    auto x = torch::randn({16, 4}, torch::dtype(torch::kFloat32).device(device));
    auto x_mean = torch::randn({1, 4}, torch::dtype(torch::kFloat32).device(device));
    auto x_std = torch::randn({1, 4}, torch::dtype(torch::kFloat32).device(device));
    auto a = torch::randn({1, 4}, torch::dtype(torch::kFloat32).device(device));
    auto y = torch::randn({2, 16, 4}, torch::dtype(torch::kFloat32).device(device));
    auto net = csd::nets::VanillaLocVolSoftplusSigmoid(4, 6, 2*4, 1.0, a, std::numeric_limits<float>::epsilon());
    net->to(device);
    net->x_mean.copy_(x_mean);
    net->x_std.copy_(x_std);
    auto uv = net->forward_complex(x, y);
    std::cout << uv.first << std::endl;
    std::cout << uv.second << std::endl;
    torch::save(net, "saved_net.pt");
    a = torch::randn({1, 4}, torch::dtype(torch::kFloat32).device(device));
    auto net_bis = csd::nets::VanillaLocVolSoftplusSigmoid(4, 6, 2*4, 1.0, a, std::numeric_limits<float>::epsilon());
    net_bis->to(device);
    torch::load(net_bis, "saved_net.pt");
    auto uv_bis = net_bis->forward_complex(x, y);
    std::cout << uv_bis.first << std::endl;
    std::cout << uv_bis.second << std::endl;
    std::cout << "u max abs err: " << (uv_bis.first-uv.first).abs_().max() << std::endl;
    std::cout << "v max abs err: " << (uv_bis.second-uv.second).abs_().max() << std::endl;
    std::cout << "(float32 machine eps: " << std::numeric_limits<float>::epsilon() << ")" << std::endl;
}

int main(int argc, char *argv[]){
    if (argc != 2) 
        throw std::invalid_argument("Only one argument please: device_idx.");
    int device_idx;
    csd::utils::args::parse_arg(1, argv[1], &device_idx);
    // AffineCSDTest(device_idx);
    // AffineCSDGradTest(device_idx);
    // ComplexSigmoidTest(device_idx);
    // ComplexSigmoidGradTest(device_idx);
    // ComplexSoftplusTest(device_idx);
    // ComplexSoftplusGradTest(device_idx);
    // CallLayerTest(device_idx);
    // CallLayerGradTest(device_idx);
    // StandardizeTest(device_idx);
    // PutFromCallLayerTest(device_idx);
    // PutFromCallLayerGradTest(device_idx);
    // VanillaLocVolTest(device_idx);
    // ComplexJointLossTest(device_idx);
    // SimLocVolTest(device_idx);
    // VanillaLocVolSensiTest(device_idx);
    VanillaLocVolSerializeTest(device_idx);
}