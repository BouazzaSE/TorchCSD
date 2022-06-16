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
#include <c10/core/DeviceType.h>
#include <c10/core/TensorOptions.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include <curand_kernel.h>
#include <iomanip>
#include <iostream>
#include <torch/optim/lbfgs.h>
#include <torch/torch.h>
#include <torch/types.h>
#include "nets/vanilla_locvol/vanilla_locvol.h"
#include "utils/parse_args.h"

int main(int argc, char *argv[]){
    if (argc != 7) 
        throw std::invalid_argument("6 arguments please: device_idx, num_layers, num_units, net_dumpname, prices_dumpname and grads_dumpname.");
    int device_idx;
    long num_layers, num_units;
    std::string net_dumpname, prices_dumpname, grads_dumpname;
    csd::utils::args::parse_arg(1, argv[1], &device_idx); // 1st arg is the device (CUDA-capable GPU) index
    csd::utils::args::parse_arg(2, argv[2], &num_layers); // 2nd arg is the number of layers
    csd::utils::args::parse_arg(3, argv[3], &num_units); // 3rd arg is the number of units per layer
    csd::utils::args::parse_arg(4, argv[4], &net_dumpname); // 4th arg name of the dump file for the neural network
    csd::utils::args::parse_arg(5, argv[5], &prices_dumpname); // 4th arg name of the dump file for the output prices
    csd::utils::args::parse_arg(6, argv[6], &grads_dumpname); // 4th arg name of the dump file for the output gradients
    std::cout << net_dumpname << std::endl;
    auto device = torch::Device(torch::kCUDA, device_idx);
    C10_CUDA_CHECK(cudaSetDevice(device.index()));
    auto stream = c10::cuda::getCurrentCUDAStream();
    cudaEvent_t evt_start, evt_stop;
    float elapsed_time;
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&evt_start, cudaEventDefault));
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&evt_stop, cudaEventDefault));
    auto mat_threshold = torch::empty({1, 28}, torch::dtype(torch::kFloat32).device(device));
    auto net = csd::nets::VanillaLocVolSoftplusSigmoid(28, num_layers, num_units, 1.0, mat_threshold, std::numeric_limits<float>::epsilon());
    net->to(device);
    torch::load(net, net_dumpname);
    long num_prices = 1024;
    auto rate = torch::tensor(0.0128, torch::dtype(torch::kFloat32).device(device));
    auto stk = torch::linspace(0.5, 2, num_prices, torch::dtype(torch::kFloat32).device(device));
    auto mat = torch::tensor(1, torch::dtype(torch::kFloat32).device(device));
    auto locvol = torch::tensor(
        {
            {1.8182, 0.8224, 0.7496, 0.4949, 1.0417},
            {1.3432, 0.9347, 0.5703, 0.4868, 0.6863},
            {1.8182, 0.3757, 0.5789, 0.5348, 0.6655},
            {0.8435, 0.1100, 0.8001, 0.7003, 0.2425},
            {1.8182, 1.1904, 0.6852, 0.1100, 1.1492}
        }, torch::dtype(torch::kFloat32).device(device)
    );
    auto pricing_params = torch::cat({rate.expand({num_prices, 1}), stk.view({-1, 1}), mat.expand({num_prices, 1}), locvol.view(-1).expand({num_prices, 25})}, 1);
    {
        torch::NoGradGuard no_grad;
        C10_CUDA_CHECK(cudaEventRecord(evt_start, stream.stream()));
        auto [prices, grads] = net->forward_fwdad_diff(pricing_params);
        C10_CUDA_CHECK(cudaEventRecord(evt_stop, stream.stream()));
        C10_CUDA_CHECK(cudaEventSynchronize(evt_stop));
        C10_CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, evt_start, evt_stop));
        std::cout << std::fixed;
        std::cout << std::setprecision(4);
        std::cout << "Computed " << num_prices << " prices and gradients (28 partial derivatives) in " << elapsed_time << " milliseconds" << std::endl;
        prices = prices.to(torch::Device(torch::kCPU));
        grads = grads.to(torch::Device(torch::kCPU));
        torch::save(prices, prices_dumpname);
        torch::save(grads, grads_dumpname);
    }
}