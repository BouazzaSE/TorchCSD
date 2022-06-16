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
#include <iomanip>
#include <iostream>
#include <torch/torch.h>
#include <torch/types.h>
#include "modules/activations/activations.h"
#include "modules/layers/layers.h"
#include "modules/modules.h"
#include "utils/parse_args.h"

int main(int argc, char *argv[]){
    // A few simple routines to check gradient computations against a slow finite difference
    if (argc != 2) 
        throw std::invalid_argument("Only one argument please: device_idx.");
    int device_idx;
    csd::utils::args::parse_arg(1, argv[1], &device_idx);
    auto device = torch::Device(torch::kCUDA, device_idx);
    C10_CUDA_CHECK(cudaSetDevice(device.index()));
    torch::manual_seed(0);
    auto module = csd::modules::layers::ComplexPutFromCallLayer();
    module->train();
    auto x = torch::randn({128}, torch::dtype(torch::kFloat64).device(device));
    auto y = torch::randn({128}, torch::dtype(torch::kFloat64).device(device));
    auto a = torch::randn({128, 3}, torch::dtype(torch::kFloat64).device(device));
    auto b = torch::randn({128, 3}, torch::dtype(torch::kFloat64).device(device));
    double S = 1;
    x.set_requires_grad(true);
    y.set_requires_grad(true);
    auto uv = module->forward_complex(x, y, a, b, S);
    (uv.first.mean()+uv.second.mean()).backward();
    auto x_grad = x.grad().data().clone();
    auto y_grad = y.grad().data().clone();
    x.set_requires_grad(false);
    y.set_requires_grad(false);
    double eps = 1e-7;
    double max_err = 0;
    auto uv_nonshifted = module->forward_complex(x, y, a, b, S);
    auto tensor_shifted = x.clone();
    for(int i=0; i<x.sizes()[0]; i++){
        for(int j=0; j<x.sizes()[1]; j++){
                tensor_shifted.copy_(x);
                tensor_shifted[i][j] += eps;
                auto uv_shifted = module->forward_complex(tensor_shifted, y, a, b, S);
                auto finite_diff = (uv_shifted.first.mean()+uv_shifted.second.mean()-uv_nonshifted.first.mean()-uv_nonshifted.second.mean())/eps;
                auto _err = (x_grad[i][j]-finite_diff).abs().item<double>();
                if(_err > max_err) max_err = _err;
        }
    }
    std::cout << "(dx) Largest absolute error: " << max_err << std::endl;
    max_err = 0;
    tensor_shifted = y.clone();
    for(int i=0; i<y.sizes()[0]; i++){
        for(int j=0; j<y.sizes()[1]; j++){
                tensor_shifted.copy_(y);
                tensor_shifted[i][j] += eps;
                auto uv_shifted = module->forward_complex(x, tensor_shifted, a, b, S);
                auto finite_diff = (uv_shifted.first.mean()+uv_shifted.second.mean()-uv_nonshifted.first.mean()-uv_nonshifted.second.mean())/eps;
                auto _err = (y_grad[i][j]-finite_diff).abs().item<double>();
                if(_err > max_err) max_err = _err;
        }
    }
    std::cout << "(dy) Largest absolute error: " << max_err << std::endl;
    // auto csoftplus = csd::modules::activations::ComplexSigmoid();
    // auto x = torch::randn({1, 1, 128}, torch::dtype(torch::kFloat64).device(device));
    // x = x.expand({2, 2, 128});
    // auto y = torch::randn({1, 1, 128}, torch::dtype(torch::kFloat64).device(device));
    // y = y.expand({2, 2, 128});
    // x.set_requires_grad(true);
    // y.set_requires_grad(true);
    // auto uv = csoftplus(x, y);
    // (uv.first.mean()+uv.second.mean()).backward();
    // auto x_grad = x.grad().data().clone();
    // auto y_grad = y.grad().data().clone();
    // x.set_requires_grad(false);
    // y.set_requires_grad(false);
    // double eps = 1e-7;
    // double max_err = 0;
    // auto uv_nonshifted = csoftplus(x, y);
    // auto tensor_shifted = x.clone();
    // for(int i=0; i<x.sizes()[0]; i++){
    //     for(int j=0; j<x.sizes()[1]; j++){
    //         for(int k=0; k<x.sizes()[2]; k++){
    //             tensor_shifted.copy_(x);
    //             tensor_shifted[i][j][k] += eps;
    //             auto uv_shifted = csoftplus(tensor_shifted, y);
    //             auto finite_diff = (uv_shifted.first.mean()+uv_shifted.second.mean()-uv_nonshifted.first.mean()-uv_nonshifted.second.mean())/eps;
    //             auto _err = (x_grad[i][j][k]-finite_diff).abs().item<double>();
    //             if(_err > max_err) max_err = _err;
    //         }
    //     }
    // }
    // std::cout << "(dx) Largest absolute error: " << max_err << std::endl;
    // max_err = 0;
    // tensor_shifted = y.clone();
    // for(int i=0; i<y.sizes()[0]; i++){
    //     for(int j=0; j<y.sizes()[1]; j++){
    //         for(int k=0; k<y.sizes()[2]; k++){
    //             tensor_shifted.copy_(y);
    //             tensor_shifted[i][j][k] += eps;
    //             auto uv_shifted = csoftplus(x, tensor_shifted);
    //             auto finite_diff = (uv_shifted.first.mean()+uv_shifted.second.mean()-uv_nonshifted.first.mean()-uv_nonshifted.second.mean())/eps;
    //             auto _err = (y_grad[i][j][k]-finite_diff).abs().item<double>();
    //             if(_err > max_err) max_err = _err;
    //         }
    //     }
    // }
    // std::cout << "(dy) Largest absolute error: " << max_err << std::endl;
}