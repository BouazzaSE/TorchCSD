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



#include <torch/torch.h>
#include "ATen/Functions.h"
#include "ATen/core/TensorBody.h"
#include "modules/activations/activations.h"
#include "modules/activations/autograd/autograd.h"
#include "modules/activations/inference/inference.h"

#define CSD_MODULE_WRAPPER_DECLARE_FWD(Name, InferenceName) \
std::pair<torch::Tensor, torch::Tensor> csd::modules::activations::Name##Impl::forward(torch::Tensor x, torch::Tensor y){ \
    if(is_training()){ \
        auto uv = csd::modules::activations::autograd::Name::apply(x, y); \
        return {uv[0], uv[1]}; \
    } else { \
        return csd::modules::activations::inference::InferenceName(x, y); \
    } \
} \

CSD_MODULE_WRAPPER_DECLARE_FWD(ComplexSoftplus, csoftplus);
CSD_MODULE_WRAPPER_DECLARE_FWD(ComplexSigmoid, csigmoid);

torch::Tensor csd::modules::activations::RealSigmoidImpl::forward(torch::Tensor x){
    return torch::sigmoid(x);
}

std::pair<torch::Tensor, torch::Tensor> csd::modules::activations::RealSigmoidImpl::forward_diff(torch::Tensor x){
    auto z = torch::sigmoid(x);
    return {z, z*(1-z)};
}

torch::Tensor csd::modules::activations::RealSoftplusImpl::forward(torch::Tensor x){
    return torch::softplus(x);
}

std::pair<torch::Tensor, torch::Tensor> csd::modules::activations::RealSoftplusImpl::forward_diff(torch::Tensor x){
    return {forward(x), torch::sigmoid(x)};
}

