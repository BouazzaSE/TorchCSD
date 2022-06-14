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

