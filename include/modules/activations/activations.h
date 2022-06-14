#pragma once
#include <torch/torch.h>

#define CSD_MODULE_WRAPPER_DEFINE(Name) \
struct Name##Impl : torch::nn::Module { \
public: \
  std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor, torch::Tensor); \
}; TORCH_MODULE(Name); \

#define FWDAD_MODULE_WRAPPER_DEFINE(Name) \
struct Name##Impl : torch::nn::Module { \
public: \
  torch::Tensor forward(torch::Tensor); \
  std::pair<torch::Tensor, torch::Tensor> forward_diff(torch::Tensor); \
}; TORCH_MODULE(Name); \

namespace csd {
namespace modules {
namespace activations {
FWDAD_MODULE_WRAPPER_DEFINE(RealSoftplus);
FWDAD_MODULE_WRAPPER_DEFINE(RealSigmoid);
CSD_MODULE_WRAPPER_DEFINE(ComplexSoftplus);
CSD_MODULE_WRAPPER_DEFINE(ComplexSigmoid);
} // namespace activations
} // namespace modules
} // namespace csd
