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
