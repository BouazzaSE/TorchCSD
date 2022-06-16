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
#include "modules/modules.h"

namespace csd{
namespace nets {
struct VanillaCallPutNetImpl : torch::nn::Module {
  torch::Tensor x_mean, x_std;
  virtual torch::Tensor forward(torch::Tensor) = 0;
  virtual std::pair<torch::Tensor, torch::Tensor> forward_complex(torch::Tensor, torch::Tensor) = 0;
  virtual std::pair<torch::Tensor, torch::Tensor> forward_diff(torch::Tensor, torch::Tensor) = 0;
  virtual std::pair<torch::Tensor, torch::Tensor> forward_call_put(torch::Tensor) = 0;
  virtual std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward_call_put_complex(torch::Tensor, torch::Tensor) = 0;
  virtual std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward_call_put_diff(torch::Tensor, torch::Tensor) = 0;
};

template <typename T_hidden_real, typename T_hidden_complex, typename T_preoutput_real, typename T_preoutput_complex>
struct VanillaLocVolImpl : VanillaCallPutNetImpl {
  static_assert(std::is_base_of<torch::nn::Module, typename T_hidden_real::Impl>::value, "template argument T_real must be a well-defined torch::nn::ModuleHolder");
  static_assert(std::is_base_of<torch::nn::Module, typename T_hidden_complex::Impl>::value, "template argument T_complex must be a well-defined torch::nn::ModuleHolder");
  const float eps, S0;
  const int num_layers;
  std::vector<csd::modules::layers::ComplexRealAffine> hidden_layers;
  csd::modules::layers::ComplexRealAffine preoutput_layer;
  modules::layers::ComplexCallLayer ccall_layer;
  modules::layers::ComplexPutFromCallLayer cput_from_call_layer;
  torch::Tensor T_threshold, x_mean, x_std;
  torch::Tensor e0, e1, e2; // only needed for gradients during inference
  T_hidden_real hidden_real_activation; T_hidden_complex hidden_complex_activation;
  T_preoutput_real preoutput_real_activation; T_preoutput_complex preoutput_complex_activation;
  VanillaLocVolImpl(int, int, int, float, torch::Tensor, float);
  void init_weights();
  torch::Tensor forward(torch::Tensor);
  std::pair<torch::Tensor, torch::Tensor> forward_complex(torch::Tensor, torch::Tensor);
  std::pair<torch::Tensor, torch::Tensor> forward_diff(torch::Tensor, torch::Tensor);
  std::pair<torch::Tensor, torch::Tensor> forward_fwdad_diff(torch::Tensor);
  torch::Tensor standardize(torch::Tensor);
  std::pair<torch::Tensor, torch::Tensor> standardize_complex(torch::Tensor, torch::Tensor);
  std::pair<torch::Tensor, torch::Tensor> forward_call_put(torch::Tensor);
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward_call_put_complex(torch::Tensor, torch::Tensor);
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward_call_put_diff(torch::Tensor, torch::Tensor);
};
using VanillaLocVolSoftplusSigmoidImpl = VanillaLocVolImpl<csd::modules::activations::RealSoftplus, csd::modules::activations::ComplexSoftplus, csd::modules::activations::RealSigmoid, csd::modules::activations::ComplexSigmoid>;
TORCH_MODULE(VanillaLocVolSoftplusSigmoid);
}
}
