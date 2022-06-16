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



#include "modules/layers/autograd/autograd.h"
#include "modules/layers/inference/inference.h"
#include "modules/layers/kernels/wrappers.h"
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

namespace csd {
namespace modules {
namespace layers {
namespace inference {
std::pair<torch::Tensor, torch::Tensor> ccall_layer(torch::Tensor x,
                         torch::Tensor y, torch::Tensor a, torch::Tensor b, double S) {
  auto u = torch::empty({x.sizes()[0], x.sizes()[1]}, torch::dtype(x.dtype()).device(x.device()));
  auto v = torch::empty_like(u.data());
  auto stream = c10::cuda::getCurrentCUDAStream();
  if (x.dtype().isScalarType(torch::ScalarType::Float)) {
    csd::modules::layers::kernels::ccall_layer::forward<float, false>(x, y, a, b, u, v, torch::Tensor(), torch::Tensor(), S, stream);
  } else if (x.dtype().isScalarType(torch::ScalarType::Double)) {
    csd::modules::layers::kernels::ccall_layer::forward<double, false>(x, y, a, b, u, v, torch::Tensor(), torch::Tensor(), S, stream);
  } else {
    throw std::invalid_argument(
        "ccall_layer only supports Float or Double tensors.");
  }
  return {u, v};
}

std::pair<torch::Tensor, torch::Tensor> cput_from_call_layer(torch::Tensor x, torch::Tensor y, torch::Tensor a, torch::Tensor b, double S) {
  auto uv = csd::modules::layers::autograd::ComplexPutFromCallLayer::apply(x, y, a, b, S);
  return {uv[0], uv[1]};
}
}
}
}
}
