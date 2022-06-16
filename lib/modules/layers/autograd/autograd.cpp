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
#include "modules/layers/kernels/wrappers.h"
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

namespace csd {
namespace modules {
namespace layers {
namespace autograd {
torch::autograd::variable_list
ComplexCallLayer::forward(torch::autograd::AutogradContext *ctx, torch::Tensor x,
                         torch::Tensor y, torch::Tensor a, torch::Tensor b, double S) {
  // TODO: ADD ASSERT CHECK FOR DIMENSIONS
  auto u = torch::empty({x.sizes()[0], x.sizes()[1]}, torch::dtype(x.dtype()).device(x.device()));
  auto v = torch::empty({x.sizes()[0], x.sizes()[1]}, torch::dtype(x.dtype()).device(x.device()));
  auto dudx1 = torch::empty({x.sizes()[0], x.sizes()[1]}, torch::dtype(x.dtype()).device(x.device()));
  auto dvdx1 = torch::empty({x.sizes()[0], x.sizes()[1]}, torch::dtype(x.dtype()).device(x.device()));
  auto stream = c10::cuda::getCurrentCUDAStream();
  if (x.dtype().isScalarType(torch::ScalarType::Float)) {
    csd::modules::layers::kernels::ccall_layer::forward<float, true>(x, y, a, b, u, v, dudx1, dvdx1, S, stream);
  } else if (x.dtype().isScalarType(torch::ScalarType::Double)) {
    csd::modules::layers::kernels::ccall_layer::forward<double, true>(x, y, a, b, u, v, dudx1, dvdx1, S, stream);
  } else {
    throw std::invalid_argument(
        "ComplexCallLayer::forward only supports Float or Double tensors.");
  }
  ctx->saved_data["S"] = S;
  ctx->save_for_backward({dudx1, dvdx1});
  // auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(u.device().index());
  // auto allocated_bytes = stats.allocated_bytes[(unsigned int)c10::cuda::CUDACachingAllocator::StatType::AGGREGATE];
  return {u, v};
}

torch::autograd::variable_list
ComplexCallLayer::backward(torch::autograd::AutogradContext *ctx,
                          torch::autograd::variable_list grad_output) {
  auto S = ctx->saved_data["S"].toDouble();
  auto saved_vars = ctx->get_saved_variables();
  auto dudx1 = saved_vars[0]; auto dvdx1 = saved_vars[1];
  /*auto dudx1 = ctx->saved_data["dudx1"].toTensor();
  auto dvdx1 = ctx->saved_data["dvdx1"].toTensor();*/
  auto dx = torch::empty({dudx1.sizes()[0], dudx1.sizes()[1], 2}, torch::dtype(dudx1.dtype()).device(dudx1.device()));
  auto dy = torch::empty_like(dx.data());
  auto du = grad_output[0];
  auto dv = grad_output[1];
  auto stream = c10::cuda::getCurrentCUDAStream();
  if (dudx1.dtype().isScalarType(torch::ScalarType::Float)) {
    csd::modules::layers::kernels::ccall_layer::backward<float>(du, dv, dudx1, dvdx1, dx, dy, S, stream);
  } else if (dudx1.dtype().isScalarType(torch::ScalarType::Double)) {
    csd::modules::layers::kernels::ccall_layer::backward<double>(du, dv, dudx1, dvdx1, dx, dy, S, stream);
  } else {
    throw std::invalid_argument(
        "ComplexCallLayer::backward only supports Float or Double tensors.");
  }
  // auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(dx.device().index());
  // auto allocated_bytes = stats.allocated_bytes[(unsigned int)c10::cuda::CUDACachingAllocator::StatType::AGGREGATE];
  return {dx, dy, torch::Tensor(), torch::Tensor(), torch::Tensor()};
}

torch::autograd::variable_list
ComplexPutFromCallLayer::forward(torch::autograd::AutogradContext *ctx, torch::Tensor x, torch::Tensor y, torch::Tensor a, torch::Tensor b, double S) {
  auto u = torch::empty_like(x.data());
  auto v = torch::empty_like(x.data());
  auto stream = c10::cuda::getCurrentCUDAStream();
  if (x.dtype().isScalarType(torch::ScalarType::Float)) {
    csd::modules::layers::kernels::cput_from_call_layer::forward<float>(x, y, a, b, u, v, S, stream);
  } else if (x.dtype().isScalarType(torch::ScalarType::Double)) {
    csd::modules::layers::kernels::cput_from_call_layer::forward<double>(x, y, a, b, u, v, S, stream);
  } else {
    throw std::invalid_argument(
        "ComplexCallFromPutLayer::forward only supports Float or Double tensors.");
  }
  // auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(u.device().index());
  // auto allocated_bytes = stats.allocated_bytes[(unsigned int)c10::cuda::CUDACachingAllocator::StatType::AGGREGATE];
  return {u, v};
}

torch::autograd::variable_list
ComplexPutFromCallLayer::backward(torch::autograd::AutogradContext *ctx, torch::autograd::variable_list grad_output) {
  auto du = grad_output[0];
  auto dv = grad_output[1];
  // auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(du.device().index());
  // auto allocated_bytes = stats.allocated_bytes[(unsigned int)c10::cuda::CUDACachingAllocator::StatType::AGGREGATE];
  // return {du, du.neg(), torch::Tensor(), torch::Tensor(), torch::Tensor()};
  return {du, dv, torch::Tensor(), torch::Tensor(), torch::Tensor()};
}
} // namespace autograd
} // namespace layers
} // namespace modules
} // namespace csd
