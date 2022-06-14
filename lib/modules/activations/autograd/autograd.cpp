#include "modules/activations/autograd/autograd.h"
#include "modules/activations/kernels/wrappers.h"
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

namespace csd {
namespace modules {
namespace activations {
namespace autograd {
torch::autograd::variable_list
ComplexSoftplus::forward(torch::autograd::AutogradContext *ctx, torch::Tensor x,
                         torch::Tensor y) {
  auto u = torch::empty_like(x.data());
  auto v = torch::empty_like(x.data());
  auto dudx = torch::empty_like(x.data());
  auto dvdx = torch::empty_like(x.data());
  auto stream = c10::cuda::getCurrentCUDAStream();
  if (x.dtype().isScalarType(torch::ScalarType::Float)) {
    csd::modules::activations::kernels::csoftplus::forward<float, true>(x, y, u, v, dudx, dvdx, stream);
  } else if (x.dtype().isScalarType(torch::ScalarType::Double)) {
    csd::modules::activations::kernels::csoftplus::forward<double, true>(x, y, u, v, dudx, dvdx, stream);
  } else {
    throw std::invalid_argument(
        "ComplexSoftplus::forward only supports Float or Double tensors.");
  }
  ctx->save_for_backward({dudx, dvdx});
  return {u, v};
}

torch::autograd::variable_list
ComplexSoftplus::backward(torch::autograd::AutogradContext *ctx,
                          torch::autograd::variable_list grad_output) {
  auto saved_vars = ctx->get_saved_variables();
  auto dudx = saved_vars[0]; auto dvdx = saved_vars[1];
  auto dx = torch::empty_like(dudx.data());
  auto dy = torch::empty_like(dudx.data());
  auto du = grad_output[0];
  auto dv = grad_output[1];
  auto stream = c10::cuda::getCurrentCUDAStream();
  if (dx.dtype().isScalarType(torch::ScalarType::Float)) {
    csd::modules::activations::kernels::csoftplus::backward<float>(du, dv, dudx, dvdx, dx, dy, stream);
  } else if (dx.dtype().isScalarType(torch::ScalarType::Double)) {
    csd::modules::activations::kernels::csoftplus::backward<double>(du, dv, dudx, dvdx, dx, dy, stream);
  } else {
    throw std::invalid_argument(
        "ComplexSoftplus::backward only supports Float or Double tensors.");
  }
  return {dx, dy};
}

torch::autograd::variable_list
ComplexSigmoid::forward(torch::autograd::AutogradContext *ctx, torch::Tensor x,
                         torch::Tensor y) {
  auto u = torch::empty_like(x.data());
  auto v = torch::empty_like(x.data());
  auto dudx = torch::empty_like(x.data());
  auto dvdx = torch::empty_like(x.data());
  auto stream = c10::cuda::getCurrentCUDAStream();
  if (x.dtype().isScalarType(torch::ScalarType::Float)) {
    csd::modules::activations::kernels::csigmoid::forward<float, true>(x, y, u, v, dudx, dvdx, stream);
  } else if (x.dtype().isScalarType(torch::ScalarType::Double)) {
    csd::modules::activations::kernels::csigmoid::forward<double, true>(x, y, u, v, dudx, dvdx, stream);
  } else {
    throw std::invalid_argument(
        "ComplexSoftplus::forward only supports Float or Double tensors.");
  }
  ctx->save_for_backward({dudx, dvdx});
  return {u, v};
}

torch::autograd::variable_list
ComplexSigmoid::backward(torch::autograd::AutogradContext *ctx,
                          torch::autograd::variable_list grad_output) {
  auto saved_vars = ctx->get_saved_variables();
  auto dudx = saved_vars[0]; auto dvdx = saved_vars[1];
  auto dx = torch::empty_like(dudx.data());
  auto dy = torch::empty_like(dudx.data());
  auto du = grad_output[0];
  auto dv = grad_output[1];
  auto stream = c10::cuda::getCurrentCUDAStream();
  if (dx.dtype().isScalarType(torch::ScalarType::Float)) {
    csd::modules::activations::kernels::csigmoid::backward<float>(du, dv, dudx, dvdx, dx, dy, stream);
  } else if (dx.dtype().isScalarType(torch::ScalarType::Double)) {
    csd::modules::activations::kernels::csigmoid::backward<double>(du, dv, dudx, dvdx, dx, dy, stream);
  } else {
    throw std::invalid_argument(
        "ComplexSoftplus::backward only supports Float or Double tensors.");
  }
  return {dx, dy};
}
} // namespace autograd
} // namespace activations
} // namespace modules
} // namespace csd
