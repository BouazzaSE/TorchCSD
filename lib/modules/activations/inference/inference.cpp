#include "modules/activations/autograd/autograd.h"
#include "modules/activations/inference/inference.h"
#include "modules/activations/kernels/wrappers.h"
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

namespace csd {
namespace modules {
namespace activations {
namespace inference {
std::pair<torch::Tensor, torch::Tensor> csoftplus(torch::Tensor x, torch::Tensor y) {
  auto u = torch::empty_like(x.data());
  auto v = torch::empty_like(x.data());
  auto stream = c10::cuda::getCurrentCUDAStream();
  if (x.dtype().isScalarType(torch::ScalarType::Float)) {
    csd::modules::activations::kernels::csoftplus::forward<float, false>(x, y, u, v, torch::Tensor(), torch::Tensor(), stream);
  } else if (x.dtype().isScalarType(torch::ScalarType::Double)) {
    csd::modules::activations::kernels::csoftplus::forward<double, false>(x, y, u, v, torch::Tensor(), torch::Tensor(), stream);
  } else {
    throw std::invalid_argument("csoftplus only supports Float or Double tensors.");
  }
  return {u, v};
}

std::pair<torch::Tensor, torch::Tensor> csigmoid(torch::Tensor x, torch::Tensor y) {
  auto uv = csd::modules::activations::autograd::ComplexSigmoid::apply(x, y);
  return {uv[0], uv[1]};
}
}
}
}
}
