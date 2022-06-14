#pragma once
#include <torch/torch.h>

namespace csd {
namespace modules {
namespace activations {
namespace autograd {
struct ComplexSoftplus : public torch::autograd::Function<ComplexSoftplus> {
  static torch::autograd::variable_list
  forward(torch::autograd::AutogradContext *, torch::Tensor, torch::Tensor);
  static torch::autograd::variable_list
  backward(torch::autograd::AutogradContext *, torch::autograd::variable_list);
};
struct ComplexSigmoid : public torch::autograd::Function<ComplexSigmoid> {
  static torch::autograd::variable_list
  forward(torch::autograd::AutogradContext *, torch::Tensor, torch::Tensor);
  static torch::autograd::variable_list
  backward(torch::autograd::AutogradContext *, torch::autograd::variable_list);
};
} // namespace autograd
} // namespace activations
} // namespace modules
} // namespace csd