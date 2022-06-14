#pragma once
#include <torch/torch.h>

namespace csd {
namespace modules {
namespace layers {
namespace autograd {
struct ComplexCallLayer : public torch::autograd::Function<ComplexCallLayer> {
  static torch::autograd::variable_list
  forward(torch::autograd::AutogradContext *, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double);
  static torch::autograd::variable_list
  backward(torch::autograd::AutogradContext *, torch::autograd::variable_list);
};

struct ComplexPutFromCallLayer : public torch::autograd::Function<ComplexPutFromCallLayer> {
  static torch::autograd::variable_list
  forward(torch::autograd::AutogradContext *, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double);
  static torch::autograd::variable_list
  backward(torch::autograd::AutogradContext *, torch::autograd::variable_list);
};
} // namespace autograd
} // namespace layers
} // namespace modules
} // namespace csd