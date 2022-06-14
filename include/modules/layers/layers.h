#pragma once
#include <torch/torch.h>

namespace csd {
namespace modules {
namespace layers {
struct ComplexRealAffineImpl : torch::nn::Module {
  torch::Tensor W, b;
  ComplexRealAffineImpl() = default;
  ComplexRealAffineImpl(int, int);
  torch::Tensor forward(torch::Tensor);
  std::pair<torch::Tensor, torch::Tensor> forward_complex(torch::Tensor,
                                                  torch::Tensor);
  std::pair<torch::Tensor, torch::Tensor> forward_diff(torch::Tensor);
};
TORCH_MODULE_IMPL(ComplexRealAffine, ComplexRealAffineImpl);

struct RealAffineImpl : torch::nn::Module {
  torch::Tensor W, b;
  RealAffineImpl() = default;
  RealAffineImpl(int, int);
  torch::Tensor forward(torch::Tensor);
  std::pair<torch::Tensor, torch::Tensor> forward_diff(torch::Tensor);
};
TORCH_MODULE_IMPL(RealAffine, RealAffineImpl);

struct ComplexCallLayerImpl : torch::nn::Module {
  ComplexCallLayerImpl() = default;
  // torch::Tensor forward(torch::Tensor, torch::Tensor, double);
  std::pair<torch::Tensor, torch::Tensor> forward_complex(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double);
};
TORCH_MODULE_IMPL(ComplexCallLayer, ComplexCallLayerImpl);

struct ComplexPutFromCallLayerImpl : torch::nn::Module {
  ComplexPutFromCallLayerImpl() = default;
  // torch::Tensor forward(torch::Tensor, torch::Tensor, double);
  std::pair<torch::Tensor, torch::Tensor> forward_complex(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double);
};
TORCH_MODULE_IMPL(ComplexPutFromCallLayer, ComplexPutFromCallLayerImpl);

struct StochasticDirectionalLossImpl : torch::nn::Module {
  double gamma_01, gamma_02, gamma_11, gamma_12, gamma_2, y_var, y_put_var;
  StochasticDirectionalLossImpl() = default;
  StochasticDirectionalLossImpl(double, double, double, double, double, double, double, double, double, unsigned long);
  torch::Tensor forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
};
TORCH_MODULE_IMPL(StochasticDirectionalLoss, StochasticDirectionalLossImpl);

struct NoDiffOnlyCallsLossImpl : torch::nn::Module {
  double gamma, y_std, y_var;
  NoDiffOnlyCallsLossImpl() = default;
  NoDiffOnlyCallsLossImpl(double, double);
  torch::Tensor forward(torch::Tensor, torch::Tensor, torch::Tensor);
};
TORCH_MODULE_IMPL(NoDiffOnlyCallsLoss, NoDiffOnlyCallsLossImpl);
} // namespace layers
} // namespace modules
} // namespace csd