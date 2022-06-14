#include "modules/layers/layers.h"
#include "ATen/Functions.h"
#include "modules/layers/autograd/autograd.h"
#include "modules/layers/inference/inference.h"
#include "torch/nn/modules/loss.h"
#include <torch/torch.h>

namespace csd {
namespace modules {
namespace layers {
ComplexRealAffineImpl::ComplexRealAffineImpl(int dim_in, int dim_out) {
  W = register_parameter(
      "W", torch::randn({dim_in, dim_out},
                        torch::TensorOptions().dtype(torch::kFloat32)));
  b = register_parameter(
      "b", torch::zeros({1, dim_out},
                        torch::TensorOptions().dtype(torch::kFloat32)));
  W.data() *= std::sqrt(2./dim_in);
}
torch::Tensor
ComplexRealAffineImpl::forward(torch::Tensor x) {
  return torch::matmul(x, W) + b; 
}
std::pair<torch::Tensor, torch::Tensor>
ComplexRealAffineImpl::forward_complex(torch::Tensor x, torch::Tensor y) {
  return {torch::matmul(x, W) + b, torch::matmul(y, W)};
}
std::pair<torch::Tensor, torch::Tensor>
ComplexRealAffineImpl::forward_diff(torch::Tensor x) {
  return {torch::matmul(x, W) + b, W.unsqueeze(0)};
}

RealAffineImpl::RealAffineImpl(int dim_in, int dim_out){
  W = register_parameter(
      "W", torch::randn({dim_in, dim_out},
                        torch::TensorOptions().dtype(torch::kFloat32)));
  b = register_parameter(
      "b", torch::zeros({1, dim_out},
                        torch::TensorOptions().dtype(torch::kFloat32)));
  W.data() *= std::sqrt(2./dim_in);
}
torch::Tensor RealAffineImpl::forward(torch::Tensor x){
  return torch::matmul(x, W) + b;
}
std::pair<torch::Tensor, torch::Tensor> RealAffineImpl::forward_diff(torch::Tensor x){
  return {torch::matmul(x, W) + b, W.unsqueeze(0)};
}

std::pair<torch::Tensor, torch::Tensor> ComplexCallLayerImpl::forward_complex(torch::Tensor x, torch::Tensor y, torch::Tensor a, torch::Tensor b, double S){
    if(is_training()){
      auto uv = autograd::ComplexCallLayer::apply(x, y, a, b, S);
      return {uv[0], uv[1]};
    } else {
      return inference::ccall_layer(x, y, a, b, S);
    }
}

std::pair<torch::Tensor, torch::Tensor> ComplexPutFromCallLayerImpl::forward_complex(torch::Tensor x, torch::Tensor y, torch::Tensor a, torch::Tensor b, double S){
  auto uv = autograd::ComplexPutFromCallLayer::apply(x, y, a, b, S);
  return {uv[0], uv[1]};
}

StochasticDirectionalLossImpl::StochasticDirectionalLossImpl(double gamma_01, double gamma_02, double gamma_11, double gamma_12, double gamma_2, double y_var, double y_put_var, double sum_inv_diffs_var, double sum_inv_put_diffs_var, unsigned long d): 
  gamma_01(gamma_01/(y_var+std::numeric_limits<float>::epsilon())), gamma_02(gamma_02/(y_put_var+std::numeric_limits<float>::epsilon())), gamma_11(gamma_11*sum_inv_diffs_var/d), gamma_12(gamma_12*sum_inv_put_diffs_var/d), gamma_2(gamma_2/sqrt(y_var+std::numeric_limits<float>::epsilon())), y_var(y_var), y_put_var(y_put_var){}

torch::Tensor StochasticDirectionalLossImpl::forward(torch::Tensor y_pred, torch::Tensor y_put_pred, torch::Tensor dy_pred, torch::Tensor dy_put_pred, torch::Tensor u, torch::Tensor y, torch::Tensor y_put, torch::Tensor dy, torch::Tensor dy_put, torch::Tensor lb){
  auto loss = gamma_01*torch::mse_loss(y_pred, y);
  loss = loss + gamma_02*torch::mse_loss(y_put_pred, y_put);
  auto dir_dy = torch::bmm(u[0].unsqueeze(1), dy.unsqueeze(2)).select(2, 0);
  auto dir_dy_put = torch::bmm(u[1].unsqueeze(1), dy_put.unsqueeze(2)).select(2, 0);
  loss = loss + gamma_11*torch::mse_loss(dy_pred, dir_dy);
  loss = loss + gamma_12*torch::mse_loss(dy_put_pred, dir_dy_put);
  loss = loss + gamma_2*torch::relu(lb-y_pred).mean();
  return loss;
}

NoDiffOnlyCallsLossImpl::NoDiffOnlyCallsLossImpl(double gamma, double y_var):
  gamma(gamma/sqrt(y_var)), y_var(y_var) {}

torch::Tensor NoDiffOnlyCallsLossImpl::forward(torch::Tensor y_pred, torch::Tensor y, torch::Tensor lb){
  auto loss = torch::mse_loss(y_pred, y)/y_var;
  loss = loss + gamma*torch::relu(lb-y_pred).mean();
  return loss;
}
} // namespace layers
} // namespace modules
} // namespace csd