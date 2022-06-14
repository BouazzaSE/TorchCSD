#include <torch/torch.h>
#include "modules/modules.h"
#include "nets/vanilla_locvol/vanilla_locvol.h"
#include "torch/types.h"

namespace csd {
namespace nets {
template <typename T_hidden_real, typename T_hidden_complex, typename T_preoutput_real, typename T_preoutput_complex>
VanillaLocVolImpl<T_hidden_real, T_hidden_complex, T_preoutput_real, T_preoutput_complex>::VanillaLocVolImpl(
  int dim_in, int num_layers, int num_units, float S0, torch::Tensor T_threshold, float eps
  ): S0(S0), eps(eps), num_layers(num_layers), T_threshold(T_threshold.clone())
{
  hidden_layers = std::vector<csd::modules::layers::ComplexRealAffine>();
  hidden_layers.push_back(csd::modules::layers::ComplexRealAffine(dim_in, num_units));
  for (int i = 1; i < num_layers; i++) hidden_layers.push_back(csd::modules::layers::ComplexRealAffine(num_units, num_units));
  preoutput_layer = csd::modules::layers::ComplexRealAffine(num_units, 2);
  ccall_layer = csd::modules::layers::ComplexCallLayer();
  cput_from_call_layer = csd::modules::layers::ComplexPutFromCallLayer();
  hidden_real_activation = T_hidden_real(); hidden_complex_activation = T_hidden_complex();
  preoutput_real_activation = T_preoutput_real(); preoutput_complex_activation = T_preoutput_complex();
  x_mean = torch::zeros({1, dim_in}, torch::dtype(torch::kFloat32));
  x_std = torch::ones({1, dim_in}, torch::dtype(torch::kFloat32));
  init_weights();
  int count = 0;
  for (const auto &layer: hidden_layers){
    register_module("hidden_ComplexRealAffine_"+std::to_string(count), layer);
    count++;
  }
  register_module("hidden_real_activation", hidden_real_activation);
  register_module("hidden_complex_activation", hidden_complex_activation);
  register_module("preoutput_ComplexRealAffine", preoutput_layer);
  register_module("preoutput_real_activation", preoutput_real_activation);
  register_module("preoutput_complex_activation", preoutput_complex_activation);
  register_module("ccall_layer", ccall_layer);
  register_module("cput_from_call_layer", cput_from_call_layer);
  register_buffer("T_threshold", this->T_threshold);
  register_buffer("x_mean", x_mean);
  register_buffer("x_std", x_std);
  // the following is only needed for the gradients during inference
  e0 = torch::zeros({1, dim_in}, torch::dtype(torch::kFloat32));
  e1 = torch::zeros({1, dim_in}, torch::dtype(torch::kFloat32));
  e2 = torch::zeros({1, dim_in}, torch::dtype(torch::kFloat32));
  e0.index_put_({0, 0}, 1);
  e1.index_put_({0, 1}, 1);
  e2.index_put_({0, 2}, 1);
  register_buffer("e0", e0);
  register_buffer("e1", e1);
  register_buffer("e2", e2);
}

template <typename T_hidden_real, typename T_hidden_complex, typename T_preoutput_real, typename T_preoutput_complex>
void VanillaLocVolImpl<T_hidden_real, T_hidden_complex, T_preoutput_real, T_preoutput_complex>::init_weights(){
  for(auto &layer: hidden_layers){
    layer->W.data().normal_();
    layer->W.data() *= std::sqrt(2./layer->W.size(0));
    layer->b.data().zero_();
  }
  preoutput_layer->W.data().normal_();
  preoutput_layer->W.data() /= std::sqrt(preoutput_layer->W.size(0));
  preoutput_layer->b.data().zero_();
}

template <typename T_hidden_real, typename T_hidden_complex, typename T_preoutput_real, typename T_preoutput_complex>
torch::Tensor VanillaLocVolImpl<T_hidden_real, T_hidden_complex, T_preoutput_real, T_preoutput_complex>::standardize(torch::Tensor x){
  return (x - x_mean) / x_std * (x.select(-1, 2).unsqueeze(-1) > T_threshold);
}

template <typename T_hidden_real, typename T_hidden_complex, typename T_preoutput_real, typename T_preoutput_complex>
std::pair<torch::Tensor, torch::Tensor> VanillaLocVolImpl<T_hidden_real, T_hidden_complex, T_preoutput_real, T_preoutput_complex>::standardize_complex(torch::Tensor x, torch::Tensor y){
  auto alpha = (x.select(-1, 2).unsqueeze(-1) > T_threshold) / x_std;
  return {(x - x_mean)*alpha, y*alpha};
}

template <typename T_hidden_real, typename T_hidden_complex, typename T_preoutput_real, typename T_preoutput_complex>
torch::Tensor VanillaLocVolImpl<T_hidden_real, T_hidden_complex, T_preoutput_real, T_preoutput_complex>::forward(torch::Tensor x){
  auto y = standardize(x);
  for(auto &layer: hidden_layers){
    y = layer(y);
    y = hidden_real_activation(y);
  }
  y = preoutput_layer(y);
  y = preoutput_real_activation(y);
  // TODO: fuse the layer below
  return (S0*y.select(-1, 0)-x.select(-1, 1)*torch::exp(-x.select(-1, 0)*x.select(-1, 2))*y.select(-1, 1)).unsqueeze(-1);
}

template <typename T_hidden_real, typename T_hidden_complex, typename T_preoutput_real, typename T_preoutput_complex>
std::pair<torch::Tensor, torch::Tensor> VanillaLocVolImpl<T_hidden_real, T_hidden_complex, T_preoutput_real, T_preoutput_complex>::forward_fwdad_diff(torch::Tensor x){
  auto y = standardize(x);
  torch::Tensor dy;
  int i=0;
  for(auto &layer: hidden_layers){
    auto y_ydiff = layer->forward_diff(y);
    auto y_ydiff_activation = hidden_real_activation->forward_diff(y_ydiff.first);
    y = y_ydiff_activation.first;
    if(i>0){
      dy = torch::matmul(dy, y_ydiff.second * y_ydiff_activation.second.unsqueeze(1));
    } else {
      dy = y_ydiff.second*y_ydiff_activation.second.unsqueeze(1)/x_std.unsqueeze(-1)*(x.select(-1, 2).unsqueeze(-1) > T_threshold).unsqueeze(-1);
    }
    i++;
  }
  auto y_ydiff = preoutput_layer->forward_diff(y);
  auto y_ydiff_activation = preoutput_real_activation->forward_diff(y_ydiff.first);
  y = y_ydiff_activation.first;
  dy = torch::matmul(dy, y_ydiff.second * y_ydiff_activation.second.unsqueeze(1));
  dy = S0*dy.select(2, 0)-(x.select(1, 1)*torch::exp(-x.select(1, 0)*x.select(1, 2))).unsqueeze(1)*dy.select(2, 1)+(x.select(1, 1)*x.select(1, 2)*torch::exp(-x.select(1, 0)*x.select(1, 2))*y.select(1, 1)).unsqueeze(1)*e0-(torch::exp(-x.select(1, 0)*x.select(1, 2))*y.select(1, 1)).unsqueeze(1)*e1+(x.select(1, 0)*x.select(1, 1)*torch::exp(-x.select(1, 0)*x.select(1, 2))*y.select(1, 1)).unsqueeze(1)*e2;
  y = (S0*y.select(1, 0)-x.select(1, 1)*torch::exp(-x.select(1, 0)*x.select(1, 2))*y.select(1, 1)).unsqueeze(1);
  return {y, dy};
}

template <typename T_hidden_real, typename T_hidden_complex, typename T_preoutput_real, typename T_preoutput_complex>
std::pair<torch::Tensor, torch::Tensor> VanillaLocVolImpl<T_hidden_real, T_hidden_complex, T_preoutput_real, T_preoutput_complex>::forward_complex(torch::Tensor x, torch::Tensor y){
  auto uv = standardize_complex(x, y);
  if(y.dim() > x.dim()){
    x = x.expand_as(y);
    uv.first = uv.first.expand_as(y);
  }
  for(auto &layer: hidden_layers){
    uv = layer->forward_complex(uv.first, uv.second);
    uv = hidden_complex_activation(uv.first, uv.second);
  }
  uv = preoutput_layer->forward_complex(uv.first, uv.second);
  uv = preoutput_complex_activation(uv.first, uv.second);
  uv = ccall_layer->forward_complex(uv.first, uv.second, x, y, S0);
  return {uv.first.unsqueeze(-1), uv.second.unsqueeze(-1)};
}

template <typename T_hidden_real, typename T_hidden_complex, typename T_preoutput_real, typename T_preoutput_complex>
std::pair<torch::Tensor, torch::Tensor> VanillaLocVolImpl<T_hidden_real, T_hidden_complex, T_preoutput_real, T_preoutput_complex>::forward_diff(torch::Tensor x, torch::Tensor y){
  auto uv = forward_complex(x, y*eps);
  return {uv.first, uv.second/eps};
}

template <typename T_hidden_real, typename T_hidden_complex, typename T_preoutput_real, typename T_preoutput_complex>
std::pair<torch::Tensor, torch::Tensor> VanillaLocVolImpl<T_hidden_real, T_hidden_complex, T_preoutput_real, T_preoutput_complex>::forward_call_put(torch::Tensor x){
  auto y = forward(x);
  auto y_put = y + (x.select(-1, 1)*torch::exp(-x.select(-1, 0)*x.select(-1, 2))-S0).unsqueeze(-1); // using call-put parity
  return {y, y_put};
}

template <typename T_hidden_real, typename T_hidden_complex, typename T_preoutput_real, typename T_preoutput_complex>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> VanillaLocVolImpl<T_hidden_real, T_hidden_complex, T_preoutput_real, T_preoutput_complex>::forward_call_put_complex(torch::Tensor x, torch::Tensor y){
  auto uv = forward_complex(x, y);
  auto put_uv = cput_from_call_layer->forward_complex(uv.first[1].select(-1, 0), uv.second[1].select(-1, 0), x, y[1], S0); // using call-put parity
  // FIXME: debug cput_from_call_layer, it returns u and v of shape (1024, 2), while we expect something like (2, 1024)
  return {uv.first[0], put_uv.first.unsqueeze(-1), uv.second[0], put_uv.second.unsqueeze(-1)};
}

template <typename T_hidden_real, typename T_hidden_complex, typename T_preoutput_real, typename T_preoutput_complex>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> VanillaLocVolImpl<T_hidden_real, T_hidden_complex, T_preoutput_real, T_preoutput_complex>::forward_call_put_diff(torch::Tensor x, torch::Tensor y){
  torch::Tensor u, u_put, v, v_put;
  std::tie(u, u_put, v, v_put) = forward_call_put_complex(x, y*eps);
  return {u, u_put, v/eps, v_put/eps};
}

template struct VanillaLocVolImpl<csd::modules::activations::RealSoftplus, csd::modules::activations::ComplexSoftplus, csd::modules::activations::RealSigmoid, csd::modules::activations::ComplexSigmoid>;
} // namespace nets
} // namespace csd
