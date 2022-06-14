#pragma once
#include <algorithm>
#include <torch/torch.h>
#include "modules/layers/layers.h"
#include "nets/vanilla_locvol/vanilla_locvol.h"

namespace csd {
namespace training {
template<typename T>
using IsVanillaCallPutNetImpl = typename std::enable_if<std::is_base_of<csd::nets::VanillaCallPutNetImpl, T>::value, bool>::type;
void generate_diff_direction(torch::Tensor, torch::Tensor, torch::Tensor, double, double);

template<typename T, IsVanillaCallPutNetImpl<T> = true>//, typename = typename std::enable_if<std::is_base_of<csd::nets::VanillaCallPutNetImpl, T>::value>::type>
void train_call_put(torch::nn::ModuleHolder<T>, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, long, long, double, double, long, long, double, double, double, double, double, double, int, int, int);

template<typename T, IsVanillaCallPutNetImpl<T> = true>//, typename = typename std::enable_if<std::is_base_of<csd::nets::VanillaCallPutNetImpl, T>::value>::type>
void compute_stats(torch::nn::ModuleHolder<T>, torch::Tensor, torch::Tensor, torch::Tensor, long);

template<typename T, IsVanillaCallPutNetImpl<T> = true>//, typename = typename std::enable_if<std::is_base_of<csd::nets::VanillaCallPutNetImpl, T>::value>::type>
void compute_twin_stats(torch::nn::ModuleHolder<T>, torch::Tensor, torch::Tensor, torch::Tensor, long);
} // namespace training
} // namespace csd
