#pragma once
#include <torch/torch.h>

namespace csd {
namespace modules {
namespace layers {
namespace inference {
std::pair<torch::Tensor, torch::Tensor> ccall_layer(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double);
}
}
}
}
