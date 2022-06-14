#pragma once
#include <torch/torch.h>

namespace csd {
namespace modules {
namespace activations {
namespace inference {
std::pair<torch::Tensor, torch::Tensor> csoftplus(torch::Tensor x, torch::Tensor y);
std::pair<torch::Tensor, torch::Tensor> csigmoid(torch::Tensor x, torch::Tensor y);
}
}
}
}
