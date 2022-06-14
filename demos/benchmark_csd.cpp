#include "cuda_runtime.h"
#include "modules/modules.h"
#include "utils/parse_args.h"
#include <ATen/Functions.h>
#include <ATen/core/TensorBody.h>
#include <c10/core/Device.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cmath>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <torch/torch.h>
#include <vector>

struct CSDTestImpl : torch::nn::Module {
  std::vector<csd::modules::layers::ComplexRealAffine> layers;
  csd::modules::activations::ComplexSoftplus complex_softplus;
  CSDTestImpl(int dim_in, int dim_out, int num_layers, int num_units){
    layers = std::vector<csd::modules::layers::ComplexRealAffine>();
    layers.push_back(csd::modules::layers::ComplexRealAffine(dim_in, num_units));
    for (int i = 1; i < num_layers - 1; i++) layers.push_back(csd::modules::layers::ComplexRealAffine(num_units, num_units));
    layers.push_back(csd::modules::layers::ComplexRealAffine(num_units, dim_out));
    int count = 0;
    for (const auto &layer: layers){
      register_module("ComplexRealAffine_"+std::to_string(count), layer);
      count++;
    }
    complex_softplus = csd::modules::activations::ComplexSoftplus();
    register_module("ComplexSoftPlus_0", complex_softplus);
  }
  torch::Tensor forward(torch::Tensor x){
    for (auto &layer : layers){
      x = layer(x);
      x = torch::softplus(x);
    }
    return x;
  }
  std::pair<torch::Tensor, torch::Tensor> forward_diff(torch::Tensor x, torch::Tensor y){
    std::pair<torch::Tensor, torch::Tensor> uv = std::pair<torch::Tensor, torch::Tensor>();
    uv.first = x; uv.second = y.squeeze() * 1e-7;
    for (auto &layer : layers){
      uv = layer->forward_complex(uv.first, uv.second);
      uv = complex_softplus(uv.first, uv.second);
    }
    return uv;
  }
}; TORCH_MODULE(CSDTest);

struct ForwardFullGradTestImpl : torch::nn::Module {
  std::vector<csd::modules::layers::RealAffine> layers;
  ForwardFullGradTestImpl(int dim_in, int dim_out, int num_layers, int num_units){
    layers = std::vector<csd::modules::layers::RealAffine>();
    layers.push_back(csd::modules::layers::RealAffine(dim_in, num_units));
    for (int i = 1; i < num_layers - 1; i++) layers.push_back(csd::modules::layers::RealAffine(num_units, num_units));
    layers.push_back(csd::modules::layers::RealAffine(num_units, dim_out));
    int count = 0;
    for (const auto &layer: layers){
      register_module("RealAffine_"+std::to_string(count), layer);
      count++;
    }
  }
  torch::Tensor forward(torch::Tensor x){
    for (auto &layer : layers){
      x = layer(x);
      x = torch::softplus(x);
    }
    return x;
  }
  std::pair<torch::Tensor, torch::Tensor> forward_diff(torch::Tensor x){
    std::pair<torch::Tensor, torch::Tensor> ffdiff;
    torch::Tensor dx;
    int count = 0;
    for (auto &layer : layers){
      ffdiff = layer->forward_diff(x);
      x = torch::softplus(ffdiff.first);
      if(count > 0){
        dx = torch::matmul(dx, ffdiff.second) * torch::sigmoid(ffdiff.first).unsqueeze(1);
      } else {
        dx = ffdiff.second * torch::sigmoid(ffdiff.first).unsqueeze(1);
      }
      count++;
    }
    return {x, dx};
  }
}; TORCH_MODULE(ForwardFullGradTest);

struct BackwardFullGradTestImpl : torch::nn::Module {
  std::vector<csd::modules::layers::RealAffine> layers;
  BackwardFullGradTestImpl(int dim_in, int dim_out, int num_layers, int num_units){
    layers = std::vector<csd::modules::layers::RealAffine>();
    layers.push_back(csd::modules::layers::RealAffine(dim_in, num_units));
    for (int i = 1; i < num_layers - 1; i++) layers.push_back(csd::modules::layers::RealAffine(num_units, num_units));
    layers.push_back(csd::modules::layers::RealAffine(num_units, dim_out));
    int count = 0;
    for (const auto &layer: layers){
      register_module("RealAffine_"+std::to_string(count), layer);
      count++;
    }
  }
  torch::Tensor forward(torch::Tensor x){
    for (auto &layer : layers){
      x = layer(x);
      x = torch::softplus(x);
    }
    return x;
  }
  std::pair<torch::Tensor, torch::Tensor> forward_diff(torch::Tensor x){
    std::pair<torch::Tensor, torch::Tensor> ffdiff;
    torch::Tensor dx;
    std::vector<torch::Tensor> diffs = std::vector<torch::Tensor>();
    for (auto &layer : layers){
      ffdiff = layer->forward_diff(x);
      x = torch::softplus(ffdiff.first);
      dx = ffdiff.second * torch::sigmoid(ffdiff.first).unsqueeze(1);
      diffs.push_back(dx);
    }
    for (int i=0; i<layers.size()-1; i++) dx = torch::matmul(diffs[layers.size()-2-i], dx);
    return {x, dx};
  }
}; TORCH_MODULE(BackwardFullGradTest);

struct ForwardDirectionalTestImpl : torch::nn::Module {
  std::vector<csd::modules::layers::RealAffine> layers;
  ForwardDirectionalTestImpl(int dim_in, int dim_out, int num_layers, int num_units){
    layers = std::vector<csd::modules::layers::RealAffine>();
    layers.push_back(csd::modules::layers::RealAffine(dim_in, num_units));
    for (int i = 1; i < num_layers - 1; i++) layers.push_back(csd::modules::layers::RealAffine(num_units, num_units));
    layers.push_back(csd::modules::layers::RealAffine(num_units, dim_out));
    int count = 0;
    for (const auto &layer: layers){
      register_module("RealAffine_"+std::to_string(count), layer);
      count++;
    }
  }
  torch::Tensor forward(torch::Tensor x){
    for (auto &layer : layers){
      x = layer(x);
      x = torch::softplus(x);
    }
    return x;
  }
  std::pair<torch::Tensor, torch::Tensor> forward_diff(torch::Tensor x, torch::Tensor y){
    std::pair<torch::Tensor, torch::Tensor> ffdiff;
    int count = 0;
    for (auto &layer : layers){
      ffdiff = layer->forward_diff(x);
      x = torch::softplus(ffdiff.first);
      y = torch::matmul(y, ffdiff.second) * torch::sigmoid(ffdiff.first).unsqueeze(1);
      count++;
    }
    return {x, y};
  }
}; TORCH_MODULE(ForwardDirectionalTest);

struct BackwardDirectionalTestImpl : torch::nn::Module {
  std::vector<csd::modules::layers::RealAffine> layers;
  BackwardDirectionalTestImpl(int dim_in, int dim_out, int num_layers, int num_units){
    layers = std::vector<csd::modules::layers::RealAffine>();
    layers.push_back(csd::modules::layers::RealAffine(dim_in, num_units));
    for (int i = 1; i < num_layers - 1; i++) layers.push_back(csd::modules::layers::RealAffine(num_units, num_units));
    layers.push_back(csd::modules::layers::RealAffine(num_units, dim_out));
    int count = 0;
    for (const auto &layer: layers){
      register_module("RealAffine_"+std::to_string(count), layer);
      count++;
    }
  }
  torch::Tensor forward(torch::Tensor x){
    for (auto &layer : layers){
      x = layer(x);
      x = torch::softplus(x);
    }
    return x;
  }
  std::pair<torch::Tensor, torch::Tensor> forward_diff(torch::Tensor x, torch::Tensor y){
    std::pair<torch::Tensor, torch::Tensor> ffdiff;
    torch::Tensor dx;
    std::vector<torch::Tensor> diffs = std::vector<torch::Tensor>();
    for (auto &layer : layers){
      ffdiff = layer->forward_diff(x);
      x = torch::softplus(ffdiff.first);
      dx = ffdiff.second * torch::sigmoid(ffdiff.first).unsqueeze(1);
      diffs.push_back(dx);
    }
    for (int i=0; i<layers.size()-1; i++) dx = torch::matmul(diffs[layers.size()-2-i], dx);
    dx = torch::matmul(y, dx);
    return {x, dx};
  }
}; TORCH_MODULE(BackwardDirectionalTest);

template <typename T>
void benchmark_fullgrad(std::string name, T module, int batch_size, int num_runs, torch::Device device){
  module->train();
  auto module_optimizer = torch::optim::Adam(module->parameters());
  std::pair<torch::Tensor, torch::Tensor> uv;
  auto x = torch::empty({batch_size, 28}, torch::dtype(torch::kFloat32).device(device));
  float dt; double dram;
  double elapsed_time = 0;
  double elapsed_time_stdev = 0;
  double ram_usage = 0;
  double ram_usage_stdev = 0;
  long mem_start, mem_stop;
  c10::cuda::CUDACachingAllocator::DeviceStats stats;
  auto stream = c10::cuda::getCurrentCUDAStream();
  cudaEvent_t evt_start, evt_stop;
  C10_CUDA_CHECK(cudaEventCreateWithFlags(&evt_start, cudaEventDefault));
  C10_CUDA_CHECK(cudaEventCreateWithFlags(&evt_stop, cudaEventDefault));
  for (int i = 0; i < num_runs + 10; i++) {
    for(auto &layer: module->layers){
      layer->W.data().normal_();
      layer->W.data() /= std::sqrt(layer->W.sizes()[0]);
      layer->b.data().zero_();
    }
    c10::cuda::CUDACachingAllocator::resetPeakStats(device.index());
    stats = c10::cuda::CUDACachingAllocator::getDeviceStats(device.index());
    mem_start = stats.allocated_bytes[(unsigned int)c10::cuda::CUDACachingAllocator::StatType::AGGREGATE].peak;
    C10_CUDA_CHECK(cudaEventRecord(evt_start, stream.stream()));
    module_optimizer.zero_grad();
    uv = module->forward_diff(x);
    (uv.first.squeeze() + uv.second.squeeze().mean(1)).mean().backward();
    module_optimizer.step();
    C10_CUDA_CHECK(cudaEventRecord(evt_stop, stream.stream()));
    C10_CUDA_CHECK(cudaEventSynchronize(evt_stop));
    C10_CUDA_CHECK(cudaEventElapsedTime(&dt, evt_start, evt_stop));
    stats = c10::cuda::CUDACachingAllocator::getDeviceStats(device.index());
    mem_stop = stats.allocated_bytes[(unsigned int)c10::cuda::CUDACachingAllocator::StatType::AGGREGATE].peak;
    dram = ((double)(mem_stop - mem_start)) / (1 << 20);
    if (i >= 10) {
      elapsed_time += dt / num_runs;
      ram_usage += dram / num_runs;
      elapsed_time_stdev += dt * dt / num_runs;
      ram_usage_stdev += dram * dram / num_runs;
    }
  }
  elapsed_time_stdev -= elapsed_time * elapsed_time;
  elapsed_time_stdev = elapsed_time_stdev > 0 ? elapsed_time_stdev : 0;
  elapsed_time_stdev = sqrt(elapsed_time_stdev);
  ram_usage_stdev -= ram_usage * ram_usage;
  ram_usage_stdev = ram_usage_stdev > 0 ? ram_usage_stdev : 0;
  ram_usage_stdev = sqrt(ram_usage_stdev);
  std::cout.precision(3);
  std::cout << "[" << name << "] Elapsed time: mean=" << elapsed_time
            << ", stdev=" << elapsed_time_stdev << std::endl;
  std::cout << "[" << name << "] GPU RAM usage (MB): mean=" << ram_usage
            << ", stdev=" << ram_usage_stdev << std::endl;
}

template <typename T>
void benchmark_directional(std::string name, T module, int batch_size, int num_runs, torch::Device device){
  module->train();
  auto module_optimizer = torch::optim::Adam(module->parameters());
  std::pair<torch::Tensor, torch::Tensor> uv;
  auto x = torch::empty({batch_size, 28}, torch::dtype(torch::kFloat32).device(device));
  auto y = torch::empty({batch_size, 1, 28}, torch::dtype(torch::kFloat32).device(device));
  float dt; double dram;
  double elapsed_time = 0;
  double elapsed_time_stdev = 0;
  double ram_usage = 0;
  double ram_usage_stdev = 0;
  long mem_start, mem_stop;
  c10::cuda::CUDACachingAllocator::DeviceStats stats;
  auto stream = c10::cuda::getCurrentCUDAStream();
  cudaEvent_t evt_start, evt_stop;
  C10_CUDA_CHECK(cudaEventCreateWithFlags(&evt_start, cudaEventDefault));
  C10_CUDA_CHECK(cudaEventCreateWithFlags(&evt_stop, cudaEventDefault));
  for (int i = 0; i < num_runs + 10; i++) {
    for(auto &layer: module->layers){
      layer->W.data().normal_();
      layer->W.data() /= std::sqrt(layer->W.sizes()[0]);
      layer->b.data().zero_();
    }
    c10::cuda::CUDACachingAllocator::resetPeakStats(device.index());
    stats = c10::cuda::CUDACachingAllocator::getDeviceStats(device.index());
    mem_start =stats.allocated_bytes[(unsigned int)c10::cuda::CUDACachingAllocator::StatType::AGGREGATE].peak;
    C10_CUDA_CHECK(cudaEventRecord(evt_start, stream.stream()));
    module_optimizer.zero_grad();
    uv = module->forward_diff(x, y);
    (uv.first.squeeze() + uv.second.squeeze()).mean().backward();
    module_optimizer.step();
    C10_CUDA_CHECK(cudaEventRecord(evt_stop, stream.stream()));
    C10_CUDA_CHECK(cudaEventSynchronize(evt_stop));
    C10_CUDA_CHECK(cudaEventElapsedTime(&dt, evt_start, evt_stop));
    stats = c10::cuda::CUDACachingAllocator::getDeviceStats(device.index());
    mem_stop = stats.allocated_bytes[(unsigned int)c10::cuda::CUDACachingAllocator::StatType::AGGREGATE].peak;
    dram = ((double)(mem_stop - mem_start)) / (1 << 20);
    if (i >= 10) {
      elapsed_time += dt / num_runs;
      ram_usage += dram / num_runs;
      elapsed_time_stdev += dt * dt / num_runs;
      ram_usage_stdev += dram * dram / num_runs;
    }
  }
  elapsed_time_stdev -= elapsed_time * elapsed_time;
  elapsed_time_stdev = elapsed_time_stdev > 0 ? elapsed_time_stdev : 0;
  elapsed_time_stdev = sqrt(elapsed_time_stdev);
  ram_usage_stdev -= ram_usage * ram_usage;
  ram_usage_stdev = ram_usage_stdev > 0 ? ram_usage_stdev : 0;
  ram_usage_stdev = sqrt(ram_usage_stdev);
  std::cout.precision(3);
  std::cout << "[" << name << "] Elapsed time: mean=" << elapsed_time
            << ", stdev=" << elapsed_time_stdev << std::endl;
  std::cout << "[" << name << "] GPU RAM usage (MB): mean=" << ram_usage
            << ", stdev=" << ram_usage_stdev << std::endl;
}

int main(int argc, char *argv[]) {
  if (argc != 5)
    throw std::invalid_argument("Only four arguments please: device_idx, "
                                "num_layers, num_units, num_runs.");
  int device_idx, num_layers, num_units, num_runs;
  csd::utils::args::parse_arg(1, argv[1], &device_idx);
  csd::utils::args::parse_arg(2, argv[2], &num_layers);
  csd::utils::args::parse_arg(3, argv[3], &num_units);
  csd::utils::args::parse_arg(4, argv[4], &num_runs);
  auto device = torch::Device(torch::kCUDA, device_idx);
  int batch_size = 16384;
  C10_CUDA_CHECK(cudaSetDevice(device_idx));
  auto csd_test = CSDTest(28, 1, num_layers, num_units);
  auto forward_fullgrad_test = ForwardFullGradTest(28, 1, num_layers, num_units);
  auto backward_fullgrad_test = BackwardFullGradTest(28, 1, num_layers, num_units);
  auto forward_directional_test = ForwardDirectionalTest(28, 1, num_layers, num_units);
  auto backward_directional_test = BackwardDirectionalTest(28, 1, num_layers, num_units);
  csd_test->to(device);
  forward_fullgrad_test->to(device);
  backward_fullgrad_test->to(device);
  forward_directional_test->to(device);
  backward_directional_test->to(device);
  benchmark_fullgrad("ForwardFullGrad", forward_fullgrad_test, batch_size, num_runs, device);
  benchmark_fullgrad("BackwardFullGrad", backward_fullgrad_test, batch_size, num_runs, device);
  benchmark_directional("ForwardDirectionalGrad", forward_directional_test, batch_size, num_runs, device);
  benchmark_directional("BackwardDirectionalGrad", backward_directional_test, batch_size, num_runs, device);
  benchmark_directional("CSD", csd_test, batch_size, num_runs, device);
}
