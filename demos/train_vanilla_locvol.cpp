/*
Copyright 2022 Bouazza SAADEDDINE

This file is part of TorchCSD.

TorchCSD is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

TorchCSD is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with TorchCSD.  If not, see <https://www.gnu.org/licenses/>.
*/



#include <ATen/Context.h>
#include <ATen/Functions.h>
#include <c10/core/TensorOptions.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include <curand_kernel.h>
#include <iomanip>
#include <iostream>
#include <torch/optim/lbfgs.h>
#include <torch/torch.h>
#include <torch/types.h>
#include "nets/vanilla_locvol/vanilla_locvol.h"
#include "simulators/simulators.h"
#include "training/training.h"
#include "utils/parse_args.h"

int main(int argc, char *argv[]){
    if (argc != 21) 
        throw std::invalid_argument("20 arguments please: device_idx, train_eval, fit_locvol, num_paths, num_inner_samples, num_layers, num_units_multiplier, num_epochs, num_epochs_warmup, batch_size, batch_size_valid, lr, lr_warmup, gamma_01, gamma_02, gamma_11, gamma_12, gamma_2, halve_every and dumpname.");
    int device_idx;
    bool train_eval, fit_locvol;
    long num_paths, num_inner_samples, num_layers, num_units_multiplier, num_epochs, num_epochs_warmup, batch_size, batch_size_valid, halve_every;
    float lr, lr_warmup, gamma_01, gamma_02, gamma_11, gamma_12, gamma_2;
    std::string dumpname;
    csd::utils::args::parse_arg(1, argv[1], &device_idx); // 1st arg is the device (CUDA-capable GPU) index
    csd::utils::args::parse_arg(2, argv[2], &train_eval); // 2nd arg is 1 if we want to train a new model, 0 if we want to evaluate an existing one instead
    csd::utils::args::parse_arg(3, argv[3], &fit_locvol); // 3rd arg is 1 if we want to fit the option data found in options_data.bin, 0 if not 
    csd::utils::args::parse_arg(4, argv[4], &num_paths); // 4th arg is the number of paths, ex: 1<<23
    csd::utils::args::parse_arg(5, argv[5], &num_inner_samples); // 5th arg is the number of inner paths, ex: 32
    csd::utils::args::parse_arg(6, argv[6], &num_layers); // 6th arg is the number of layers, ex: 6
    csd::utils::args::parse_arg(7, argv[7], &num_units_multiplier); // 7th arg is a multiplier K such that the number of units per layer is equal to the input dimension multiplied by K, ex: 2
    csd::utils::args::parse_arg(8, argv[8], &num_epochs); // 8th arg is the number of epochs for training, ex: 300
    csd::utils::args::parse_arg(9, argv[9], &num_epochs_warmup); // 9th arg is the number of warmup epochs, ex: 30
    csd::utils::args::parse_arg(10, argv[10], &batch_size); // 10th arg is the batch size used for training, ex: 4*4096
    csd::utils::args::parse_arg(11, argv[11], &batch_size_valid); // 11th arg is the batch size used for validation, ex: 8*4096
    csd::utils::args::parse_arg(12, argv[12], &lr); // 12th arg is the learning rate for training, ex: 0.01
    csd::utils::args::parse_arg(13, argv[13], &lr_warmup); // 13th arg is the learning rate during warmup, ex: 0.005
    csd::utils::args::parse_arg(14, argv[14], &gamma_01); // 14th arg is the weight for the call payoffs error term, ex: 1
    csd::utils::args::parse_arg(15, argv[15], &gamma_02); // 15th arg is the weight for the put payoffs error term, ex: 1
    csd::utils::args::parse_arg(16, argv[16], &gamma_11); // 16th arg is the weight for the call payoff sensitivities error term, ex: 1
    csd::utils::args::parse_arg(17, argv[17], &gamma_12); // 17th arg is the weight for the put payoff sensitivities error term, ex: 1
    csd::utils::args::parse_arg(18, argv[18], &gamma_2); // 18th arg is the weight for the no-arbitrage lower bound penalization term, ex: 1
    csd::utils::args::parse_arg(19, argv[19], &halve_every); // 19th arg is a number halve_every such that the learning rate is halved every halve_every epochs, ex: 50
    csd::utils::args::parse_arg(20, argv[20], &dumpname); // 20th arg name of the dump file for the neural network
    std::cout << dumpname << std::endl;
    auto device = torch::Device(torch::kCUDA, device_idx);
    // setting the bounds for the model parameters
    UniformBounds<float> r_b = {0., 0.05};
    UniformBounds<float> stk_b = {0.25, 2.1};
    UniformBounds<float> mat_b = {0.05, 2.5};
    UniformBounds<float> lv_b = {0.1, 2.};
    // we stop shortly before the MC horizon in the local vol grid to ensure last nodes are sufficiently explored
    UniformBounds<float> nodes_x_b = {1.0/12, 2.};
    UniformBounds<float> nodes_y_b = {static_cast<float>(log(0.25)), static_cast<float>(log(2.1))}; //{static_cast<float>(log(0.25)), static_cast<float>(log(2.1))};
    // we are considering a local volatility grid of size 5x5
    long num_nodes_x = 5;
    long num_nodes_y = 5;
    const long num_inputs = 3+num_nodes_x*num_nodes_y;
    C10_CUDA_CHECK(cudaSetDevice(device.index()));
    auto stream = c10::cuda::getCurrentCUDAStream();
    cudaEvent_t evt_start, evt_stop;
    float elapsed_time;
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&evt_start, cudaEventDefault));
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&evt_stop, cudaEventDefault));
    torch::manual_seed(0);
    auto mat_threshold = torch::empty({1, num_inputs}, torch::dtype(torch::kFloat32).device(device));
    auto net = csd::nets::VanillaLocVolSoftplusSigmoid(num_inputs, num_layers, num_units_multiplier*num_inputs, 1.0, mat_threshold, std::numeric_limits<float>::epsilon());
    net->to(device);
    if(train_eval){
        long num_steps_per_year = 1000;
        curandStateMRG32k3a_t *curand_states;
        auto nodes_x = torch::linspace(nodes_x_b.min, nodes_x_b.max, num_nodes_x, torch::dtype(torch::kFloat32).device(device));
        auto nodes_y = torch::linspace(nodes_y_b.min, nodes_y_b.max, num_nodes_y, torch::dtype(torch::kFloat32).device(device));
        // the mat_threshold array constructed here will be used to help ensure zero partial derivatives with respect to nodes corresponding to time steps beyond the maturity given in the input
        mat_threshold.zero_();
        mat_threshold.slice(1, 3+num_nodes_y) = nodes_x.slice(0, c10::nullopt, -1).repeat_interleave(num_nodes_y);
        C10_CUDA_CHECK(cudaMalloc((void **)&curand_states, num_paths*sizeof(curandState))); // TODO: we probably don't need that many states
        csd::simulators::kernels::curand::init_states(curand_states, num_paths, stream);
        auto x = torch::empty({num_paths, 3+num_nodes_x*num_nodes_y}, torch::dtype(torch::kFloat32).device(device));
        auto r = x.select(-1, 0);
        auto stk = x.select(-1, 1);
        auto mat = x.select(-1, 2);
        auto lv = x.slice(1, 3).view({-1, num_nodes_x, num_nodes_y});
        C10_CUDA_CHECK(cudaEventRecord(evt_start, stream.stream()));
        // we generate random samples of model and product parameters
        csd::simulators::kernels::locvol::generate_params(curand_states, r_b, stk_b, mat_b, lv_b, r, stk, mat, lv, num_steps_per_year, stream);
        C10_CUDA_CHECK(cudaEventRecord(evt_stop, stream.stream()));
        C10_CUDA_CHECK(cudaEventSynchronize(evt_stop));
        C10_CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, evt_start, evt_stop));
        std::cout << std::fixed;
        std::cout << std::setprecision(4);
        std::cout << "Generated " << num_paths << " instances of model parameters in " << elapsed_time/1000 << " seconds" << std::endl;
        float S0 = 1.;
        auto call_payoffs = torch::empty({num_paths, 1}, torch::dtype(torch::kFloat32).device(device));
        auto put_payoffs = torch::empty({num_paths, 1}, torch::dtype(torch::kFloat32).device(device));
        auto call_dpayoffs = torch::empty({num_paths, 3+num_nodes_x*num_nodes_y}, torch::dtype(torch::kFloat32).device(device));
        auto put_dpayoffs = torch::empty({num_paths, 3+num_nodes_x*num_nodes_y}, torch::dtype(torch::kFloat32).device(device));
        C10_CUDA_CHECK(cudaEventRecord(evt_start, stream.stream()));
        // in front of each realization of model and product parameters, we will generate one payoff conditional on those params
        // (or more precisely an average of num_inner_samples where the latter is usually 32 if we want to reduce variance at a small computational cost)
        csd::simulators::kernels::locvol::generate_payoffs<curandStateMRG32k3a_t, float, 5, 5>(curand_states, r, stk, mat, lv, nodes_x, nodes_y, call_payoffs.select(-1, 0), put_payoffs.select(-1, 0), call_dpayoffs, put_dpayoffs, S0, num_steps_per_year, num_inner_samples, stream);
        C10_CUDA_CHECK(cudaEventRecord(evt_stop, stream.stream()));
        C10_CUDA_CHECK(cudaEventSynchronize(evt_stop));
        C10_CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, evt_start, evt_stop));
        std::cout << "Generated " << num_paths << " labels in " << elapsed_time/1000 << " seconds" << std::endl;
        // we compute the no-arbitrage lower bound for the call price that needs to be satisfied
        // the lower bound will be enforced via a penalization during training
        auto lb = (S0-(-r*mat).exp_().mul_(stk)).relu_().unsqueeze(1);
        auto train_slice = torch::indexing::Slice(0, num_paths>>1);
        auto valid_slice = torch::indexing::Slice(num_paths>>1, num_paths);
        C10_CUDA_CHECK(cudaEventRecord(evt_start, stream.stream()));
        // we launch the training of our neural network
        csd::training::train_call_put(net, x.index({train_slice}), x.index({valid_slice}), call_payoffs.index({train_slice}), call_payoffs.index({valid_slice}), put_payoffs.index({train_slice}), put_payoffs.index({valid_slice}), call_dpayoffs.index({train_slice}), call_dpayoffs.index({valid_slice}), put_dpayoffs.index({train_slice}), put_dpayoffs.index({valid_slice}), lb.index({train_slice}), lb.index({valid_slice}), batch_size, batch_size_valid, lr, lr_warmup, num_epochs, num_epochs_warmup, gamma_01, gamma_02, gamma_11, gamma_12, gamma_2, log(2.)/halve_every, 5, 5, 5);
        C10_CUDA_CHECK(cudaEventRecord(evt_stop, stream.stream()));
        C10_CUDA_CHECK(cudaEventSynchronize(evt_stop));
        C10_CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, evt_start, evt_stop));
        // dumping the trained neural network
        // it can then be trivially loaded for inference using torch::load
        torch::save(net, dumpname);
        std::cout << "Neural network trained in " << elapsed_time/1000 << " seconds" << std::endl;
        C10_CUDA_CHECK(cudaEventRecord(evt_start, stream.stream()));
        csd::simulators::kernels::locvol::generate_params(curand_states, r_b, stk_b, mat_b, lv_b, r, stk, mat, lv, num_steps_per_year, stream);
        C10_CUDA_CHECK(cudaEventRecord(evt_stop, stream.stream()));
        C10_CUDA_CHECK(cudaEventSynchronize(evt_stop));
        C10_CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, evt_start, evt_stop));
        std::cout << "Generated out-of-sample " << num_paths << " instances of model parameters in " << elapsed_time/1000 << " seconds" << std::endl;
        C10_CUDA_CHECK(cudaEventRecord(evt_start, stream.stream()));
        csd::simulators::kernels::locvol::generate_payoffs<curandStateMRG32k3a_t, float, 5, 5>(curand_states, r, stk, mat, lv, nodes_x, nodes_y, call_payoffs.select(-1, 0), put_payoffs.select(-1, 0), call_dpayoffs, put_dpayoffs, S0, num_steps_per_year, num_inner_samples, stream);
        C10_CUDA_CHECK(cudaEventRecord(evt_stop, stream.stream()));
        C10_CUDA_CHECK(cudaEventSynchronize(evt_stop));
        C10_CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, evt_start, evt_stop));
        std::cout << "Generated out-of-sample " << num_paths << " labels in " << elapsed_time/1000 << " seconds" << std::endl;
        csd::training::compute_stats(net, x, call_payoffs, call_dpayoffs, batch_size_valid);
    } else {
        // Launching a twin-simulation procedure to evaluate the distance against ground-truth prices
        torch::load(net, dumpname);
        long num_steps_per_year = 1000;
        curandStateMRG32k3a_t *curand_states;
        auto nodes_x = torch::linspace(nodes_x_b.min, nodes_x_b.max, num_nodes_x, torch::dtype(torch::kFloat32).device(device));
        auto nodes_y = torch::linspace(nodes_y_b.min, nodes_y_b.max, num_nodes_y, torch::dtype(torch::kFloat32).device(device));
        mat_threshold.zero_();
        mat_threshold.slice(1, 3+num_nodes_y) = nodes_x.slice(0, c10::nullopt, -1).repeat_interleave(num_nodes_y);
        C10_CUDA_CHECK(cudaMalloc((void **)&curand_states, num_paths*sizeof(curandState))); // TODO: we probably don't need that many states
        csd::simulators::kernels::curand::init_states(curand_states, num_paths, stream);
        auto x = torch::empty({num_paths, 3+num_nodes_x*num_nodes_y}, torch::dtype(torch::kFloat32).device(device));
        auto r = x.select(-1, 0);
        auto stk = x.select(-1, 1);
        auto mat = x.select(-1, 2);
        auto lv = x.slice(1, 3).view({-1, num_nodes_x, num_nodes_y});
        C10_CUDA_CHECK(cudaEventRecord(evt_start, stream.stream()));
        for(int i=0; i<2; i++){
            csd::simulators::kernels::locvol::generate_params(curand_states, r_b, stk_b, mat_b, lv_b, r, stk, mat, lv, num_steps_per_year, stream);
        }
        C10_CUDA_CHECK(cudaEventRecord(evt_stop, stream.stream()));
        C10_CUDA_CHECK(cudaEventSynchronize(evt_stop));
        C10_CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, evt_start, evt_stop));
        std::cout << std::fixed;
        std::cout << std::setprecision(8);
        std::cout << "Generated " << num_paths << " instances of model parameters in " << elapsed_time/1000 << " seconds" << std::endl;
        float S0 = 1.;
        auto call_payoffs = torch::empty({num_paths, 1}, torch::dtype(torch::kFloat32).device(device));
        auto call_payoffs_twin = torch::empty({num_paths, 1}, torch::dtype(torch::kFloat32).device(device));
        auto put_payoffs = torch::empty({num_paths, 1}, torch::dtype(torch::kFloat32).device(device));
        auto put_payoffs_twin = torch::empty({num_paths, 1}, torch::dtype(torch::kFloat32).device(device));
        auto call_dpayoffs = torch::empty({num_paths, 3+num_nodes_x*num_nodes_y}, torch::dtype(torch::kFloat32).device(device));
        auto call_dpayoffs_twin = torch::empty({num_paths, 3+num_nodes_x*num_nodes_y}, torch::dtype(torch::kFloat32).device(device));
        auto put_dpayoffs = torch::empty({num_paths, 3+num_nodes_x*num_nodes_y}, torch::dtype(torch::kFloat32).device(device));
        auto put_dpayoffs_twin = torch::empty({num_paths, 3+num_nodes_x*num_nodes_y}, torch::dtype(torch::kFloat32).device(device));
        C10_CUDA_CHECK(cudaEventRecord(evt_start, stream.stream()));
        for(int i=0; i<2; i++){
            csd::simulators::kernels::locvol::generate_payoffs<curandStateMRG32k3a_t, float, 5, 5>(curand_states, r, stk, mat, lv, nodes_x, nodes_y, call_payoffs.select(-1, 0), put_payoffs.select(-1, 0), call_dpayoffs, put_dpayoffs, S0, num_steps_per_year, num_inner_samples, stream);
            csd::simulators::kernels::locvol::generate_payoffs<curandStateMRG32k3a_t, float, 5, 5>(curand_states, r, stk, mat, lv, nodes_x, nodes_y, call_payoffs_twin.select(-1, 0), put_payoffs_twin.select(-1, 0), call_dpayoffs_twin, put_dpayoffs_twin, S0, num_steps_per_year, num_inner_samples, stream);
        }
        C10_CUDA_CHECK(cudaEventRecord(evt_stop, stream.stream()));
        C10_CUDA_CHECK(cudaEventSynchronize(evt_stop));
        C10_CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, evt_start, evt_stop));
        std::cout << "Generated " << num_paths << " label twins in " << elapsed_time/1000 << " seconds" << std::endl;
        csd::training::compute_stats(net, x, call_payoffs, call_dpayoffs, batch_size_valid);
        std::cout << "----- CALL PRICE ERROR -----" << std::endl;
        csd::training::compute_twin_stats(net, x, call_payoffs, call_payoffs_twin, batch_size_valid);
    }
    
    if(fit_locvol){
        // Here a stochastic gradient descent is used for calibration and is ONLY FOR ILLUSTRATION PURPOSES.
        // More serious calibration routines implementations should use more advanced algorithms like Levenberg-Marquardt.
        // Calibration is entirely orthogonal to the presented approach and is NOT the focus of the paper or this library.
        // The main idea is that, whatever optimization-based calibration routine is used,
        // the calibration will be fast if the pricing is fast,
        // and having the latter is exactly the goal of the neural network training above
        // (focused on call options, and via put-call parity, on put options too, hence tailored for calibration
        // procedures where those are the calibration instruments).

        std::ifstream f("options_data.bin", std::ios::out | std::ios::binary);
        if(!f) {
            std::cout << "Could not open options data." << std::endl;
            return 1;
        }
        std::uint32_t calls_arr_shape[2];
        f.read((char*) (&calls_arr_shape[0]), sizeof(std::uint32_t));
        f.read((char*) (&calls_arr_shape[1]), sizeof(std::uint32_t));
        auto calls_arr = torch::empty({(long) calls_arr_shape[0], (long) calls_arr_shape[1]}, torch::dtype(torch::kFloat32).device(device));
        for(int i=0; i<calls_arr.numel(); i++){
            double tmp_float64;
            f.read((char*) &tmp_float64, sizeof(double));
            calls_arr.view(-1)[i] = (float) tmp_float64;
        }
        f.close();
        std::cout << "Successfully loaded " << calls_arr_shape[0] << " option prices." << std::endl;
        std::cout << std::fixed;
        std::cout << std::setprecision(6);
        auto locvol_params = torch::empty({num_nodes_x*num_nodes_y}, torch::dtype(torch::kFloat32).device(device));
        locvol_params.fill_(0.5);
        locvol_params.requires_grad_();
        std::cout << locvol_params.view({num_nodes_x, num_nodes_y}) << std::endl;
        for(auto& param: net->parameters()){
            param.set_requires_grad(false);
        }
        auto optimizer = torch::optim::Adam({locvol_params}, torch::optim::AdamOptions().lr(0.01).amsgrad(true));
        auto rate = torch::tensor(0.0128, torch::dtype(torch::kFloat32).device(device));
        int num_iters = 20000;
        auto calls_arr_var = calls_arr.select(1, 2).var();
        for(int i=0; i<num_iters; i++){
            optimizer.zero_grad();
            auto locvol_params_aug = torch::cat({rate.expand({calls_arr.size(0), 1}), calls_arr.slice(1, 0, -1), locvol_params.unsqueeze(0).expand({calls_arr.size(0), num_nodes_x*num_nodes_y})}, 1);
            auto loss = (net->forward(locvol_params_aug).squeeze()-calls_arr.select(1, 2)).square().mean()/calls_arr_var;
            loss.backward();
            optimizer.step();
            locvol_params.data().clamp_(lv_b.min*1.1, lv_b.max/1.1);
            if(i%500==0 || i==num_iters-1) std::cout << "[CALIBRATION][i = " << i << "] " << loss.item<double>() << " | calls_arr_var = " << calls_arr_var.item<double>() << std::endl;
        }
        std::cout << locvol_params.view({num_nodes_x, num_nodes_y}) << std::endl;
    }
}