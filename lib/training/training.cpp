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



#include <algorithm>
#include <limits>
#include <torch/torch.h>
#include <type_traits>
#include "ATen/Functions.h"
#include "modules/layers/layers.h"
#include "nets/vanilla_locvol/vanilla_locvol.h"
#include "training/training.h"

namespace csd {
namespace training {
void generate_diff_direction(torch::Tensor u, torch::Tensor dy_std, torch::Tensor dy_put_std, double sqrt_sum_inv_diffs_var, double sqrt_sum_inv_put_diffs_var){
    u.bernoulli_().mul_(2).sub_(1);
    u[0] /= (dy_std*sqrt_sum_inv_diffs_var).unsqueeze(0);
    u[1] /= (dy_put_std*sqrt_sum_inv_put_diffs_var).unsqueeze(0);
}

template<typename T, IsVanillaCallPutNetImpl<T>>
void train_call_put(torch::nn::ModuleHolder<T> net, torch::Tensor pricing_params, torch::Tensor pricing_params_valid, torch::Tensor y, torch::Tensor y_valid, torch::Tensor y_put, torch::Tensor y_put_valid, torch::Tensor dy, torch::Tensor dy_valid, torch::Tensor dy_put, torch::Tensor dy_put_valid, torch::Tensor lb, torch::Tensor lb_valid, long batch_size, long batch_size_valid, double lr, double lr_warmup, long num_epochs, long num_epochs_warmup, double gamma_01, double gamma_02, double gamma_11, double gamma_12, double gamma_2, double lr_decay, int warmup_print_every, int train_print_every, int compute_valid_loss_every){
    assert(pricing_params.sizes()[0]==y.sizes()[0]);
    assert(y.sizes()[1]==1);
    assert(pricing_params.sizes()[0] % batch_size == 0);
    assert(pricing_params_valid.sizes()[0] % batch_size_valid == 0);
    auto device = net->parameters()[0].device();
    auto warmup_optimizer = torch::optim::Adam(net->parameters(), torch::optim::AdamOptions().lr(lr_warmup).amsgrad(true));
    net->x_mean.copy_(pricing_params.mean(0).unsqueeze(0));
    net->x_std.copy_(pricing_params.std(0).unsqueeze(0));
    auto y_var = y.var().item().toDouble();
    auto y_put_var = y_put.var().item().toDouble();
    auto dy_var = dy.var(0); dy_var.clamp_min_(1e-3); auto dy_std = dy_var.sqrt();
    auto dy_put_var = dy_put.var(0); dy_put_var.clamp_min_(1e-3); auto dy_put_std = dy_put_var.sqrt();
    auto sum_inv_diffs_var = (1/dy_var).sum().item().toDouble();
    auto sum_inv_put_diffs_var = (1/dy_put_var).sum().item().toDouble();
    auto sqrt_sum_inv_diffs_var = sqrt(sum_inv_diffs_var);
    auto sqrt_sum_inv_put_diffs_var = sqrt(sum_inv_put_diffs_var);
    auto u = torch::empty({2, batch_size, pricing_params.sizes()[1]}, torch::dtype(torch::kFloat32).device(device));
    auto u_valid = torch::empty({2, batch_size_valid, pricing_params.sizes()[1]}, torch::dtype(torch::kFloat32).device(device));
    auto sto_directional_loss = modules::layers::StochasticDirectionalLoss(gamma_01, gamma_02, gamma_11, gamma_12, gamma_2, y_var, y_put_var, sum_inv_diffs_var, sum_inv_put_diffs_var, pricing_params.size(1));
    auto nodiff_onlycalls_loss = modules::layers::NoDiffOnlyCallsLoss(gamma_2, y_var);
    auto num_batches = (pricing_params.sizes()[0]+batch_size-1)/batch_size;
    auto num_batches_valid = (pricing_params_valid.sizes()[0]+batch_size_valid-1)/batch_size_valid;
    torch::indexing::Slice slice;
    double valid_loss, best_valid_loss;
    auto losses = std::vector<std::tuple<int, double, double>>();
    torch::Tensor tmp_loss_tensor = torch::tensor(0, torch::dtype(torch::kFloat32).device(device));
    auto best_params = std::vector<torch::Tensor>();
    for(auto& param: net->parameters()) best_params.push_back(param.data().clone());
    net->eval();
    {
        torch::NoGradGuard no_grad;
        tmp_loss_tensor.zero_();
        for(int b=0; b<num_batches_valid; b++){
            slice = torch::indexing::Slice(b*batch_size_valid, (b+1)*batch_size_valid);
            generate_diff_direction(u_valid, dy_std, dy_put_std, sqrt_sum_inv_diffs_var, sqrt_sum_inv_put_diffs_var);
            auto [y_pred, y_put_pred, dy_pred, dy_put_pred] = net->forward_call_put_diff(pricing_params_valid.index({slice}), u_valid);
            tmp_loss_tensor += sto_directional_loss(y_pred, y_put_pred, dy_pred, dy_put_pred, u_valid, y_valid.index({slice}), y_put_valid.index({slice}), dy_valid.index({slice}), dy_put_valid.index({slice}), lb_valid.index({slice}))/num_batches_valid;
        }
        valid_loss = tmp_loss_tensor.item<double>();
        best_valid_loss = valid_loss;
        losses.push_back({0, valid_loss, best_valid_loss});
    }
    std::cout << "[INIT] " << valid_loss << std::endl;
    for(int e=0; e<num_epochs_warmup; e++){
        net->train();
        for(int b=0; b<num_batches; b++){
            warmup_optimizer.zero_grad();
            slice = torch::indexing::Slice(b*batch_size, (b+1)*batch_size);
            auto y_pred = net(pricing_params.index({slice}));
            nodiff_onlycalls_loss(y_pred, y.index({slice}), lb.index({slice})).backward();
            warmup_optimizer.step();
        }
        if(e % warmup_print_every == 0){
            net->eval();
            {
                torch::NoGradGuard no_grad;
                tmp_loss_tensor.zero_();
                for(int b=0; b<num_batches; b++){
                    slice = torch::indexing::Slice(b*batch_size, (b+1)*batch_size);
                    auto y_pred = net(pricing_params.index({slice}));
                    tmp_loss_tensor += nodiff_onlycalls_loss(y_pred, y.index({slice}), lb.index({slice}))/num_batches;
                }
                std::cout << "[WARMUP][e = " << e << "] " << tmp_loss_tensor.item<double>() << std::endl;
            }
        }
    }
    auto optimizer = torch::optim::Adam(net->parameters(), torch::optim::AdamOptions().lr(lr).amsgrad(true));
    for(int e=0; e<num_epochs; e++){
        net->train();
        for (auto& param_group: optimizer.param_groups()){
            if(param_group.has_options()){
                param_group.options().set_lr(lr*exp(-e*lr_decay));
            }
        }
        for(int b=0; b<num_batches; b++){
            optimizer.zero_grad();
            slice = torch::indexing::Slice(b*batch_size, (b+1)*batch_size);
            generate_diff_direction(u, dy_std, dy_put_std, sqrt_sum_inv_diffs_var, sqrt_sum_inv_put_diffs_var);
            auto [y_pred, y_put_pred, dy_pred, dy_put_pred] = net->forward_call_put_diff(pricing_params.index({slice}), u);
            sto_directional_loss(y_pred, y_put_pred, dy_pred, dy_put_pred, u, y.index({slice}), y_put.index({slice}), dy.index({slice}), dy_put.index({slice}), lb.index({slice})).backward();
            for(auto& param: net->parameters()) param.grad().data().clamp_(-5, 5);
            optimizer.step();
        }
        if((e % compute_valid_loss_every == 0)||(e==num_epochs-1)){
            net->eval();
            {
                torch::NoGradGuard no_grad;
                tmp_loss_tensor.zero_();
                for(int b=0; b<num_batches_valid; b++){
                    slice = torch::indexing::Slice(b*batch_size_valid, (b+1)*batch_size_valid);
                    generate_diff_direction(u_valid, dy_std, dy_put_std, sqrt_sum_inv_diffs_var, sqrt_sum_inv_put_diffs_var);
                    auto [y_pred, y_put_pred, dy_pred, dy_put_pred] = net->forward_call_put_diff(pricing_params_valid.index({slice}), u_valid);
                    tmp_loss_tensor += sto_directional_loss(y_pred, y_put_pred, dy_pred, dy_put_pred, u_valid, y_valid.index({slice}), y_put_valid.index({slice}), dy_valid.index({slice}), dy_put_valid.index({slice}), lb_valid.index({slice}))/num_batches_valid;
                }
                valid_loss = tmp_loss_tensor.item<double>();
                if(valid_loss < best_valid_loss){
                    best_valid_loss = valid_loss;
                    for(int i=0; i<net->parameters().size(); i++) best_params[i].copy_(net->parameters()[i].data());
                }
                losses.push_back({e+1, valid_loss, best_valid_loss});
            }
        }
        if(e % train_print_every == 0){
            std::cout << "[TRAIN][e = " << e << "] " << valid_loss << "|" << best_valid_loss << std::endl;
        }
    }
    for(int i=0; i<net->parameters().size(); i++) net->parameters()[i].data().copy_(best_params[i]);
    net->eval();
}

template<typename T, IsVanillaCallPutNetImpl<T>>
void compute_stats(torch::nn::ModuleHolder<T> net, torch::Tensor pricing_params, torch::Tensor y, torch::Tensor dy, long batch_size){
    assert(pricing_params.sizes()[0]==y.sizes()[0]);
    assert(y.sizes()[1]==1);
    assert(pricing_params.sizes()[0] % batch_size == 0);
    auto device = net->parameters()[0].device();
    auto y_var = y.var().item().toDouble();
    auto dy_var = dy.var(0); dy_var.clamp_min_(1e-3); auto dy_std = dy_var.sqrt();
    auto sum_inv_diffs_var = (1/dy_var).sum().item().toDouble();
    auto sqrt_sum_inv_diffs_var = sqrt(sum_inv_diffs_var);
    auto stats_onlycalls_loss = modules::layers::NoDiffOnlyCallsLoss(0, y_var);
    auto stats_onlycallsdiff_loss = modules::layers::StochasticDirectionalLoss(0, 0, 1, 0, 0, y_var, 1, sum_inv_diffs_var, 1, pricing_params.size(1));
    auto u = torch::empty({2, batch_size, pricing_params.sizes()[1]}, torch::dtype(torch::kFloat32).device(device));
    auto num_batches = (pricing_params.sizes()[0]+batch_size-1)/batch_size;
    torch::Tensor tmp_loss_tensor = torch::tensor(0, torch::dtype(torch::kFloat32).device(device));
    torch::indexing::Slice slice;
    {
        torch::NoGradGuard no_grad;
        net->eval();
        tmp_loss_tensor.zero_();
        for(int b=0; b<num_batches; b++){
            slice = torch::indexing::Slice(b*batch_size, (b+1)*batch_size);
            auto y_pred = net(pricing_params.index({slice}));
            tmp_loss_tensor += stats_onlycalls_loss(y_pred, y.index({slice}), y.index({slice}))/num_batches;
        }
        std::cout << "> CALL PAYOFF PROJ MSE " << tmp_loss_tensor.item<double>() << std::endl;
        tmp_loss_tensor.zero_();
        for(int b=0; b<num_batches; b++){
            slice = torch::indexing::Slice(b*batch_size, (b+1)*batch_size);
            generate_diff_direction(u, dy_std, dy_std, sqrt_sum_inv_diffs_var, sqrt_sum_inv_diffs_var);
            auto [y_pred, dy_pred] = net->forward_diff(pricing_params.index({slice}), u);
            tmp_loss_tensor += stats_onlycallsdiff_loss(y_pred, y_pred, dy_pred, dy_pred, u, y.index({slice}), y.index({slice}), dy.index({slice}), dy.index({slice}), y.index({slice}))/num_batches;
        }
        std::cout << "> CALL PAYOFF DIFF PROJ MSE " << tmp_loss_tensor.item<double>() << std::endl;
    }
}

template<typename T, IsVanillaCallPutNetImpl<T>>
void compute_twin_stats(torch::nn::ModuleHolder<T> net, torch::Tensor pricing_params, torch::Tensor y, torch::Tensor y_twin, long batch_size){
    assert(pricing_params.sizes()[0]==y.sizes()[0]);
    assert(pricing_params.sizes()[0] % batch_size == 0);
    assert(y.sizes()[1]==1);
    assert(y_twin.sizes()[1]==1);
    auto device = net->parameters()[0].device();
    auto num_batches = (pricing_params.sizes()[0]+batch_size-1)/batch_size;
    torch::Tensor loss_tensor = torch::tensor(0, torch::dtype(torch::kFloat32).device(device));
    torch::Tensor loss_tensor_std = torch::tensor(0, torch::dtype(torch::kFloat32).device(device));
    torch::Tensor norm = torch::tensor(0, torch::dtype(torch::kFloat32).device(device));
    torch::indexing::Slice slice;
    {
        torch::NoGradGuard no_grad;
        net->eval();
        loss_tensor.zero_();
        norm.zero_();
        for(int b=0; b<num_batches; b++){
            slice = torch::indexing::Slice(b*batch_size, (b+1)*batch_size);
            auto y_pred = net(pricing_params.index({slice}));
            auto twins_prod = (y.index({slice})*y_twin.index({slice})).mean()/num_batches;
            loss_tensor += (y_pred*(y_pred-y.index({slice})-y_twin.index({slice}))).mean()/num_batches+twins_prod;
            loss_tensor_std += (y_pred*(y_pred-y.index({slice})-y_twin.index({slice}))+y.index({slice})*y_twin.index({slice})).square_().mean()/num_batches;
            norm += twins_prod;
        }
        loss_tensor_std.sub_(loss_tensor.square()).sqrt_();
        auto loss_tensor_sq = loss_tensor.clone();
        loss_tensor.sqrt_();
        norm.sqrt_();
        std::cout << "> TWIN MSE ESTIMATE: " << loss_tensor_sq.item<double>() << std::endl;
        std::cout << "> TWIN MSE STD ESTIMATE: " << loss_tensor_std.square().item<double>() << std::endl;
        std::cout << "> TWIN L2 ERROR ESTIMATE: " << loss_tensor.item<double>() << std::endl;
        std::cout << "> TWIN L2 SQRT STD ESTIMATE: " << loss_tensor_std.item<double>() << std::endl;
        std::cout << "> TWIN L2 NORM ESTIMATE: " << norm.item<double>() << std::endl;
        std::cout << "> TWIN L2 NORMALIZED ERROR ESTIMATE: " << loss_tensor.item<double>()/(norm.item<double>()+std::numeric_limits<double>::epsilon()) << std::endl;
    }
}

template void train_call_put(torch::nn::ModuleHolder<csd::nets::VanillaLocVolSoftplusSigmoidImpl>, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, long, long, double, double, long, long, double, double, double, double, double, double, int, int, int);
template void compute_stats(torch::nn::ModuleHolder<csd::nets::VanillaLocVolSoftplusSigmoidImpl>, torch::Tensor, torch::Tensor, torch::Tensor, long);
template void compute_twin_stats(torch::nn::ModuleHolder<csd::nets::VanillaLocVolSoftplusSigmoidImpl>, torch::Tensor, torch::Tensor, torch::Tensor, long);
} // namespace training
} // namespace csd