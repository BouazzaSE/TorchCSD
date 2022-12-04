# TorchCSD

TorchCSD is a library written for my paper **"Fast Calibration using Complex-Step Sobolev Training"** (see _Citing_ below, preprint is [available at SSRN](https://ssrn.com/abstract=4275942) or [on this Github repo](https://github.com/BouazzaSE/TorchCSD/raw/main/preprint.pdf)). It is a libtorch-based C++ implementation of Sobolev training using stochastic directional derivatives. The latter are computed using complex-step differentiation.

## Prerequisites
The code was tested with the following software/library versions:
* `cmake = 3.22.5`
* `clang = 6.0.0`;
* `gcc = 11.2.0`;
* `libtorch = 1.11.0+cu102`

A CUDA-capable GPU is needed along with a CUDA toolkit version >= 10.2.

This project is for the advanced libtorch developer. It is assumed that the developer is familiar with neural networks (obviously), the C++ API of Pytorch (libtorch), its automatic differentiation mechanism and the `torch::autograd::Function` class. We invite the user to check out the documentation of `libtorch` at https://pytorch.org/cppdocs/frontend.html and the various unit tests in https://github.com/pytorch/pytorch/blob/master/test/cpp/api/autograd.cpp for a comprehensive _tour_ of the `torch::autograd::Function` class.

The code was tested in various Linux environments and on RTX 2070 Super, V100, A100 and different Google Colaboratory GPUs but is for now untested under Windows. However in theory if one can compile code against libtorch in Visual C++ using for example the extension found at https://marketplace.visualstudio.com/items?itemName=YiZhang.LibTorch001, then there should be no issues to build this code on Windows.

An example Python Notebook `Demo_for_Google_Colab.ipynb` serving as a launcher for the build process is provided and it can be used in a Google Colaboratory instance. When testing in Google Colaboratory, please make sure the GPU is enabled by navigating to *"Edit"* -> *"Notebook Settings"* and choosing *"GPU"* in *"Hardware Accelerator"*.

## Quick Walkthrough
The complex arithmetic uses a custom representation allowing for contiguous real and imaginary parts of complex tensors, as opposed to the PyTorch implementation of complex tensors for which these are not perfectly contiguous.

The chaining in the backpropagation process would normally require, for example in complex activation functions, four real partial derivatives. However, we explicitly exploit the Cauchy-Riemann equations thanks to our assumption of holomorphism and we use only the derivatives with respect to the real part of the input of the real and imaginary part of the output.

Indeed, assume one has the following computation graph in the scalar case for simplicity:
<p align="center">
  <img src="https://github.com/BouazzaSE/TorchCSD/blob/main/images/computation_graph_example.png?raw=true", alt="Example of a computation graph for computing z(f(a(x), b(x)), g(a(x), b(x)))">
</p>

for computing  $z(f(a(x), b(x)), g(a(x), b(x)))$ where  $x\in\mathbb{R}$  and the functions  $a:\mathbb{R}\rightarrow\mathbb{R}$,  $b:\mathbb{R}\rightarrow\mathbb{R}$,  $f:\mathbb{R}^2\rightarrow\mathbb{R}$,  $g:\mathbb{R}^2\rightarrow\mathbb{R}$  and $z:\mathbb{R}^2\rightarrow\mathbb{R}$  are assumed to be differentiable in their arguments.

The common question in backpropagation that needs to be answered for the latter to work and properly differentiate  $z$  is, in the context of this example, the following: being given the adjoints  $\bar{f}=\frac{\partial z}{\partial f}$  and  $\bar{g}=\frac{\partial z}{\partial g}$, how can the adjoints  $\bar{a}=\frac{\partial z}{\partial a}$  and  $\bar{b}=\frac{\partial z}{\partial b}$  be computed? Via the chain rule, we have:

$$\begin{aligned}\bar{a} &= \bar{f} \frac{\partial f}{\partial a} + \bar{g} \frac{\partial g}{\partial a}\\\\ \bar{b} &= \bar{f} \frac{\partial f}{\partial b} + \bar{g} \frac{\partial g}{\partial b}\end{aligned}$$

Hence, in general, one would need the following four partial derivatives:  $\frac{\partial f}{\partial a} $,  $\frac{\partial f}{\partial b}$,  $\frac{\partial g}{\partial a}$  and  $\frac{\partial g}{\partial b}$.

However, if one were to add the assumption that the function $\mathbb{C}\ni z=a+i b \mapsto f(a+i b)+i g(a+i b)$ is holomorphic, then the latter will satisfy the Cauchy-Riemann equations and we would have:

$$\begin{aligned}\frac{\partial f}{\partial a} &= \frac{\partial g}{\partial b}\\\\ \frac{\partial f}{\partial b} &= -\frac{\partial g}{\partial a}\end{aligned}$$

Thus, one can compute for example only  $\frac{\partial f}{\partial a}$  and  $\frac{\partial g}{\partial a}$  and deduce the adjoints  $\bar{a}$  and  $\bar{b}$  as follows:

$$\begin{aligned}\bar{a} &= \bar{f} \frac{\partial f}{\partial a} + \bar{g} \frac{\partial g}{\partial a}\\\\ \bar{b} &= \bar{g} \frac{\partial f}{\partial a} - \bar{f} \frac{\partial g}{\partial a}\end{aligned}$$

This simple but powerful idea allows to speed-up the backpropagation and reduce the memory footprints by explicitly hardcoding the fact that we need to compute and store only 2 partial derivatives, whenever holomorphism is assumed.

The analytic extensions of the activations and different custom layers are implemented by subclassing `torch::autograd::Function` and exploiting the Cauchy-Riemann equations in the `backward` methods of the different `torch::autograd::Function` subclasses. More precisely, `torch::autograd::Function` subclasses are implemented in the `modules/activations/autograd` and `modules/layers/autograd` subdirectories and act as wrappers for specialized CUDA code in the respective subdirectories `modules/activations/kernels` and `modules/layers/kernels`.

In order to further optimize the backpropagation process, certain expressions can be pre-computed during the forward pass in order to be later reused to speedup the backward pass at the cost of additional computations during the forward pass. Hence, when such an optimization is possible, the relevant modules will have a state indicating whether we are in training or inference mode in order to chose the best version of the CUDA kernel for the given situation (_i.e._ to pre-compute or not to pre-compute the given quantities that would only be needed for the backward pass). Hence, the `.train()` and `.eval()` methods of the modules must be used to indicate which version one wants to use.

CUDA simulation code needs to be implemented in a manner consistent with the paper referenced below (see _Citing_). A substantial example for the case of fixed-grid local volatility is given in the `simulators` subdirectory.

Lastly, an example code training a neural network to price calls (and puts using put-call parity) in a fixed-grid local volatility model is provided in `demos/train_vanilla_locvol.cpp`.

As I am not allowed to redistribute the options data, one has to constuct one's own options data file (using for example an options chain from Yahoo Finance) if one wants to test the calibration routine (which is not the highlight of the approach presented as it is orthogonal, but a simplistic example code was given for completeness). In the example code, the simplistic calibration routine assumes the presence of a binary file where the two first 32-bits unsigned integers (byte-order = little-endian) contain respectively the number of rows and columns (necessarily 3) of the option chain matrix, and the rest of the file is the binary blob representing the num_rows by num_columns C-contiguous prices matrix where the type of each entry is assumed to be a 64-bits float and each row contains the following three columns in this order: the moneyness, the maturity in years and the price. Calibration instruments are assumed, only for simplicity of the example, to be calls. Hence, one needs to convert all put prices to call prices using put-call parity if one wants to explicitly use this example. 

## Structure

The directory structure is as follows:
```
TorchCSD
│   build_demo.ipynb                (notebook with build instructions for Google Colab)
└───lib
│   │──modules                     (code for the analytic activations and custom layers)
│   │──nets                        (implementations of neural network architectures for pricing)
│   │──simulators                  (Monte Carlo simulators for the data generation)
│   │──training                    (training routines)
└───include                         (same structure as lib, contains include files)
└───demos
    │──benchmark_csd.cpp         (benchmarking code)
    │──grad_check.cpp            (code for checking gradients)
    │──layers_test.cpp           (code for testing out the different activations and layers)
    └──train_vanilla_locvol.cpp  (a demo code implementing the local volatility experiment in the paper)
```

## TODO

The following are upcoming features I am or will be working on:
* Extending to higher-order sensitivities;
* Extending to the fast pricing of structured products;
* Adding more advanced model calibration applications;
* Adding support for arbitrary shapes in the complex tensors;
* Adding AVX acceleration for CPU-only training and inference;
* Implementing TorchScript extensions to allow the use of the modules from PyTorch and integration with the ops-fusion mechanism;
* Adding more analytic extensions of activations.

## Citing
If you use this code in your work, you are strongly encouraged to both cite this Github repository (with the corresponding identifier for the commit you are looking at) and the paper describing this training approach for calibration:
```latex
@unpublished{csd2022saadeddine,
  title={Fast Calibration using Complex-Step Sobolev Training},
  author={Saadeddine, Bouazza},
  year={2022},
  note={Preprint available at SSRN: https://ssrn.com/abstract=4275942}
}
```
