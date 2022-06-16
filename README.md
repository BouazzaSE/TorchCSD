# TorchCSD

TorchCSD is a library written for the paper **"Fast Calibration using Complex-Step Sobolev Training"** by Saadeddine Bouazza. It is a libtorch-based C++ implementation of Sobolev training using stochastic directional derivatives. The latter are computed using complex-step differentiation.

The complex arithmetic uses a custom representation allowing for contiguous real and imaginary parts of complex tensors, as opposed to the PyTorch implementation of complex tensors for which these are not perfectly contiguous.

The analytic extensions of the activations and different custom layers are implemented with the complex numbers treated as vectors in $\mathbb{R}^2$, hence allowing the reuse of libtorch's autograd mechanism by limiting ourselves to the implementation of the chaining code in the `::backward` methods of the different `torch::autograd::Function` subclasses.

The chaining would normally require, for example in activation functions, four partial derivatives. However, we explicitly exploit the Cauchy-Riemann equations thanks to our assumption of holomorphism and we use only the derivatives with respect to the real part of the input of the real and imaginary part of the output.

Indeed, assume one has the following computation graph in the scalar case for simplicity:
![Example of a computation graph for computing z(f(a(x), b(x)), g(a(x), b(x)))](/images/computation_graph_example.png)
for computing $z(f(a(x), b(x)), g(a(x), b(x)))$ where $x\in\mathbb{R}$ and the functions $a:\mathbb{R}\rightarrow\mathbb{R}$, $b:\mathbb{R}\rightarrow\mathbb{R}$, $f:\mathbb{R}^2\rightarrow\mathbb{R}$, $g:\mathbb{R}^2\rightarrow\mathbb{R}$ and $z:\mathbb{R}^2\rightarrow\mathbb{R}$ are assumed to be differentiable in their arguments.

The common question in backpropagation that needs to be answered for the latter to work and properly differentiate $z$ is, in the context of this example, the following: being given the adjoints $\bar{f}=\frac{\partial z}{\partial f}$ and $\bar{g}=\frac{\partial z}{\partial g}$, how can the adjoints $\bar{a}=\frac{\partial z}{\partial a}$ and $\bar{b}=\frac{\partial z}{\partial b}$ be computed? Via the chain rule, we have:
$$
\begin{aligned}
\bar{a} &= \bar{f} \frac{\partial f}{\partial a} + \bar{g} \frac{\partial g}{\partial a}\\
\bar{b} &= \bar{f} \frac{\partial f}{\partial b} + \bar{g} \frac{\partial g}{\partial b}
\end{aligned}
$$

Hence, in general, one would need the following four partial derivatives: $\frac{\partial f}{\partial a}$, $\frac{\partial f}{\partial b}$, $\frac{\partial g}{\partial a}$ and $\frac{\partial g}{\partial b}$.

However, if one were to add the assumption that the function $\mathbb{C}\ni z=a+i b \mapsto f(a+i b)+i g(a+i b)$ is holomorphic, then the latter will satisfy the Cauchy-Riemann equations and we would have:

$$
\begin{aligned}
\frac{\partial f}{\partial a} &= \frac{\partial g}{\partial b}\\
\frac{\partial f}{\partial b} &= -\frac{\partial g}{\partial a}
\end{aligned}
$$

Thus, one can compute for example only $\frac{\partial f}{\partial a}$ and $\frac{\partial g}{\partial a}$ and deduce the adjoints $\bar{a}$ and $\bar{b}$ as follows:
$$
\begin{aligned}
\bar{a} &= \bar{f} \frac{\partial f}{\partial a} + \bar{g} \frac{\partial g}{\partial a}\\
\bar{b} &= \bar{g} \frac{\partial f}{\partial a} - \bar{f} \frac{\partial g}{\partial a}
\end{aligned}
$$
This simple but powerful idea allows to speed-up the backpropagation and reduce the memory footprints by explicitly hardcoding the fact that we need to compute and store only 2 partial derivatives, whenever holomorphism is assumed.

## Citing
If you use this code in your work, we strongly encourage you to both cite this Github repository (with the corresponding identifier for the commit you are looking at) and the paper describing this training approach for calibration:
```latex
@unpublished{csd2022saadeddine,
  title={Fast Calibration using Complex-Step Sobolev Training},
  author={Saadeddine, Bouazza},
  year={\ndd},
  note={unpublished}
}
```

## Prerequisites
The code was tested with the following software/library versions:
* `cmake = 3.22.5`
* `clang >= 6.0.0`;
* `libtorch = 1.11.0+cu102`

A CUDA-capable GPU is needed along with a CUDA toolkit version >= 10.2.

Familiarity with CMake, the C++ API of Pytorch (`libtorch`), its automatic differentiation mechanism and the `torch::autograd::Function` class is assumed. We invite the user to check out the documentation of `libtorch` at https://pytorch.org/cppdocs/. A simple tutorial on the Python counterpart `torch.autograd.Function` is available at https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html. We invite the advanced user to check the unit tests of the C++ API for a detailed _tour_ of the `torch::autograd::Function` class.

The code was tested in Linux environments and is for now untested under Windows. We provide an example Python Notebook `build_demo.ipynb` serving as a launcher for the build process and which can be used in a Google Colaboratory instance. When testing in Google Colaboratory, please make sure the GPU is enabled by navigating to *"Edit"* -> *"Notebook Settings"* and choosing *"GPU"* in *"Hardware Accelerator"*.

## Structure

The directory structure is as follows:
```
TorchCSD
│   build_demo.ipynb                (notebook with build instructions for Google Colab)
└───lib
│   │   modules                     (code for the analytic activations and custom layers)
│   │   nets                        (implementations of neural network architectures for pricing)
│   │   simulators                  (Monte Carlo simulators for the data generation)
│   │   training                    (training routines)
└───include                         (same structure as lib, contains include files)
└───demos
│   │──   benchmark_csd.cpp         (benchmarking code)
│   │──   grad_check.cpp            (code for checking gradients)
│   │──   layers_test.cpp           (code for testing out the different activations and layers)
│   └──   train_vanilla_locvol.cpp  (a demo code implementing the local volatility experiment in the paper)
```
