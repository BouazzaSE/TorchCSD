# TorchCSD

TorchCSD is a library written for the paper **"Fast Calibration using Complex-Step Sobolev Training"** by Saadeddine Bouazza. It is a C++ implementation, based on `libtorch`, of Sobolev training using stochastic directional derivatives. The latter are computed using complex-step differentiation.

The complex arithmetic uses a custom representation allowing for contiguous real and imaginary parts of complex tensors, as opposed to the PyTorch implementation of complex tensors for which these are not perfectly contiguous.

The analytic extensions of the activations and different custom layers are implemented with the complex numbers treated as vectors in $\mathbb{R}^2$, hence allowing the reuse of `libtorch`'s autograd mechanism by limiting ourselves to the implementation of the chaining code in the `::backward` methods of the different `torch::autograd::Function` subclasses.

The chaining would normally require, for example in activation functions, four partial derivatives. However, we explicitly exploit the Cauchy-Riemann equations thanks to our assumption of holomorphism and we use only the derivatives with respect to the real part of the input of the real and imaginary part of the output.

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
* `clang = 6.0.0`;
* `libtorch = 1.11.0+cu102`

Familiarity with CMake, the C++ API of Pytorch (`libtorch`), its automatic differentiation mechanism and the `torch::autograd::Function` class is assumed. We invite the user to check out the documentation of `libtorch` at https://pytorch.org/cppdocs/. A simple tutorial on the Python counterpart `torch.autograd.Function` is available at https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html. We invite the advanced user to check the unit tests of the C++ API for a detailed _tour_ of the `torch::autograd::Function` class.

The code was tested in Linux environments and is for now untested under Windows. We provide an example Python Notebook `build_demo.ipynb` serving as a launcher for the build process and which can be used in a Google Colaboratory instance. When testing in Google Colaboratory, please make sure the GPU is enabled by navigating to *"Edit"* -> *"Notebook Settings"* and choosing *"GPU"* in *"Hardware Accelerator"*.
