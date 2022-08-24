## Introduction

DLOP-Bench is an open source benchmark suite for deep learning operators. It has the following three major features:

- **Operators at the deep leaning framework level**
We focus on the operator at the deep learning framework level (such as torch.convolution) and do not dive into the implementation details of each operator (implicit gemm implementation or winograd implementation and the related algorithm selection). One can easily benchmark the operators on a certain AI accelerator as long as they finish the adaption on a deep learning framework.

- **Basic Operators and Long-tail operators**
Besides basic operators, we also collect many representative domain-specific operators. These operators have no dedicated implementation on deep learning accelerators and have to resort to the Python interpreter. As such, they will always be broken down into large numbers of basic operators. They incur a lot of function calls, as well as data transfer and context switching costs. We name them long-tail operators.

- **Benchmarking deep learning accelerators and their compilers**
From the operator level, this benchmark suite can provide a more microscopic assessment from both accelerator hardware specifications and compiler design.

