---
layout: post
current: post
cover: assets/images/torch_err.jpeg
navigation: True
title: Libtorch的GPU使用问题记录
date: 2021-1-30 20:21:00
tags: [pytorch,DeepLearning]
excerpt: 介绍pytorch的C++版本的gpu使用的解决问题的过程记录
class: post-template
subclass: 'post'
---


> 这里得吹逼下自己领导，10min解决了困扰我2天的问题（好吧，也许是我太蠢）。

### 一. 问题描述

由于项目需要使用libtorch（pytorch的C++版本）的GPU版本，但是发现无法使用GPU，因此将问题和解决过程记录下来，方便日后观看和反思。

### 二. 解决问题的过程

#### 2.1 使用的torch版本

这里需要说下pytorch和libtorch的版本一定要一致，且和cuda的版本一致。这里都是通过pytorch官网上进行安装即可。

* pytorch1.6.0（GPU）：使用pip安装

  ```
  # CUDA 10.1
  pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
  ```

* pytorch1.6.0（CPU）：使用pip安装

  ```
  # CPU only
  pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
  ```
  
* libtorch1.6.0（GPU）：选择使用release版本即可（据说debug有问题）

  ```
  https://download.pytorch.org/libtorch/cu101/libtorch-win-shared-with-deps-1.6.0%2Bcu101.zip
  ```
  
* libtorch1.6.0（CPU）：选择使用release版本即可（据说debug有问题）
  ```
  https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-1.6.0%2Bcpu.zip
  ```

#### 2.2 使用cmakelist的搭建工程

```cmake
cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(torch_gpu_test)

# add CMAKE_PREFIX_PATH
list(APPEND CMAKE_PREFIX_PATH "D:/software/opencv/opencv/build/x64/vc15/lib")
list(APPEND CMAKE_PREFIX_PATH "D:/software/libtorch_gpu")
list(APPEND CUDA_TOOLKIT_ROOT_DIR "D:/software/cuda/development")


find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

if(NOT Torch_FOUND)
    message(FATAL_ERROR "Pytorch Not Found!")
endif(NOT Torch_FOUND)

message(STATUS "Pytorch status:")
message(STATUS "    libraries: ${TORCH_LIBRARIES}")
message(STATUS "Find Torch VERSION: ${Torch_VERSION}")

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")


add_executable(torch_gpu_test torch_gpu_test.cpp)
target_link_libraries(torch_gpu_test ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET torch_gpu_test PROPERTY CXX_STANDARD 11)
```

* 这里利用vs2019生成项目之后，编写以下代码进行测试：

```C++
#include <torch/torch.h>
#include <torch/script.h>

using namespace torch;

int main()
{
    torch::DeviceType device_type = at::kCPU;
    if (torch::cuda::is_available()) {
        cout << "cuda!" << endl;
        torch::DeviceType device_type = at::kCUDA;
    }
    else
    {
        cout << "cpu" << endl;
    }
    
}
```

* 到了这里我开始了我的问题之旅：由于是Release版本，不能debug，只能主观的认为这里应该是cuda的环境没配好导致torch无法使用gpu的，因此一直在找cmake的cuda环境配置问题。

#### 2.3 Release with Debug改变了我的想法

>  这里得说下本人第一次知道release版本也可以debug！（本人也算一C++小白哈，别计较）

这里顺带记录下如何使用vs2019的Release with debug的过程：

* 直接在项目中将`Release`版本选择为`RelWithDebInfo`

![](https://i.loli.net/2021/02/01/FcP7UNy4fqTxkR1.png)

* 禁用代码优化功能：这里是防止出现“变量已被优化掉 因而不可用”这种问题

![](https://i.loli.net/2021/02/01/KSBZQE8f7IkenWD.png)

在这里debug的时候发现，`device`这个我定义的变量是可以加载cuda的！因此可以推翻我之前想的（cuda环境的问题）。

#### 2.4 libtorch1.6GPU版本问题

这里就可以肯定是libtorch的GPU问题了。为啥`torch::cuda::is_available()`会是`false`呢？

* 网上的思路是：

在“属性 --> 链接器 --> 命令行 --> 其他选项”中添加：

```
  /INCLUDE:?warp_size@cuda@at@@YAHXZ
```

* 本人实验了下，按照网上的添加会报错，因此以下是本人实验可行的结果：

在“链接器 --> 输入 --> 附加依赖项”中进行添加：

```
D:\software\libtorch_gpu\lib\torch_cuda.lib
D:\software\libtorch_gpu\lib\torch_cpu.lib
-INCLUDE:?warp_size@cuda@at@@YAHXZ
```

> 这里很奇怪，cmakelist明明已经配置好了libtorch的gpu，但是这里却没有`torch_cuda.lib`

至此，问题解决了！



