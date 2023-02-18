---
layout: post
current: post
cover: assets/images/yolov5.jpeg
navigation: True
title: windows下搭建libtorch和paddle的C++环境搭建
date: 2020-12-25 20:21:00
tags: [pytorch,paddle,Cplusplus,DeepLearning]
excerpt: 介绍在C++平台下搭建torch和paddle的环境
class: post-template
subclass: 'post'
---


> 参考文章：[NSTALLING C++ DISTRIBUTIONS OF PYTORCH](https://pytorch.org/cppdocs/installing.html)，[安装与编译 Windows 预测库](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-rc1/guides/05_inference_deployment/inference/windows_cpp_inference.html)，[在C++中加载PYTORCH模型](https://pytorch.apachecn.org/docs/1.0/cpp_export.html)

### 一. 必要软件

* [vs2019](https://visualstudio.microsoft.com/zh-hans/vs/)：paddle和torch这里的编译都是由Visual Studio 2019完成的
* [libtorch](https://pytorch.org/get-started/locally/)：直接在官网上进行下载压缩包，这里说明下分为release和debug版本，直接下载release版本即可。
* [paddle](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-rc1/guides/05_inference_deployment/inference/windows_cpp_inference.html)：这里选择2.0-rc1的cpu版本的直接进行解压安装。
* [opencv](https://opencv.org/releases/)：windows下直接安装exe到本地即可。
* cmake：直接用scoop安装`scoop install cmake`

### 二. 安装libtorch环境

#### 2.1 构建一个C++项目

目录层级如下：

```
├─example-app
	 ├─build // 新建一个空目录
	 ├─CMakeLists.txt // 构建一个cmakelist
	 └─example-app.cpp // 构建一个cpp文件用于测试
```

其中，`CMakeList.txt`具体设置如下：

```cmake
cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(example-app)

# add CMAKE_PREFIX_PATH
#增加opencv和libtorch的路径
list(APPEND CMAKE_PREFIX_PATH "D:/software/opencv/opencv/build/x64/vc15/lib") 
# 注意这里如果是vs2015的版本，需要改成 /build/x64/vc14/lib
list(APPEND CMAKE_PREFIX_PATH "D:/software/libtorch")


find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

if(NOT Torch_FOUND)
    message(FATAL_ERROR "Pytorch Not Found!")
endif(NOT Torch_FOUND)

message(STATUS "Pytorch status:")
message(STATUS "    libraries: ${TORCH_LIBRARIES}")

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")


add_executable(example-app example-app.cpp)
target_link_libraries(example-app ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET example-app PROPERTY CXX_STANDARD 11)
```

C++测试代码（`example-app.cpp`）如下（测试opencv和libtorch）：

```C++
#include <torch/torch.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int main() {
  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;
  std::cout << "ok!" << std::endl;
  Mat img = imread("1.jpg");
  imshow("1",img);
  waitKey(0);
  return 0;
}
```

#### 2.2 编译和生成项目

* 进入到`build`目录：`cd build`
* 利用cmake进行编译： `cmake ..`
* 编译顺利的话，就可以看到`build`目录下生成了如下所示：

![](https://raw.githubusercontent.com/yy2lyx/picgo/admin/img/libtorch_1.jpg)

* 利用vs2019打开项目`example-app.sln`
* 点击`example-app` 右键选择`设为启动项`，并且将版本选择`release`版本，点击`本地Windows调试器`

![](https://raw.githubusercontent.com/yy2lyx/picgo/admin/img/libtorch_2.jpg)

#### 2.3 调试问题的解决

* 报错信息：`由于找不到c10.dll`，`torch.dll`这种找不到dll文件的，直接将dll文件(这些dll文件都在`libtorch/lib`路径下)复制到`build/release`文件夹下
* `opencv_world3411.dll`和`opencv_ffmpeg3411_64.dll`等都在opencv的`opencv\opencv\build\x64\vc15\lib`路径下。
* 这里注意测试opencv的时候，需要将图片放置到和`example-app.vcxproj`同级目录下

#### 2.4 exe生成文件的平台移植

* 如果需要将生成的exe文件移植到其他PC上面，只需要将release文件夹下所有文件（包括dll文件和exe文件）复制到其他PC即可。
* 生成的exe文件在找图片的时候也是同级目录下找，因此需要将图片放置到`exe`文件的同级目录下。

#### 2.5 pytorch模型在C++平台的使用

PyTorch模型从Python到C++的转换由[Torch Script](https://pytorch.org/docs/master/jit.html)实现。Torch Script是PyTorch模型的一种表示，可由Torch Script编译器理解，编译和序列化。一般利用trace将PyTorch模型转换为Torch脚本,必须将模型的实例以及样本输入传递给`torch.jit.trace`函数。这将生成一个 `torch.jit.ScriptModule`对象，并在模块的`forward`方法中嵌入模型评估的跟踪。

### 三. 安装paddle的C++环境

#### 3.1 下载安装paddle

这里官网有2种方式在windows上安装paddle环境：一个是通过git下载paddle源码进行编译安装，另一种直接从官网下载zip编译好的文件（本文使用该种方式）。

![](https://raw.githubusercontent.com/yy2lyx/picgo/admin/img/libtorch_3.jpg)

#### 3.2 结合paddleOCR测试并使用paddle预测库

* paddleOCR的git地址：https://github.com/PaddlePaddle/PaddleOCR
* 下载到本地之后，`cd PaddleOCR\deploy\cpp_infer`，修改`CMakeList.txt`文件

```cmake
SET(PADDLE_LIB "D:/software/paddle_inference_install_dir") # 这里是下载的paddle预测库的路径
SET(OPENCV_DIR "D:/software/opencv/opencv") # 这里是下载的opencv的路径
find_package(OpenCV REQUIRED)
```

* 新建一个build文件夹：`mkdir build`
* 进入build：`cd build` ， 编译：`cmake ..`
* 同样的利用vs2019打开项目`ocr_system.sln`，生成即可。
* 这里注意需要将`paddle_fluid.dll`放入到`Release`目录下。

