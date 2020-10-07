---
layout: post
title: Windows下安装C++ IDE（clion）和opencv环境
date: 2019-10-01 00:00:00 +0300
description: 在Windows下安装Clion和Opencv的运行环境 # Add post description (optional)
img: software.jpg # Add image post (optional)
tags: [Clion, OpenCV] # add tag
---


#### 1. 下载软件

* [clion](https://www.macw.com/mac/1893.html)：C++的IDE
* [cmake](https://cmake.org/download/) : 这里需要添加到环境变量中 `D:\Profile\mingw64\bin`
* [MinGW]([https://sourceforge.net/projects/mingw-w64/files/Toolchains%20targetting%20Win64/Personal%20Builds/mingw-builds/8.1.0/threads-posix/seh/x86_64-8.1.0-release-posix-seh-rt_v6-rev0.7z/download](https://sourceforge.net/projects/mingw-w64/files/Toolchains targetting Win64/Personal Builds/mingw-builds/8.1.0/threads-posix/seh/x86_64-8.1.0-release-posix-seh-rt_v6-rev0.7z/download)) ：添加到环境变量 `D:\Profile\mingw64\bin`
* [opencv3.4.10](https://opencv.org/releases/)：开源的计算机视觉库

#### 2. MinGW和OpenCV

主要是如何用你的编译器来编译OpenCV。我们需要有include文件夹，这个在写代码时就用的到，还有lib和dll，这俩货我也不是很懂，dll的话没有是可以编译成功的，但运行是要失败的，所以我们是肯定要把dll加入到系统环境变量Path里的。lib是编译时就需要的，所以我们得把lib放在CLion的CMakeLists里面。

下载完Windows的OpenCV，其实我们只有给Visual Studio用的dll和lib，可是我们想要g++来编译和运行，所以就得自己根据OpenCV的sources文件夹来自己编译OpenCV。

* 这里需要在cmake中加入`OPENCV_ALLOCATOR_STATS_COUNTER_TYPE=int64_t`，`add Entry` ==> `string`，这里参考[报错信息1](https://github.com/opencv/opencv/issues/17065)
* 这里还需要再cmake中加入`OPENCV_ENABLE_ALLOCATOR_STATS=OFF`，参考[报错信息2](https://answers.opencv.org/question/228737/gcc-error-long-no-such-file-or-directory/)

![](https://tva1.sinaimg.cn/large/007S8ZIlgy1gjehgzemrbj30fl0ep0ug.jpg)

* 需要2次Configure和1次Genrate即可编译完成。
* `cd opencv\mingw-build`目录下输入`mingw32-make`
* 等待完成，`mingw32-make install`
* 打开你的mingw-build文件夹，里面有个install目录就是你要的，可以复制一下这个文件夹，以后就不用重新编译了。我在C盘建立了OpenCV目录，并且把install文件夹下的文件复制进去了,`C:\OpenCV\x64\mingw\bin`加入系统环境变量Path中。

#### 3. 写CMakeList

其实就是加入lib目录和include目录

```
cmake_minimum_required(VERSION 3.16)
project(opencv_test)

set(CMAKE_CXX_STANDARD 14)

add_executable(opencv_test main.cpp)

## 添加的OpenCVConfig.cmake的路径
set(OpenCV_DIR "D:/Profile/opencv_builded")

## 搜索OpenCV目录
find_package(OpenCV REQUIRED)

## 添加OpenCV头文件目录
include_directories("D:/Profile/opencv_builded/include")

## 链接OpenCV库文件
target_link_libraries(opencv_test ${OpenCV_LIBS})
```

#### 4. 编译成可执行文件

`main.cpp`文件中写完后，`cd 项目目录`，`cmake .`，即可看到项目中新加了文件夹`cmake-build-debug`中里面存在`.exe`可执行文件。


