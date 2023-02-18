---
layout: post
current: post
cover: assets/images/jetson.jpg
navigation: True
title: Jetson Xavier NX 的使用记录
date: 2021-8-24 21:11:00
tags: [pytorch,Jetson,ComputerVision]
excerpt: 记录使用Jetson Xavier NX的使用体验中遇到的问题和记录
class: post-template
subclass: 'post'
---

### 一. 远程桌面

> 在windows10远程上操作jetson Xavier，远程的前提：jetson xavier和Windows的PC在同一个局域网内（我这里是直接在windows10上开启热点）。

* 安装xrdp：`sudo apt-get install xrdp vnc4server xbase-clients`

#### 1.1：桌面共享没反应

> 桌面共享其实就是一个vnc-server（因此没有必要再在linux上安装vnc-server了），如果要远程，必须要先开启共享，允许其他人控制自己电脑。

这里发现**双击了桌面共享，没反应**。

**解决方案**：

*  安装dconf-editor：`sudo apt-get install dconf-editor`
* 运行dconf-editor，更改系统配置，org ==> gnome ==> desktop ==> remote-access，关闭以下两个：`promotion-enabled`和`requre-encryption`
* 开启桌面共享：`/usr/lib/vino/vino-server`

#### 1.2 开启远程

* 在Windows10上安装vnc-client，官网地址：https://www.realvnc.com/en/connect/download/viewer/windows/

![](https://raw.githubusercontent.com/yy2lyx/picgo/admin/img/jet_1.png)

* 输入linux的ip后，直接连接即可。

  ![](https://raw.githubusercontent.com/yy2lyx/picgo/admin/img/jet_2.png)

#### 1.3 建立双击可执行文件.desktop

> 由于每次我开启远程桌面的时候，都需要在命令行输入相关指令，很麻烦，就想说有没有可以直接在像windows一样快捷方式

* 写一个shell脚本来开启桌面共享：`vim ~/vnc-server.sh`

  ```bash
  #!/bin/sh
  /usr/lib/vino/vino-server
  ```

* 在桌面上新建一个.desktop文件：`vim ~/Desktop/vnc-server.desktop`

  ```bash
  [Desktop Entry]
  Encoding=UTF-8
  Type=Application
  Categories=true
  Version=1.0
  Name=vnc-server
  Exec=sh /home/yy/vnc-server.sh #注意这里是绝对路径
  Path=/home/yy
  Terminal=false # 是否保留终端
  StartupNotify=true # 开机自启动
  ```

* 给.desktop文件加上可执行权限：`sudo chmod +x ~/Desktop/vnc-server.desktop` 

#### 1.4 开机自动开启

* 创建开机自启动文件夹：`mkdir ~/.config/autostart`
* 复制.desktop文件到该文件夹下：`cp ~/Desktop/vnc-server.desktop ~/.config/autostart/`
* 给.desktop文件加上可执行权限：`sudo chmod +x ~/.config/autostart/vnc-server.desktop` 

### 二. 配置cuda和cuDNN

* 用JetPack刷机（本人选用的是Jetpack4.4.1），jetpack地址：https://developer.nvidia.com/embedded/jetpack-archive

![](https://raw.githubusercontent.com/yy2lyx/picgo/admin/img/jet_3.png)

* 安装完成后，我们可以查看jetpack的版本：`cat /etc/nv_tegra_release`，同时已经给我们安装好了CUDA 10.2 、cudnn 8.0 、opencv、python3.6。

#### 2.1 设置cuda环境变量

​	查询cuda版本：`nvcc -V`基本就能看到cuda信息了，但是这里却**报错没有nvcc指令**。解决方案：将cuda添加到环境变量中

* `sudo vim /etc/profile `，这里说下其实在`/usr/local`下有2个cuda相关文件夹，分别是`cuda`和`cuda-10.2`，你会发现其实是一致的。

  ```bash
  export PATH=/usr/local/cuda-10.2/bin:$PATH
  export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64$LD_LIBRARY_PATH
  export CUDA_HOME=/usr/local/cuda-10.2
  ```

* `source /etc/profile`，再去命令行输入`nvcc -V`就可以看到cuda10.2的信息了

#### 2.2 设置cuDNN

安装好的cuDNN的头文件：`/usr/include/cudnn.h`
安装好的cuDNN的库文件：`/usr/lib/aarch64-linux-gnu/libcudnn*`

(1) 而**这些头文件和库文件都不在cuda目录下**，因此要复制到cuda目录下：

* 复制头文件：`sudo cp /usr/include/cudnn.h /usr/local/cuda/include`
* 复制库文件：`sudo cp /usr/lib/aarch64-linux-gnu/libcudnn* /usr/local/cuda/lib64/`

(2) 修改文件权限：`sudo chmod 777 /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*`

(3) 重新软连接：

```bash
   cd /usr/local/cuda/lib64
   sudo ln -sf libcudnn.so.8.0.0 libcudnn.so.8
   sudo ln -sf libcudnn_ops_train.so.8.0.0 libcudnn_ops_train.so.8
   sudo ln -sf libcudnn_ops_infer.so.8.0.0 libcudnn_ops_infer.so.8
   sudo ln -sf libcudnn_adv_infer.so.8.0.0 libcudnn_adv_infer.so.8
   sudo ln -sf libcudnn_cnn_infer.so.8.0.0 libcudnn_cnn_infer.so.8
   sudo ln -sf libcudnn_cnn_train.so.8.0.0 libcudnn_cnn_train.so.8
   sudo ln -sf libcudnn_adv_train.so.8.0.0 libcudnn_adv_train.so.8

   sudo ln -sf libcudnn_ops_train.so.7.3.1 libcudnn_ops_train.so.7
   sudo ln -sf libcudnn_ops_infer.so.7.3.1libcudnn_ops_infer.so.7
   sudo ln -sf libcudnn_adv_infer.so.7.3.1 libcudnn_adv_infer.so.7
   sudo ln -sf libcudnn_cnn_infer.so.7.3.1 libcudnn_cnn_infer.so.7
   sudo ln -sf libcudnn_cnn_train.so.7.3.1 libcudnn_cnn_train.so.7
   sudo ln -sf libcudnn_adv_train.so.7.3.1 libcudnn_adv_train.so.7
```

(4) 编译与验证：

```
sudo ldconfig
sudo cp -r /usr/src/cudnn_samples_v8/ ~/
cd ~/cudnn_samples_v8/mnistCUDNN
sudo chmod 777 ~/cudnn_samples_v8
sudo make clean
sudo make
./mnistCUDNN # 验证cuDNN
```

(5) 查询cuDNN版本：`cat /usr/include/cudnn.h | grep CUDNN_MAJOR -A 2`

### 三. 配置torch和torchvision

#### 3.1 配置torch

> 这里安装torch的方式和普通的windows和linux下安装不一样，pytorch有专门的jetson版本

Pytorch的jetson版本下载地址：https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-9-0-now-available/72048

![](https://raw.githubusercontent.com/yy2lyx/picgo/admin/img/jet_4.png)

* 下载完成后直接使用pip进行安装：`sudo pip3 install numpy torch-1.9.0-cp36-cp36m-linux_aarch64.whl`

* 安装必须的依赖：`sudo apt-get install libopenblas-base libopenmpi-dev `

#### 3.2 配置torchvision

由于安装的是1.9.0版本的pytorch，直接安装于其对应的torchvision(0.10.0)：`sudo pip3 install torchvision==0.10.0`

#### 3.3 验证GPU可用

* 命令行打开python，导入torch包：`import torch`

* 查看pytorch可调用的cuda版本：`torch.version.cuda`
* 查看cuda是否可用：`torch.cuda.is_available()`

#### 3.4 使用yolo遇到的问题

问题：`RuntimeError: No such operator torchvision::nms`

实验过程：在网上搜到的都是torch和torchvision的版本不对，要升级torchvision之类的，实验将torchvision升级到0.10.0，依旧不行。

解决方案：其实在使用yolo时的非极大值抑制的时候，可以将`utils.general.py`中nms相关代码替换

* 原始代码：`i = torch.ops.torchvision.nms(boxes, scores, iou_thres) `

* 替换代码：

  ```python
  import torchvision
  i = torchvision.ops.nms(boxes, scores, iou_thres)
  ```

