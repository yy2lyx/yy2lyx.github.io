---
layout: post
current: post
cover: assets/images/pointcloud.png
navigation: True
title: windows下安装python-pcl
date: 2021-4-10 10:21:00
tags: [ComputerVision,PCL]
excerpt: 介绍如何在win10下安装python版本的PCL点云库
class: post-template
subclass: 'post'
---

### 一. 准备工作

* **python 版本：3.7.9**
  * cython
  * numpy
* **python-pcl:1.9.1**
  * [python-pcl源码](https://github.com/strawlab/python-pcl)：后面需要进行编译
  * [PCL1.9.1的All-In-One Installer](https://github.com/PointCloudLibrary/pcl/releases/) ：目前安装仅支持1.6到1.9的版本
* **visual studio 2019**
* **[Windows Gtk](http://www.tarnyko.net/dl/gtk.htm)**

### 二. 安装

* 将下载好的ALL-In-One Installer进行安装，这里会要求你添加到环境变量（必须添加啊），并且会安装OpenNI这个工具。

* 解压下载好的windows Gtk，将`bin`目录下所有文件复制到python-pcl源码目录下的`pkg-config`目录下。

* 在`pkg-config`目录下，运行脚本`InstallWindowsGTKPlus.bat`，该脚本会下载必须的内容，下载完成后会多出这些文件夹，如下图所示

  ![](https://i.loli.net/2021/04/12/CtZmlOTNWhnakYU.png)

* 安装python的pcl包：
  * `cd 你安装python-pcl源码目录`
  * `python setup.py build_ext -i`
  * `python setup.py install`

### 三. 安装遇到的坑

#### 3.1 坑一：cannot find PCL

* 问题：当你运行`python setup.py build_ext -i`的时候报出：`setup.py: error: cannot find PCL, tried 		pkg-config pcl_common-1.7 		pkg-config pcl_common-1.6 		pkg-config pcl_common`
* 解决方案：这里就是上面说的，别下除了1.6到1.9版本的pcl的All-In-One Installer啊。

#### 3.2 坑二：DLL load failed

* 问题：全部安装完成之后，一切没有问题了，当你打开python，运行`import pcl`的时候报出：`DLL load failed`。
* 解决方案：重启电脑！

### 四. python版本的使用

#### 4.1  点云数据的展示（python）

构建点云--Point_XYZRGBA格式(需要点云数据是N*4，分别表示x,y,z,RGB ,其中RGB 用一个整数表示颜色)，下面是python版本的点云数据展示

```python
import pcl
import pcl.pcl_visualization as viewer  #可视化库
import numpy as np

# cloud = pcl.load("cloud.pcd")
cloud_np = np.load("cloud.npy")
cloud = pcl.PointCloud_PointXYZRGBA(cloud_np)
visual = pcl.pcl_visualization.CloudViewing()
visual.ShowColorACloud(cloud)

v = True
while v:
    v = not (visual.WasStopped())
```

#### 4.2 命令行展示

由于上面已经下载了PCL1.9.1了，可以直接在命令行中进行展示：`pcl_viewer_release H cloud.PCD`，下面的是来自Middlebury 2014数据集中经过立体匹配后的3D点云图。

![](https://i.loli.net/2021/04/12/cPJwuA8LgHUmFDf.png)