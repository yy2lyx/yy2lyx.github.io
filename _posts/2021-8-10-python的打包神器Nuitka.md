---
layout: post
current: post
cover: assets/images/nuitka.jpeg
navigation: True
title: python的打包神器——Nuitka
date: 2021-8-10 15:11:00
tags: [python]
excerpt: 对比pyinstaller和nuitka打包工具及nuitka的使用过程
class: post-template
subclass: 'post'
---


### 一. pyinstaller和Nuitka使用感受

#### 1.1 使用需求

> 这次也是由于项目需要，要将python的代码转成exe的程序，在找了许久后，发现了2个都能对python项目打包的工具——pyintaller和nuitka。

这2个工具同时都能满足项目的需要：

* **隐藏源码**。这里的pyinstaller是通过设置key来对源码进行加密的；而nuitka则是将python源码转成C++（这里得到的是二进制的pyd文件，防止了反编译），然后再编译成可执行文件。
* **方便移植**。用户使用方便，不用再安装什么python啊，第三方包之类的。

#### 1.2 使用感受

2个工具使用后的最大的感受就是：

* pyinstaller体验很差！
  * 一个深度学习的项目最后转成的exe竟然有近3个G的大小（**pyinstaller是将整个运行环境进行打包**），对，你没听错，一个EXE有3个G！
  * 打包超级慢，启动超级慢。
* nuitka真香！
  * 同一个项目，生成的exe只有7M！
  * 打包超级快（1min以内），启动超级快。

### 二. Nuitka的安装及使用

#### 2.1 nuitka的安装

* 直接利用pip即可安装：`pip install Nuitka `
* 下载vs2019(MSVS)或者MinGW64，反正都是C++的编译器，随便下。

#### 2.2 使用过程

对于第三方依赖包较多的项目（比如需要import torch,tensorflow,cv2,numpy,pandas,geopy等等）而言，这里最好打包的方式是**只将属于自己的代码转成C++，不管这些大型的第三方包！**

以下是我demo的一个目录结构（这里使用了pytq5框架写的界面）：

```json
├─utils //源码1文件夹
├─src // 源码2文件夹
├─logo.ico // demo的图标
└─demo.py // main文件
```

使用以下命令（调试）直接生成exe文件：

```bash
nuitka --standalone --show-memory --show-progress --nofollow-imports --plugin-enable=qt-plugins --follow-import-to=utils,src --output-dir=out --windows-icon-from-ico=./logo.ico demo.py
```

这里简单介绍下我上面的nuitka的命令：

* `--standalone`：方便移植到其他机器，不用再安装python
* `--show-memory --show-progress`：展示整个安装的进度过程
* `--nofollow-imports`：不编译代码中所有的import，比如keras，numpy之类的。
* `--plugin-enable=qt-plugins`：我这里用到pyqt5来做界面的，这里nuitka有其对应的插件。
* `--follow-import-to=utils,src`：需要编译成C++代码的指定的2个包含源码的文件夹，这里用`,`来进行分隔。
* `--output-dir=out`：指定输出的结果路径为out。
* `--windows-icon-from-ico=./logo.ico`：指定生成的exe的图标为logo.ico这个图标，这里推荐一个将图片转成ico格式文件的网站（比特虫）。
* `--windows-disable-console`：运行exe取消弹框。这里没有放上去是因为我们还需要调试，可能哪里还有问题之类的。

经过1min的编译之后，你就能在你的目录下看到：

```json
├─utils //源码1文件夹
├─src // 源码2文件夹
├─out // 生成的exe文件夹
    ├─demo.build 
    └─demo.dist
		└─demo.exe // 生成的exe文件
├─logo.ico // demo的图标
└─demo.py // main文件
```

当然这里你会发现真正运行exe的时候，会报错：`no module named torch,cv2,tensorflow`等等这些没有转成C++的第三方包。

这里需要找到这些包（我的是在software\python3.7\Lib\site-packages下）复制（比如numpy,cv2这个文件夹）到`demo.dist`路径下。

至此，exe能完美运行啦！

