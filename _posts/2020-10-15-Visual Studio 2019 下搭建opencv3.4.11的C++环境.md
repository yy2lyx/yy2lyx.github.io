---
layout: post
title: Visual Studio 2019 下搭建opencv3.4.11的C++环境
date: 2020-10-15 00:00:00 +0300
description: 在Windows环境下利用Visual Studio 2019搭建基于C++的opencv3环境 # Add post description (optional)
img: balls.jpg # Add image post (optional)
tags: [Visual Studio, Windows, Opencv3, C++] # add tag
---

### 一. 下载需要的软件

* [visual studio 2019 社区版](https://visualstudio.microsoft.com/zh-hans/downloads/)
* [opencv3.4.11](https://opencv.org/releases/)


### 二. 基于C++的环境搭建
#### 2.1 创建系统环境变量
* 解压opencv，到`D:\software`

* 配置系统变量：Path下添加Opencv的路径`D:\software\opencv\opencv\build\x64\vc15\bin`（这里选择vc15更适合vs2019，如果是vs2015就选择vc14）

#### 2.2 在Visual Studio2019中配置Opencv
* 选择视图-属性管理器- 选择Debugx64-添加新项目属性表-这里选择保存的名称和位置

![](https://tva1.sinaimg.cn/large/007S8ZIlgy1gjrjssvz0fj30qc0hydg5.jpg)
* 选择VC++目录-包含目录中添加以下
```
D:\software\opencv\opencv\build\include
D:\software\opencv\opencv\build\include\opencv
D:\software\opencv\opencv\build\include\opencv2
```

* 选择VC++目录-库目录中添加`D:\software\opencv\opencv\build\x64\vc15\lib`
* 选择链接器-输入-附加依赖项中添加`opencv_world3411d.lib`
* 保存即可，注意这里构建的新建项目属性表可以保存下来，直接其他的项目直接导入用即可（视图-属性管理器- 选择Debugx64-添加现有属性表）
* 回到解决方案资源管理器-项目-属性-配置管理器-活动解决方案平台-选择x64-Debug

![](https://tva1.sinaimg.cn/large/007S8ZIlgy1gjrjt4rjq6j30j50dfmx9.jpg)

### 三. 构建代码测试

* 构建cpp源码：解决方案-源文件-添加-新建项-cpp文件

用以下代码进行测试

```c
#include<iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/opencv.hpp>
#include<math.h>


using namespace cv;
using namespace std;

int main()
{

	//Mat img = imread("D:\\vs_project\\opencvtest\\1.jpg");
	Mat img = imread("D:/vs_project/opencvtest/1.jpg");
	if (img.empty()) {
		cout << "Could not load img..." << endl;
		return -1;
	}
	namedWindow("ori_img", WINDOW_AUTOSIZE);
	imshow("ori_img", img);

	// 图像转成灰度图像
	Mat gray_img;
	cvtColor(img, gray_img, CV_RGB2GRAY);
	namedWindow("gray_img", WINDOW_AUTOSIZE);
	imshow("gray_img", gray_img);
	waitKey(0);


	return 0;
}
```



