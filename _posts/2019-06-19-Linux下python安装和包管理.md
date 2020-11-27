---
layout: post
current: post
cover: assets/images/linux.jpg
navigation: True
title: Linux下python安装和包管理
date: 2019-06-19 20:20:00
tags: Linux, Python]
excerpt: 讲述在Linux环境下python包编译及安装过程，以及包管理工具virtualenv
class: post-template
subclass: 'post'
---


#### 1. 上传python文件并打包编译

* 下载python版本：https://www.python.org/ftp/python/
* 解压：`tar -xf Python-3..1.tgz`
* 编译：`sudo ./configure --prefix=/path/you/want/to/install/ --with-ssl && make && make install`(这里需要加--prefix是因为可以直接在指定文件夹下删除软件即可，加入with ssl是由于pip需要ssl),在编译结束后，正常程序会装在 /usr/local/bin 下（注意这里如果不加--with-ssl**默认安装的软件涉及到ssl的功能不可用**）
* 创建软连接：`ln -sf /usr/local/bin/python3.8 /usr/bin/python`和`ln -sf /usr/local/bin/python3.8-config /usr/bin/python-config`

#### 2. venv管理和包安装

* 安装virtualenvs：`pip3 install virtualenv`

* 创建环境：`sudo virtualenv --python=python3.6 环境名字`

* 安装第三方包：进入环境下的bin目录，`sudo ./pip3 install -r requirements.txt  -i 指定的pip安装源   `这里指定安装源较快。

#### 3. 创建软连接
```
ln -sf /usr/local/bin/python3.8 /usr/bin/python
ln -sf /usr/local/bin/python3.8-config /usr/bin/python-config
```
