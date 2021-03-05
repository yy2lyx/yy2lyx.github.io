---
layout: post
current: post
cover: assets/images/linux.jpg
navigation: True
title: Linux下python安装和包管理
date: 2019-06-19 20:20:00
tags: [python,环境搭建]
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

#### 4. 设置pip镜像源，下载提速
之前利用pip进行安装的时候，要不是直接在pip下载的中途断掉，要不就是网速特别慢。这里推荐设置下国内的源进行pip下载。

> 临时使用的方式：`pip install tensorflow -i 国内源`

**国内源**：
* 清华：https://pypi.tuna.tsinghua.edu.cn/simple
* 阿里云：http://mirrors.aliyun.com/pypi/simple/
* 中国科技大学 https://pypi.mirrors.ustc.edu.cn/simple/
* 华中理工大学：http://pypi.hustunique.com/
* 山东理工大学：http://pypi.sdutlinux.org/ 
* 豆瓣：http://pypi.douban.com/simple/

这里最好不要一味的相信某一个源（比如清华源），吐槽下：下其他的包速度都很快，某些包的时候不仅慢，它还中途断掉！

所以推荐最好每个都试试！

> 永久配置某个源：这里就不需要再加`-i 国内源`

linux：修改 `~/.pip/pip.conf`
windows：直接在user目录中创建一个pip目录，如：`C:\Users\xx\pip`，新建文件`pip.ini`

linux和windows的具体内容都一致，如下：

```bash
[global]
index-url = 国内源
[install]
trusted-host=mirrors.aliyun.com
```