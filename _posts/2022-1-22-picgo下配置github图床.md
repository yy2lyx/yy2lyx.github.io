---
layout: post
current: post
cover: assets/images/picgo.jpg
navigation: True
title: picgo下配置github图床
date: 2022-1-20 11:11:00
tags: [图床,博客]
excerpt: 记录在博客中上传图片的好用工具和github图床相关配置
class: post-template
subclass: 'post'
---

### 一. 必须的安装
* picgo：强大的能快速创建图片url的工具（支持多种图床），简直不要太好用。我们可以直接在官网下载相应版本的[picgo](https://picgo.github.io/PicGo-Doc/zh/guide/#%E4%B8%8B%E8%BD%BD%E5%AE%89%E8%A3%85)，我这里选用的是windows版本的，当然也有mac版本的。
* git：这里推荐用scoop进行安装`scoop install git`

### 二. 搭建属于自己的github图床
#### 2.1 新建一个共有仓库

首先，要搭建一个github图床，我们需要创建一个**共有仓库**（注意：如果是创建私有仓库根本无法显示图片出来）来存储上传的图片。
![](https://raw.githubusercontent.com/yy2lyx/picgo/admin/img/new-rep.jpg)

#### 2.2 创建admin分支
这里如果不会用git创建分支的同学可参考我上一篇文章[GIT命令学习](https://www.lixiaofei2yy.website/git%E5%91%BD%E4%BB%A4)。

这里注意最好选择帮你添加一个README.md文件。
* 克隆刚创建的远程仓库：`git clone XX.git`
* 创建并切换到admin分支：`git checkout -b admin`
* 向远端仓库推送admin分支：`git push orgin admin`

这时，我们就可以在仓库中看到admin分支了，如下图。
![](https://raw.githubusercontent.com/yy2lyx/picgo/admin/img/admin.jpg)

#### 2.3 设置token
这里我们还需要设置token，直接点击该[链接](https://github.com/settings/tokens)即可。注意，这里需要将repo选择。如下图所示。
![](https://raw.githubusercontent.com/yy2lyx/picgo/admin/img/token.jpg)
然后点击generate token就完成了。

### 三. 在picgo下配置github图床
记住上面刚刚配置好的token和仓库名字及分支名admin，按照下图进行配置即可。
![](https://raw.githubusercontent.com/yy2lyx/picgo/admin/img/picgo.jpg)

至此，我们就可以愉快的上传图片咯！



