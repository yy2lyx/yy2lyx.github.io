---
layout: post
title: Mac下利用jekyll和github pages搭建个人博客
date: 2020-10-17 00:00:00 +0300
description: 在Mac下利用Jekyll和githubpages来搭建个人的博客 # Add post description (optional)
img: man.png # Add image post (optional)
tags: [Mac, Jekyll, Github Pages] # add tag
---

### 一. 搭建环境

#### 1.1 下载软件

* [jekyll](https://www.jekyll.com.cn/)：这个是将纯文本转化为静态网站和博客，使用gem安装下载：`gem install bundler jekyll`。
* [github pages]()：免费开源，并且可以自动生成域名，自己去构建一个属于自己的github账号和新建一个仓库（名字为：`XX.github.io`，这里`XX`就是你自己的账号名称）

![](https://tva1.sinaimg.cn/large/007S8ZIlgy1gjs9dyt6cwj31lk0gigp9.jpg)

* mac或者linux平台：原因是jekyll无法在Windows平台上安装。

#### 1.2 选择一个适合自己的博客主题

> jekyll主题：http://jekyllthemes.org/
>
> jekyll插件：http://www.jekyll-plugins.com/

* 本人选择的jekyll主题：[Flexible-Jekyll](https://github.com/artemsheludko/flexible-jekyll)，如下图所示：

![](https://tva1.sinaimg.cn/large/007S8ZIlgy1gjs9xls4u4j31b40om7du.jpg)

* 选择一个地方存放下载的主题（直接git下载）：`git clone https://github.com/artemsheludko/flexible-jekyll `
* 这里直接在本地利用jekyll生成一个网页进行测试和调试：`bundle exec jekyll server`或者是`bundle exec jekyll s`，这里会生成一个本地的博客地址：`http://127.0.0.1:4000/`，可以直接查看

![](https://tva1.sinaimg.cn/large/007S8ZIlgy1gjsa7qt24yj31d40fuk2x.jpg)

#### 1.3 将主题放入自己的仓库中

* git下载创建好的xx.github.io该仓库到自己本地：`git clone https://github.com/xx/xx.github.io.git`
* 将之前下载的主题放到自己创建的github.io这个仓库中
* git上传修改的地方：
  * git添加修改的地方：`git add .`
  * git提交：`git commit -m "修改"`
  * git推送：`git push`
* 这里可以直接在网页上打开`https://xx.github.io/`查看自己的博客，当然这里还是别人的内容，还需要自己修改成属于自己专属的博客。
* 这里注意：如果无法访问github pages，这里需要修改dns服务器就可以了，修改为114.114.114.114，具体如下所示：

![](https://tva1.sinaimg.cn/large/007S8ZIlgy1gjs9wjy9fwj30db0gcjrz.jpg)

### 二. 创建一个专属于自己的个人博客

#### 2.1 了解下载主题的文件和文件夹的作用和内容

整个下载jekyll主题的代码结构如下图所示：

![](https://tva1.sinaimg.cn/large/007S8ZIlgy1gjsa15n5gsj30bq0l0ta3.jpg)

* _drafts文件夹：主要是存放一些自己还没有写好的markdown文档，这里是不会在网页上展示的，但是没有写完的却可以通过git来保存
* _posts文件夹：保存已经写好的markdown文章
* assets文件夹：保存一些图片和css的一些文件
* _config.yml：用来修改主页上的一些个人信息
* Gemfile：这里是该主题下需要的一些gem依赖，这里直接在当前目录下`bundle install`即可下载安装依赖，注意这里如果gem下载很慢，可以设定source源，直接在Gemfile文件开头写`source 'https://gems.ruby-china.com/'`

#### 2.2 如何在博客中展示公式

如果自己的博客中需要展示数学公式的，那么需要在`_layouts/default.html`文件中进行添加：

```js
</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
```

#### 2.3 打开自己博客网页响应很慢

这里一般是自己博客中展示的图片资源过大导致的，这里需要将图片进行无损压缩，重新上传即可。这里推荐几个图片无损压缩的网站。

> TinyPNG：https://tinypng.com/
>
> Compressor：https://compressor.io/

