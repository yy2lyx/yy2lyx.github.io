---
layout: post
current: post
cover: assets/images/hannah-cover.jpg
navigation: True
title: Postman和Jmeter进行上传文件及压力测试
date: 2020-03-23 15:21:00
tags: [压力测试]
excerpt: 讲述如何利用Postman和Jmeter对网络接口进行压力测试
class: post-template
subclass: 'post'
---



### 一. 准备工作

> postman下载链接：https://www.postman.com/downloads/
>
> Jmeter下载链接：http://jmeter.apache.org/download_jmeter.cgi
>
> flask代码地址：https://github.com/yy2lyx/FlaskTutorial/tree/master/Flask-7-upload
>
> windows下scoop下载jdk(这里是由于Jmeter需要)：`scoop install ojdkbuild`

### 二. 构建接口的flask服务

其中包含前端表单index.html文件如下和flask的后端

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>upload</title>
</head>
<body>
    <form action = "/success" method = "post" enctype="multipart/form-data">
        <input type="file" name="file">
        <input type = "submit" value="Upload">
    </form>
</body>
</html>
```

### 三. Postman的http接口测试

postman分别在header和body中填入下图：

* headers中需要填入value:multipart/form-data

![](https://tva1.sinaimg.cn/large/007S8ZIlgy1gjgo0qznj0j30p605qmxn.jpg)

* body 中需要填入key：file(这里参考index.html文件中name="file")，value:eml文件地址

![](https://tva1.sinaimg.cn/large/007S8ZIlgy1gjgo1floa4j30m20463yp.jpg)

* 然后将写好的保存在collections当中，并构建tests选项（如果不填入，后面的串行压力测试无法开始，报错）

![](https://tva1.sinaimg.cn/large/007S8ZIlgy1gjgo31nohoj30on084aaz.jpg)

* 通过collections中选中保存好的请求，run即可

### 四. Jmeter的http接口测试

在下载解压好的jmeter二进制文件中打开：`apache-jmeter-5.3\bin\jmeter.bat`

* 新建一个线程组，如下图，包括http请求及监听

![](https://tva1.sinaimg.cn/large/007S8ZIlgy1gjgo1myv4aj308903vq2y.jpg)



* 线程中填入线程总数，和全部线程开启总的时间（这里由于需要测试并发1小时2万次访问）

![](https://tva1.sinaimg.cn/large/007S8ZIlgy1gjgo3saf9cj30am0a63z4.jpg)

* 在http请求页面填入请求的参数

![](https://tva1.sinaimg.cn/large/007S8ZIlgy1gjgo41hoa3j30t1089jsj.jpg)

* http页面下面不用填Parameters和BodyData，在Files Upload中填入下图，其中file和上面一致，而MIME Type需要访问https://www.freeformatter.com/mime-types-list.html，找到其中.eml格式前面的

* 运行，即可看到并行的接口请求情况

### 五. 总结

* 一般的网络接口测试，功能性测试postman较为好用。
  
* 需要测试高并发的情况下，只能用Jmeter来进行测试，因为postman是串行，而Jmeter是多线程并行测试。

