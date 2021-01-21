---
layout: post
current: post
cover: assets/images/grapes.jpg
navigation: True
title: Windows下安装mysql和导入sql文件
date: 2020-10-16 19:21:00
tags: [mysql, 数据库]
excerpt: 介绍在Windows环境下本地安装mysql和导入sql文件的使用
class: post-template
subclass: 'post'
---




### 一. 下载软件
* mysql：这里使用的是scoop来进行安装：`scoop install mysql`，这里的优势是自动帮你配好环境了
* 安装[Navicat Premium](https://www.navicat.com.cn/download/navicat-premium)
* 破解[Navicat Premium](https://www.nrgh.net/archives/navicat-premium.html)

### 二. 初始化mysql
* 初始化数据库：`mysqld --initialize --console`,并记录红色标注的字符，这是随机生成的密码
![](https://tva1.sinaimg.cn/large/007S8ZIlgy1gjrjtkoau2j30sg0lcqbu.jpg)
* 输入`mysqld -install`将mysql安装为Windows的服务：
* 启动mysql：`net start mysql`
* 首次进入mysql：`mysql -u root -p`，输入第一次的系统生成的密码
* 输入`ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'mysql的密码'`;回车  别漏了后面有个分号  mysql的密码是安装mysql时设置的密码
* 输入`FLUSH PRIVILEGES;`，这里一定要输入，不然用navicat链接的时候会报`1251连接不成功`
* 修改my.ini文件：首先进入scoop安装的mysql文件夹中（C:\Users\Administrator\scoop\apps\mysql\8.0.21），修改my.ini文件，如果不加`secure_file_priv=''`，会导致无法导入导出数据。

```bash
[mysqld]
datadir=D:/yeyan/mysql_data/data
secure_file_priv=''   
[client]
user=root
```

### 三.  导入.sql文件

* 启动mysql：`net start mysql`
* 首次进入mysql：`mysql -u root -p`，输入自己的密码
* 查看数据库：`show databases;`
* 使用某个数据库：`use test`
* 查看该数据库下的表：`show tables;`
* 导入sql文件：`source D:/git_repo/Trace/data.sql;`