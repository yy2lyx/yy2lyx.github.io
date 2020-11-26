## 李小肥的YY博客

* Jekyll主题：Jasper2

## 主题应用需要注意的地方
* 本地clone了Jasper2这个主题之后，需要利用命令：`bundle exec jekyll serve`生成html文件（文件夹在`../jasper2-pages`），这里需要在本地新建一个目录`_site`来存放这些html文件。

* 利用netlify进行加速的时候，需要注意的是他所build的是`_site`文件夹下的内容，因此每次修改项目代码的时候同时记得将`_site`文件夹更新。