## 李小肥的YY博客

* Jekyll主题：Jasper2


## 主题应用需要注意的地方

* 验证方式：登陆git page来查看：`https://yy2lyx.github.io/`。

* 本地clone了Jasper2这个主题之后，需要利用命令：`bundle exec jekyll serve`生成html文件（文件夹在`../jasper2-pages`），这里需要在本地新建一个目录`_site`来存放这些html文件。

* 利用[netlify](https://www.netlify.com/)进行加速的时候，可能存在deploy失败的情况，本次遇到的是由于ubuntu镜像版本过老导致的，可以在`build & deploy`下的`Build image selection`进行更新。
