## 凡人炼丹传-博客


## 主题应用需要注意的地方

* 验证方式：登陆git page来查看：`https://yy2lyx.github.io/`。

* 本地clone了Jasper2这个主题之后，需要利用命令：`bundle exec jekyll serve`生成html文件（文件夹在`../jasper2-pages`），这里需要在本地新建一个目录`_site`来存放这些html文件。

* 利用[netlify](https://www.netlify.com/)进行加速的时候，可能存在deploy失败的情况，本次遇到的是由于ubuntu镜像版本过老导致的，可以在`build & deploy`下的`Build image selection`进行更新。

## 发布前检查

图床目录或文章图片地址变更后，需要重新生成 `_site`，避免发布页面继续引用旧图片路径：

```bash
bundle exec jekyll build
ruby scripts/check-picgo-images.rb
```

校验脚本会同时检查文章源码、生成页面和相邻目录中的 `../picgo/img` 本地图床仓库。

更新文章封面或首页轮播图后，先生成轻量 WebP 图片再构建：

```bash
python3 scripts/optimize_site_images.py
bundle exec jekyll build
```
