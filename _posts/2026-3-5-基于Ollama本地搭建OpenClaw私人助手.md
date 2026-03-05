---
layout: post
current: post
cover: assets/images/claw.jpg
navigation: True
title: 基于Ollama本地搭建OpenClaw私人助手
date: 2026-03-05 00:00:00
tags: [NLP,DeepLearning]
excerpt: 记录学习openclaw的过程
class: post-template
subclass: 'post'
---



> 官网：https://openclaw.ai/，目前已经从clawdbot更名为moltbot了，现在又更名为openclaw了。

### 一. OpenClaw为什么那么火爆

OpenClaw（原 Clawdbot/Moltbot）能快速爆火，核心是精准击中本地隐私、全场景自动化、低门槛使用三大刚需，叠加开源社区与社交传播助力，形成现象级增长。GitHub 短时间星标破 10 万，一周网站访问量超 200 万，成为 2026 年初增长最快的开源 AI 项目之一。下面则是爆火的原因：

* 隐私可控：所有数据完全存储与本地，而且支持本地大模型，隐私更加可控。
* 任务自动化：基于AI agent，可自动处理邮件、日程、文件、代码、网页自动化。
* 多渠道接入：支持Telegram、WhatsApp、Discord、iMessage 等应用，但是微信目前还不支持。

### 二. mac上本地安装

#### 2.1 下载ollama本地模型

> ollama官网下载ollama注意需要下载最新版本，下载必要的本地/cloud模型
>
> 参考ollama的文档：https://docs.ollama.com/integrations/openclaw

这里使用`ollama run --verbose qwen3-coder:latest`来测试模型的生成速度：

* `total duration`：总耗时
* `load duration`：模型加载时间
* `eval rate`：推理生成tokens

下载推荐的llm，下载模型：`ollama pull llm`，下面测试是基于Macbook pro的M4芯片测试的结果：

| 模型          | 大小 | token的速度    | 简介                                                         |
| ------------- | ---- | -------------- | ------------------------------------------------------------ |
| qwen3-coder   | 18G  | 68.37 tokens/s | **千问团队于2025年7月推出的新一代开源AI编程大模型系列，专为代码生成、智能代理（Agent）和仓库级编程优化** |
| glm-4.7-flash | 19G  | 48.96 tokens/s | 智谱AI于2025年12月发布的GLM-4系列大模型，**主打高吞吐量和高性价比，特别强化了本地编程和Agent任务能力** |
| gpt-oss:20b   | 13G  | 51.01 tokens/s | **OpenAI于2025年8月发布的一款200亿参数的开源权重混合专家模型**它专为轻量级推理和工具调用场景设计，性能接近o3-mini，能在16GB显存的消费级硬件或边缘设备上高效运行，具备极高的本地部署性价比 |



#### 2.2 搭建OpenClaw环境

> 参考ollama的openclaw的博客：https://ollama.com/blog/openclaw

* 构建conda环境：`conda create -n clawdbot python=3.13 -y`

* 激活环境：`conda activate clawdbot`

* 下载[nodes](https://nodejs.org/en/download)：

  ```bash
  # Download and install nvm:
  curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
  
  # in lieu of restarting the shell
  \. "$HOME/.nvm/nvm.sh"
  
  # Download and install Node.js:
  nvm install 24
  
  # Verify the Node.js version:
  node -v # Should print "v24.13.0".
  
  # Verify npm version:
  npm -v # Should print "11.6.2".
  ```

* 安装/更新open claw：`npm install -g openclaw@latest`

* 安装线上wizard：`openclaw onboard --install-daemon`

* 从ollama启动openclaw：`ollama launch openclaw`

* 将node，npm，openclaw设置环境变量：`sudo vim ~/.zshrc`

  ```bash
  export PATH=~/.nvm/versions/node/v24.13.0/bin:$PATH
  export PATH="$(npm prefix -g)/bin:$PATH"
  ```

本地快速查看openclaw，聊天/查看配置：`http://127.0.0.1:18789/chat`

#### 2.3 让openclaw爬取互联网信息

1. 安装chromium（这里可以理解为开源的google chrome浏览器）：

   ```
   # 安装 Chromium
   brew install --cask chromium
   
   # 移除隔离属性（首次打开可能需要）
   xattr -dr com.apple.quarantine /Applications/Chromium.app
   ```

2. 更改openclaw的json文件，其目录在`~/.openclaw/openclaw.json`，或者直接通过打开本地的dashboard（`openclaw dashboard`）来查看和更改config。

   ```
   "browser":{
       "enabled":true,
       "executablePath":"/Applications/Chromium.app/Contents/MacOS/Chromium",
       "headless":false,
       "noSandbox":true,
       "attachOnly":false,
       "defaultProfile":"openclaw"
     }
   ```

3. 重启openclaw的网关：`openclaw gateway restart`

4. 查看browser的状态：`openclaw browser status`

5. 启动专属的浏览器：`openclaw browser start`

6. 修改web配置：`openclaw configure --section web`

### 三. open claw配置agent skills

> clawhub地址：https://docs.openclaw.ai/tools/clawhub#clawhub

#### 3.1 agent skills的优势

相较于没有skills的open claw，有agent skills的优势：

* 相同的workflow，每次使用，不需要重复去解释
* 持久化：对话的数据不会持久化，而skill的数据会在本地存储
* 节约tokens：没有skill每次需要重复的对话占用token

#### 3.2 skills 安装和使用

> Clawhub 官网：https://clawhub.ai

这里推荐使用的是我们直接clawhub的官网搜索需要的skills，找到对应的名字，比如我这里找到的就是浏览器的agent skills——Agent Browser，它可以帮助我们实现自动爬虫的功能（模拟人浏览页面的操作），我们在安装的时候也能发现，安装这个skill，需要安装playwright这个框架。

![](https://raw.githubusercontent.com/yy2lyx/picgo/admin/img/clawhub.png)

1. 安装clawhub插件：`npm i -g clawhub`
2. 安装agent-browser：`clawhub install "agent-browser"`

如果上面第二步下载不成功，也可以基于git来安装：

```bash
npm install -g pnpm
git clone https://github.com/vercel-labs/agent-browser
cd agent-browser
pnpm install
pnpm build
agent-browser install
```

至此，就安装好了！

