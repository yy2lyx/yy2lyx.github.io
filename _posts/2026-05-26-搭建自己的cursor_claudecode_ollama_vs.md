---
layout: post
current: post
cover: assets/images/my_cursor.png
navigation: True
title: 打造属于自己的免费Cursor：claude code+ollama+vscode
date: 2026-05-26 00:00:00
tags: [NLP,DeepLearning]
excerpt: 讲述利用claude code和ollama来打造属于自己免费的coding agent
class: post-template
subclass: 'post'
---


### 一. AI Coding软件的横向对比

| 工具名称      | 厂商              | 付费模式（个人）                                             | 模型能力（默认 / 上限）                                      | 核心特色                                                     |
| ------------- | ----------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Cursor**    | Anysphere（美国） | 免费版（基础）；**Pro $20 / 月**；企业定制                   | 默认：**Claude 3.5/4 Sonnet/Opus**；可接 GPT-5、Gemini；长上下文强（200K+） | VS Code 魔改 IDE；全仓库索引 + 实时代码编辑；**AI 体验最顺滑**；国内网络友好；生态插件多 |
| **Codex**     | OpenAI（美国）    | 网页版：ChatGPT Plus **$20/月**；CLI：开源免费；Pro $120 / 月 | 仅限 **GPT-5.3/5.4 Codex**；推理强、代码准确率高；上下文 128K | 云端沙箱安全执行；异步批量任务强；CLI 轻量；适合**自动化脚本、批量生成**；封闭生态 |
| **Trae**      | 字节（中国）      | 基础版**永久免费**；Pro **¥60 / 月（≈$10）**                 | 自研 **Doubao 大模型**；兼容 GPT-4/5、Claude、本地 Ollama；中文理解强、长文本友好 | 国产 AI IDE（VS Code 分支）；**中文优化 + 国内速度快**；本地优先、隐私友好；适合国内团队 / 学生 |
| **CodeBuddy** | 腾讯（中国）      | 基础免费；Pro **¥40 / 月**；学生特惠                         | 自研 **GLM-4/5**；可接入 GPT-4/5、Claude、Ollama；编码 + 工具调用均衡 | 轻量 IDE 插件 + 独立客户端；**国内合规 + 数据不出境**；文档 / 注释生成强；适合政企 / 学术场景 |
| **OpenCode**  | 开源              | **完全开源免费**；仅付第三方 API 费用                        | 无绑定：支持 **75+ LLM**（Claude、GPT、Gemini、Kimi、Ollama）；本地 / 云端自由切换 | Claude Code 开源平替；**TUI 终端界面友好**；多会话 Agent；国内访问稳定；高度可定制 |

说实话，如果咱们有钱，那肯定**无脑入cursor**啊，奈何钱包有限，只能用免费方案了，那么下面就是一种**免费平替**方案：

* 大模型：利用ollama的免费cloud模型或者直接使用本地模型
* Coding Agent：这里选择claude code作为coding agent
* 界面：选择强大的vscode

### 二. 本地安装相关软件

> claude code 文档：https://code.claude.com/docs/en/overview
>
> Ollama 官网：https://ollama.com

#### 2.1 下载必须软件

Claude code 安装：

* 下载Mac：`curl -fsSL https://claude.ai/install.sh | bash`
* 给zshrc加可写权限：`sudo chmod 777 ~/.zshrc`
* 加载环境变量：`echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc && source ~/.zshrc`

ollama安装：`curl -fsSL https://ollama.com/install.sh | sh`

#### 2.2 ollama模型的选型

针对coding场景，ollama适配claude code的模型需要具备的能力：

* 支持ools/function calling
* 长上下文：这里推荐至少 32k
* 适合编码
* 最好能支持图片

针对下面不同的场景，这里推荐的ollama模型：

| 使用诉求                       | 推荐模型              | 一键启动命令                                         | 厂商 + 简介                                                  |
| ------------------------------ | --------------------- | ---------------------------------------------------- | ------------------------------------------------------------ |
| 离线 / 隐私优先、低配电脑      | qwen2.5-coder:7b      | `ollama launch claude --model qwen2.5-coder:7b`      | 阿里巴巴（通义千问）：轻量强编码、工具调用稳、本地首选       |
| 免费云端主力（首选）           | minimax-m2.5:cloud    | `ollama launch claude --model minimax-m2.5:cloud`    | MiniMax（稀宇科技）：国内速度快、Claude Code 适配好、免费最优 |
| 长代码 / 大项目（免费）        | kimi-k2.5:cloud       | `ollama launch claude --model kimi-k2.5:cloud`       | Moonshot（月之暗面）：百万级上下文、读大库强                 |
| 重度开发、不限额度（Pro 订阅） | minimax-m2.7:cloud    | `ollama launch claude --model minimax-m2.7:cloud`    | MiniMax：M2.5 升级版、Agent 更强、付费主力                   |
| 架构 / 底层开发（Pro 订阅）    | deepseek-v4-pro:cloud | `ollama launch claude --model deepseek-v4-pro:cloud` | DeepSeek（深度求索）：顶尖推理 + 架构能力、复杂系统首选      |

### 三.  在VScode中使用claude code

#### 3.1 vscode配置claude code

打开vscode的plugins，然后搜索`Claude Code for VS Code`这个插件，进行安装。

claude code在vscode中使用的是2种方式，这里可以使用`cmd` + `shift` +`p`，然后选择`Claude Code: Open In New Tap`来进行切换：

* **CLI**：即命令行模式，**说实话其实也很好看**。
* **原生UI**：使用命令行可能大多数人在界面上不适应，这里可以选择使用原生UI。

#### 3.2 配置Claude code的ollama模型

**临时在CLI中使用ollama模型**

当我们直接在命令行中输入`claude`后，会发现模型还是claude自带收费模型`Opus 4.7 (1M context)`，我们这里需要推出后，使用ollama模型：`claude --model minimax-m2.5:cloud`，就可以发现你可以直接在vscode中使用ollama模型的claude了，如下图。

![claude_cli](/Users/carlyye/LLM/博客/claude_cli.png)

**永久在原生UI和CLI中使用ollama模型**

vscode 设置全局变量`settings.json`，这里需要将可选的模型和url改成ollama的模型：

```
{
    "claudeCode.preferredLocation": "panel",
    "terminal.integrated.inheritEnv": false,
    "terminal.integrated.mouseWheelScrollSensitivity": 3,
    "claudeCode.environmentVariables": [
    
        {
            "name": "ANTHROPIC_BASE_URL",
            "value": "http://localhost:11434"
        },
        {
            "name": "ANTHROPIC_AUTH_TOKEN",
            "value": "ollama"
        },
        {
            "name": "ANTHROPIC_DEFAULT_HAIKU_MODEL",
            "value": "minimax-m2.5:cloud"
        },
        {
            "name": "ANTHROPIC_DEFAULT_SONNET_MODEL",
            "value": "minimax-m2.5:cloud"
        },
        {
            "name": "ANTHROPIC_DEFAULT_OPUS_MODEL",
            "value": "minimax-m2.5:cloud"
        },

    ]
}
```

配置完上述的环境变量后，当你再次打开vscode后就发现默认使用的就是ollama的`minimax-m2.5:cloud`模型了。

![claude_ui](/Users/carlyye/LLM/博客/claude_ui.png)

