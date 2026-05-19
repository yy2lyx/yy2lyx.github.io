---
layout: post
current: post
cover: assets/images/tools.png
navigation: True
title: Function call、MCP、SKILL下哪个才是大模型的最优解
date: 2026-05-19 00:00:00
tags: [NLP,DeepLearning]
excerpt: 大模型的工具对比
class: post-template
subclass: 'post'
---

### 一. Function Call和MCP

#### 1.1 Function Call

**Function Call 是大模型原生具备的「结构化输出能力」，用来让模型主动请求外部函数 / API。**

- 起源：2023 年 OpenAI GPT-3.5/4 推出

- 本质：**模型输出格式**（JSON），不是协议

- 工作方式：

  1. 你给模型「函数列表 + 参数定义」

  2. 用户提问 → 模型判断要调用哪个函数

  3. 模型输出类似：

     ```
     {"name":"get_weather","arguments":{"city":"武汉"}}
     ```

  4. 后端执行函数 → 结果回传给模型 → 模型生成回答

#### 1.2 MCP

**MCP（Model Context Protocol，模型上下文协议）是 Anthropic 2024 年底开源的「开放、标准化、双向通信协议」，用于连接任意 LLM 与任意工具 / 数据源。**

* 本质：**跨模型、跨工具的通用通信标准（像 USB‑C）**

* 架构：Client ↔ Host ↔ Server（解耦、双向）

* 能力不止函数调用：

  - **Tools**：调用工具（类似 Function Call）

  - **Resources**：直接挂载大文件 / 数据库 / 代码库

  - **Prompts**：服务端下发提示词模板

  - 会话管理、状态保持、错误处理、双向调用

#### 1.3 两者核心区别

**Function Call**：你告诉模型怎么调用工具，那么就肯定包含了一下这些必须定义的内容

 * tool 定义规范
 * 定义入参和出参argument的结构
 * 写错误处理（timeout、重试、异常）
 * 每个模型（OpenAI / 通义 / 文心）格式不一样，还要**适配 N 套**

MCP：工具自己告诉模型我能做什么，相当于上述需要自定义的内容直接不用做了，**全自动**了属于是。

![](https://raw.githubusercontent.com/yy2lyx/picgo/admin/img/mcp.png)



#### 1.4 两者适用场景

下面是用表格的形式来展示2者的不同：

| 特性                        | Function Call                            | MCP                              |
| --------------------------- | ---------------------------------------- | -------------------------------- |
| **Setup（搭建方式）**       | 手动配置，硬编码嵌入到你的应用中         | 工具可注册，支持自动发现         |
| **Scalability（可扩展性）** | 扩展困难；每新增一个工具都需要编写新代码 | 扩展性强；即插即用的工具生态     |
| **Maintenance（维护成本）** | 所有逻辑都需要你自行管理                 | 简化维护；由协议本身处理通用逻辑 |
| **Best For（适用场景）**    | 简单、单一用途的应用                     | 复杂、多智能体、企业级系统       |

### 二. Skills

skills其实是用自然语言的方式记录一套固定的流程手册或者领域的SOP，它主要是将一套agent的工作流封装成一个skill，**让AI更清晰、不偏的、按照步骤的一步一步将任务做对**。

#### 2.1 核心定位

MCP

- 层级：**底层集成层 / 协议层**
- 角色：**通用、中立、跨模型、跨工具**的连接标准
- 不关心业务：只定义 **工具发现、调用格式、参数、错误处理、权限、状态**
- 一句话：**MCP 管 “能不能连、怎么连”**

Skills

- 层级：**上层知识层 / 执行层**
- 角色：**领域专属、任务导向、可复用的技能包**（本质是结构化提示词 + 流程）
- 关心业务：定义**触发条件、步骤、判断逻辑、输出格式、边界处理、示例**
- 一句话：**Skills 管 “会不会做、怎么做对”**

#### 2.2 如何写Skills

**不需要**关心底层连接，但必须写：

- **Skill 触发条件**（什么时候用这个技能）
- **完整步骤 SOP**（先做什么、后做什么、判断分支）
- **参数要求、输出模板、示例、边界处理**
- 本质：**写一份给 AI 看的详细操作手册**

下面是一个典型的skills的目录结构：

```bash
.claude/skills
├── pdf-parsing
│   ├── script.py
│   └── SKILL.md
├── python-code-style
│   ├── REFERENCE.md
│   └── SKILL.md
└── web-scraping
    └── SKILL.md
```

对于每个skill，必须要包含`SKILL.md`文件，下面是一个`SKILL.md`的例子：

```bash
---
name: web-scraping
description: 使用scpr命令行工具，根据提供的网址抓取网页内容。
---

当需要抓取网页时，请按以下方式使用scpr命令行工具:

```bash
scpr --url <https://example.com>
```

当然了你也可以将mcp，甚至是其他的Skills封装到Skills中，比如

```bash
---
name: web-scraping
description: 使用scpr命令行工具，根据提供的网址抓取网页内容。
---

## 概述
本技能让 AI 能够通过 **MCP（模型上下文协议）** 调用 `scpr` 命令行工具，实现对公开网页内容的抓取。
所有工具定义、参数解析、错误处理和执行流程，均由 MCP 服务端统一管理。
AI 只需关注**何时触发**和**如何处理用户输入**，无需编写任何胶水代码。

## 技能用途
- 从用户提供的 URL 中抓取网页文本、元数据和主要内容
- 使用标准化的 MCP 工具调用，替代零散的自定义函数调用
- 由 MCP 自动管理超时、异常和 CLI 执行细节

## 触发条件
- 用户明确提出“抓取、爬取、提取”某个网页内容的需求
- 用户提供了具体的网页 URL，并希望获取该页面的信息

## MCP 集成（由 MCP 服务端预定义）
MCP 服务端会**自动暴露**以下工具，无需手动定义函数、参数解析或错误处理。

### MCP 工具名称
`web_scrape`

### MCP 自动提供的参数
- `url` (字符串，必填)：需要抓取的网页 URL

### MCP 自动处理的功能
- 参数校验
- CLI 命令执行：`scpr scrape --url <url>`
- 网络超时
- 无效 URL 拦截
- 连接失败处理
- 统一返回格式

## 技能行为（给 AI 的执行指令）
当用户请求抓取网页时：
1. 从用户提问中提取出目标 **URL**
2. 通过 MCP 调用 `web_scrape` 工具，并传入该 URL
3. 将抓取到的内容整理为清晰易读的格式回复给用户
4. 如果 MCP 返回错误，将其以友好的方式告知用户

## 执行流程示例
### 用户输入
帮我抓取这个网页：shturl.cc/bbhhySzVq

### AI 通过 MCP 执行的动作
调用 MCP 工具（内部格式）：
```json
{
  "toolName": "web_scrape",
  "arguments": {
    "url": "shturl.cc/bbhhySzVq"
  }
}
```

### 三. function call、MCP、SKILL三者总结

* **Function Call（全手动）**：构建管道，自定义工具，

* **MCP（全自动）**，构建管道，连接标准化

* **Skills（老司机手册）**：流程结构化，AI 会做复杂任务