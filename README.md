# 🎓 吉大校园助手 · JLU Chatbot

面向吉林大学校园信息问答的 AI Agent 项目，使用 **Streamlit** 构建交互界面，结合 **LangChain / LangGraph**、本地知识库和网页搜索，支持自然语言问答与多轮对话。

项目处于开发阶段，当前已提供聊天问答、知识库管理和历史对话功能。

## 功能

- **流式聊天**：逐步展示模型回答，支持连续追问。
- **知识库问答**：将文档切分后建立 Chroma 向量库和 BM25 索引，使用 RRF 融合候选结果，再通过本地交叉编码模型重排。
- **网页搜索**：Agent 可按需调用 Tavily 补充知识库不足的信息，页面展示本轮搜索来源链接。
- **知识库管理**：支持上传 TXT、Markdown、PDF、DOCX，预览解析正文、按文件名搜索、查看入库片段、替换同名文档和删除文档。
- **内容去重与索引恢复**：相同正文避免重复入库，提供“重建检索索引”操作以同步 BM25 索引。
- **历史对话**：通过 SQLite 保存聊天记录及 Agent 上下文，支持新建、恢复和删除对话。
- **上下文总结**：消息达到配置阈值时自动总结较早的上下文，完整聊天记录单独保存。

## 技术栈

| 模块 | 实现 |
| --- | --- |
| Web 界面 | Streamlit |
| Agent 与会话状态 | LangChain、LangGraph、SQLite Checkpointer |
| 对话与查询改写 | DeepSeek |
| 文本嵌入 | DashScope |
| 向量检索 | Chroma |
| 关键词检索 | bm25s、pkuseg 中文分词 |
| 检索融合与重排 | RRF、Sentence Transformers CrossEncoder |
| 联网搜索 | Tavily |
| 文档解析 | pypdf、python-docx |
| 依赖管理 | uv、pyproject.toml、uv.lock |

## 快速开始

### 1. 准备环境

- Python **3.13 或更高版本**，建议首次运行使用 Python 3.13。
- 已安装 `uv`，可通过 `uv --version` 检查。
- 准备 DeepSeek、DashScope 和 Tavily 的 API 密钥。
- 安装依赖、调用外部 API，以及首次下载本地重排模型时需要网络连接。

在克隆后的项目根目录执行：

```bash
uv sync
```

当前配置在 Windows 上使用 PyTorch CUDA 13.0 软件源。检索重排会根据 `torch.cuda.is_available()` 自动选择 CUDA 或 CPU；首次安装与首次检索可能需要较长时间。

### 2. 配置密钥

复制根目录的 `.env.example` 为 `.env`。已有 `.env` 时，直接补充配置即可。

Windows PowerShell：

```powershell
Copy-Item .env.example .env
```

macOS / Linux：

```bash
cp .env.example .env
```

编辑 `.env`：

```dotenv
DEEPSEEK_API_KEY=your_deepseek_api_key
TAVILY_API_KEY=your_tavily_api_key
DASHSCOPE_API_KEY=your_dashscope_api_key
```

| 变量 | 用途 |
| --- | --- |
| `DEEPSEEK_API_KEY` | 对话生成、上下文总结与检索问题改写 |
| `TAVILY_API_KEY` | 网页搜索；当前在 Agent 初始化时校验 |
| `DASHSCOPE_API_KEY` | 文档嵌入和向量检索；当前在知识库服务初始化时校验 |

系统环境变量优先于项目根目录的 `.env`。`.env` 已被 `.gitignore` 排除，请将真实密钥保留在本地。

### 3. 启动应用

```bash
uv run streamlit run streamlit_app/main.py
```

根据终端显示的地址打开应用，默认通常为 `http://localhost:8501`。

### 4. 上传资料并开始问答

1. 在侧边栏进入 **知识库管理**，上传校园资料。
2. 检查解析后的正文，点击 **上传到知识库**。
3. 进入 **聊天问答**，输入问题，例如“吉林大学有哪些校区？”或“根据上传的资料介绍学校历史”。
4. 在侧边栏新建、恢复或删除历史对话。

`src/JLU_agent/repo/materials/` 中保存了校园资料，可选择其中的文本文件上传。该目录的文件不会仅因放入目录就自动入库；实际知识库以 Chroma 中保存的切片为准。

## 工作流程

```text
上传文档 → 解析正文 → 文本切分 → DashScope 嵌入 → Chroma 持久化
                                            └→ 同步 BM25 索引

用户提问 → Agent（会话上下文）
             ├→ 知识库工具：查询改写 → 向量检索 + BM25 → RRF → 本地重排
             └→ 网页搜索工具：Tavily → 标题、链接、摘要
          → 流式回答 → 保存完整聊天记录与网页来源
```

Agent 根据问题和工具结果决定检索与回答流程。知识库工具返回正文及来源文件名；页面的“搜索来源”按钮展示 Tavily 返回的网页链接。

## 项目结构

```text
JLU-chatbot-main/
├── streamlit_app/
│   ├── main.py                  # 应用入口与页面导航
│   └── pages/
│       ├── app_qa.py            # 流式聊天与历史对话
│       └── app_knowledge.py     # 知识文档管理
├── src/JLU_agent/
│   ├── agents/                 # 校园问答 Agent
│   ├── config/                 # 模型、密钥读取与存储配置
│   ├── schemas/                # 提示词与回答结构
│   ├── tools/                  # 知识库及网页检索工具
│   ├── services/
│   │   ├── chat_history.py     # 完整聊天记录存储
│   │   └── RAG/                # 解析、切分、索引、检索与重排
│   ├── ui/                     # 浏览器历史归属标识
│   └── repo/
│       ├── materials/          # 校园资料
│       ├── chroma_db/          # 向量数据库
│       ├── index_db/           # BM25 索引
│       └── short-term_memory/  # Agent 检查点与聊天记录数据库
├── tests/                      # unittest 测试
├── .env.example                # 密钥配置示例
├── pyproject.toml              # 项目元数据与依赖配置
└── uv.lock                     # 依赖锁定文件
```

## 配置说明

模型名称等参数目前通过 Python 配置文件设置，`.env` 用于配置 API 密钥。

- [`agent_config.py`](src/JLU_agent/config/agent_config.py)：对话模型、查询改写模型、超时、重试、总结阈值、Tavily 参数与会话数据库路径。
- [`chroma_config.py`](src/JLU_agent/config/chroma_config.py)：嵌入模型、重排模型、文本切分参数、检索数量与知识库存储路径。

当前配置如下，使用前可根据 API 账户支持的模型调整：

| 配置项 | 当前值 |
| --- | --- |
| 对话模型 | `deepseek-v4-pro` |
| 查询改写模型 | `deepseek-flash` |
| 嵌入模型 | `qwen3.7-text-embedding-flash` |
| 本地重排模型 | `Qwen/Qwen3-Reranker-0.6B` |
| 文本切片大小 / 重叠 | `1000` / `100` 字符 |
| 向量与 BM25 单路检索数量 | 最多各 `3` 条 |
| 重排输出数量 | 最多 `3` 条，默认值定义于 `docs_reranker.py` |
| 总结触发 / 保留近期消息数 | `20` / `6` |

更换嵌入模型后，应重新建立知识库向量数据；“重建检索索引”仅从现有向量库切片重建 BM25 索引。

## 测试

在项目根目录运行已有测试：

```bash
uv run python -m unittest discover -s tests -v
```

测试覆盖文件解析、上传与替换、索引持久化、融合重排、工具调用、Agent 流式响应、聊天历史和 Streamlit 页面流程。模型与外部服务相关流程使用模拟对象，测试通过不代表真实 API 或模型下载已验证。

## 使用说明与限制

- TXT 和 Markdown 文件需要采用 UTF-8 编码。
- PDF 支持提取已有文本，暂不支持 OCR 和加密 PDF；图片资料不能通过当前上传流程入库。
- 历史记录按浏览器本地存储中的标识归属，同一浏览器、同一站点可在刷新或新标签页后恢复。清除站点存储或更换浏览器后，无法通过原标识访问历史记录。
- 当前没有登录系统或知识库管理权限控制，知识库由应用实例共享；浏览器标识用于区分聊天历史，不构成账户认证。
- 回答质量取决于资料、检索和模型输出，涉及学校政策、通知或办事要求时，请核对原始来源。

## 提交到 GitHub 前

- 仅提交 `.env.example`，检查待提交内容中没有真实 API 密钥。
- 当前仓库已跟踪 `repo/chroma_db/` 和 `repo/index_db/` 中的数据。公开前检查其中的知识文本是否适合发布；如需移除，应同时处理 Git 跟踪状态，单独新增忽略规则不会移除已跟踪文件。
- 聊天记录和 Agent 检查点位于 `repo/short-term_memory/`，对应数据库及日志文件已有忽略规则。
- 当前项目尚未提供 `LICENSE` 文件；如计划允许他人按开源协议使用，请选择合适的许可证后补充。
