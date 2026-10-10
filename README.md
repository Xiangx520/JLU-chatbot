# 🎓 吉大校园助手 · JLU Chatbot

面向吉林大学校园信息问答的 AI Agent 项目，使用 **Streamlit** 构建交互界面，结合 **LangChain / LangGraph**、本地知识库和网页搜索，支持自然语言问答与多轮对话。

项目处于开发阶段，当前已提供聊天问答、知识库管理和历史对话功能。

## 功能

- **流式聊天**：逐步展示模型回答，支持连续追问。
- **知识库问答**：文档正文、元数据、向量和内置 BM25 保存在同一个 Milvus Collection，服务端用 RRF 融合候选，按融合顺序直接返回给 Agent。
- **网页搜索**：Agent 可按需调用 Tavily 补充知识库不足的信息，页面展示本轮搜索来源链接。
- **知识库管理**：支持上传 TXT、Markdown、PDF、DOCX，预览解析正文、按文件名搜索、查看入库片段、替换同名文档和删除文档。
- **内容去重与重试恢复**：相同正文避免重复入库；稳定切片 ID 支持补写缺失片段，替换时确认新切片完整入库后才删除旧片段。
- **历史对话**：通过 SQLite 保存聊天记录及 Agent 上下文，支持新建、恢复和删除对话。
- **上下文总结**：消息达到配置阈值时自动总结较早的上下文，完整聊天记录单独保存。

## 技术栈

| 模块 | 实现 |
| --- | --- |
| Web 界面 | Streamlit |
| Agent 与会话状态 | LangChain、LangGraph、SQLite Checkpointer |
| 对话与查询改写 | DeepSeek |
| 文本嵌入 | DashScope |
| 知识库存储与向量检索 | Milvus 3.0.2、langchain-milvus、pymilvus |
| 关键词检索 | Milvus 内置 BM25、chinese 分析器 |
| 检索融合 | Milvus RRF |
| 保留的本地重排代码（应用不调用） | Sentence Transformers CrossEncoder，可选 `rerank` 依赖 |
| 联网搜索 | Tavily |
| 文档解析 | pypdf、python-docx |
| 依赖管理 | uv、pyproject.toml、uv.lock |

## 快速开始

### 1. 准备环境

- Python **3.13 或更高版本**，建议首次运行使用 Python 3.13。
- 已安装 `uv`，可通过 `uv --version` 检查。
- 已安装 Docker 与 Docker Compose V2。Windows 使用 Docker Desktop，开启 WSL 2 后端和 Linux containers；先启动 Docker Desktop，再运行 `docker version` 和 `docker compose version` 检查。
- 准备 DeepSeek、DashScope 和 Tavily 的 API 密钥。
- 安装依赖和调用外部 API 时需要网络连接；应用不下载或加载本地重排模型。

在克隆后的项目根目录执行：

```bash
uv sync --locked
```

默认安装不包含 PyTorch 和 Sentence Transformers，检索结果由 Milvus RRF 融合后直接返回，无需本地重排模型或 GPU。

`docs_reranker.py` 中的独立重排代码及模型配置仍然保留。需要单独调用它时，先安装可选依赖：

```bash
uv sync --locked --extra rerank
```

安装 `rerank` 扩展不会让应用启用重排；应用没有重排开关。保留的函数在被手动调用时才加载模型，并根据 `torch.cuda.is_available()` 选择 CUDA 或 CPU；首次调用需要下载模型。该扩展在 Windows 上继续使用 PyTorch CUDA 13.0 软件源。

中文分词和 BM25 索引由 Milvus 服务维护，应用无需下载独立分词模型或维护第二份索引。

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
MILVUS_URI=http://localhost:19530
MILVUS_TOKEN=
MILVUS_DB_NAME=default
MILVUS_COLLECTION_NAME=jlu_knowledge
```

| 变量 | 用途 |
| --- | --- |
| `DEEPSEEK_API_KEY` | 对话生成、上下文总结与检索问题改写 |
| `TAVILY_API_KEY` | 网页搜索；当前在 Agent 初始化时校验 |
| `DASHSCOPE_API_KEY` | 文档嵌入和向量检索；当前在知识库服务初始化时校验 |
| `MILVUS_URI` | Milvus 服务地址，默认 `http://localhost:19530`，支持远程 http/https 服务 |
| `MILVUS_TOKEN` | 服务认证 Token；本地默认部署留空 |
| `MILVUS_DB_NAME` | 数据库名，默认 `default`；其他数据库需事先创建 |
| `MILVUS_COLLECTION_NAME` | Collection 名，默认 `jlu_knowledge`，首次上传时自动创建 |

系统环境变量优先于项目根目录的 `.env`。`.env` 已被 `.gitignore` 排除，请将真实密钥保留在本地。

### 3. 启动 Milvus 与应用

在项目根目录启动知识库服务：

```bash
docker compose up -d
docker compose ps -a
```

等待 `standalone`、`etcd` 和 `minio` 显示 healthy。`milvus-volume-init` 是一次性卷权限初始化服务，正常状态为退出码 0。部署基于 [Milvus v3.0.2 官方 Compose 模板](https://github.com/milvus-io/milvus/releases/download/v3.0.2/milvus-standalone-docker-compose.yaml)，使用 Docker 命名卷持久化数据；Milvus 和 WebUI 端口仅绑定本机。可通过 [Milvus WebUI](http://localhost:9091/webui/) 查看服务状态。

```bash
docker compose logs --tail=100 standalone
docker compose down
```

`down` 停止服务并保留数据卷；`docker compose down -v` 会删除全部知识库数据。日常服务重启使用 `docker compose restart standalone`。Milvus 服务无法连接时，页面会报告初始化或操作失败，不能视作空知识库。

服务就绪后启动应用：

```bash
uv run streamlit run streamlit_app/main.py
```

根据终端显示的地址打开应用，默认通常为 `http://localhost:8501`。

### 4. 上传资料并开始问答

1. 在侧边栏进入 **知识库管理**，上传校园资料。
2. 检查解析后的正文，点击 **上传到知识库**。
3. 进入 **聊天问答**，输入问题，例如“吉林大学有哪些校区？”或“根据上传的资料介绍学校历史”。
4. 在侧边栏新建、恢复或删除历史对话。

`src/JLU_agent/repo/materials/` 中保存了校园资料，可选择其中的文本文件上传。该目录的文件不会仅因放入目录就自动入库；实际知识库以 Milvus 中保存的切片为准。本次切换已移除工作区中的旧 Chroma、BM25 和 MD5 数据，请在页面重新上传资料；原始资料和聊天历史保留。

## 工作流程

```text
上传文档 → 解析正文 → 文本切分 → DashScope 嵌入 → Milvus 单 Collection
                                            └→ 服务端生成 BM25 稀疏向量

用户提问 → Agent（会话上下文）
             ├→ 知识库工具：查询改写 → Milvus 向量 + BM25 → Milvus RRF → 返回 Agent
             └→ 网页搜索工具：Tavily → 标题、链接、摘要
          → 流式回答 → 保存完整聊天记录与网页来源
```

Agent 根据问题和工具结果决定检索与回答流程。知识库工具按 RRF 融合顺序返回全部候选（当前最多 6 条），包含正文、来源文件名和 RRF 融合得分；该得分不是本地模型重排分数。页面的“搜索来源”按钮展示 Tavily 返回的网页链接。

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
│   │   └── RAG/                # 解析、切分、Milvus 存储、检索及保留的重排代码
│   ├── ui/                     # 浏览器历史归属标识
│   └── repo/
│       ├── materials/          # 校园资料
│       └── short-term_memory/  # Agent 检查点与聊天记录数据库
├── docker-compose.yaml         # Milvus、etcd、MinIO 与持久化卷
├── .env.example                # 密钥配置示例
├── pyproject.toml              # 项目元数据与依赖配置
└── uv.lock                     # 依赖锁定文件
```

## 配置说明

模型名称等参数通过 Python 配置文件设置，`.env` 用于配置 API 密钥和 Milvus 连接。

- [`agent_config.py`](src/JLU_agent/config/agent_config.py)：对话模型、查询改写模型、超时、重试、总结阈值、Tavily 参数与会话数据库路径。
- [`rag_config.py`](src/JLU_agent/config/rag_config.py)：嵌入模型、保留的重排模型配置、文本切分参数、检索数量与 Milvus 连接读取。

当前配置如下，使用前可根据 API 账户支持的模型调整：

| 配置项 | 当前值 |
| --- | --- |
| 对话模型 | `deepseek-v4-pro` |
| 查询改写模型 | `deepseek-flash` |
| 嵌入模型 | `qwen3.7-text-embedding-flash` |
| 保留的本地重排模型（应用不调用） | `Qwen/Qwen3-Reranker-0.6B` |
| 文本切片大小 / 重叠 | `1000` / `100` 字符 |
| 向量与 BM25 单路检索数量 | 最多各 `3` 条 |
| Milvus RRF 返回 Agent 的数量 / 平滑参数 | 最多 `6` 条 / `60` |
| Milvus 索引与一致性 | dense COSINE/AUTOINDEX、sparse BM25/SPARSE_INVERTED_INDEX、Strong |
| 独立重排函数的默认输出数量（应用不使用） | 最多 `3` 条，默认值定义于 `docs_reranker.py` |
| 总结触发 / 保留近期消息数 | `20` / `6` |

更换嵌入模型或向量维度后，配置新的 Collection 名并重新上传全部资料。初始化保留已有 Collection；应用不提供旧双库兼容或“重建检索索引”操作。文件替换支持失败重试，但多步骤替换不是文档级事务；当前写锁仅协调同一个应用进程。

## 使用说明与限制

- TXT 和 Markdown 文件需要采用 UTF-8 编码。
- PDF 支持提取已有文本，暂不支持 OCR 和加密 PDF；图片资料不能通过当前上传流程入库。
- 历史记录按浏览器本地存储中的标识归属，同一浏览器、同一站点可在刷新或新标签页后恢复。清除站点存储或更换浏览器后，无法通过原标识访问历史记录。
- 当前没有登录系统或知识库管理权限控制，知识库由应用实例共享；浏览器标识用于区分聊天历史，不构成账户认证。
- 回答质量取决于资料、检索和模型输出，涉及学校政策、通知或办事要求时，请核对原始来源。

## 提交到 GitHub 前

- 仅提交 `.env.example`，检查待提交内容中没有真实 API 密钥。测试、RAG 评估代码和报告保留在本地，不随发布提交。
- Milvus 数据由 Docker 命名卷保存，不提交到 Git；旧版知识库数据已从当前工作区移除，本次清理不重写历史提交。
- 聊天记录和 Agent 检查点位于 `repo/short-term_memory/`，对应数据库及日志文件已有忽略规则。
- 当前项目尚未提供 `LICENSE` 文件；如计划允许他人按开源协议使用，请选择合适的许可证后补充。
