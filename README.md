# 📱 AI手机参数提取与评测系统

面向手机选购场景的 AI 辅助工具：输入手机官网参数页 URL，自动提取结构化参数，结合本地知识库生成大白话解读与短板标签，并支持单机评测和多机横向对比。

## 核心功能

- **自动识别机型**：从网页内容中识别品牌与完整型号，无需手动输入
- **结构化参数提取**：网页抓取 → 智能分段 → 分段并行提取 → 深度合并（冲突检测），输出统一 JSON
- **专业知识解读**：本地 YAML 知识库覆盖芯片（三级匹配）、屏幕、电池、相机、品牌等维度
- **参数标签**：基于明确证据自动检测短板与亮点（如不支持 n79、分辨率较低、三频/四频北斗），不做"未提及即不支持"的推断
- **单机评测 / 多机对比**：调用大模型生成 300-500 字客观评测或 Markdown 对比表格与推荐
- **缓存与重试**：网页原文与提取结果按 URL 缓存，API 调用失败自动重试退避

## 技术栈

| 层 | 方案 |
| --- | --- |
| 前端 | Streamlit |
| 后端 | Python（`cshi.py` 核心流程 + `knowledge_base.py` 知识库查询） |
| 大模型 | DeepSeek API（参数提取 / 评测生成） |
| 网页抓取 | Jina Reader（本地 Docker 部署） |
| 知识库 | `phone_knowledge.yaml`（静态 YAML，人工维护） |

## 快速开始

### 1. 安装依赖

```bash
git clone https://github.com/kjhgfk687-alt/ai-phone-review-system.git
cd ai-phone-review-system
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
copy .env.example .env         # Windows
# cp .env.example .env         # macOS / Linux
```

编辑 `.env`，填入你的 `DEEPSEEK_API_KEY`（在 [DeepSeek 开放平台](https://platform.deepseek.com/) 申请）。

### 3. 启动本地 Jina Reader

系统通过本地 Jina Reader 抓取网页（避免外部服务的爬虫限制）：

```bash
docker run -d -p 3001:3000 jinaai/reader
```

> 端口映射如与示例不同，请在 `.env` 中修改 `JINA_READER_URL`。

### 4. 运行

```bash
streamlit run streamlit_app.py
```

浏览器打开提示的地址（默认 `http://localhost:8501`），粘贴手机官网参数页 URL 即可。

也可以使用命令行交互模式：

```bash
python cshi.py
```

## 使用说明

1. 在输入框粘贴一个或多个手机参数页 URL（每行一个），点击 **开始提取**
2. 查看每部手机的参数标签、专业知识解读、完整参数（带冲突提示 ⚠️）
3. 点击 **生成评测** 得到单机评测，或选择多部手机生成 **对比评测**
4. 所有结果均可一键下载（JSON / Markdown）
5. 同一 URL 再次提取会命中缓存；如需重新抓取，可在侧边栏清除缓存

## 项目结构

```
├── streamlit_app.py        # Web 界面（Streamlit）
├── cshi.py                 # 核心流程：抓取/分段/提取/合并/标签/评测
├── knowledge_base.py       # 知识库查询与匹配逻辑
├── phone_knowledge.yaml    # 静态知识库（芯片/屏幕/电池/相机/品牌等）
├── output/                 # 提取结果 JSON 输出（gitignore）
├── cache/                  # 网页与结果缓存（gitignore）
├── .env.example            # 配置模板
└── requirements.txt
```

## 已知限制与规划

- 知识库为静态 YAML，存在更新滞后风险；计划引入定期抓取或 RAG 动态扩充
- 评测基于官方参数生成，无法覆盖实际手感、发热、拍照样张等体验维度
- 规划中：云端部署、个性化推荐、多模态评测

## 许可与引用

产品背景调研与设计详见《AI手机参数提取与评测系统产品文档》。本项目为个人作品，仅供学习交流。
