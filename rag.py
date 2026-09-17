# rag.py — 便于维护的本地 RAG 系统（V3.0）
# 设计原则：
#   ①零第三方依赖（venv 的 pip 已损坏）：检索用字符二元组 BM25，中文友好
#   ②维护方式：往 knowledge/docs/ 丢 Markdown 即可，一键重建索引
#   ③语料四来源：docs 手写文档 / knowledge/params 提取参数 / phone_registry 注册表 / phone_knowledge 策展事实
#   ④检索结果带来源元数据，生成时要求模型标注引用——把"客观"从口号变成机制

import os
import re
import json
import math
import yaml
from datetime import date
from typing import Dict, List, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "knowledge", "docs")
PARAMS_DIR = os.path.join(BASE_DIR, "knowledge", "params")
INDEX_PATH = os.path.join(BASE_DIR, "knowledge", "rag_index.json")

K1, B = 1.5, 0.75          # BM25 参数
CHUNK_MAX = 420             # 语料块目标长度（字符）

# ==================== 分词：拉丁词 + 中文字符二元组/一元 ====================
def tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    tokens = re.findall(r'[a-z0-9]+', text)               # 英文/数字词
    for seg in re.findall(r'[\u4e00-\u9fff]+', text):     # 中文段
        tokens.append(seg)                                # 整段（短查询直配）
        if len(seg) == 1:
            tokens.append(seg)
        else:
            tokens += [seg[i:i + 2] for i in range(len(seg) - 1)]   # 二元组
            tokens += list(seg)                            # 一元组（保证"快""屏"等单字可召回）
    return tokens

# ==================== 语料分块 ====================
def _chunks_from_text(text: str, max_len: int = CHUNK_MAX) -> List[str]:
    """按空行分段聚合成长度≤max_len的语料块；超长段落硬切"""
    blocks, cur = [], ""
    for para in re.split(r'\n\s*\n', text or ""):
        para = para.strip()
        if not para:
            continue
        if len(cur) + len(para) + 1 <= max_len:
            cur = (cur + "\n" + para).strip()
        else:
            if cur:
                blocks.append(cur)
            while len(para) > max_len:
                blocks.append(para[:max_len])
                para = para[max_len:]
            cur = para
    if cur:
        blocks.append(cur)
    return blocks

# ==================== 语料采集（四来源） ====================
def _doc_from_markdown(path: str) -> Optional[Dict]:
    """解析 Markdown 文档：支持可选 frontmatter（--- 包裹的 标题/来源/日期/机型）"""
    try:
        with open(path, encoding="utf-8") as f:
            raw = f.read()
    except Exception:
        return None
    meta: Dict = {}
    if raw.startswith("---"):
        m = re.match(r'---\s*\n(.*?)\n---\s*\n?', raw, re.S)
        if m:
            try:
                loaded = yaml.safe_load(m.group(1))
                if isinstance(loaded, dict):
                    meta = loaded
                raw = raw[m.end():]
            except Exception:
                pass
    if not raw.strip():
        return None
    title_m = re.search(r'^#\s+(.+)$', raw, re.M)
    return {
        "title": str(meta.get("标题") or meta.get("title") or (title_m.group(1) if title_m else os.path.basename(path))),
        "source": str(meta.get("来源") or meta.get("source") or "本地文档"),
        "date": str(meta.get("日期") or meta.get("date") or date.today().isoformat()),
        "models": meta.get("机型") or [],
        "text": raw.strip(),
        "file": os.path.basename(path),
    }

def _docs_from_params() -> List[Dict]:
    """提取参数库（knowledge/params/*.json）→ 每台机型一份结构化文档"""
    out = []
    if not os.path.isdir(PARAMS_DIR):
        return out
    for f in os.listdir(PARAMS_DIR):
        if not f.endswith(".json"):
            continue
        try:
            with open(os.path.join(PARAMS_DIR, f), encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception:
            continue
        model = payload.get("model") or f
        p = payload.get("params") or {}
        lines = [f"{model} 关键参数（来自官方参数页提取）："]
        for cat in ("basic_info", "processor", "memory_storage", "display",
                    "battery_charging", "camera", "connectivity", "sensors",
                    "audio", "physical"):
            cat_data = p.get(cat)
            if not isinstance(cat_data, dict):
                continue
            for k, v in cat_data.items():
                if str(k).startswith("_") or v in ("", None, "未提及"):
                    continue
                lines.append(f"{k}: {v}")
        out.append({
            "title": f"{model} 完整参数",
            "source": "参数知识库（官网提取）",
            "date": payload.get("saved_at") or "",
            "models": [model],
            "text": "\n".join(lines),
            "file": f,
        })
    return out

def _docs_from_knowledge_base() -> List[Dict]:
    """策展知识库（phone_knowledge.yaml）的芯片与品牌条目 → 策展事实语料"""
    out = []
    try:
        with open(os.path.join(BASE_DIR, "phone_knowledge.yaml"), encoding="utf-8") as f:
            kb = yaml.safe_load(f) or {}
    except Exception:
        return out
    for name, info in (kb.get("chipsets") or {}).items():
        if not isinstance(info, dict):
            continue
        parts = [f"芯片 {name}：定位 {info.get('tier','未知')}，性能评分 {info.get('performance_score','?')}/100"]
        if info.get("description"): parts.append(str(info["description"]))
        if info.get("strengths"): parts.append("优势：" + "；".join(info["strengths"]))
        if info.get("weaknesses"): parts.append("注意：" + "；".join(info["weaknesses"]))
        if info.get("typical_price_range"): parts.append(f"常见价格区间：{info['typical_price_range']}")
        out.append({"title": f"芯片 {name}", "source": "策展知识库（人工维护）",
                    "date": f"{info.get('year','')}" if info.get("year") else "",
                    "models": [], "text": "\n".join(parts), "file": "phone_knowledge.yaml"})
    for name, info in (kb.get("brands") or {}).items():
        if not isinstance(info, dict):
            continue
        parts = [f"品牌 {name}：{info.get('brand_positioning','')}"]
        if info.get("target_users"): parts.append(f"目标用户：{info['target_users']}")
        if info.get("key_technologies"): parts.append("核心技术：" + "、".join(info["key_technologies"]))
        out.append({"title": f"品牌 {name}", "source": "策展知识库（人工维护）",
                    "date": "", "models": [], "text": "\n".join(parts), "file": "phone_knowledge.yaml"})
    return out

def build_corpus() -> List[Dict]:
    """采集全部语料文档（docs 手写 + 参数库 + 策展知识库）"""
    docs: List[Dict] = []
    if os.path.isdir(DOCS_DIR):
        for f in sorted(os.listdir(DOCS_DIR)):
            if f.lower().startswith("readme"):
                continue
            if f.lower().endswith((".md", ".markdown", ".txt")):
                d = _doc_from_markdown(os.path.join(DOCS_DIR, f))
                if d:
                    docs.append(d)
    docs += _docs_from_params()
    docs += _docs_from_knowledge_base()
    for i, d in enumerate(docs):
        d["doc_id"] = i
    return docs

# ==================== 索引（BM25） ====================
def _build_index() -> Dict:
    docs = build_corpus()
    chunks = []
    for d in docs:
        for j, block in enumerate(_chunks_from_text(d["text"])):
            chunks.append({
                "doc_id": d["doc_id"], "title": d["title"], "source": d["source"],
                "date": d.get("date", ""), "file": d.get("file", ""),
                "chunk_idx": j, "text": block,
            })
    doc_tokens = [tokenize(c["text"]) for c in chunks]
    doc_len = [len(t) for t in doc_tokens]
    avg_len = (sum(doc_len) / len(doc_len)) if doc_len else 0.0
    df: Dict[str, int] = {}
    for toks in doc_tokens:
        for t in set(toks):
            df[t] = df.get(t, 0) + 1
    index = {
        "built_at": date.today().isoformat(),
        "corpus_hash": corpus_hash(docs),
        "chunks": chunks,
        "doc_tokens": doc_tokens,
        "doc_len": doc_len,
        "avg_len": avg_len,
        "df": df,
        "n": len(chunks),
    }
    try:
        os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
        with open(INDEX_PATH, "w", encoding="utf-8") as f:
            json.dump({"built_at": index["built_at"], "corpus_hash": index["corpus_hash"],
                       "chunks": chunks, "doc_len": doc_len, "avg_len": avg_len,
                       "df": df, "n": index["n"]}, f, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️ 索引持久化失败（内存索引仍可用）：{e}")
    return index

def corpus_hash(docs: List[Dict]) -> str:
    import hashlib
    h = hashlib.md5()
    for d in docs:
        h.update((d.get("file", "") + d.get("title", "") + d.get("text", "")).encode("utf-8"))
    return h.hexdigest()

_index_cache: Optional[Dict] = None

def get_index(force_rebuild: bool = False) -> Dict:
    """获取索引：内存缓存 → 磁盘（语料未变时直用）→ 重建"""
    global _index_cache
    if _index_cache and not force_rebuild:
        return _index_cache
    docs = build_corpus()
    cur_hash = corpus_hash(docs)
    if not force_rebuild and os.path.exists(INDEX_PATH):
        try:
            with open(INDEX_PATH, encoding="utf-8") as f:
                disk = json.load(f)
            if disk.get("corpus_hash") == cur_hash:
                disk["doc_tokens"] = [tokenize(c["text"]) for c in disk["chunks"]]
                disk["doc_len"] = [len(t) for t in disk["doc_tokens"]]
                disk["avg_len"] = (sum(disk["doc_len"]) / len(disk["doc_len"])) if disk["doc_len"] else 0.0
                disk["df"] = {}
                for toks in disk["doc_tokens"]:
                    for t in set(toks):
                        disk["df"][t] = disk["df"].get(t, 0) + 1
                disk["n"] = len(disk["chunks"])
                _index_cache = disk
                return _index_cache
        except Exception:
            pass
    _index_cache = _build_index()
    return _index_cache

def rebuild_index() -> Dict:
    """强制重建（开发者按钮/语料变更时）"""
    global _index_cache
    _index_cache = _build_index()
    return _index_cache

# ==================== 检索 ====================
def search(query: str, top_k: int = 4, boost_model: str = None) -> List[Dict]:
    """BM25 检索，返回 [{text,title,source,date,score,file}]；
    boost_model：与该机型相关的语料块获得加分（同机型不同写法仍可命中）"""
    idx = get_index()
    if not idx.get("n"):
        return []
    q_tokens = tokenize(query)
    boost_tokens = set(tokenize(boost_model)) if boost_model else set()
    k1, b, avg, n = K1, B, idx["avg_len"] or 1.0, idx["n"]
    df = idx["df"]
    scored = []
    for toks, chunk, dlen in zip(idx["doc_tokens"], idx["chunks"], idx["doc_len"]):
        tf: Dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        score = 0.0
        for t in q_tokens:
            f = tf.get(t)
            if not f:
                continue
            idf = math.log(1 + (n - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5))
            score += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * dlen / avg))
        if not score:
            continue
        text_l = chunk["text"].lower()
        for bt in boost_tokens:
            if len(bt) >= 2 and bt in text_l:
                score += 1.5
                break
        scored.append({**chunk, "score": round(score, 3)})
    scored.sort(key=lambda x: -x["score"])
    seen_titles = set()
    out = []
    for c in scored:
        key = (c["title"], c["chunk_idx"])
        if key in seen_titles:
            continue
        seen_titles.add(key)
        out.append(c)
        if len(out) >= top_k:
            break
    return out

# ==================== 生成集成 ====================
def build_context(query: str, model: str = None, top_k: int = 4) -> str:
    """为生成 prompt 构建参考资料块（含来源标注）；无可检索内容返回空串"""
    try:
        hits = search(query, top_k=top_k, boost_model=model)
    except Exception as e:
        print(f"⚠️ RAG 检索失败（跳过资料注入）：{str(e)[:60]}")
        return ""
    if not hits:
        return ""
    blocks = []
    for i, h in enumerate(hits, 1):
        src = h.get("source") or "知识库"
        dt = f"，{h['date']}" if h.get("date") else ""
        blocks.append(f"[资料{i}｜来源：{src}{dt}｜主题：{h['title']}]\n{h['text']}")
    return "\n\n".join(blocks)

# ==================== 统计（开发者面板用） ====================
def rag_stats() -> Dict:
    docs = build_corpus()
    idx = get_index()
    by_source: Dict[str, int] = {}
    for d in docs:
        by_source[d.get("source", "未知")] = by_source.get(d.get("source", "未知"), 0) + 1
    return {"docs": len(docs), "chunks": idx.get("n", 0),
            "built_at": idx.get("built_at", ""), "by_source": by_source}
