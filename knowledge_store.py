# knowledge_store.py — 参数知识库（V2.2 数据层扩展，RAG 语料地基）
# 定位：把每台已提取机型的"完整结构化参数"沉淀成本地知识库，
# 型号查询优先从这里直取（秒回、离线可用），未命中才走注册表/搜索。
# 未来 RAG：本目录即结构化语料库，检索引擎（BM25/向量）直接挂载。

import os
import json
from datetime import date
from typing import Optional, Dict

from resolvers import normalize_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KB_DIR = os.path.join(BASE_DIR, "knowledge", "params")

def _slug(model: str) -> str:
    return normalize_model(model) or "phone"

def _path(model: str) -> str:
    return os.path.join(KB_DIR, _slug(model) + ".json")

def save_params(model: str, params: Dict, spec_url: str = None) -> str:
    """提取完成后写回知识库（含内部键 _knowledge_summary/_tags，读取时直接复用）"""
    os.makedirs(KB_DIR, exist_ok=True)
    payload = {
        "model": model,
        "saved_at": date.today().isoformat(),
        "spec_url": spec_url,
        "params": params,
    }
    path = _path(model)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return path

def load_params(model: str) -> Optional[Dict]:
    """按型号读取完整参数。解析顺序：输入归一化 slug → 注册表别名桥接
    （用户写法与官方名不同时，经注册表登记名换算主键）。未命中返回 None。"""
    tried = set()

    def _try(m):
        k = _slug(m)
        if not k or k in tried:
            return None
        tried.add(k)
        path = _path(m)
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ 知识库读取失败：{e}")
        return None

    hit = _try(model)
    if hit:
        return hit
    try:
        import registry as _reg
        entry = _reg.lookup(model)
        if entry and entry.get("model"):
            hit = _try(entry["model"])
            if hit:
                return hit
    except Exception:
        pass
    return None

def count() -> int:
    if not os.path.isdir(KB_DIR):
        return 0
    return len([f for f in os.listdir(KB_DIR) if f.endswith(".json")])
