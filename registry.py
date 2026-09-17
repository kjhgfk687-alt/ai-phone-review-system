# registry.py — 机型注册表（V2.0 数据层）
# B精修 schema：别名 + keywords + 溯源 sources + 三时间戳 + rag 状态位。
# 定位：配置型数据用 YAML 人工可维护；未来部署版多用户写入时再迁 SQLite（schema 对齐即可）。
# 数据飞轮：用户每次"确认"搜索结果都会写回这里，长尾特例降维成一条数据。

import os
import re
import yaml
from datetime import date
from typing import Dict, List, Optional

from resolvers import normalize_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REGISTRY_PATH = os.path.join(BASE_DIR, "phone_registry.yaml")
LEGACY_MAP_PATH = os.path.join(BASE_DIR, "cache", "model_url_map.json")

# ==================== 基础读写 ====================
def load() -> Dict:
    """读取注册表；文件缺失时返回空骨架"""
    if not os.path.exists(REGISTRY_PATH):
        return {"version": 2, "updated_at": None, "phones": []}
    try:
        with open(REGISTRY_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"⚠️ 注册表读取失败（使用空表）：{e}")
        return {"version": 2, "updated_at": None, "phones": []}
    data.setdefault("version", 2)
    data.setdefault("phones", [])
    return data

def save(data: Dict):
    data["updated_at"] = date.today().isoformat()
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

def all_phones() -> List[dict]:
    return load().get("phones", [])

# ==================== 查询 ====================
def _entry_keys(entry: dict) -> List[str]:
    """一条记录的所有可比对键（model + aliases 的归一化形式）"""
    keys = [entry.get("model", "")] + list(entry.get("aliases") or [])
    return [normalize_model(k) for k in keys if k]

def lookup(model_input: str) -> Optional[dict]:
    """按型号/别名精确匹配注册表（归一化比对）；命中返回记录副本"""
    key = normalize_model(model_input)
    if not key:
        return None
    for entry in all_phones():
        if key in _entry_keys(entry):
            return dict(entry)
    return None

def find_by_spec_url(url: str) -> Optional[dict]:
    """按参数页/主图页 URL 反查注册表条目（外观分析取人工补充图用）"""
    key = str(url).strip().rstrip('/')
    for p in all_phones():
        for u in (p.get("spec_url"), p.get("landing_url")):
            if u and str(u).strip().rstrip('/') == key:
                return dict(p)
    return None

def stats() -> Dict:
    """注册表统计（UI 可视化用）"""
    phones = all_phones()
    brands = {}
    for p in phones:
        brands[p.get("brand", "未知")] = brands.get(p.get("brand", "未知"), 0) + 1
    return {
        "total": len(phones),
        "brands": len(brands),
        "brand_dist": brands,
        "by_source": _count(phones, "source"),
        "rag_pending": sum(1 for p in phones if (p.get("rag") or {}).get("corpus_status") != "indexed"),
    }

def _count(phones: List[dict], field: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for p in phones:
        k = p.get(field) or "未知"
        out[k] = out.get(k, 0) + 1
    return out

# ==================== 写入（数据飞轮） ====================
def upsert(model: str, spec_url: str, brand: Optional[str] = None,
           landing_url: Optional[str] = None, aliases: Optional[List[str]] = None,
           keywords: Optional[List[str]] = None, images: Optional[List[str]] = None,
           appearance: Optional[dict] = None,
           source: str = "confirmed", note: str = None) -> dict:
    """
    新增或更新一条机型记录（按归一化 model 判重）。
    source: manual=手动补录 | confirmed=搜索确认 | template=模板直配 | pending=待补录
    images: 策展外观图（远程 URL 或 assets 相对路径）
    appearance: {analysis, model} 外观分析结果缓存（访客直读，零 API 消耗）
    """
    data = load()
    key = normalize_model(model)
    now = date.today().isoformat()

    entry = None
    for p in data["phones"]:
        if key in _entry_keys(p):
            entry = p
            break

    if entry is None:
        entry = {
            "model": model, "aliases": aliases or [], "brand": brand or "",
            "spec_url": "", "landing_url": "", "keywords": keywords or [],
            "images": [], "sources": [], "added_at": now, "updated_at": now,
            "verified_at": None, "rag": {"corpus_status": "pending"},
            "source": source,
        }
        data["phones"].append(entry)

    if spec_url:
        entry["spec_url"] = spec_url
        entry["verified_at"] = now
    if landing_url:
        entry["landing_url"] = landing_url
    if brand:
        entry["brand"] = brand
    if aliases:
        merged = list(entry.get("aliases") or [])
        for a in aliases:
            if a not in merged and normalize_model(a) != normalize_model(entry["model"]):
                merged.append(a)
        entry["aliases"] = merged
    if keywords:
        merged = list(entry.get("keywords") or [])
        for k in keywords:
            if k not in merged:
                merged.append(k)
        entry["keywords"] = merged
    if images:
        merged = list(entry.get("images") or [])
        for im in images:
            if im not in merged:
                merged.append(im)
        entry["images"] = merged
    if appearance and appearance.get("analysis"):
        entry["appearance"] = {
            "analysis": appearance["analysis"],
            "generated_at": now,
        }
    if note:
        entry["note"] = note
    entry["source"] = source if spec_url else entry.get("source", source)
    entry["updated_at"] = now

    save(data)
    return dict(entry)

def mark_pending(model: str, brand: Optional[str] = None, note: str = None):
    """登记"待补录"条目（机型暂未定位到参数页）"""
    return upsert(model, spec_url="", brand=brand, source="pending", note=note)

# ==================== 旧数据迁移 ====================
def migrate_legacy_map() -> int:
    """把 V1.4 的 cache/model_url_map.json（型号→URL 简单映射）迁移进注册表"""
    if not os.path.exists(LEGACY_MAP_PATH):
        return 0
    try:
        import json
        with open(LEGACY_MAP_PATH, encoding="utf-8") as f:
            legacy = json.load(f)
    except Exception:
        return 0
    moved = 0
    for model, url in (legacy or {}).items():
        if not url:
            continue
        if lookup(model) is None:
            upsert(model, url, source="confirmed")
            moved += 1
    return moved
