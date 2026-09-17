# 批量导入.py — 批量参数页 URL 导入工具（开发者用）
# 用法：把 URL 列表写入 urls.txt（每行一个），运行：python 批量导入.py
# 自动分流：参数页→单机提取入库；Apple 支持页（多机型规格总表）→ RAG 文档语料；
#           支持页/搜索页/比价页等垃圾链接→跳过；已入库 URL→跳过；失败→重试一轮→清单。

import io
import re
import sys
import os
import time
import uuid
from datetime import date
from typing import Dict, Optional

if sys.stdout and hasattr(sys.stdout, "buffer"):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    except Exception:
        pass

from cshi import extract_single_phone, fetch_markdown
import registry
import knowledge_store
import rag

URLS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "urls.txt")

# 垃圾链接特征（确定不含参数内容的页面）
JUNK_PATTERNS = ("search?", "compare", "service/", "support.oppo", "/answer/",
                 "help/", "instructions", "questions/", "/shop/help")
# Apple 支持页：多机型规格总表，作为 RAG 文档语料价值极高
APPLE_SUPPORT = "support.apple.com"


def classify(url: str) -> str:
    ul = url.lower()
    if any(k in ul for k in JUNK_PATTERNS):
        return "junk"
    if APPLE_SUPPORT in ul:
        return "apple_doc"
    return "phone"


def save_apple_doc(url: str) -> Optional[str]:
    """Apple 支持页 → knowledge/docs/ 文档语料（带 frontmatter）"""
    md = fetch_markdown(url, max_retries=1, timeout=45)
    if not md or len(md.strip()) < 800:
        return None
    m = re.search(r'/(\d+)', url)
    doc_id = m.group(1) if m else str(abs(hash(url)) % 100000)
    fname = f"apple-support-{doc_id}.md"
    today = date.today().isoformat()
    content = (f"---\n标题: Apple 机型识别与规格页（{doc_id}）\n来源: Apple 官方支持页\n"
               f"日期: {today}\n---\n\n" + md)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "knowledge", "docs", fname), "w", encoding="utf-8") as f:
        f.write(content)
    return fname


def quality_ok(params: Dict) -> bool:
    """质量门：有效参数大类 ≥5 才入库（防垃圾页污染知识库）"""
    cats = [k for k, v in params.items() if not str(k).startswith("_")
            and isinstance(v, dict) and v]
    return len(cats) >= 5


def main():
    if not os.path.exists(URLS_FILE):
        print(f"未找到 {URLS_FILE}，请先创建并每行写一个 URL")
        return
    urls = []
    seen = set()
    with open(URLS_FILE, encoding="utf-8") as f:
        for line in f:
            u = line.strip()
            if u.startswith("http") and u not in seen:
                seen.add(u)
                urls.append(u)
    print(f"共读取 {len(urls)} 条 URL\n")

    results = {"extracted": [], "apple_docs": [], "skipped_junk": [],
               "skipped_dup": [], "low_quality": [], "failed": []}

    phones_to_extract = []
    apple_urls = []
    for u in urls:
        kind = classify(u)
        if kind == "junk":
            results["skipped_junk"].append(u)
        elif kind == "apple_doc":
            apple_urls.append(u)
        else:
            phones_to_extract.append(u)

    # Apple 支持页 → 文档语料
    for u in apple_urls:
        try:
            fname = save_apple_doc(u)
        except Exception as e:
            fname = None
            print(f"  ⚠️ Apple 文档保存失败：{str(e)[:60]}")
        if fname:
            results["apple_docs"].append(fname)
            print(f"  📄 Apple 支持页已存为语料文档：{fname}")
        else:
            results["failed"].append((u, "Apple 页面抓取失败"))

    # 单机提取（已入库 URL 跳过：注册表与知识库双查）
    for i, u in enumerate(phones_to_extract, 1):
        if registry.find_by_spec_url(u):
            results["skipped_dup"].append(u)
            print(f"[{i}/{len(phones_to_extract)}] ⏭️ 已在注册表：{u[:70]}")
            continue
        t0 = time.time()
        print(f"[{i}/{len(phones_to_extract)}] 提取中：{u[:70]} ...")
        try:
            result, name, error = extract_single_phone(u, use_cache=True)
        except Exception as e:
            result, name, error = None, None, str(e)[:80]
        if result and name:
            if quality_ok(result):
                registry.upsert(name, u, brand=None, source="confirmed")
                results["extracted"].append(name)
                print(f"  ✅ {name}（{time.time()-t0:.1f}s，"
                      f"{len([k for k in result if not str(k).startswith('_')])} 大类）")
            else:
                results["low_quality"].append(u)
                print(f"  ⚠️ 参数类别不足（疑似非参数页），不入库：{u[:70]}")
        else:
            results["failed"].append((u, error or "未知错误"))
            print(f"  ❌ 失败：{str(error)[:70]}")
        time.sleep(2)

    # 失败重试一轮
    if results["failed"]:
        print(f"\n=== 重试 {len(results['failed'])} 条失败 URL ===\n")
        still = []
        for u, err in results["failed"]:
            try:
                result, name, error = extract_single_phone(u, use_cache=True)
            except Exception as e:
                result, name, error = None, None, str(e)[:60]
            if result and name and quality_ok(result):
                registry.upsert(name, u, source="confirmed")
                results["extracted"].append(name)
                print(f"  ✅ 重试成功：{name}")
            else:
                still.append((u, err))
            time.sleep(2)
        results["failed"] = still

    print(f"\n=== 批量导入报告 ===")
    print(f"单机提取成功：{len(results['extracted'])}")
    print(f"Apple 语料文档：{len(results['apple_docs'])}")
    print(f"垃圾链接跳过：{len(results['skipped_junk'])}")
    print(f"重复跳过：{len(results['skipped_dup'])}")
    print(f"低质量未入库：{len(results['low_quality'])}")
    print(f"失败：{len(results['failed'])}")
    for u, e in results["failed"]:
        print(f"  ❌ {u[:70]} | {e[:50]}")
    print(f"\n知识库现收录：{knowledge_store.count()} 台 | "
          f"RAG 重建请到开发者面板点击，或运行 rag.rebuild_index()")


if __name__ == "__main__":
    main()
