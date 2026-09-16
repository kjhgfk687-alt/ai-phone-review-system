# model_finder.py — 型号→自动定位官网参数页（V2.0 三层架构·执行层）
# 查找管线（数据飞轮）：
#   ①注册表命中（数据层，秒回）→ ②适配器模板（策略层）→ ③必应站内搜索+验证闭环（兜底）
#   → 用户确认 → 写回注册表。V1.4e 的全部健壮性修复保留（数字身份/中英归一/阈值/并行早退）。

import re
import json
import difflib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import registry
from resolvers import (
    normalize_model, detect_brand, build_spec_urls, get_adapter,
    all_domains, landing_from_spec, spec_markers, BRAND_ADAPTERS,
)
from cshi import fetch_markdown, extract_phone_name, load_cache, call_llm, client, JINA_READER_URL

CONFIDENCE_THRESHOLD = 0.5  # 低于此值的候选视为不匹配（壳页/无关页面），直接丢弃

# ==================== 匹配打分（含数字身份校验） ====================
def match_score(model_input: str, phone_name: str) -> float:
    """
    输入型号与页面识别出的机型名的匹配度 0~1。
    两道硬规则（实测教训）：
    - 正向包含满分；反向包含按长度比缩水，防"华为"短词蹭分；
    - 数字身份校验：两侧都含数字但数字序列不一致（剥掉 5G/4G 噪声后）强制压分
      （X30000≠X300、K70≠K700——编辑相似度对型号数字完全失明）。
    """
    a, b = normalize_model(model_input), normalize_model(phone_name or "")
    if not a or not b:
        return 0.0
    if a in b:
        score = 1.0
    elif b in a:
        score = round(0.85 * min(1.0, len(b) / max(len(a), 1)), 3)
    else:
        score = round(difflib.SequenceMatcher(None, a, b).ratio(), 3)

    da = re.sub(r'\D', '', re.sub(r'[45]g', '', a))
    db = re.sub(r'\D', '', re.sub(r'[45]g', '', b))
    if da and db and da != db:
        return min(score, 0.35)
    return score

def _page_h1(md: str) -> str:
    """取 Markdown 首个一级标题（产品页 H1 通常是机型名）"""
    m = re.search(r'^#\s+(.+)$', md, re.M)
    return m.group(1).strip() if m else ""

# ==================== 品牌识别（关键词表 + LLM 兜底） ====================
_LLM_BRAND_ALIAS = {
    "oppo": "OPPO", "vivo": "vivo", "iqoo": "iQOO", "xiaomi": "小米", "redmi": "Redmi",
    "honor": "荣耀", "huawei": "华为", "oneplus": "一加", "realme": "真我",
    "samsung": "三星", "apple": "苹果", "meizu": "魅族", "motorola": "摩托罗拉",
}

def detect_brand_full(model_input: str) -> Optional[str]:
    """品牌识别：适配器关键词表 → LLM 分类兜底（短超时不重试，快速降级）。"""
    name = detect_brand(model_input)
    if name:
        return name
    prompt = (f'判断手机型号「{model_input}」属于哪个品牌。只返回JSON：'
              '{{"brand": "OPPO/vivo/iQOO/Xiaomi/Redmi/Honor/Huawei/OnePlus/realme/'
              'Samsung/Apple/Meizu/Motorola 之一", "known": true/false}}。'
              '不认识该型号时 known 填 false、brand 填 null。')
    try:
        content = call_llm(client, prompt, temperature=0, timeout=12,
                           max_tokens=100, tag="品牌识别",
                           disable_thinking=True, max_retries=1)
        data = json.loads(content)
        brand_key = str(data.get("brand") or "").strip().lower()
        if data.get("known") and brand_key in _LLM_BRAND_ALIAS:
            return _LLM_BRAND_ALIAS[brand_key]
    except Exception as e:
        print(f"⚠️ LLM品牌识别失败（快速降级）：{str(e)[:60]}")
    return None

# ==================== 必应站内搜索（兜底层） ====================
def _bing_site_candidates(model_input: str, brand: Optional[str], progress) -> List[str]:
    """必应站内搜索：有品牌用 site:限定（去 www 前缀）；无品牌用多品牌域 OR 兜底。
    实测必应全文匹配需要品牌词，查询中拼接 search_brand。"""
    import requests
    ad = get_adapter(brand)
    if brand:
        site = ad["domains"][0].removeprefix("www.") if ad.get("domains") else ""
        query = f"site:{site} {model_input} 参数" if site else f"{model_input} 参数"
    else:
        domains = [d.removeprefix("www.") for d in all_domains()][:8]
        query = f"{model_input} 参数 ({' OR '.join('site:' + d for d in domains)})"
    url = f"{JINA_READER_URL.rstrip('/')}/https://cn.bing.com/search?q={requests.utils.quote(query)}"
    progress(0.30, f"🔎 必应搜索：{query[:70]}")
    try:
        res = requests.get(url, headers={"X-Respond-With": "markdown"}, timeout=90)
        res.encoding = 'utf-8'
        text = res.text
    except Exception as e:
        progress(0.35, f"⚠️ 搜索请求失败：{str(e)[:60]}")
        return []

    links = list(dict.fromkeys(re.findall(r'\((https?://[^)\s]+)\)', text)))
    allowed = all_domains()
    out = []
    for u in links:
        host = u.split('/')[2] if u.count('/') >= 2 else ''
        if not any(d in host for d in allowed):
            continue
        if brand and not any(d in host for d in get_adapter(brand).get("domains", [])):
            continue
        out.append(u)
    progress(0.45, f"🔎 搜索得到 {len(out)} 条候选链接")
    return out[:6]

# ==================== 验证闭环 ====================
def _validate_one(url: str, model_input: str, brand: Optional[str] = None) -> Optional[dict]:
    """验证单个候选：抓页面 → 识别机型 → 打分。
    识别三级递进：H1（零成本）→ 页面文本包含（零成本，需先剥 URL 防自链接污染）→ LLM 兜底。"""
    cached = load_cache(url, "page")
    md = cached["text"] if (cached and cached.get("text")) else fetch_markdown(url, max_retries=1, timeout=25)
    if not md or len(md.strip()) < 500:
        return None

    name, score = "", 0.0
    h1 = _page_h1(md)
    if h1:
        name, score = h1, match_score(model_input, h1)
    if score < 0.45:
        a = normalize_model(model_input)
        visible = re.sub(r'https?://\S+', ' ', md[:10000])   # 剥 URL 防壳页自链接污染
        if a and a in normalize_model(visible):
            score, name = max(score, 0.9), name or model_input
    if score < 0.45:
        try:
            # url_fallback=False：模板 URL 由输入拼出，URL 推断名会循环自证，
            # 识别失败宁可放弃该候选（vivox30000 壳页曾借此混入）
            llm_name = extract_phone_name(md[:2500], url, url_fallback=False)
            if not llm_name:
                return None
            s2 = match_score(model_input, llm_name)
            if s2 > score:
                name, score = llm_name, s2
        except Exception:
            pass

    # 参数页特征降权：页面无"参数"字样且 URL 无 specs 特征（可能是重定向落地页）
    if "参数" not in md and not any(h in url.lower() for h in spec_markers(brand)):
        score = round(score * 0.85, 3)
    if not name:
        return None
    return {"url": url, "how": "搜索", "matched_name": name,
            "confidence": round(min(score, 1.0), 3)}

def _slug_specs_score(model_input: str, url: str, brand: Optional[str]) -> float:
    """specs 类 URL 的 slug 与输入的匹配度（强匹配免抓取直配）"""
    if not any(h in url.lower() for h in spec_markers(brand)):
        return 0.0
    segs = [s for s in url.split('/') if s and s.lower() not in ("specs", "spec")]
    slug = segs[-1] if segs else ""
    return match_score(model_input, slug.replace('-', ' '))

def _validate_candidates(urls: List[str], model_input: str, brand: Optional[str], progress) -> List[dict]:
    """并行验证 + 满分早退 + 死链防护 + 超时收割（V1.4b/e 全部机制保留）"""
    direct, to_fetch, seen = [], [], set()
    for u in urls[:6]:
        if u in seen:
            continue
        seen.add(u)
        s = _slug_specs_score(model_input, u, brand)
        if s >= 0.95:
            direct.append({"url": u, "how": "模板直配", "matched_name": "(URL 结构匹配)",
                           "confidence": round(s * 0.95, 3)})
        else:
            to_fetch.append(u)
    if direct:
        progress(0.9, f"⚡ {len(direct)} 个候选 slug 直配，免抓取入选")

    # 缓存页优先：瞬时完成，容易触发早退
    to_fetch.sort(key=lambda u: 0 if (load_cache(u, "page") or {}).get("text") else 1)

    fetched = []
    if to_fetch:
        workers = min(3, len(to_fetch))
        progress(0.55, f"📄 并行验证 {len(to_fetch)} 个候选（{workers} 并发，满分即早退）...")
        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="validate")
        futs = {pool.submit(_validate_one, u, model_input, brand): u for u in to_fetch}
        try:
            for fut in as_completed(futs, timeout=45):
                try:
                    r = fut.result()
                except Exception:
                    continue
                if r:
                    fetched.append(r)
                    if r["confidence"] >= 0.99:
                        progress(0.9, "⚡ 满分候选出现，提前结束验证")
                        break
                    sub = [x for x in fetched if x["confidence"] < CONFIDENCE_THRESHOLD]
                    if len(sub) >= 2:
                        progress(0.9, "⏹️ 候选均不匹配，提前放弃验证")
                        break
        except TimeoutError:
            progress(0.9, "⏱️ 部分候选验证超时，按已完成的出结果")
            for f in futs:
                if f.done():
                    try:
                        r = f.result()
                        if r:
                            fetched.append(r)
                    except Exception:
                        pass
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    # 死链防护：抓取验证最高分不足时抽查直配候选，验证不过全部丢弃
    fetched_best = max((r["confidence"] for r in fetched), default=0.0)
    if direct and fetched_best < 0.9:
        progress(0.92, "🔎 抓取结果存疑，抽查直配候选是否真实存在...")
        direct.sort(key=lambda x: x["confidence"], reverse=True)
        r = _validate_one(direct[0]["url"], model_input, brand)
        if r and r["confidence"] >= CONFIDENCE_THRESHOLD:
            fetched.append(r)
            direct = direct[1:]
        else:
            progress(0.92, "⏹️ 直配候选验证失败（机型可能不存在），全部丢弃")
            direct = []

    results = fetched + direct
    # 排序：置信度为主，同分时 specs 参数页优先于主图页
    results.sort(key=lambda x: (x["confidence"] +
                                (0.05 if any(h in x["url"].lower() for h in spec_markers(brand)) else 0)),
                 reverse=True)
    return results

# ==================== 主入口 ====================
def find_model_candidates(model_input: str, progress=None) -> List[dict]:
    """
    型号→参数页候选列表（按匹配度降序），V2.0 管线：
    注册表命中 → 适配器模板 → 必应搜索 → 验证闭环 → 阈值过滤。
    用户确认后经 confirm_mapping 写回注册表（数据飞轮）。
    """
    def _p(p, msg):
        print(msg)
        if progress:
            progress(max(0.0, min(1.0, p)), msg)

    model_input = (model_input or "").strip()
    if not model_input:
        return []

    # ① 注册表命中（数据层：已收录/已确认的机型秒回）
    entry = registry.lookup(model_input)
    if entry and entry.get("spec_url"):
        _p(0.95, f"⚡ 注册表命中：{entry['model']}")
        return [{"url": entry["spec_url"], "how": "已收录",
                 "matched_name": entry["model"], "confidence": 1.0}]

    _p(0.1, f"🔍 开始查找「{model_input}」")
    brand = detect_brand_full(model_input)
    _p(0.15, f"🏷️ 识别品牌：{brand or '未识别（将通用搜索）'}")

    # ② 必应站内搜索（兜底层）
    urls = _bing_site_candidates(model_input, brand, _p)

    # ③ 适配器模板永远追加（零前期成本，验证闭环筛选）
    if brand:
        urls = urls + [u for u in build_spec_urls(model_input, brand) if u not in urls]

    if not urls:
        _p(1.0, "❌ 未找到任何候选链接")
        return []

    results = _validate_candidates(urls, model_input, brand, _p)
    results = [r for r in results if r["confidence"] >= CONFIDENCE_THRESHOLD]
    _p(1.0, f"✅ 查找完成，{len(results)} 个有效候选")
    return results

# ==================== 确认回写（数据飞轮） ====================
def confirm_mapping(model_input: str, url: str, brand: Optional[str] = None):
    """用户确认某候选后写入注册表：型号→参数页，含主图页推导与溯源"""
    brand = brand or detect_brand(url) or detect_brand(model_input)
    landing = landing_from_spec(url, brand) if brand else url
    registry.upsert(
        model=model_input, spec_url=url, brand=brand,
        landing_url=landing if landing != url else None,
        aliases=[model_input],
        source="confirmed",
    )

def get_cached_url(model_input: str) -> Optional[str]:
    """兼容旧接口：注册表命中时返回 spec_url"""
    entry = registry.lookup(model_input)
    return entry.get("spec_url") if entry else None
