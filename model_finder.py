# model_finder.py — 型号→自动定位官网参数页（V1.4 功能①）
# 策略（2026-09-14 活体实验定稿）：
#   ①映射缓存（用户确认过的直接命中）
#   ②必应站内搜索（实验证实经本地 Reader 可用且精准，为主策略；比 URL 模板更抗官网改版）
#   ③URL 模板生成（仅作搜索失败时的补充候选）
# 每个候选都经"抓取→识别机型→与输入比对"的验证闭环打分，杜绝张冠李戴。
import re
import os
import json
import difflib
from typing import Dict, List, Optional

from cshi import fetch_markdown, extract_phone_name, load_cache, save_cache, call_llm, client

MAP_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "model_url_map.json")

# 品牌关键词与官网主域（用于 site: 限定搜索与结果过滤）
BRANDS = {
    "OPPO":    {"keywords": ["oppo", "欧珀"], "domain": "www.oppo.com"},
    "vivo":    {"keywords": ["vivo"], "domain": "www.vivo.com.cn"},
    "iQOO":    {"keywords": ["iqoo"], "domain": "www.iqoo.com"},
    "小米":     {"keywords": ["xiaomi", "小米"], "domain": "www.mi.com"},
    "Redmi":   {"keywords": ["redmi", "红米"], "domain": "www.mi.com"},
    "荣耀":     {"keywords": ["honor", "荣耀"], "domain": "www.honor.cn"},
    "华为":     {"keywords": ["huawei", "华为"], "domain": "consumer.huawei.com"},
    "一加":     {"keywords": ["oneplus", "一加"], "domain": "www.oneplus.cn"},
    "真我":     {"keywords": ["realme", "真我"], "domain": "www.realme.com.cn"},
    "三星":     {"keywords": ["samsung", "三星", "galaxy"], "domain": "www.samsung.com"},
    "苹果":     {"keywords": ["iphone", "苹果"], "domain": "www.apple.com.cn"},
    "魅族":     {"keywords": ["meizu", "魅族"], "domain": "www.meizu.com"},
    "摩托罗拉":  {"keywords": ["moto", "razr"], "domain": "www.motorola.com.cn"},
}

SPECS_HINTS = ("specs", "spec", "param", "参数")


# 中英品牌名等价表：归一化时把中文品牌词翻成英文，修复"红米k70"vs"Redmi K70"
# 这类脚本不等价导致包含/相似度全部失效的问题（实测主因，2026-09-15）
_CANONICAL_BRAND_WORDS = {
    "红米": "redmi", "小米": "xiaomi", "华为": "huawei", "荣耀": "honor",
    "一加": "oneplus", "真我": "realme", "三星": "samsung", "苹果": "apple",
    "魅族": "meizu", "摩托罗拉": "motorola", "欧珀": "oppo",
}

def normalize_model(s: str) -> str:
    """型号归一化：小写、中英品牌词等价化、去空白与标点（用于比对与缓存键）"""
    s = (s or "").lower()
    for cn, en in _CANONICAL_BRAND_WORDS.items():
        if cn in s:
            s = s.replace(cn, f" {en} ")
    return re.sub(r'[^0-9a-z]+', '', s)


def detect_brand(model_input: str) -> Optional[dict]:
    """从型号输入中识别品牌（关键词表 → LLM 分类兜底）；未识别返回 None"""
    s = (model_input or "").lower()
    for name, info in BRANDS.items():
        if any(k in s for k in info["keywords"]):
            return {"name": name, **info}
    return _detect_brand_llm(model_input)

_LLM_BRAND_ALIAS = {
    "oppo": "OPPO", "vivo": "vivo", "iqoo": "iQOO", "xiaomi": "小米", "redmi": "Redmi",
    "honor": "荣耀", "huawei": "华为", "oneplus": "一加", "realme": "真我",
    "samsung": "三星", "apple": "苹果", "meizu": "魅族", "motorola": "摩托罗拉",
}

def _detect_brand_llm(model_input: str) -> Optional[dict]:
    """关键词表不认识时，让 LLM 做品牌分类（一次廉价调用，实测它能认出
    'Find X9s Pro'→OPPO 这类无品牌词输入），识别失败返回 None。
    这是装饰性步骤：短超时+不重试（max_retries=1），API 抖动时快速降级
    到多品牌 OR 搜索，绝不拖住整体流程（修复过 3 次重试拖 288 秒的事故）。"""
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
            name = _LLM_BRAND_ALIAS[brand_key]
            return {"name": name, **BRANDS[name]}
    except Exception as e:
        print(f"⚠️ LLM品牌识别失败（快速降级）：{str(e)[:60]}")
    return None


def match_score(model_input: str, phone_name: str) -> float:
    """
    输入型号与页面识别出的机型名的匹配度 0~1。
    两道硬规则（实测教训）：
    - 正向包含（输入⊂页面名）满分；反向包含按长度比缩水，防"华为"短词蹭分；
    - 数字身份校验：两侧都含数字但数字序列不一致时强制压到阈值之下
      （X30000≠X300、K70≠K700——编辑相似度对型号数字完全失明，vivox30000
      曾凭与 vivo X300 的 0.947 相似度假命中）。
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

    # 数字身份校验：剥掉"5G/4G"这类常见噪声数字后，型号数字序列必须一致
    da = re.sub(r'\D', '', re.sub(r'[45]g', '', a))
    db = re.sub(r'\D', '', re.sub(r'[45]g', '', b))
    if da and db and da != db:
        return min(score, 0.35)
    return score

CONFIDENCE_THRESHOLD = 0.5  # 低于此值的候选视为不匹配（0.45 档曾放进 vivo 壳页的编辑相似度候选），直接丢弃


def _load_map() -> Dict[str, str]:
    try:
        with open(MAP_CACHE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def confirm_mapping(model_input: str, url: str):
    """用户确认某型号→参数页映射后落盘，下次直接命中"""
    m = _load_map()
    m[normalize_model(model_input)] = url
    os.makedirs(os.path.dirname(MAP_CACHE), exist_ok=True)
    with open(MAP_CACHE, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=1)


def get_cached_url(model_input: str) -> Optional[str]:
    return _load_map().get(normalize_model(model_input))


def _bing_site_candidates(model_input: str, brand: Optional[dict], progress) -> List[str]:
    """必应站内搜索：有品牌用 site:单域限定；无品牌（或品牌识别降级）时用
    多品牌域 OR 兜底——实验证明必应只有 site: 限定才出深链。"""
    import requests
    from cshi import JINA_READER_URL
    if brand:
        # site: 操作符去掉 www 前缀：实测 site:www.vivo.com.cn 返回 0 条，
        # 而 site:vivo.com.cn 至少能出域名链接（必应对带 www 的站点限定很挑剔）
        site_domain = brand["domain"].removeprefix("www.")
        query = f"site:{site_domain} {model_input} 参数"
    else:
        main_domains = [b["domain"] for b in BRANDS.values()][:8]
        query = f"{model_input} 参数 ({' OR '.join('site:' + d for d in main_domains)})"
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
    allowed = [b["domain"] for b in BRANDS.values()]
    out = []
    for u in links:
        host = u.split('/')[2] if u.count('/') >= 2 else ''
        if not any(d in host for d in allowed):
            continue
        if brand and brand["domain"] not in host:
            continue
        # 参数页优先，主图页次之（也可提取但参数少）
        out.append(u)
    progress(0.45, f"🔎 搜索得到 {len(out)} 条候选链接")
    return out[:6]


def _template_candidates(model_input: str, brand: Optional[dict]) -> List[str]:
    """URL 模板层（兜底）：已知品牌按官网产品页 slug 规则生成候选。
    模式均经真实页面探针验证（2026-09-15），无效路径会返回无内容的软200壳页，
    由验证闭环按低分过滤，不会误推荐。"""
    if not brand:
        return []
    name = brand["name"]

    if name == "OPPO":
        slug = re.sub(r'[^0-9a-z]+', '-', model_input.lower()).strip('-')
        if slug and re.fullmatch(r'[a-z0-9-]+', slug):
            return [
                f"https://www.oppo.com/cn/smartphones/{slug}/specs/",
                f"https://www.oppo.com/cn/smartphones/series-{slug.rsplit('-', 1)[0]}/{slug}/specs/",
            ]
        return []

    if name in ("vivo", "iQOO"):
        # vivo 产品页模式：/vivo/{全小写连写slug}/（实测 /vivo/x200/、/vivo/x300ultra/）
        rest = model_input.lower()
        for kw in ("vivo", "iqoo"):
            rest = rest.replace(kw, " ")
        slug = re.sub(r'[^0-9a-z]+', '', rest)
        if slug and re.fullmatch(r'[a-z0-9]+', slug):
            return [f"https://www.vivo.com.cn/vivo/{slug}/"]
        return []

    if name in ("小米", "Redmi"):
        # 小米商城模式：/mi.com/{英文品牌}-{型号连写-}[/specs]（实测 /redmi-k70[/specs]）
        eng = "redmi" if name == "Redmi" else "xiaomi"
        rest = model_input.lower()
        for kw in ("小米", "红米", "xiaomi", "redmi"):
            rest = rest.replace(kw, " ")
        model_part = re.sub(r'[^0-9a-z]+', '-', rest).strip('-')
        if model_part:
            return [
                f"https://www.mi.com/{eng}-{model_part}",
                f"https://www.mi.com/{eng}-{model_part}/specs",
            ]
        return []

    return []


def _page_h1(md: str) -> str:
    """取 Markdown 首个一级标题（产品页 H1 通常是机型名）"""
    m = re.search(r'^#\s+(.+)$', md, re.M)
    return m.group(1).strip() if m else ""

def _validate_one(url: str, model_input: str) -> Optional[dict]:
    """验证单个候选：抓页面 → 识别机型 → 打分。
    识别走三级递进：H1 标题（零成本）→ 页面文本包含（零成本）→ LLM 识别（兜底），
    绝大多数候选在前两级即出结果，不再逐个调 API。"""
    cached = load_cache(url, "page")
    md = cached["text"] if (cached and cached.get("text")) else fetch_markdown(url, max_retries=1, timeout=25)
    if not md or len(md.strip()) < 500:
        return None

    name, score = "", 0.0
    h1 = _page_h1(md)
    if h1:
        name, score = h1, match_score(model_input, h1)
    if score < 0.45:
        # 页面文本包含归一化型号 → 高置信。必须先剥掉 URL：
        # 页面常含指向自身的链接，整页归一化会把 URL slug 变成纯字母串，
        # 让任何输入都"被包含"（vivox30000 壳页 76% 误报的根因）
        a = normalize_model(model_input)
        visible = re.sub(r'https?://\S+', ' ', md[:10000])
        if a and a in normalize_model(visible):
            score, name = max(score, 0.9), name or model_input
    if score < 0.45:
        try:
            llm_name = extract_phone_name(md[:2500], url)
            s2 = match_score(model_input, llm_name)
            if s2 > score:
                name, score = llm_name, s2
        except Exception:
            pass

    # 参数页特征精修：无"参数"字样的（可能是重定向落地页）降权，规范 URL 排前
    if "参数" not in md and not any(h in url.lower() for h in SPECS_HINTS):
        score = round(score * 0.85, 3)
    if not name:
        return None
    return {"url": url, "how": "搜索", "matched_name": name,
            "confidence": round(min(score, 1.0), 3)}

def _slug_specs_score(model_input: str, url: str) -> float:
    """specs 类 URL 的 slug 与输入的匹配度（用于强匹配免抓取直配）"""
    if not any(h in url.lower() for h in ("specs", "spec", "param")):
        return 0.0
    segs = [s for s in url.split('/') if s and s.lower() not in ("specs", "spec")]
    slug = segs[-1] if segs else ""
    return match_score(model_input, slug.replace('-', ' '))

def _validate_candidates(urls: List[str], model_input: str, progress) -> List[dict]:
    """
    并行验证 + 早退 + 兜底超时（修复"验证候选太久"）：
    - 强 slug 直配（specs URL 且 slug≈输入 ≥0.95）免抓取直接入选；
    - 已缓存页面的候选排前面（验证近乎瞬时，大概率触发早退）；
    - 并行验证，拿到首个 ≥0.99 的满分候选立即取消剩余；
    - 整体 60 秒上限，反例/慢站不再傻等。
    progress 只在调度线程调用（并行工作线程不碰界面）。
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from cshi import load_cache

    direct, to_fetch, seen = [], [], set()
    for u in urls[:6]:
        if u in seen:
            continue
        seen.add(u)
        s = _slug_specs_score(model_input, u)
        if s >= 0.95:
            direct.append({"url": u, "how": "直配", "matched_name": "(URL 结构匹配)",
                           "confidence": round(s * 0.95, 3)})
        else:
            to_fetch.append(u)
    if direct:
        progress(0.9, f"⚡ {len(direct)} 个候选 slug 直配，免抓取入选")

    # 缓存页优先排序：瞬时完成，容易触发早退
    to_fetch.sort(key=lambda u: 0 if (load_cache(u, "page") or {}).get("text") else 1)

    fetched = []
    if to_fetch:
        workers = min(3, len(to_fetch))
        progress(0.55, f"📄 并行验证 {len(to_fetch)} 个候选（{workers} 并发，满分即早退）...")
        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="validate")
        futs = {pool.submit(_validate_one, u, model_input): u for u in to_fetch}
        try:
            for fut in as_completed(futs, timeout=45):
                try:
                    r = fut.result()
                except Exception:
                    continue
                if r:
                    fetched.append(r)
                    if r["confidence"] >= 0.99:
                        progress(0.9, f"⚡ 满分候选出现，提前结束验证")
                        break
                    # 明显垃圾输入：2 个低分结果即提前放弃，不等慢站
                    sub = [x for x in fetched if x["confidence"] < CONFIDENCE_THRESHOLD]
                    if len(sub) >= 2:
                        progress(0.9, "⏹️ 候选均不匹配，提前放弃验证")
                        break
        except TimeoutError:
            # 超时不浪费：收割已完成的结果再收摊
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

    # 死链防护（实测教训：小米17T 为海外机型、国行站无页面，slug 直配会推荐
    # 打不开的链接）：当抓取验证的最高分不足 0.9 时，抽查分数最高的直配候选，
    # 验证不过则所有直配视为死链丢弃
    fetched_best = max((r["confidence"] for r in fetched), default=0.0)
    if direct and fetched_best < 0.9:
        progress(0.92, "🔎 抓取结果存疑，抽查直配候选是否真实存在...")
        direct.sort(key=lambda x: x["confidence"], reverse=True)
        r = _validate_one(direct[0]["url"], model_input)
        if r and r["confidence"] >= CONFIDENCE_THRESHOLD:
            fetched.append(r)
            direct = direct[1:]
        else:
            progress(0.92, "⏹️ 直配候选验证失败（机型可能不存在），全部丢弃")
            direct = []

    results = fetched + direct

    # 排序：置信度为主，同分时 specs 参数页优先于主图页（用户要的是参数页）
    results.sort(key=lambda x: (x["confidence"] +
                                (0.05 if any(h in x["url"].lower() for h in SPECS_HINTS) else 0)),
                 reverse=True)
    return results


def find_model_candidates(model_input: str, progress=None) -> List[dict]:
    """
    型号→参数页候选列表，按匹配度降序。
    progress(pct, msg) 用于界面反馈。候选均经过机型识别验证。
    """
    def _p(p, msg):
        print(msg)
        if progress:
            progress(max(0.0, min(1.0, p)), msg)

    model_input = (model_input or "").strip()
    if not model_input:
        return []

    # ① 映射缓存直接命中（历史确认过的）
    cached = get_cached_url(model_input)
    if cached:
        _p(0.9, f"⚡ 命中型号映射缓存：{cached}")
        return [{"url": cached, "how": "缓存", "matched_name": model_input, "confidence": 1.0}]

    _p(0.1, f"🔍 开始查找「{model_input}」")
    brand = detect_brand(model_input)
    _p(0.15, f"🏷️ 识别品牌：{brand['name'] if brand else '未识别（将通用搜索）'}")

    # ② 必应站内搜索（主策略）
    urls = _bing_site_candidates(model_input, brand, _p)

    # ②b 无品牌且 OR 兜底也 0 条时：按主流品牌域逐个 site: 试探（单域查询实测可靠）。
    # 注意必应全文匹配需要品牌词：探针查询必须拼上该域对应的品牌名
    # （实测 "site:www.oppo.com Find X9s Pro 参数"=0 条，加"OPPO"后=2 条）。
    if not urls and not brand:
        _p(0.4, "🔎 OR 兜底无结果，按主流品牌域逐个试探...")
        for name, info in list(BRANDS.items())[:4]:
            probe = {"name": name, **info}
            enriched = f"{name} {model_input}"
            urls = _bing_site_candidates(enriched, probe, _p)
            if urls:
                break

    # ③ 模板候选永远追加（零前期成本，交由验证闭环筛）：
    # 实测教训——只在"搜索为 0"时启用会漏掉 vivo 这类搜索弱、但 URL 模式
    # 规整的品牌（必应只搜回首页链接时模板被跳过，真实机型 X200 反而失败）
    if brand:
        urls = urls + [u for u in _template_candidates(model_input, brand) if u not in urls]

    if not urls:
        _p(1.0, "❌ 未找到任何候选链接")
        return []

    results = _validate_candidates(urls, model_input, _p)
    # 阈值过滤：胡编型号/无关页面不进候选，避免垃圾推荐
    results = [r for r in results if r["confidence"] >= CONFIDENCE_THRESHOLD]
    _p(1.0, f"✅ 查找完成，{len(results)} 个有效候选")
    return results
