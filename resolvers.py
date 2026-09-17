# resolvers.py — 品牌适配器与型号文本归一化（V2.0 策略层）
# 设计目标：新品牌接入 = 在 BRAND_ADAPTERS 里加一个条目，不改任何执行代码。
# 所有模式均来自真实页面探针验证（2026-09-13 ~ 2026-09-15）。

import re
from typing import Dict, List, Optional

# ==================== 中英品牌词等价表 ====================
# 归一化时把中文品牌词翻成英文，修复"红米k70"vs"Redmi K70"脚本不等价问题
CANONICAL_BRAND_WORDS = {
    "红米": "redmi", "小米": "xiaomi", "华为": "huawei", "荣耀": "honor",
    "一加": "oneplus", "真我": "realme", "三星": "samsung",
    "苹果": "iphone", "魅族": "meizu", "摩托罗拉": "motorola", "欧珀": "oppo",
}

def normalize_model(s: str) -> str:
    """型号归一化：小写、中英品牌词等价化、去空白标点与汉字（用于比对与缓存键）"""
    s = (s or "").lower()
    for cn, en in CANONICAL_BRAND_WORDS.items():
        if cn in s:
            s = s.replace(cn, f" {en} ")
    return re.sub(r'[^0-9a-z]+', '', s)

# ==================== 品牌适配器 ====================
# 字段说明（新增品牌只需填以下条目）：
#   keywords        型号输入中的品牌识别词（中文/英文/别名）
#   domains         官网域名列表（必应搜索的 site: 限定与结果过滤；可含全球站）
#   search_brand    搜索查询中拼接的品牌词（必应全文匹配需要它）
#   slug_rule       URL slug 生成规则："dash"=小写横杠（redmi-k70）| "concat"=小写连写（x300）
#   slug_strip      生成 slug 前要从输入中剔除的品牌词
#   spec_templates  参数页 URL 模板；可用占位符 {slug} {series}（series=slug 去掉末段）
#   landing_from_spec  参数页→主图页推导："strip_specs"=去 /specs | None=同页
#   image_exclude   主图页图片 URL 的排除关键字
#   spec_markers    判定"这是参数页"的 URL 特征

BRAND_ADAPTERS: Dict[str, dict] = {
    "OPPO": {
        "keywords": ["oppo", "欧珀"],
        "domains": ["www.oppo.com"],
        "search_brand": "OPPO",
        "slug_rule": "dash",
        "slug_strip": ["oppo"],
        "spec_templates": [
            "https://www.oppo.com/cn/smartphones/{slug}/specs/",
            "https://www.oppo.com/cn/smartphones/series-{series}/{slug}/specs/",
        ],
        "landing_from_spec": "strip_specs",
        "image_exclude": ["kv", "banner", "logo", "icon", ".svg"],
        "spec_markers": ["specs", "spec", "param"],
    },
    "vivo": {
        "keywords": ["vivo"],
        "domains": ["www.vivo.com.cn"],
        "search_brand": "vivo",
        "slug_rule": "concat",
        "slug_strip": ["vivo"],
        "spec_templates": [
            "https://www.vivo.com.cn/vivo/{slug}/",
            "https://www.vivo.com.cn/vivo/param/{slug}",
        ],
        "landing_from_spec": "param_to_product",   # /vivo/param/x300 → /vivo/x300/（产品页图更全）
        "image_exclude": ["kv", "banner", "logo", "icon", ".svg"],
        "spec_markers": ["param", "spec"],
    },
    "iQOO": {
        # 实测（2026-09-17 用户URL）：iQOO 产品页在 vivo.com.cn 的 /vivo/param/ 路径下
        "keywords": ["iqoo"],
        "domains": ["www.vivo.com.cn", "www.iqoo.com"],
        "search_brand": "iQOO",
        "slug_rule": "concat",
        "slug_strip": [],
        "spec_templates": [
            "https://www.vivo.com.cn/vivo/param/{slug}",
            "https://www.vivo.com.cn/vivo/{slug}/",
        ],
        "landing_from_spec": "param_to_product",
        "image_exclude": ["kv", "banner", "logo", "icon", ".svg"],
        "spec_markers": ["param", "spec"],
    },
    "小米": {
        "keywords": ["xiaomi", "小米"],
        "domains": ["www.mi.com"],
        "search_brand": "小米",
        "slug_rule": "dash",
        "slug_strip": [],   # 品牌词 xiaomi 保留在 slug 里（真实 URL：mi.com/prod/xiaomi-15/specs）
        "spec_templates": [
            "https://www.mi.com/prod/{slug}/specs",   # 新商城模式（2026-09-17 用户反馈确认）
            "https://www.mi.com/{slug}",
            "https://www.mi.com/{slug}/specs",
        ],
        "landing_from_spec": "strip_specs",
        "image_exclude": ["kv", "banner", "logo", "icon", ".svg"],
        "spec_markers": ["specs", "spec", "param", "prod"],
    },
    "Redmi": {
        "keywords": ["redmi", "红米"],
        "domains": ["www.mi.com"],
        "search_brand": "Redmi",
        "slug_rule": "dash",
        "slug_strip": ["redmi", "红米"],
        "spec_templates": [
            "https://www.mi.com/prod/redmi-{slug}/specs",   # 新商城模式（用户实测确认）
            "https://www.mi.com/redmi-{slug}",
            "https://www.mi.com/redmi-{slug}/specs",
        ],
        "landing_from_spec": "strip_specs",
        "image_exclude": ["kv", "banner", "logo", "icon", ".svg"],
        "spec_markers": ["specs", "spec", "param", "prod"],
    },
    "荣耀": {
        "keywords": ["honor", "荣耀"],
        "domains": ["www.honor.cn"],
        "search_brand": "荣耀",
        "slug_rule": "dash",
        "slug_strip": ["honor", "荣耀"],
        "spec_templates": [
            "https://www.honor.cn/{slug}/specs/",
        ],
        "landing_from_spec": "strip_specs",
        "image_exclude": ["kv", "banner", "logo", "icon", ".svg"],
        "spec_markers": ["specs", "spec", "param"],
    },
    "一加": {
        "keywords": ["oneplus", "一加"],
        "domains": ["www.oneplus.cn"],
        "search_brand": "一加",
        "slug_rule": "dash",
        "slug_strip": ["oneplus", "一加"],
        "spec_templates": [
            "https://www.oneplus.cn/{slug}/specs",
        ],
        "landing_from_spec": "strip_specs",
        "image_exclude": ["kv", "banner", "logo", "icon", ".svg"],
        "spec_markers": ["specs", "spec", "param"],
    },
    "华为": {
        "keywords": ["huawei", "华为", "mate", "pura"],
        "domains": ["consumer.huawei.com"],
        "search_brand": "华为",
        "slug_rule": "dash",
        "slug_strip": ["huawei", "华为"],
        "spec_templates": [
            "https://consumer.huawei.com/cn/phones/{slug}/specs/",
        ],
        "landing_from_spec": "strip_specs",
        "image_exclude": ["kv", "banner", "logo", "icon", ".svg"],
        "spec_markers": ["specs", "spec", "param"],
    },
    "真我": {
        # 实测（2026-09-17 用户URL）：域名 realme.com/cn，slug 含品牌词且
        # 型号内字母数字边界拆横杠（GT8 Pro → realme-gt-8-pro）
        "keywords": ["realme", "真我"],
        "domains": ["www.realme.com", "www.realme.com.cn"],
        "search_brand": "真我",
        "slug_rule": "dash",
        "slug_strip": [],
        "spec_templates": [
            "https://www.realme.com/cn/{slug}/specs",
            "https://www.realme.com.cn/{slug}/specs",
        ],
        "landing_from_spec": "strip_specs",
        "image_exclude": ["kv", "banner", "logo", "icon", ".svg"],
        "spec_markers": ["specs", "spec", "param"],
    },
    "三星": {
        # 实测（2026-09-17 用户URL）：区域路径 /hk/（大陆站 /cn/ 常无产品页），
        # slug 保留 galaxy，参数页后缀 specs
        "keywords": ["samsung", "三星", "galaxy"],
        "domains": ["www.samsung.com"],
        "search_brand": "三星",
        "slug_rule": "dash",
        "slug_strip": ["samsung", "三星"],
        "spec_templates": [
            "https://www.samsung.com/hk/smartphones/{slug}/specs/",
            "https://www.samsung.com/cn/smartphones/{slug}/specs/",
            "https://www.samsung.com/cn/smartphones/{slug}/spec/",
        ],
        "landing_from_spec": "strip_specs",
        "image_exclude": ["kv", "banner", "logo", "icon", ".svg"],
        "spec_markers": ["specs", "spec", "param"],
    },
    "苹果": {
        # 实测（2026-09-17 探针）：Apple 中国官网规格页模式 /{slug}/specs/
        # slug 保留 iphone 前缀（iphone-17-pro）；canonical 已把"苹果"映射为 iphone
        "keywords": ["iphone", "苹果"],
        "domains": ["www.apple.com.cn"],
        "search_brand": "iPhone",
        "slug_rule": "dash",
        "slug_strip": [],
        "spec_templates": [
            "https://www.apple.com.cn/{slug}/specs/",
        ],
        "landing_from_spec": "strip_specs",
        "image_exclude": ["kv", "banner", "logo", "icon", ".svg"],
        "spec_markers": ["specs"],
    },
}

# 通用兜底配置（未收录品牌走这里）
GENERIC_ADAPTER = {
    "domains": [],
    "search_brand": "",
    "image_exclude": ["kv", "banner", "logo", "icon", ".svg"],
    "spec_markers": ["specs", "spec", "param", "参数"],
}

def detect_brand(model_input: str) -> Optional[str]:
    """从型号输入识别品牌名（关键词表）；未识别返回 None"""
    s = (model_input or "").lower()
    for name, ad in BRAND_ADAPTERS.items():
        if any(k in s for k in ad["keywords"]):
            return name
    return None

def get_adapter(brand: Optional[str]) -> dict:
    """取品牌适配器；未收录品牌返回通用兜底"""
    if brand and brand in BRAND_ADAPTERS:
        return BRAND_ADAPTERS[brand]
    return dict(GENERIC_ADAPTER)

def match_brand_by_host(host: str) -> Optional[str]:
    """由 URL 主机名反查品牌（用于图片提取等只有 URL 的场景）"""
    host = (host or "").lower()
    for name, ad in BRAND_ADAPTERS.items():
        if any(d in host for d in ad.get("domains", [])):
            return name
    return None

def build_spec_urls(model_input: str, brand: Optional[str]) -> List[str]:
    """按适配器规则生成参数页候选 URL 列表。
    先做中英品牌词替换（小米17t→xiaomi 17t），再按 slug 规则整形——
    保证"红米k70"和"Redmi K70"生成同一 URL。"""
    if not brand:
        return []
    ad = get_adapter(brand)
    rest = (model_input or "").lower()
    for cn, en in CANONICAL_BRAND_WORDS.items():
        if cn in rest:
            rest = rest.replace(cn, f" {en} ")
    for kw in ad.get("slug_strip", []):
        rest = rest.replace(kw, " ")
    if ad.get("slug_rule") == "concat":
        slug = re.sub(r'[^0-9a-z]+', '', rest)
    else:  # dash
        slug = re.sub(r'[^0-9a-z]+', '-', rest).strip('-')
    if not slug or not re.fullmatch(r'[a-z0-9-]+', slug):
        return []
    series = slug.rsplit('-', 1)[0] if '-' in slug else slug
    urls = [tpl.format(slug=slug, series=series) for tpl in ad.get("spec_templates", [])]
    # 字母数字边界变体：GT8→gt-8、X9S→x-9s 这类官网分词差异的兜底
    # （真我 GT8 Pro 实测 slug 为 realme-gt-8-pro）
    alt = re.sub(r'(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])', '-', slug)
    if alt != slug:
        urls += [tpl.format(slug=alt, series=alt.rsplit('-', 1)[0] if '-' in alt else alt)
                 for tpl in ad.get("spec_templates", [])]
    return urls

def landing_from_spec(url: str, brand: Optional[str]) -> str:
    """参数页 URL → 产品主图页 URL（按适配器规则，未收录品牌走通用规则）"""
    u = str(url).strip()
    ad = get_adapter(brand)
    rule = ad.get("landing_from_spec")
    if rule == "strip_specs":
        return re.sub(r'/specs?/?$', '/', u)
    if rule == "param_to_product":
        # vivo/iQOO：/vivo/param/{slug} → /vivo/{slug}/（产品页图片资源更全）
        return re.sub(r'/param/([^/]+)$', r'/\1/', u)
    return re.sub(r'/(specs|spec|param)[^/]*/?$', '/', u)

def image_exclude_words(brand: Optional[str]) -> List[str]:
    return get_adapter(brand).get("image_exclude", GENERIC_ADAPTER["image_exclude"])

def spec_markers(brand: Optional[str]) -> List[str]:
    return get_adapter(brand).get("spec_markers", GENERIC_ADAPTER["spec_markers"])

def all_domains() -> List[str]:
    """所有已收录品牌的域名（必应结果过滤用）"""
    out = []
    for ad in BRAND_ADAPTERS.values():
        out.extend(ad.get("domains", []))
    return out
