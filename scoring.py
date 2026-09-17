# scoring.py — 五维透明评分（0~100），用于对比可视化
# 设计原则：规则全部显式写在这里，评分依据可向用户解释；
# 数据缺失的维度返回 None（图表中按 0 处理但表格里标"未知"）。

import re
from typing import Dict, Optional


def _first_num(s) -> Optional[float]:
    if not s:
        return None
    m = re.search(r'\d+\.?\d*', str(s))
    return float(m.group()) if m else None


def _mega_pixels(s) -> Optional[float]:
    """解析主摄像素（统一为百万单位）："2亿像素"→200、"5000万像素"→50、"50MP"→50"""
    if not s:
        return None
    s = str(s)
    m = re.search(r'(\d+\.?\d*)\s*亿', s)
    if m:
        return float(m.group(1)) * 100
    m = re.search(r'(\d+\.?\d*)\s*万', s)
    if m:
        return float(m.group(1)) / 100
    return _first_num(s)


def _scale(v40: Optional[int]) -> Optional[int]:
    """内部 0~40 分制映射到 0~100"""
    return None if v40 is None else min(100, round(v40 * 2.5))


def _resolution_pts(res: str) -> Optional[int]:
    """分辨率长边分级（与标签系统同一套阈值）"""
    nums = [int(x) for x in re.findall(r'\d{3,4}', str(res))]
    le = max(nums) if nums else 0
    if le <= 0: return None
    if le >= 3800: return 40
    if le >= 3000: return 36
    if le >= 2600: return 32
    if le >= 2300: return 26
    if le >= 2000: return 20
    return 12


def score_phone(params: dict, knowledge_summary: dict = None) -> Dict[str, Optional[int]]:
    """返回 {性能/屏幕/续航充电/影像/性价比参考: 0~100 或 None}"""
    p = params or {}
    ks = knowledge_summary or {}
    display = p.get("display", {}) or {}
    battery = p.get("battery_charging", {}) or {}
    camera = p.get("camera", {}) or {}
    rear_main = camera.get("rear_main", {}) or {}

    # ---- 性能：直接用知识库芯片评分（0~100） ----
    performance = (ks.get("chipset", {}).get("info", {}) or {}).get("performance_score")

    # ---- 屏幕：分辨率(0~24) + 刷新率(0~16) → 换算 0~100 ----
    res_pts = _resolution_pts(display.get("resolution"))
    refresh = _first_num(display.get("refresh_rate"))
    if refresh is not None:
        ref_pts = 16 if refresh >= 144 else 13 if refresh >= 120 else 9 if refresh >= 90 else 5 if refresh >= 60 else 2
    else:
        ref_pts = None
    screen = None if (res_pts is None and ref_pts is None) else round(
        ((res_pts or 0) + (ref_pts or 0)) * 2.0)

    # ---- 续航充电：容量(0~24) + 有线功率(0~16) → 0~100 ----
    cap = _first_num(battery.get("capacity"))
    if cap is not None:
        cap_pts = 24 if cap >= 7000 else 21 if cap >= 6000 else 18 if cap >= 5500 else 15 if cap >= 5000 else 10 if cap >= 4500 else 5
    else:
        cap_pts = None
    watt = _first_num(battery.get("wired_charging"))
    if watt is not None:
        w_pts = 16 if watt >= 120 else 14 if watt >= 100 else 12 if watt >= 80 else 10 if watt >= 66 else 8 if watt >= 50 else 5 if watt >= 30 else 2
    else:
        w_pts = None
    battery_score = None if (cap_pts is None and w_pts is None) else round(((cap_pts or 0) + (w_pts or 0)) * 2.5)

    # ---- 影像：主摄像素(0~16) + OIS(0~6) → 0~100 ----
    mp = _mega_pixels(rear_main.get("resolution"))
    if mp is not None:
        mp_pts = 16 if mp >= 200 else 12 if mp >= 100 else 10 if mp >= 64 else 8 if mp >= 50 else 5
    else:
        mp_pts = None
    ois_pts = 6 if "支持" in str(rear_main.get("ois", "")) else 0
    camera_pts = None if mp_pts is None else mp_pts + ois_pts
    camera_score = None if camera_pts is None else min(100, camera_pts * 4)

    # ---- 性价比参考：由价格区间映射（无价格则 None） ----
    price = _first_num(p.get("basic_info", {}).get("price"))
    if price is not None:
        # 同性能下的价格效率：性能分 / 价格(千元)，映射到 0~100
        perf_val = performance or 50
        value_ratio = perf_val / max(price / 1000, 0.5)
        value_score = min(100, round(value_ratio * 8))
    else:
        value_score = None

    return {
        "性能": round(performance) if performance is not None else None,
        "屏幕": screen,
        "续航充电": battery_score,
        "影像": camera_score,
        "性价比参考": value_score,
    }
