# cshi.py — AI手机参数提取与评测系统 核心后端
# V1.1：新增 .env 配置 / 缓存 / API重试 / 分段并行提取 / 标签系统修正
import requests
import json
from openai import OpenAI
import re
import time
import os
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, Callable, Tuple

from knowledge_base import PhoneKnowledgeBase

# ================= 1. 配置（支持 .env / 环境变量） =================
def _load_dotenv(path: str = None):
    """极简 .env 加载器（无额外依赖）；已在环境变量中存在的值不会被覆盖"""
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                value = value.strip().strip('"').strip("'")
                os.environ.setdefault(key.strip(), value)
    except FileNotFoundError:
        pass

_load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# API Key：提取与评测可分别配置；评测 Key 留空时与提取共用
EXTRACT_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
REVIEW_API_KEY = os.environ.get("DEEPSEEK_REVIEW_API_KEY", "") or EXTRACT_API_KEY

JINA_READER_URL = os.environ.get("JINA_READER_URL", "http://127.0.0.1:3001/")
MODEL_NAME = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
MAX_PARALLEL_CHUNKS = int(os.environ.get("MAX_PARALLEL_CHUNKS", "4"))
API_MAX_RETRIES = int(os.environ.get("API_MAX_RETRIES", "3"))
CACHE_DIR = os.path.join(BASE_DIR, "cache")

client = OpenAI(api_key=EXTRACT_API_KEY or "sk-not-configured", base_url="https://api.deepseek.com")
review_client = OpenAI(api_key=REVIEW_API_KEY or "sk-not-configured", base_url="https://api.deepseek.com")

# ================= 2. 缓存（网页原文 + 最终结果，按 URL 哈希） =================
def _cache_file(url: str, kind: str) -> str:
    digest = hashlib.md5(url.encode("utf-8")).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"{kind}_{digest}.json")

def load_cache(url: str, kind: str):
    try:
        with open(_cache_file(url, kind), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def save_cache(url: str, kind: str, data):
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(_cache_file(url, kind), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️ 缓存写入失败: {e}")

def clear_cache() -> int:
    """清空缓存目录，返回删除的文件数"""
    removed = 0
    if os.path.isdir(CACHE_DIR):
        for name in os.listdir(CACHE_DIR):
            try:
                os.remove(os.path.join(CACHE_DIR, name))
                removed += 1
            except OSError:
                pass
    return removed

# ================= 3. 抓取本地 Jina Reader 数据（带重试） =================
def fetch_markdown(url: str, max_retries: int = 2) -> Optional[str]:
    """抓取网页内容，失败自动重试"""
    reader_url = f"{JINA_READER_URL.rstrip('/')}/{url}"
    headers = {"X-Respond-With": "markdown"}

    for attempt in range(1, max_retries + 1):
        try:
            print(f"正在抓取网页: {url} ...")
            res = requests.get(reader_url, headers=headers, timeout=60)
            res.raise_for_status()
            res.encoding = 'utf-8'
            if len(res.text.strip()) > 200:
                return res.text
            print(f"⚠️ Reader 返回内容过短（{len(res.text)} 字符），可能是错误页")
        except Exception as e:
            print(f"⚠️ 抓取失败（第 {attempt}/{max_retries} 次）: {str(e)[:80]}")
        if attempt < max_retries:
            time.sleep(2 * attempt)
    return None

# ================= 4. LLM 调用封装（带重试与退避） =================
def call_llm(client_obj, prompt: str, *, temperature: float = 0.1, timeout: int = 120,
             max_tokens: int = 8000, json_mode: bool = True, tag: str = "LLM调用") -> str:
    last_error = None
    for attempt in range(1, API_MAX_RETRIES + 1):
        try:
            kwargs = dict(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                timeout=timeout,
                max_tokens=max_tokens,
            )
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            response = client_obj.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            if not (content and content.strip()):
                # 思考型模型偶发把整个 token 预算花在思考上，正文为空
                print(f"⚠️ {tag} 正文为空（finish_reason={response.choices[0].finish_reason}）")
            return content
        except Exception as e:
            last_error = e
            msg = str(e)
            # 认证类错误重试无意义，直接抛出并给出可读提示
            if "401" in msg or "Authentication" in msg or "api_key" in msg.lower():
                raise RuntimeError("API 认证失败：请检查 .env 中的 DEEPSEEK_API_KEY 是否正确") from e
            if attempt < API_MAX_RETRIES:
                wait = min(2 ** attempt, 8)
                print(f"⚠️ {tag} 第 {attempt} 次失败：{msg[:80]}，{wait} 秒后重试...")
                time.sleep(wait)
    raise RuntimeError(f"{tag}连续 {API_MAX_RETRIES} 次失败：{str(last_error)[:100]}")

def call_llm_text(client_obj, prompt: str, *, temperature: float = 0.5, timeout: int = 180,
                  max_tokens: int = 8000, tag: str = "文本生成", attempts: int = 2) -> Optional[str]:
    """
    生成纯文本（评测等，非JSON）。
    deepseek-flash 属"先思考后正文"的模型：长输入下思考可能耗尽 token 预算导致正文为空，
    或推理耗时超过单次超时——call_llm 只对异常重试，这里对"空正文"再做软重试，
    且预算给足（8000，只按实际生成计费）。
    """
    last_error = None
    for i in range(attempts):
        try:
            content = call_llm(client_obj, prompt, temperature=temperature, timeout=timeout,
                               max_tokens=max_tokens, json_mode=False, tag=tag)
            if content and content.strip():
                return content.strip()
            print(f"⚠️ {tag} 第 {i + 1} 次返回空正文（思考耗尽额度），重试...")
        except Exception as e:
            last_error = e
            print(f"⚠️ {tag} 失败: {str(e)[:100]}")
    if last_error:
        raise last_error
    return None

# ================= 5. 自动识别手机名称 =================
def extract_phone_name(markdown_text: str, url: str) -> str:
    """从网页内容中自动识别手机名称"""
    print("🔍 正在识别手机型号...")

    prompt = f"""
    从以下手机官网的Markdown内容中识别手机的品牌和完整型号名称。

    网页URL：{url}

    网页内容（前2000字符）：
    {markdown_text[:2000]}

    请只返回JSON格式，不要包含其他文字：
    {{
        "brand": "品牌名称",
        "device_name": "完整型号名称（不包含品牌）",
        "full_name": "完整名称（品牌+型号）"
    }}
    """

    try:
        content = call_llm(client, prompt, temperature=0.1, timeout=60,
                           max_tokens=1000, tag="型号识别")
        result = json.loads(content)
        full_name = result.get("full_name", "未知手机")
        print(f"✅ 识别到手机：{full_name}")
        return full_name
    except Exception as e:
        print(f"⚠️ 识别失败（{str(e)[:60]}），从URL提取名称")
        return extract_name_from_url(url)

def extract_name_from_url(url: str) -> str:
    """从URL中提取可能的手机名称"""
    path = url.replace("https://", "").replace("http://", "")
    parts = [p for p in path.split("/") if p and p not in ["cn", "www", "specs", "smartphones", "#"]]

    if parts:
        name = parts[-1].replace("-", " ").replace("_", " ").title()
        return name
    return "未知手机"

# ================= 6. 参数架构 =================
def get_full_schema():
    """完整的参数架构"""
    return {
        "basic_info": {
            "device_name": "手机型号",
            "brand": "品牌",
            "model_number": "具体型号编号",
            "release_date": "发布日期",
            "price": "发售价",
            "os": "操作系统版本",
            "colors": "可选颜色"
        },
        "processor": {
            "soc": "处理器具体型号",
            "cpu_cores": "CPU核心数和架构",
            "cpu_frequency": "CPU频率",
            "gpu": "GPU型号",
            "process_technology": "制程工艺"
        },
        "memory_storage": {
            "ram": "运行内存",
            "ram_type": "内存类型",
            "rom": "存储容量",
            "storage_type": "存储类型"
        },
        "display": {
            "size": "屏幕尺寸",
            "type": "屏幕类型",
            "resolution": "分辨率",
            "pixel_density": "像素密度",
            "refresh_rate": "最高刷新率",
            "touch_sampling_rate": "触控采样率",
            "peak_brightness": "峰值亮度",
            "hdr_support": "HDR支持",
            "color_gamut": "色域",
            "protection": "屏幕保护玻璃"
        },
        "battery_charging": {
            "capacity": "电池容量典型值",
            "rated_capacity": "电池额定容量",
            "wired_charging": "有线快充最高瓦数",
            "wireless_charging": "无线充电瓦数",
            "reverse_wireless_charging": "反向无线充电",
            "charging_technology": "充电技术名称"
        },
        "camera": {
            "rear_main": {
                "sensor": "主摄传感器型号",
                "resolution": "主摄像素",
                "aperture": "主摄光圈",
                "sensor_size": "传感器尺寸",
                "ois": "光学防抖",
                "focal_length": "焦距"
            },
            "rear_ultrawide": {
                "resolution": "超广角像素",
                "aperture": "超广角光圈",
                "fov": "视场角"
            },
            "rear_telephoto": {
                "resolution": "长焦像素",
                "aperture": "长焦光圈",
                "optical_zoom": "光学变焦倍数",
                "digital_zoom": "数码变焦"
            },
            "front_main": {
                "resolution": "前置主摄像素",
                "aperture": "前置光圈",
                "autofocus": "前置自动对焦"
            },
            "video": {
                "rear_max": "后置最大视频规格",
                "front_max": "前置最大视频规格",
                "slow_motion": "慢动作规格"
            }
        },
        "connectivity": {
            "network": "网络制式",
            "5g_bands": "5G频段",
            "4g_bands": "4G频段",
            "wifi": "WiFi版本",
            "bluetooth": "蓝牙版本",
            "nfc": "NFC支持",
            "usb": "USB接口类型",
            "gps": "定位系统",
            "infrared": "红外遥控",
            "sim": "SIM卡类型"
        },
        "sensors": {
            "fingerprint": "指纹识别类型和位置",
            "face_unlock": "面部识别",
            "other_sensors": "其他传感器"
        },
        "audio": {
            "speakers": "扬声器配置",
            "headphone_jack": "耳机孔",
            "audio_chip": "音频芯片"
        },
        "physical": {
            "dimensions": "尺寸(长宽高)",
            "weight": "重量",
            "thickness": "厚度",
            "material": "机身材质",
            "water_resistance": "防水等级"
        }
    }

# ================= 7. 智能分段 =================
def split_markdown_smartly(markdown_text: str, max_chunk_size: int = 5000) -> List[str]:
    """智能分段（拼接无损）"""
    if len(markdown_text) <= max_chunk_size:
        return [markdown_text]

    # split 对以换行结尾的文本会产生一个末尾空元素，先去掉并记录，
    # 避免拼接结果比原文多出一个换行
    had_trailing_newline = markdown_text.endswith('\n')
    lines = markdown_text.split('\n')
    if had_trailing_newline:
        lines = lines[:-1]

    chunks = []
    current_chunk = ""

    for line in lines:
        if len(current_chunk) + len(line) > max_chunk_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = line + '\n'
        else:
            current_chunk += line + '\n'

    if current_chunk:
        if had_trailing_newline:
            chunks.append(current_chunk)
        else:
            chunks.append(current_chunk[:-1])  # 去掉对最后一行额外补的换行

    return chunks

# ================= 8. 修复截断的JSON =================
def fix_truncated_json(json_str: str):
    """
    修复被截断的JSON。
    记录最近一次"可以安全截断"的位置（完整的键值对或闭合的对象），
    从该处截断并补齐所需数量的右花括号。
    """
    depth = 0
    in_string = False
    escape = False
    last_safe = None  # (截断位置, 截断处的括号深度)

    for i, char in enumerate(json_str):
        if escape:
            escape = False
            continue
        if char == '\\':
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                last_safe = (i + 1, 0)   # 顶层对象完整结束
            elif depth > 0:
                last_safe = (i + 1, depth)  # 嵌套对象完整闭合
        elif char == ',' and depth >= 1:
            last_safe = (i, depth)       # 逗号前是完整的键值对

    if last_safe:
        cut, d = last_safe
        if d == 0:
            return json_str[:cut]
        return json_str[:cut] + '}' * d
    return None

# ================= 9. 提取单个分段 =================
def extract_from_chunk(chunk_text: str, phone_name: str, schema: dict,
                       chunk_index: int, total_chunks: int) -> dict:
    """从单个文本块中提取参数"""
    print(f"  处理第 {chunk_index + 1}/{total_chunks} 段...")

    prompt = f"""
    你是一个严谨的数据提取专家。请从以下手机官网的 Markdown 文本片段中提取硬件参数。
    这是第 {chunk_index + 1} 段，共 {total_chunks} 段。

    注意：
    1. 文本中包含大量的网页导航链接和冗余广告信息，请直接无视它们
    2. 如果文本中确实缺失某项参数，请将其值设为 "未提及"
    3. 不要编造任何参数，只提取文本中明确提到的信息
    4. 对于数值型参数，保留原始单位和精度

    当前提取的手机型号为：{phone_name}

    必须严格按照以下 JSON 结构输出，不要包含任何多余的解释性文字或 Markdown 代码块标记：
    {json.dumps(schema, ensure_ascii=False, indent=2)}

    【网页内容片段】：
    {chunk_text}
    """

    try:
        content = call_llm(client, prompt, temperature=0.1, timeout=120,
                           max_tokens=8000, tag=f"第{chunk_index + 1}段提取")
    except RuntimeError:
        # 认证/连续失败等致命错误向上传播，让整次提取快速失败
        raise
    except Exception as e:
        print(f"  ❌ 第 {chunk_index + 1} 段失败: {str(e)[:80]}")
        return {}

    try:
        result = json.loads(content)
        print(f"  ✅ 第 {chunk_index + 1} 段完成")
        return result
    except json.JSONDecodeError:
        print(f"  ⚠️ 第 {chunk_index + 1} 段JSON解析失败，尝试修复...")
        fixed = fix_truncated_json(content)
        if fixed:
            try:
                result = json.loads(fixed)
                print(f"  ✅ 修复成功，提取到部分参数")
                return result
            except json.JSONDecodeError:
                pass
        print(f"  ❌ 第 {chunk_index + 1} 段无法解析")
        return {}

# ================= 10. 深度合并（带冲突检测） =================
def deep_merge_with_conflicts(dict1: dict, dict2: dict, conflicts: list = None, path: str = ""):
    """
    深度合并两个字典，并检测同一路径下不同非空值之间的冲突。
    冲突时保留先出现的值（按分段顺序合并）。
    """
    if conflicts is None:
        conflicts = []

    result = dict1.copy()

    for key, value in dict2.items():
        current_path = f"{path}.{key}" if path else key

        if key not in result:
            result[key] = value
        elif isinstance(value, dict) and isinstance(result[key], dict):
            result[key], conflicts = deep_merge_with_conflicts(result[key], value, conflicts, current_path)
        else:
            if (result[key] not in ["未提及", "", None, []]) and (value not in ["未提及", "", None, []]):
                if result[key] != value:
                    conflicts.append({
                        "path": current_path,
                        "value1": result[key],
                        "value2": value,
                        "final_value": result[key]
                    })
            if result[key] in ["未提及", "", None, []] and value not in ["未提及", "", None, []]:
                result[key] = value

    return result, conflicts

# ================= 11. 清理空值 =================
EMPTY_VALUES = ["", "未提及", None, [], {}]
# 注意："无" 是有效信息（如"耳机孔: 无"），不再被清理

def clean_empty_values(data):
    """递归移除空值和未提及，保留内部键（以 _ 开头）"""
    if isinstance(data, dict):
        cleaned = {}
        for key, value in data.items():
            if key.startswith('_'):
                cleaned[key] = value
                continue
            if value not in EMPTY_VALUES:
                if isinstance(value, (dict, list)):
                    cleaned_value = clean_empty_values(value)
                    if cleaned_value:
                        cleaned[key] = cleaned_value
                else:
                    cleaned[key] = value
        return cleaned
    elif isinstance(data, list):
        cleaned_list = []
        for item in data:
            cleaned_item = clean_empty_values(item)
            if cleaned_item:
                cleaned_list.append(cleaned_item)
        return cleaned_list
    else:
        return data

# ================= 12. 键名翻译映射 =================
KEY_TRANSLATION = {
    "basic_info": "基本信息",
    "device_name": "手机型号",
    "brand": "品牌",
    "model_number": "型号编号",
    "release_date": "发布日期",
    "price": "发售价",
    "os": "操作系统",
    "colors": "可选颜色",

    "processor": "处理器",
    "soc": "芯片型号",
    "cpu_cores": "CPU核心",
    "cpu_frequency": "CPU频率",
    "gpu": "GPU",
    "process_technology": "制程工艺",

    "memory_storage": "内存与存储",
    "ram": "运行内存",
    "ram_type": "内存类型",
    "rom": "存储容量",
    "storage_type": "存储类型",

    "display": "显示屏",
    "size": "屏幕尺寸",
    "type": "屏幕类型",
    "resolution": "分辨率",
    "pixel_density": "像素密度",
    "refresh_rate": "刷新率",
    "touch_sampling_rate": "触控采样率",
    "peak_brightness": "峰值亮度",
    "hdr_support": "HDR支持",
    "color_gamut": "色域",
    "protection": "屏幕保护",

    "battery_charging": "电池与充电",
    "capacity": "电池容量",
    "rated_capacity": "额定容量",
    "wired_charging": "有线充电",
    "wireless_charging": "无线充电",
    "reverse_wireless_charging": "反向无线充电",
    "charging_technology": "充电技术",

    "camera": "摄像头",
    "rear_main": "后置主摄",
    "rear_ultrawide": "后置超广角",
    "rear_telephoto": "后置长焦",
    "rear_periscope": "潜望长焦",
    "front_main": "前置摄像头",
    "sensor": "传感器",
    "aperture": "光圈",
    "sensor_size": "传感器尺寸",
    "ois": "光学防抖",
    "focal_length": "焦距",
    "fov": "视场角",
    "optical_zoom": "光学变焦",
    "digital_zoom": "数码变焦",
    "autofocus": "自动对焦",
    "video": "视频拍摄",
    "rear_max": "后置最高规格",
    "front_max": "前置最高规格",
    "slow_motion": "慢动作",

    "connectivity": "连接性",
    "network": "网络制式",
    "5g_bands": "5G频段",
    "4g_bands": "4G频段",
    "wifi": "WiFi",
    "bluetooth": "蓝牙",
    "nfc": "NFC",
    "usb": "USB接口",
    "gps": "定位系统",
    "infrared": "红外遥控",
    "sim": "SIM卡",

    "sensors": "传感器",
    "fingerprint": "指纹识别",
    "face_unlock": "面部识别",
    "other_sensors": "其他传感器",

    "audio": "音频",
    "speakers": "扬声器",
    "headphone_jack": "耳机接口",
    "audio_chip": "音频芯片",

    "physical": "机身参数",
    "dimensions": "机身尺寸",
    "weight": "重量",
    "thickness": "厚度",
    "material": "机身材质",
    "water_resistance": "防水等级"
}

def translate_keys(data):
    """将英文键名翻译成中文，跳过内部键"""
    if isinstance(data, dict):
        translated = {}
        for key, value in data.items():
            if key.startswith('_'):
                translated[key] = value
                continue
            new_key = KEY_TRANSLATION.get(key, key)
            if isinstance(value, dict):
                translated[new_key] = translate_keys(value)
            elif isinstance(value, list):
                translated[new_key] = [
                    translate_keys(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                translated[new_key] = value
        return translated
    else:
        return data

# ================= 13. 生成参数标签（V1.1 修正误判） =================
def _resolution_class(res_str) -> Optional[float]:
    """
    判断分辨率等级：4 / 2 / 1.5 / 1(1080P) / 0.75(720P)。
    优先解析"宽×高"取长边像素（行业惯例按长边分级），无法解析时回退关键字。
    """
    s = str(res_str)
    m = re.search(r'(\d{3,4})\s*[×xX*]\s*(\d{3,4})', s)
    if m:
        long_edge = max(int(m.group(1)), int(m.group(2)))
        if long_edge >= 3800: return 4.0
        if long_edge >= 3000: return 2.0
        if long_edge >= 2600: return 1.5
        if long_edge >= 2300: return 1.0
        return 0.75
    s_lower = s.lower()
    if "4k" in s_lower: return 4.0
    if "2k" in s_lower or "qhd" in s_lower: return 2.0
    if "1.5k" in s_lower: return 1.5
    if "1080" in s_lower or "fhd" in s_lower: return 1.0
    if "720" in s_lower or " hd" in s_lower or s_lower == "hd": return 0.75
    return None

def _count_beidou_bands(gps_text: str) -> Optional[int]:
    """
    从定位系统信息中解析北斗频段数量（B1I/B1C/B2a/B2b/B3I 等）。
    B 开头的频段是北斗独有写法（GPS 用 L、GLONASS 用 G、Galileo 用 E、QZSS/NavIC 用 L），
    不会与其他系统混淆；参数页未写出频段明细时返回 None（不做推断）。
    """
    s = str(gps_text).upper()
    bands = set(re.findall(r'(?<![A-Z0-9])B[123](?:I|C|A|B)?(?![0-9])', s))
    return len(bands) if bands else None

def _is_empty(value) -> bool:
    return value is None or str(value).strip() in ("", "未提及")

def generate_parameter_tags(params: dict) -> List[str]:
    """
    根据参数生成提示标签。
    原则：只在参数中有"明确证据"时才打短板标签——
    官网没写的（未提及）不等于不支持，不做推断。
    """
    tags = []

    connectivity = params.get("connectivity", {}) or {}
    display = params.get("display", {}) or {}

    # --- n79 频段：仅当频段被明确列出（内容含 n+数字）且确实没有 n79 ---
    bands_5g = str(connectivity.get("5g_bands", ""))
    if not _is_empty(bands_5g):
        if re.search(r'n\d+', bands_5g, re.IGNORECASE) and "n79" not in bands_5g.lower():
            tags.append("不支持 n79 频段")

    # --- GPS：仅当明确写出"单频"或只写了 L1（无 L5/双频字样）时提示 ---
    gps_info = str(connectivity.get("gps", ""))
    if not _is_empty(gps_info):
        gps_lower = gps_info.lower()
        has_dual = ("双频" in gps_info) or ("l5" in gps_lower)
        if ("单频" in gps_info) or (("l1" in gps_lower) and not has_dual):
            tags.append("单频 GPS，定位精度一般")

        # --- 北斗：解析 B1I/B1C/B2a/B2b 等频段明细，按实际数量标注 ---
        if ("北斗" in gps_info) or ("BDS" in gps_info.upper()):
            band_count = _count_beidou_bands(gps_info)
            if band_count is None:
                # 未写频段明细时不推断；文案明确写了"四频/三频北斗"的除外
                if "四频" in gps_info:
                    tags.append("四频北斗")
                elif "三频" in gps_info:
                    tags.append("三频北斗")
            elif band_count >= 4:
                tags.append("四频北斗")
            elif band_count == 3:
                tags.append("三频北斗")
            elif band_count == 2:
                tags.append("双频北斗")
            else:
                tags.append("单频北斗")

    # --- 分辨率：按"宽×高"长边像素分级，避免 FHD+ 2640×1216 这类 1.5K 屏被误判 ---
    resolution = display.get("resolution", "")
    if not _is_empty(resolution):
        res_class = _resolution_class(resolution)
        if res_class is not None:
            if res_class < 1.5:
                tags.append("分辨率较低清晰度一般")
            elif res_class > 1.5:
                tags.append("清晰度很高")

    # --- 刷新率：解析第一个数字 ---
    refresh_rate = display.get("refresh_rate", "")
    if not _is_empty(refresh_rate):
        match = re.search(r'(\d+(?:\.\d+)?)', str(refresh_rate))
        if match:
            rate_num = float(match.group(1))
            if rate_num < 120:
                tags.append("刷新率较低可能体感较卡顿")
            elif rate_num > 120:
                tags.append("刷新率很高使用流畅")

    return tags

# ================= 14. 剥离内部键（下划线开头，避免污染评测 prompt） =================
def _strip_internal(data):
    if isinstance(data, dict):
        return {k: _strip_internal(v) for k, v in data.items() if not str(k).startswith("_")}
    if isinstance(data, list):
        return [_strip_internal(x) for x in data]
    return data

# ================= 15. 提取单个手机（并行分段 + 缓存 + 进度回调） =================
def extract_single_phone(url: str, phone_name: str = None,
                         progress_callback: Callable[[float, str], None] = None,
                         use_cache: bool = True) -> Tuple[Optional[dict], Optional[str], Optional[str]]:
    """
    提取单个手机的完整参数。
    返回 (result, phone_name, error)：成功时 error 为 None；
    失败时 result 为 None，error 为面向用户的错误说明。
    progress_callback(progress: 0~1, message: str) 用于界面实时反馈。
    """
    def report(p: float, msg: str):
        print(msg)
        if progress_callback:
            try:
                progress_callback(max(0.0, min(1.0, p)), msg)
            except Exception:
                pass

    start_time = time.time()
    report(0.02, f"🔍 开始处理：{url}")

    # --- 结果级缓存：同 URL 直接返回 ---
    if use_cache:
        cached = load_cache(url, "phone")
        if cached and cached.get("result"):
            report(1.0, f"⚡ 命中缓存：{cached.get('phone_name', '未知手机')}")
            return cached["result"], cached.get("phone_name"), None

    # --- 抓取网页（原文缓存 + 重试） ---
    markdown_data = None
    if use_cache:
        page_cache = load_cache(url, "page")
        if page_cache and page_cache.get("text"):
            markdown_data = page_cache["text"]
            report(0.10, "⚡ 网页原文命中缓存")
    if not markdown_data:
        report(0.05, "📡 正在抓取网页...")
        markdown_data = fetch_markdown(url)
        if not markdown_data:
            return None, None, ("网页抓取失败：请确认本地 Jina Reader 已启动"
                                f"（默认地址 {JINA_READER_URL}），且 URL 可访问")
        if use_cache:
            save_cache(url, "page", {"url": url, "text": markdown_data})
    report(0.10, "✅ 网页抓取完成")

    if not markdown_data.strip():
        return None, None, "网页内容为空，无法提取"

    # --- 识别型号 ---
    if not phone_name:
        report(0.15, "🔍 正在识别手机型号...")
        phone_name = extract_phone_name(markdown_data, url)
    report(0.20, f"📱 手机型号：{phone_name}")

    # --- 分段并行提取 ---
    chunks = split_markdown_smartly(markdown_data, max_chunk_size=5000)
    report(0.22, f"📄 内容分为 {len(chunks)} 段，并行提取中（最多 {MAX_PARALLEL_CHUNKS} 并发）...")

    schema = get_full_schema()
    results_by_index: Dict[int, dict] = {}
    chunk_errors: List[str] = []

    with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_CHUNKS, len(chunks))) as pool:
        future_map = {
            pool.submit(extract_from_chunk, chunk, phone_name, schema, i, len(chunks)): i
            for i, chunk in enumerate(chunks)
        }
        done = 0
        for fut in as_completed(future_map):
            idx = future_map[fut]
            try:
                results_by_index[idx] = fut.result()
            except Exception as e:
                chunk_errors.append(str(e)[:120])
                results_by_index[idx] = {}
            done += 1
            report(0.22 + 0.68 * done / len(chunks), f"📄 分段提取进度 {done}/{len(chunks)}")

    # 按分段顺序合并，保证冲突处理结果稳定
    all_results = [results_by_index[i] for i in sorted(results_by_index) if results_by_index[i]]

    if not all_results:
        detail = chunk_errors[0] if chunk_errors else "未知原因"
        if "认证失败" in detail:
            # 认证失败与余额无关，直接给出针对性提示
            return None, phone_name, detail
        return None, phone_name, f"所有分段提取失败：{detail}（请检查网络与账户余额）"

    # --- 合并 + 冲突检测 ---
    report(0.92, "🔄 正在合并结果（检测冲突）...")
    final_result: Dict[str, Any] = {}
    conflicts: List[dict] = []
    for result in all_results:
        final_result, conflicts = deep_merge_with_conflicts(final_result, result, conflicts)

    if conflicts:
        final_result["_conflicts"] = conflicts
        print(f"⚠️ 发现 {len(conflicts)} 处参数冲突")

    final_result = clean_empty_values(final_result)

    # --- 知识解读 + 标签 ---
    report(0.96, "📚 正在生成知识解读与标签...")
    kb = PhoneKnowledgeBase()
    knowledge_summary = kb.generate_knowledge_summary(final_result)
    if knowledge_summary:
        final_result["_knowledge_summary"] = knowledge_summary
        print("✅ 知识解读生成成功")
    else:
        print("⚠️ 未能生成知识解读")

    tags = generate_parameter_tags(final_result)
    if tags:
        final_result["_tags"] = tags
        print(f"🏷️ 生成 {len(tags)} 个标签")

    save_result(phone_name, final_result)
    if use_cache:
        save_cache(url, "phone", {"phone_name": phone_name, "result": final_result})

    total_time = time.time() - start_time
    report(1.0, f"✅ {phone_name} 提取完成，总耗时 {total_time:.1f} 秒")

    print("\n" + "=" * 60)
    print(f"📱 {phone_name} 完整参数")
    print("=" * 60)
    print(json.dumps(translate_keys(final_result), indent=2, ensure_ascii=False))

    return final_result, phone_name, None

def save_result(phone_name: str, result: dict) -> str:
    """保存提取结果到 output 目录"""
    output_dir = os.path.join(BASE_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)

    safe_name = str(phone_name).replace(' ', '_').replace('/', '_').replace('\\', '_')
    filename = os.path.join(output_dir, f"{safe_name}_params.json")

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n💾 参数已保存到 {filename}")
    return filename

# ================= 16. 生成单手机评测（使用 review_client） =================
def _compact(d: dict, keys: list) -> dict:
    """从字典中挑选非空字段组成精简摘要"""
    return {k: d[k] for k in keys if d.get(k) not in ("", None, "未提及")}

def _build_review_brief(phone_name: str, params: dict, knowledge_summary) -> dict:
    """
    把完整参数压缩成评测所需的简明摘要。
    直接塞原始JSON会让模型在冗长的官网原文（如两三百字的WiFi兼容列表）中迷失，
    导致评测空泛——这里只保留写作需要的字段，并带上知识库解读要点。
    """
    p = _strip_internal(params)
    basic = p.get("basic_info", {}) or {}
    processor = p.get("processor", {}) or {}
    display = p.get("display", {}) or {}
    battery = p.get("battery_charging", {}) or {}
    camera = p.get("camera", {}) or {}
    physical = p.get("physical", {}) or {}
    rear_main = camera.get("rear_main", {}) or {}
    tele = camera.get("rear_telephoto", {}) or {}
    wide = camera.get("rear_ultrawide", {}) or {}
    audio = p.get("audio", {}) or {}

    ks = knowledge_summary or {}
    chipset = ks.get("chipset", {}) or {}
    chip_info = chipset.get("info", {}) or {}

    brief = {
        "机型": phone_name,
        "价格": basic.get("price") or "未提供",
        "外观": _compact(physical, ["dimensions", "weight", "thickness", "material", "water_resistance"]),
        "性能": {
            "芯片": processor.get("soc", "未知"),
            "市场定位": chip_info.get("tier", ""),
            "知识库评价": _compact(chip_info, ["description", "strengths", "weaknesses"]),
        },
        "屏幕": _compact(display, ["size", "type", "resolution", "refresh_rate", "peak_brightness"]),
        "屏幕解读": ks.get("display", {}),
        "电池与充电": _compact(battery, ["capacity", "wired_charging", "wireless_charging"]),
        "续航解读": ks.get("battery", {}),
        "影像": {
            "主摄": _compact(rear_main, ["resolution", "aperture", "sensor", "sensor_size", "ois"]),
            "长焦": _compact(tele, ["resolution", "aperture", "optical_zoom"]),
            "超广角": _compact(wide, ["resolution", "aperture", "fov"]),
            "视频": _compact(camera.get("video", {}) or {}, ["rear_max", "slow_motion"]),
        },
        "其他": _compact(audio, ["speakers", "headphone_jack"]),
        # 程序基于参数自动检测的短板/亮点，要求评测必须向读者解释
        "系统检测标签": params.get("_tags", []),
    }
    # 移除整体为空的分区，进一步降噪
    return {k: v for k, v in brief.items() if v not in ("", None, {}, [])}

def generate_review(phone_name: str, params: dict, knowledge_summary) -> Optional[str]:
    """生成单手机专业评测"""
    brief = _build_review_brief(phone_name, params, knowledge_summary)

    prompt = f"""
    你是一位以客观中立著称的专业手机评测编辑，请基于下面的参数摘要为 {phone_name} 撰写一篇评测。

    写作要求：
    1. 用 Markdown 小标题分节，建议结构：**开篇一句话定位** → **性能** → **屏幕** → **续航与充电** → **影像** → **总结**，正文 400-600 字
    2. 只依据摘要中的数据说话，严禁编造摘要之外的参数；价格显示"未提供"就不要谈价格
    3. "系统检测标签"是程序从官方参数中自动检测出的短板或亮点，如非空，必须用大白话向读者解释其含义和影响
    4. "知识库评价"是行业通用认知，可作为专业背景引用
    5. 面向普通消费者，专业但易懂；不与其他机型对比；不带营销吹嘘语气
    6. 直接输出评测正文，不要输出任何思考过程、前言或额外说明

    参数摘要：
    {json.dumps(brief, ensure_ascii=False, indent=2)}
    """
    try:
        return call_llm_text(review_client, prompt, temperature=0.5, timeout=180,
                             max_tokens=8000, tag="单机评测")
    except Exception as e:
        print(f"生成评测失败: {str(e)[:100]}")
        return None

# ================= 17. 生成多手机对比评测（使用 review_client） =================
def generate_comparison(phones_data: list) -> Optional[str]:
    """生成多手机对比评测"""
    if len(phones_data) < 2:
        return "至少需要两部手机才能进行对比。"

    phones_brief = []
    for phone in phones_data:
        params = phone["params"]
        # 芯片定位来自知识库，给模型更直接的对比依据
        chipset = ((phone.get("knowledge_summary") or {}).get("chipset") or {})
        chip_info = chipset.get("info", {}) or {}
        # 主摄：参数页通常不写传感器尺寸，缺失时回退到"像素+光圈"
        rear_main = params.get("camera", {}).get("rear_main", {}) or {}
        main_cam = rear_main.get("sensor_size") or " ".join(
            x for x in (rear_main.get("resolution"), rear_main.get("aperture")) if x
        ) or "未知"
        tele = params.get("camera", {}).get("rear_telephoto", {}) or {}
        tele_cam = " ".join(
            x for x in (tele.get("optical_zoom"), tele.get("resolution")) if x
        ) or "未知"
        brief = {
            "name": phone["phone_name"],
            "key_params": {
                "处理器": params.get("processor", {}).get("soc", "未知"),
                "芯片定位": f"{chip_info.get('tier', '未知')}（评分 {chip_info.get('performance_score', '?')}/100）" if chip_info else "未知",
                "屏幕": f"{params.get('display', {}).get('size', '?')} {params.get('display', {}).get('resolution', '?')} {params.get('display', {}).get('refresh_rate', '?')}",
                "电池": params.get("battery_charging", {}).get("capacity", "未知"),
                "充电": params.get("battery_charging", {}).get("wired_charging", "未知"),
                "主摄": main_cam,
                "长焦": tele_cam,
                "价格": params.get("basic_info", {}).get("price", "未知")
            },
            # 短板标签直接给模型，作为对比时的客观依据
            "参数标签": params.get("_tags", [])
        }
        phones_brief.append(brief)

    prompt = f"""
    你是一位以客观中立著称的专业手机评测编辑，请根据以下多部手机的参数摘要，生成一份对比评测。

    写作要求：
    1. 先用一张 Markdown 表格汇总关键参数（处理器/屏幕/电池/充电/主摄/价格），差异明显处加粗
    2. 按 性能 → 屏幕 → 续航充电 → 影像 → 性价比 分节展开分析，明确指出每一项谁更强、依据是什么参数
    3. "参数标签"是程序从官方参数自动检测的短板/亮点，分析时必须客观引用并向读者解释
    4. 只依据给定数据，严禁编造；某项数据为"未知"时直接说明缺失，不要猜测
    5. 最后按用户群体给出推荐（如：游戏玩家/拍照用户/续航焦虑用户/预算优先），没有明显差异时如实说明
    6. 直接输出正文（Markdown），不要输出思考过程或额外说明

    手机信息：
    {json.dumps(phones_brief, ensure_ascii=False, indent=2)}
    """
    try:
        return call_llm_text(review_client, prompt, temperature=0.3, timeout=180,
                             max_tokens=8000, tag="对比评测")
    except Exception as e:
        print(f"生成对比评测失败: {str(e)[:100]}")
        return None

# ================= 18. 交互式输入（命令行模式） =================
def interactive_mode():
    """交互式输入模式"""
    print("=" * 60)
    print("📱 AI手机参数提取器")
    print("=" * 60)

    urls = []
    print("\n请输入手机参数页URL（每行一个，输入空行结束）：")

    while True:
        url = input().strip()
        if not url:
            break
        if url.startswith("http"):
            urls.append(url)
        else:
            print(f"⚠️ 无效URL（跳过）：{url}")

    if not urls:
        print("❌ 未输入有效URL")
        return

    print(f"\n✅ 共输入 {len(urls)} 个URL，开始处理...")

    results = []
    for i, url in enumerate(urls, 1):
        print(f"\n[{i}/{len(urls)}] 处理中...")
        result, phone_name, error = extract_single_phone(url)
        if result:
            results.append({
                "url": url,
                "phone_name": phone_name,
                "parameters": result
            })
        else:
            print(f"❌ 提取失败：{error}")

    print(f"\n{'=' * 60}")
    print(f"✅ 处理完成！成功提取 {len(results)}/{len(urls)} 个手机参数")
    print(f"{'=' * 60}")

    return results

# ================= 19. 主程序 =================
def main():
    """主程序入口"""
    interactive_mode()

if __name__ == "__main__":
    main()
