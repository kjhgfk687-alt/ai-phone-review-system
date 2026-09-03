import requests
import json
from openai import OpenAI
import re
import time
import os
from typing import Dict, Any, List, Optional

from knowledge_base import PhoneKnowledgeBase

# ================= 1. 配置 DeepSeek API（双 Key） =================
# Key 1：参数提取专用
client = OpenAI(
    api_key="sk-你的apikey",
    base_url="https://api.deepseek.com"
)

# Key 2：评测生成与对比专用
review_client = OpenAI(
    api_key="sk-你的apikey",
    base_url="https://api.deepseek.com"
)

# ================= 2. 抓取本地 Jina Reader 数据 =================
def get_clean_markdown(url):
    """抓取网页内容"""
    local_reader = f"http://127.0.0.1:3001/{url}"
    headers = {"X-Respond-With": "markdown"}
    try:
        print(f"正在抓取网页: {url} ...")
        res = requests.get(local_reader, headers=headers, timeout=60)
        res.raise_for_status()
        res.encoding = 'utf-8'
        return res.text
    except Exception as e:
        print(f"本地 Reader 调用失败: {e}")
        return None

# ================= 3. 自动识别手机名称 =================
def extract_phone_name(markdown_text, url):
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
        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"},
            timeout=30
        )
        result = json.loads(response.choices[0].message.content)
        full_name = result.get("full_name", "未知手机")
        print(f"✅ 识别到手机：{full_name}")
        return full_name
    except Exception as e:
        print(f"⚠️ 识别失败，从URL提取名称")
        return extract_name_from_url(url)

def extract_name_from_url(url):
    """从URL中提取可能的手机名称"""
    path = url.replace("https://", "").replace("http://", "")
    parts = [p for p in path.split("/") if p and p not in ["cn", "www", "specs", "smartphones", "#"]]

    if parts:
        name = parts[-1].replace("-", " ").replace("_", " ").title()
        return name
    return "未知手机"

# ================= 4. 参数架构 =================
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

# ================= 5. 智能分段 =================
def split_markdown_smartly(markdown_text, max_chunk_size=5000):
    """智能分段"""
    if len(markdown_text) <= max_chunk_size:
        return [markdown_text]

    chunks = []
    current_chunk = ""

    lines = markdown_text.split('\n')

    for line in lines:
        if len(current_chunk) + len(line) > max_chunk_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = line
        else:
            current_chunk += line + '\n'

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

# ================= 6. 修复截断的JSON =================
def fix_truncated_json(json_str):
    """修复被截断的JSON"""
    depth = 0
    in_string = False
    escape = False
    last_complete = 0

    for i, char in enumerate(json_str):
        if escape:
            escape = False
            continue
        if char == '\\':
            escape = True
            continue
        if char == '"':
            in_string = not in_string
        elif not in_string:
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    last_complete = i + 1
            elif char == ',' and depth == 1:
                last_complete = i

    if last_complete > 0:
        fixed = json_str[:last_complete] + '}'
        return fixed
    return None

# ================= 7. 提取单个分段 =================
def extract_from_chunk(chunk_text, phone_name, schema, chunk_index, total_chunks):
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
        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"},
            timeout=120,
            max_tokens=8000
        )

        content = response.choices[0].message.content
        result = json.loads(content)
        print(f"  ✅ 第 {chunk_index + 1} 段完成")
        return result

    except json.JSONDecodeError as e:
        print(f"  ⚠️ 第 {chunk_index + 1} 段JSON解析失败，尝试修复...")
        try:
            content = response.choices[0].message.content
            fixed_content = fix_truncated_json(content)
            if fixed_content:
                result = json.loads(fixed_content)
                print(f"  ✅ 修复成功，提取到部分参数")
                return result
            else:
                print(f"  ❌ 无法修复")
                return {}
        except Exception as fix_error:
            print(f"  ❌ 修复失败: {str(fix_error)[:50]}")
            return {}
    except Exception as e:
        print(f"  ❌ 第 {chunk_index + 1} 段失败: {str(e)[:50]}")
        return {}

# ================= 8. 深度合并（带冲突检测） =================
def deep_merge_with_conflicts(dict1, dict2, conflicts=None, path=""):
    """
    深度合并两个字典，并检测同一路径下不同非空值之间的冲突。
    conflicts: 列表，用于记录冲突信息，每个元素为字典 {path, value1, value2, final_value}
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

# ================= 9. 清理空值 =================
def clean_empty_values(data):
    """递归移除空值和未提及，保留内部键"""
    if isinstance(data, dict):
        cleaned = {}
        for key, value in data.items():
            if key.startswith('_'):
                cleaned[key] = value
                continue
            if value not in ["", "未提及", "无", None, [], {}]:
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

# ================= 10. 键名翻译映射 =================
def translate_keys(data):
    """将英文键名翻译成中文，跳过内部键"""
    key_translation = {
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

    if isinstance(data, dict):
        translated = {}
        for key, value in data.items():
            if key.startswith('_'):
                translated[key] = value
                continue
            new_key = key_translation.get(key, key)
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

# ================= 11. 生成参数标签 =================
def generate_parameter_tags(params):
    """根据参数生成提示标签"""
    tags = []

    connectivity = params.get("connectivity", {})
    display = params.get("display", {})

    bands_5g = connectivity.get("5g_bands", "")
    if bands_5g and bands_5g != "未提及":
        if "n79" not in bands_5g.lower():
            tags.append("不支持 n79 频段")

    gps_info = connectivity.get("gps", "")
    if gps_info and gps_info != "未提及":
        gps_lower = gps_info.lower()
        if "双频" not in gps_lower and "l5" not in gps_lower:
            tags.append("无双频 GPS")
        if "四频" not in gps_lower and "b2b" not in gps_lower:
            tags.append("无四频北斗")

    resolution = display.get("resolution", "")
    if resolution and resolution != "未提及":
        def resolution_to_num(res_str):
            res_lower = res_str.lower()
            if "4k" in res_lower:
                return 4.0
            elif "2k" in res_lower or "qhd" in res_lower:
                return 2.0
            elif "1.5k" in res_lower:
                return 1.5
            elif "1080" in res_lower or "fhd" in res_lower:
                return 1.0
            elif "720" in res_lower or "hd" in res_lower:
                return 0.75
            else:
                return None

        res_num = resolution_to_num(resolution)
        if res_num is not None:
            if res_num < 1.5:
                tags.append("分辨率较低清晰度一般")
            elif res_num > 1.5:
                tags.append("清晰度很高")

    refresh_rate = display.get("refresh_rate", "")
    if refresh_rate and refresh_rate != "未提及":
        match = re.search(r'(\d+(?:\.\d+)?)', refresh_rate)
        if match:
            rate_num = float(match.group(1))
            if rate_num < 120:
                tags.append("刷新率较低可能体感较卡顿")
            elif rate_num > 120:
                tags.append("刷新率很高使用流畅")

    return tags

# ================= 12. 提取单个手机 =================
def extract_single_phone(url, phone_name=None):
    """提取单个手机的完整参数，并附带知识总结、冲突信息和标签"""
    print(f"\n{'=' * 60}")
    print(f"🔍 开始处理：{url}")
    print(f"{'=' * 60}")

    start_time = time.time()

    markdown_data = get_clean_markdown(url)
    if not markdown_data:
        print("❌ 无法获取网页内容")
        return None, None

    if not phone_name:
        phone_name = extract_phone_name(markdown_data, url)

    print(f"📱 手机型号：{phone_name}")

    chunks = split_markdown_smartly(markdown_data, max_chunk_size=5000)
    print(f"📄 网页内容已分为 {len(chunks)} 段")

    schema = get_full_schema()

    all_results = []
    for i, chunk in enumerate(chunks):
        result = extract_from_chunk(chunk, phone_name, schema, i, len(chunks))
        if result:
            all_results.append(result)

    if not all_results:
        print("❌ 所有分段提取均失败")
        return None, phone_name

    print("\n🔄 正在合并结果（检测冲突）...")
    final_result = {}
    conflicts = []
    for result in all_results:
        final_result, conflicts = deep_merge_with_conflicts(final_result, result, conflicts)

    if conflicts:
        final_result["_conflicts"] = conflicts
        print(f"⚠️ 发现 {len(conflicts)} 处参数冲突")

    final_result = clean_empty_values(final_result)

    print("\n📚 正在生成知识解读...")
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

    total_time = time.time() - start_time
    print(f"\n⏱️ 总耗时: {total_time:.1f}秒")

    print("\n" + "=" * 60)
    print(f"📱 {phone_name} 完整参数")
    print("=" * 60)

    translated_result = translate_keys(final_result)
    print(json.dumps(translated_result, indent=2, ensure_ascii=False))

    return final_result, phone_name

def save_result(phone_name, result):
    """保存提取结果"""
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    safe_name = phone_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
    filename = os.path.join(output_dir, f"{safe_name}_params.json")

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n💾 参数已保存到 {filename}")
    return filename

# ================= 13. 生成单手机评测（使用 review_client） =================
def generate_review(phone_name, params, knowledge_summary):
    """生成单手机专业评测"""
    price_info = ""
    basic_info = params.get("basic_info", {})
    if "price" in basic_info and basic_info["price"] not in ["", "未提及", None]:
        price_info = basic_info["price"]

    prompt = f"""
    你是一位专业的手机评测编辑，请根据以下手机参数和知识库信息，为 {phone_name} 撰写一段专业、客观的评测。
    要求：
    - 语言简洁、专业，避免主观情绪
    - 涵盖外观手感、性能、屏幕、续航充电、拍照、总结
    - 重点突出知识库中的专业解读
    - 字数在300-500字之间
    - 提及价格（如果有价格信息）
    - 不要与其他手机对比

    手机参数：
    {json.dumps(params, ensure_ascii=False, indent=2)}

    知识解读：
    {json.dumps(knowledge_summary, ensure_ascii=False, indent=2)}

    价格信息：{price_info if price_info else "未提供"}

    请直接返回评测正文，不要包含标题或额外说明。
    """
    try:
        response = review_client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            timeout=60,
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"生成评测失败: {e}")
        return None

# ================= 14. 生成多手机对比评测（使用 review_client） =================
def generate_comparison(phones_data):
    """生成多手机对比评测"""
    if len(phones_data) < 2:
        return "至少需要两部手机才能进行对比。"

    phones_brief = []
    for phone in phones_data:
        brief = {
            "name": phone["phone_name"],
            "key_params": {
                "处理器": phone["params"].get("processor", {}).get("soc", "未知"),
                "屏幕": f"{phone['params'].get('display', {}).get('size', '?')} {phone['params'].get('display', {}).get('resolution', '?')} {phone['params'].get('display', {}).get('refresh_rate', '?')}",
                "电池": phone["params"].get("battery_charging", {}).get("capacity", "未知"),
                "充电": phone["params"].get("battery_charging", {}).get("wired_charging", "未知"),
                "主摄": phone["params"].get("camera", {}).get("rear_main", {}).get("sensor_size", "未知"),
                "价格": phone["params"].get("basic_info", {}).get("price", "未知")
            }
        }
        phones_brief.append(brief)

    prompt = f"""
    你是一位专业的手机评测编辑，请根据以下多部手机的参数和知识解读，生成一份客观、专业的对比评测。
    要求：
    - 语言简洁、专业，避免主观情绪
    - 包含性能、屏幕、续航充电、拍照、价格等维度的对比
    - 使用 Markdown 表格展示关键参数差异
    - 最后给出综合推荐（适合不同用户群体）
    - 字数不限，但要全面

    手机信息：
    {json.dumps(phones_brief, ensure_ascii=False, indent=2)}

    请直接返回对比评测正文，包含 Markdown 表格。
    """
    try:
        response = review_client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            timeout=120,
            max_tokens=2000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"生成对比评测失败: {e}")
        return None

# ================= 15. 交互式输入（保留） =================
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
        result, phone_name = extract_single_phone(url)
        if result:
            results.append({
                "url": url,
                "phone_name": phone_name,
                "parameters": result
            })

    print(f"\n{'=' * 60}")
    print(f"✅ 处理完成！成功提取 {len(results)}/{len(urls)} 个手机参数")
    print(f"{'=' * 60}")

    return results

# ================= 16. 主程序 =================
def main():
    """主程序入口"""
    results = interactive_mode()

if __name__ == "__main__":
    main()