import streamlit as st
import sys
import os
import json
import time

# 导入核心函数
sys.path.append(os.path.dirname(__file__))
from cshi import extract_single_phone, translate_keys, generate_review, generate_comparison

# 页面配置
st.set_page_config(
    page_title="AI手机参数提取与评测系统",
    page_icon="📱",
    layout="wide"
)

st.title("📱 AI手机参数提取与评测系统")
st.markdown("---")

# 初始化会话状态
if "phones" not in st.session_state:
    st.session_state.phones = []          # 存储提取的手机信息
if "review_results" not in st.session_state:
    st.session_state.review_results = {}  # 存储评测结果，键为手机名索引
if "comparison_result" not in st.session_state:
    st.session_state.comparison_result = None

# 输入区域
url_input = st.text_area(
    "输入手机参数页URL（每行一个）",
    height=100,
    placeholder="https://www.oppo.com/cn/smartphones/series-find-x/find-x9s-pro/specs/"
)

def display_params(params, container, indent=0, conflicts=None):
    """
    递归展示参数，跳过下划线开头的内部键。
    如果 conflicts 不为空，则在有冲突的参数后显示警告图标。
    """
    if conflicts is None:
        conflicts = []

    for key, value in params.items():
        if key.startswith('_'):
            continue

        # 简单冲突检测：若冲突路径中包含当前键名，则显示警告
        conflict_found = any(key in c["path"] for c in conflicts)

        if isinstance(value, dict):
            container.markdown(f"{'  ' * indent}**{key}**")
            display_params(value, container, indent + 1, conflicts)
        elif isinstance(value, list):
            container.markdown(f"{'  ' * indent}**{key}**")
            for item in value:
                if isinstance(item, dict):
                    display_params(item, container, indent + 1, conflicts)
                else:
                    container.markdown(f"{'  ' * indent}- {item}")
        else:
            if value and value != "未提及":
                if conflict_found:
                    container.markdown(
                        f"{'  ' * indent}**{key}**: {value} <span title='⚠️ 此参数在不同来源中存在冲突，已自动选择较可信值'>⚠️</span>",
                        unsafe_allow_html=True
                    )
                else:
                    container.markdown(f"{'  ' * indent}**{key}**: {value}")

def display_knowledge_summary(summary, container):
    """展示知识总结（中文键名）"""
    if not summary:
        return

    container.subheader("📚 专业知识解读")

    display_key_map = {
        "resolution_desc": "分辨率",
        "refresh_desc": "刷新率",
        "screen_type_desc": "屏幕类型",
        "brightness_desc": "亮度",
        "capacity_desc": "电池容量",
        "charging_desc": "有线充电",
        "wireless_charging_desc": "无线充电",
        "sensor_desc": "传感器",
        "aperture_desc": "光圈",
        "pixel_desc": "像素",
        "zoom_desc": "变焦",
    }

    # 芯片解读
    if "chipset" in summary:
        chipset = summary["chipset"]
        name = chipset.get("name", "")
        info = chipset.get("info", {})
        container.markdown("### 🔥 芯片性能")
        container.markdown(f"**{name}**")
        if info.get("description"):
            container.markdown(f"📝 {info['description']}")
        if info.get("performance_score"):
            container.markdown(f"⭐ 性能评分：{info['performance_score']}/100")
        if info.get("tier"):
            container.markdown(f"🏷️ 定位：{info['tier']}")
        if info.get("strengths"):
            container.markdown("**优势**：")
            for s in info["strengths"]:
                container.markdown(f"- {s}")
        if info.get("weaknesses"):
            container.markdown("**注意**：")
            for w in info["weaknesses"]:
                container.markdown(f"- {w}")
        if info.get("typical_price_range"):
            container.markdown(f"💰 常见价格区间：{info['typical_price_range']}")
        if info.get("note"):
            container.markdown(f"📌 {info['note']}")
        container.markdown("---")

    # 屏幕解读
    if "display" in summary:
        container.markdown("### 🖥️ 屏幕表现")
        for key, desc in summary["display"].items():
            label = display_key_map.get(key, key.replace('_desc', ''))
            container.markdown(f"- **{label}**：{desc}")
        container.markdown("---")

    # 电池解读
    if "battery" in summary:
        container.markdown("### 🔋 电池与充电")
        for key, desc in summary["battery"].items():
            label = display_key_map.get(key, key.replace('_desc', ''))
            container.markdown(f"- **{label}**：{desc}")
        container.markdown("---")

    # 相机解读
    if "camera" in summary:
        container.markdown("### 📷 相机能力")
        for key, desc in summary["camera"].items():
            label = display_key_map.get(key, key.replace('_desc', ''))
            container.markdown(f"- **{label}**：{desc}")
        container.markdown("---")

    # 品牌解读
    if "brand" in summary:
        brand = summary["brand"]
        container.markdown("### 🏢 品牌特色")
        if brand.get("brand_positioning"):
            container.markdown(f"**定位**：{brand['brand_positioning']}")
        if brand.get("target_users"):
            container.markdown(f"**目标用户**：{brand['target_users']}")
        if brand.get("key_technologies"):
            container.markdown("**核心技术**：")
            for tech in brand["key_technologies"]:
                container.markdown(f"- {tech}")
        if brand.get("new_technologies_2025_2026"):
            container.markdown("**最新技术（2025-2026）**：")
            for tech in brand["new_technologies_2025_2026"]:
                container.markdown(f"- {tech}")
        container.markdown("---")

# 提取按钮
if st.button("🚀 开始提取", type="primary"):
    urls = [u.strip() for u in url_input.split('\n') if u.strip().startswith("http")]

    if not urls:
        st.error("请输入至少一个有效URL")
    else:
        st.info(f"共{len(urls)}个URL，开始处理...")
        progress_bar = st.progress(0)

        for i, url in enumerate(urls):
            progress_bar.progress(i / len(urls))
            start_time = time.time()
            result, phone_name = extract_single_phone(url)
            elapsed = time.time() - start_time

            if result:
                # 翻译键名（内部字段保留）
                translated_result = translate_keys(result)
                # 获取知识总结
                knowledge_summary = result.get("_knowledge_summary")
                # 获取冲突信息
                conflicts = result.get("_conflicts", [])
                # 获取标签
                tags = result.get("_tags", [])

                # 保存到 session_state
                st.session_state.phones.append({
                    "phone_name": phone_name,
                    "params": result,
                    "translated_params": translated_result,
                    "knowledge_summary": knowledge_summary,
                    "conflicts": conflicts,
                    "tags": tags,
                })
                st.success(f"✅ {phone_name} 提取成功（{elapsed:.1f}秒）")
            else:
                st.error(f"❌ 提取失败: {url}")

        progress_bar.progress(1.0)

# 如果已有手机数据，展示信息
if st.session_state.phones:
    st.markdown("---")
    st.subheader("📋 提取结果")

    for idx, phone in enumerate(st.session_state.phones):
        st.markdown(f"### 📱 {phone['phone_name']}")

        # 展示参数标签
        if phone.get('tags'):
            st.markdown("**🏷️ 参数标签**")
            tags_html = " ".join([
                f"<span style='background-color:#f0f0f0; padding:4px 8px; margin-right:6px; border-radius:4px; font-size:0.9em;'>{tag}</span>"
                for tag in phone['tags']
            ])
            st.markdown(tags_html, unsafe_allow_html=True)
            st.markdown("")

        # 展示知识总结
        if phone['knowledge_summary']:
            with st.expander("📚 专业知识解读", expanded=False):
                display_knowledge_summary(phone['knowledge_summary'], st)

        # 展示参数（带冲突提示）
        with st.expander("🔍 完整参数", expanded=False):
            display_params(phone['translated_params'], st, conflicts=phone.get('conflicts', []))

        # 评测按钮
        button_key = f"review_btn_{idx}"
        if st.button(f"📝 生成评测", key=button_key):
            with st.spinner("正在生成评测..."):
                review = generate_review(phone['phone_name'], phone['params'], phone['knowledge_summary'])
                if review:
                    st.session_state.review_results[idx] = review
                else:
                    st.error("评测生成失败，请重试")

        # 显示评测结果
        if idx in st.session_state.review_results:
            st.markdown("**📝 专业评测**")
            st.markdown(st.session_state.review_results[idx])

        st.markdown("---")

    # 对比评测按钮（至少两部手机）
    if len(st.session_state.phones) >= 2:
        if st.button("🔍 生成对比评测", key="compare_btn"):
            with st.spinner("正在生成对比评测..."):
                phones_data = [
                    {
                        "phone_name": p["phone_name"],
                        "params": p["params"],
                        "knowledge_summary": p["knowledge_summary"]
                    }
                    for p in st.session_state.phones
                ]
                comparison = generate_comparison(phones_data)
                if comparison:
                    st.session_state.comparison_result = comparison
                else:
                    st.error("对比评测生成失败，请重试")

        if st.session_state.comparison_result:
            st.subheader("🔍 对比评测")
            st.markdown(st.session_state.comparison_result)