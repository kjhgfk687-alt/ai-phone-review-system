import streamlit as st
import sys
import os
import json
import time
import html
import uuid

# 导入核心函数
sys.path.append(os.path.dirname(__file__))
from cshi import (
    extract_single_phone, generate_review, generate_comparison,
    KEY_TRANSLATION, clear_cache, EXTRACT_API_KEY
)

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
    st.session_state.phones = []          # 已提取的手机列表
if "review_results" not in st.session_state:
    st.session_state.review_results = {}  # 评测结果，键为手机唯一 id
if "comparison_result" not in st.session_state:
    st.session_state.comparison_result = None

# ================= 侧边栏 =================
with st.sidebar:
    st.header("⚙️ 设置")
    use_cache = st.checkbox("启用缓存（同一 URL 秒出结果）", value=True)
    st.caption("缓存目录：`cache/`，可在下方手动清空")

    if st.button("🗑️ 清空全部结果", use_container_width=True):
        st.session_state.phones = []
        st.session_state.review_results = {}
        st.session_state.comparison_result = None
        st.rerun()

    if st.button("🧹 清除缓存", use_container_width=True):
        removed = clear_cache()
        st.toast(f"已清除 {removed} 个缓存文件", icon="🧹")

    st.divider()
    # API Key 配置状态检查（避免误把认证失败当成余额问题）
    key_ok = EXTRACT_API_KEY and not EXTRACT_API_KEY.startswith("sk-你的") \
        and EXTRACT_API_KEY != "sk-not-configured"
    if key_ok:
        st.success("API Key 已配置 ✅", icon="🔑")
    else:
        st.error(
            "API Key 未配置 ❌\n\n"
            "请把真实的 DeepSeek Key 填入项目根目录 `.env` 的 `DEEPSEEK_API_KEY=` 一行，"
            "保存后重启程序。当前提取功能无法使用。",
            icon="🔑"
        )
    st.caption(
        "使用前提：\n\n"
        "1. 本地 Jina Reader 已启动（默认 `http://127.0.0.1:3001/`）\n\n"
        "2. 项目根目录 `.env` 已配置 `DEEPSEEK_API_KEY`"
    )

# ================= 展示辅助函数 =================
def display_params(params, container, indent=0, conflict_paths=None, path=""):
    """
    递归展示参数（遍历英文原始键，展示时翻译为中文）。
    conflict_paths 为发生冲突的完整参数路径集合（英文路径，与提取时记录一致），
    只有精确命中路径的参数才显示 ⚠️，避免同名键误报。
    """
    if conflict_paths is None:
        conflict_paths = set()

    indent_html = "&nbsp;&nbsp;&nbsp;&nbsp;" * indent

    for key, value in params.items():
        if str(key).startswith('_'):
            continue

        current_path = f"{path}.{key}" if path else key
        label = KEY_TRANSLATION.get(key, key)
        conflict_found = current_path in conflict_paths

        if isinstance(value, dict):
            container.markdown(f"{indent_html}**{label}**", unsafe_allow_html=True)
            display_params(value, container, indent + 1, conflict_paths, current_path)
        elif isinstance(value, list):
            container.markdown(f"{indent_html}**{label}**", unsafe_allow_html=True)
            for item in value:
                if isinstance(item, dict):
                    display_params(item, container, indent + 1, conflict_paths, current_path)
                else:
                    container.markdown(f"{indent_html}- {item}")
        else:
            if value and value != "未提及":
                if conflict_found:
                    container.markdown(
                        f"{indent_html}**{label}**: {html.escape(str(value))} "
                        "<span title='此参数在不同分段提取结果中存在差异，已保留首个非空值，请注意核对'>⚠️</span>",
                        unsafe_allow_html=True
                    )
                else:
                    container.markdown(f"{indent_html}**{label}**: {html.escape(str(value))}", unsafe_allow_html=True)

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

# ================= 输入与提取 =================
url_input = st.text_area(
    "输入手机参数页URL（每行一个）",
    height=100,
    placeholder="https://www.oppo.com/cn/smartphones/series-find-x/find-x9s-pro/specs/"
)

if st.button("🚀 开始提取", type="primary"):
    urls = [u.strip() for u in url_input.split('\n') if u.strip().startswith("http")]

    if not urls:
        st.error("请输入至少一个有效URL（以 http 开头）")
    else:
        existing_urls = {p.get("url") for p in st.session_state.phones}
        progress_bar = st.progress(0.0, text="准备中...")

        for i, url in enumerate(urls):
            if url in existing_urls:
                st.info(f"⏭️ 已跳过（本次会话已提取过）：{url}")
                continue

            with st.status(f"📡 [{i + 1}/{len(urls)}] 正在处理：{url}", expanded=True) as status:
                def update_progress(p, msg):
                    progress_bar.progress(p, text=f"{msg}")

                start_time = time.time()
                result, phone_name, error = extract_single_phone(
                    url, use_cache=use_cache, progress_callback=update_progress
                )

                if result:
                    st.session_state.phones.append({
                        "id": uuid.uuid4().hex,
                        "url": url,
                        "phone_name": phone_name,
                        "params": result,
                        "knowledge_summary": result.get("_knowledge_summary"),
                        "conflicts": result.get("_conflicts", []),
                        "tags": result.get("_tags", []),
                    })
                    status.update(
                        label=f"✅ {phone_name} 提取成功（{time.time() - start_time:.1f}秒）",
                        state="complete", expanded=False
                    )
                else:
                    status.update(label=f"❌ 提取失败：{url}", state="error", expanded=True)
                    st.error(f"提取失败：{error or '未知错误'}")

        progress_bar.progress(1.0, text="全部处理完成")

# ================= 提取结果展示 =================
if st.session_state.phones:
    st.markdown("---")
    st.subheader(f"📋 提取结果（共 {len(st.session_state.phones)} 部）")

    for phone in st.session_state.phones:
        phone_id = phone["id"]
        name = phone["phone_name"]

        # 标题行 + 删除按钮
        title_col, del_col = st.columns([6, 1])
        title_col.markdown(f"### 📱 {name}")
        if del_col.button("🗑️ 删除", key=f"del_{phone_id}"):
            st.session_state.phones = [p for p in st.session_state.phones if p["id"] != phone_id]
            st.session_state.review_results.pop(phone_id, None)
            st.rerun()

        # 参数标签
        if phone.get('tags'):
            st.markdown("**🏷️ 参数标签**")
            tags_html = " ".join([
                f"<span style='background-color:#f0f0f0; padding:4px 8px; margin-right:6px; "
                f"border-radius:4px; font-size:0.9em;'>{html.escape(str(tag))}</span>"
                for tag in phone['tags']
            ])
            st.markdown(tags_html, unsafe_allow_html=True)
            st.markdown("")

        # 知识总结
        if phone['knowledge_summary']:
            with st.expander("📚 专业知识解读", expanded=False):
                display_knowledge_summary(phone['knowledge_summary'], st)

        # 完整参数（冲突提示精确到参数路径）
        conflict_count = len(phone.get('conflicts', []))
        expander_title = f"🔍 完整参数" + (f"（{conflict_count} 处冲突 ⚠️）" if conflict_count else "")
        with st.expander(expander_title, expanded=False):
            if conflict_count:
                st.caption("⚠️ 表示该参数在不同分段提取结果中存在差异，已保留首个非空值")
            conflict_paths = {c["path"] for c in phone.get('conflicts', [])}
            display_params(phone['params'], st, conflict_paths=conflict_paths)

        # 操作行：生成评测 + 下载参数
        action_col1, action_col2 = st.columns(2)
        if action_col1.button(f"📝 生成评测：{name}", key=f"review_{phone_id}"):
            with st.spinner("正在生成评测..."):
                review = generate_review(phone['phone_name'], phone['params'], phone['knowledge_summary'])
            if review:
                st.session_state.review_results[phone_id] = review
            else:
                st.error("评测生成失败，请重试（可在终端查看具体原因）")

        params_json = json.dumps(phone['params'], ensure_ascii=False, indent=2)
        safe_name = str(name).replace(' ', '_').replace('/', '_')
        action_col2.download_button(
            "⬇️ 下载参数 JSON",
            data=params_json,
            file_name=f"{safe_name}_params.json",
            mime="application/json",
            key=f"dl_params_{phone_id}",
            use_container_width=True
        )

        # 评测结果
        if phone_id in st.session_state.review_results:
            review_md = st.session_state.review_results[phone_id]
            st.markdown("**📝 专业评测**")
            st.markdown(review_md)
            st.download_button(
                "⬇️ 下载评测（Markdown）",
                data=review_md,
                file_name=f"{safe_name}_评测.md",
                mime="text/markdown",
                key=f"dl_review_{phone_id}"
            )

        st.markdown("---")

# ================= 对比评测 =================
if len(st.session_state.phones) >= 2:
    st.subheader("🔍 对比评测")

    phone_labels = {p["id"]: p["phone_name"] for p in st.session_state.phones}
    selected_ids = st.multiselect(
        "选择要对比的手机（建议 2-4 部）",
        options=list(phone_labels.keys()),
        default=list(phone_labels.keys()),
        format_func=lambda i: phone_labels[i]
    )

    if st.button("🔍 生成对比评测", disabled=len(selected_ids) < 2):
        with st.spinner("正在生成对比评测..."):
            phones_data = [
                {
                    "phone_name": p["phone_name"],
                    "params": p["params"],
                    "knowledge_summary": p["knowledge_summary"]
                }
                for p in st.session_state.phones if p["id"] in selected_ids
            ]
            comparison = generate_comparison(phones_data)
        if comparison:
            st.session_state.comparison_result = comparison
        else:
            st.error("对比评测生成失败，请重试（可在终端查看具体原因）")

    if st.session_state.comparison_result:
        st.markdown(st.session_state.comparison_result)
        st.download_button(
            "⬇️ 下载对比评测（Markdown）",
            data=st.session_state.comparison_result,
            file_name="手机对比评测.md",
            mime="text/markdown",
            key="dl_comparison"
        )
