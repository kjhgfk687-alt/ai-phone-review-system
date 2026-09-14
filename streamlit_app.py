import streamlit as st
import sys
import os
import json
import time
import html
import uuid
from concurrent.futures import ThreadPoolExecutor

# 导入核心函数
sys.path.append(os.path.dirname(__file__))
from cshi import (
    extract_single_phone, generate_review, generate_comparison,
    generate_personalized_advice, KEY_TRANSLATION, clear_cache,
    MAX_PARALLEL_URLS, EXTRACT_API_KEY,
    submit_generation_task, get_generation_task
)

# 页面配置
st.set_page_config(
    page_title="AI手机参数提取与评测系统",
    page_icon="📱",
    layout="wide"
)

st.title("📱 AI手机参数提取与评测系统")
st.markdown("---")

# 紧凑键值对样式（参数双栏区使用）
st.markdown("""<style>
.compact-kv p { margin: 0 0 5px 0; font-size: 0.88em; line-height: 1.5; }
</style>""", unsafe_allow_html=True)

# 初始化会话状态
if "phones" not in st.session_state:
    st.session_state.phones = []          # 已提取的手机列表
if "user_profile" not in st.session_state:
    st.session_state.user_profile = None  # 侧边栏用户画像
# 评测/建议/对比结果改由 cshi 后台任务存储管理（get_generation_task），
# 界面轮询渲染，因此不再需要 review_results/advice_results/comparison_result

# ================= 侧边栏 =================
with st.sidebar:
    st.header("⚙️ 设置")
    use_cache = st.checkbox("启用缓存（同一 URL 秒出结果）", value=True)
    st.caption("缓存目录：`cache/`，可在下方手动清空")

    if st.button("🗑️ 清空全部结果", use_container_width=True):
        st.session_state.phones = []
        # 正在后台生成的任务不中断，完成后结果因手机列表已清空而不再展示
        st.rerun()

    if st.button("🧹 清除缓存", use_container_width=True):
        removed = clear_cache()
        st.toast(f"已清除 {removed} 个缓存文件", icon="🧹")

    # ---- 用户画像（个性化推荐的独立 UI 块，位置可整体迁移） ----
    st.divider()
    st.header("👤 用户画像")
    profile_enabled = st.toggle(
        "启用个性化", value=True,
        help="开启后，评测/对比/建议会按你的身份和侧重要素调整语言风格与内容深度"
    )
    user_type = st.radio(
        "你的身份",
        ["数码小白", "数码爱好者", "购机决策者"],
        index=0,
        help="对应调研问卷中的三类用户画像"
    )
    priorities = st.multiselect(
        "最看重什么？（可多选）",
        ["性能/游戏", "续航/充电", "拍照", "屏幕", "性价比", "品牌"],
        default=[]
    )
    budget = st.slider(
        "预算上限（元）", 0, 15000, 0, step=500,
        help="0 表示不限预算"
    )
    if profile_enabled:
        st.session_state.user_profile = {
            "user_type": user_type,
            "priorities": priorities,
            "budget": budget if budget > 0 else None,
        }
    else:
        st.session_state.user_profile = None

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
CATEGORY_ICONS = {
    "basic_info": "📋", "processor": "🔥", "memory_storage": "💾", "display": "🖥️",
    "battery_charging": "🔋", "camera": "📷", "connectivity": "📡", "sensors": "🧭",
    "audio": "🔊", "physical": "📏",
}

def _valid_leaf(label: str, value) -> bool:
    """叶子参数是否值得展示：非空、非'未提及'、值不等于键名（冗余）"""
    return bool(value) and value != "未提及" and str(value).strip() != label

def _flat_leaves(d: dict) -> list:
    """把子树拍平为 [(label, value_str)]（值经 HTML 转义）"""
    out = []
    for k, v in d.items():
        if str(k).startswith("_"):
            continue
        label = KEY_TRANSLATION.get(k, k)
        if isinstance(v, dict):
            out.extend(_flat_leaves(v))
        elif _valid_leaf(label, v):
            out.append((label, html.escape(str(v))))
    return out

def _render_kv_two_columns(container, pairs):
    """键值对双栏紧凑渲染"""
    if not pairs:
        return
    half = (len(pairs) + 1) // 2
    cols = container.columns(2)
    for col, chunk in zip(cols, (pairs[:half], pairs[half:])):
        body = "".join(f"<p><b>{l}</b>: {v}</p>" for l, v in chunk)
        col.markdown(f"<div class='compact-kv'>{body}</div>", unsafe_allow_html=True)

def render_params_tabs(container, params):
    """完整参数：分类标签页 + 双栏键值对"""
    categories = [(k, v) for k, v in params.items()
                  if not str(k).startswith("_") and isinstance(v, dict)]
    if not categories:
        container.caption("无参数数据")
        return
    labels = [f"{CATEGORY_ICONS.get(k, '📁')} {KEY_TRANSLATION.get(k, k)}"
              for k, _ in categories]
    tabs = container.tabs(labels)
    for tab, (cat_key, cat_val) in zip(tabs, categories):
        leaves, groups = [], []
        for k, v in cat_val.items():
            if str(k).startswith("_"):
                continue
            label = KEY_TRANSLATION.get(k, k)
            if isinstance(v, dict):
                sub = _flat_leaves(v)
                if sub:
                    groups.append((label, sub))
            elif _valid_leaf(label, v):
                leaves.append((label, html.escape(str(v))))
        _render_kv_two_columns(tab, leaves)
        for gtitle, gpairs in groups:
            tab.markdown(f"**{gtitle}**")
            _render_kv_two_columns(tab, gpairs)
        if not leaves and not groups:
            tab.caption("本分类未提取到有效参数")

def translate_path(path: str) -> str:
    """把英文参数路径翻译成中文展示（display.refresh_rate → 显示屏 › 刷新率）"""
    return " › ".join(KEY_TRANSLATION.get(seg, seg) for seg in str(path).split("."))

def render_conflicts_table(container, conflicts):
    """冲突明细：独立折叠表格（替代参数行内 ⚠️ 标记）"""
    if not conflicts:
        return
    with container.expander(f"⚠️ 冲突明细（{len(conflicts)} 处，已自动保留首个非空值）",
                            expanded=False):
        rows = [{
            "参数": translate_path(c.get("path", "")),
            "保留值": str(c.get("final_value", c.get("value1", ""))),
            "另一分段提取值": str(c.get("value2", "")),
        } for c in conflicts]
        st.dataframe(rows, use_container_width=True, hide_index=True)

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

    for section_key, title in [("display", "🖥️ 屏幕表现"), ("battery", "🔋 电池与充电"),
                               ("camera", "📷 相机能力")]:
        if section_key in summary:
            container.markdown(f"### {title}")
            for key, desc in summary[section_key].items():
                label = display_key_map.get(key, key.replace('_desc', ''))
                container.markdown(f"- **{label}**：{desc}")
            container.markdown("---")

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

# ---- 后台生成任务的轮询渲染（fragment 自动刷新） ----
@st.fragment(run_every="2s")
def task_fragment(task_id: str, header: str, download_name: str = None):
    """轮询渲染后台生成任务：运行中显示耗时与流式正文，完成显示结果与下载"""
    t = get_generation_task(task_id)
    if not t:
        return
    if t["status"] == "running":
        elapsed = time.time() - t["started"]
        extra = f"，正文已到 {len(t['text'])} 字" if t["text"] else ""
        st.info(f"⏳ {t['title']} 生成中… 已用 {elapsed:.0f} 秒{extra}")
        if t["text"]:
            st.markdown(t["text"])
    elif t["status"] == "done":
        st.markdown(f"**{header}**")
        st.markdown(t["text"])
        if download_name:
            st.download_button(
                "⬇️ 下载（Markdown）", data=t["text"],
                file_name=download_name, mime="text/markdown",
                key=f"dl_{task_id}"
            )
    else:
        st.error(f"{t['title']} 生成失败：{t['error']}（可重试）")

# ================= 输入与提取 =================
url_input = st.text_area(
    "输入手机参数页URL（每行一个）",
    height=100,
    placeholder="https://www.oppo.com/cn/smartphones/series-find-x/find-x9s-pro/specs/"
)

if st.button("🚀 开始提取", type="primary"):
    # 输入去重 + 过滤无效行
    urls = list(dict.fromkeys(u.strip() for u in url_input.split('\n') if u.strip().startswith("http")))

    if not urls:
        st.error("请输入至少一个有效URL（以 http 开头）")
    else:
        existing_urls = {p.get("url") for p in st.session_state.phones}
        new_urls = [u for u in urls if u not in existing_urls]
        if len(new_urls) < len(urls):
            st.info(f"⏭️ 跳过 {len(urls) - len(new_urls)} 个本次会话已提取过的URL")
        if not new_urls:
            st.info("没有需要提取的新URL")
        else:
            workers = min(MAX_PARALLEL_URLS, len(new_urls))
            st.info(f"🚀 并行提取 {len(new_urls)} 个URL（{workers} 个同时进行）")

            status_map = {u: st.status(f"📡 待开始：{u}", expanded=False) for u in new_urls}
            progress_bar = st.progress(0.0, text="准备中...")

            # 线程安全约定：工作线程只写 progress_map，主线程轮询渲染
            progress_map = {}

            def make_callback(url):
                def cb(p, msg):
                    progress_map[url] = (p, msg)
                return cb

            pending_futs = {}
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for u in new_urls:
                    pending_futs[pool.submit(
                        extract_single_phone, u,
                        use_cache=use_cache, progress_callback=make_callback(u)
                    )] = u

                remaining = set(new_urls)
                deadline = time.time() + 900
                while remaining and time.time() < deadline:
                    for u in remaining:
                        if u in progress_map:
                            _, msg = progress_map[u]
                            status_map[u].update(label=f"{msg}")
                    done_ps = [progress_map.get(u, (0.0, ""))[0] for u in new_urls]
                    overall = sum(done_ps) / len(new_urls)
                    progress_bar.progress(overall, text=f"总体进度 {overall * 100:.0f}%")

                    for fut in list(pending_futs):
                        if fut.done():
                            u = pending_futs.pop(fut)
                            remaining.discard(u)
                            try:
                                result, phone_name, error = fut.result()
                            except Exception as e:
                                result, phone_name, error = None, None, f"线程异常：{e}"

                            if result:
                                st.session_state.phones.append({
                                    "id": uuid.uuid4().hex,
                                    "url": u,
                                    "phone_name": phone_name,
                                    "params": result,
                                    "knowledge_summary": result.get("_knowledge_summary"),
                                    "conflicts": result.get("_conflicts", []),
                                    "tags": result.get("_tags", []),
                                })
                                status_map[u].update(
                                    label=f"✅ {phone_name} 提取成功",
                                    state="complete", expanded=False
                                )
                            else:
                                status_map[u].update(label=f"❌ 提取失败：{u}", state="error", expanded=True)
                                with status_map[u]:
                                    st.error(f"提取失败：{error or '未知错误'}")
                    time.sleep(0.4)

            progress_bar.progress(1.0, text="全部处理完成")

# ================= 提取结果展示 =================
if st.session_state.phones:
    st.markdown("---")
    st.subheader(f"📋 提取结果（共 {len(st.session_state.phones)} 部）")

    for phone in st.session_state.phones:
        phone_id = phone["id"]
        name = phone["phone_name"]
        safe_name = str(name).replace(' ', '_').replace('/', '_')

        # 标题行 + 删除按钮
        title_col, del_col = st.columns([6, 1])
        title_col.markdown(f"### 📱 {name}")
        if del_col.button("🗑️ 删除", key=f"del_{phone_id}"):
            st.session_state.phones = [p for p in st.session_state.phones if p["id"] != phone_id]
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

        # 完整参数：分类标签页 + 双栏；冲突明细独立表格
        conflict_count = len(phone.get('conflicts', []))
        expander_title = f"🔍 完整参数" + (f"（{conflict_count} 处冲突 ⚠️）" if conflict_count else "")
        with st.expander(expander_title, expanded=False):
            render_params_tabs(st, phone['params'])
            render_conflicts_table(st, phone.get('conflicts', []))

        # 操作行：提交后台生成任务（点击即返回，界面不锁，可同时生成多部）
        profile = st.session_state.get("user_profile")
        action_col1, action_col2, action_col3 = st.columns(3)

        if action_col1.button(f"📝 生成评测：{name}", key=f"review_{phone_id}"):
            accepted = submit_generation_task(
                f"review_{phone_id}", f"{name} 的评测",
                lambda cb, p=phone: generate_review(
                    p["phone_name"], p["params"], p["knowledge_summary"],
                    profile, on_text=cb)
            )
            st.toast("评测已开始后台生成，可继续其他操作 🚀" if accepted
                     else "该评测已在生成中，请稍候", icon="🚀")

        if action_col2.button("🎯 个性化建议", key=f"advice_{phone_id}",
                              disabled=profile is None,
                              help="先在左侧『用户画像』设置身份与侧重要素"):
            accepted = submit_generation_task(
                f"advice_{phone_id}", f"{name} 的个性化建议",
                lambda cb, p=phone: generate_personalized_advice(
                    p["phone_name"], p["params"], p["knowledge_summary"],
                    profile, on_text=cb)
            )
            st.toast("建议已开始后台生成 🚀" if accepted
                     else "该建议已在生成中，请稍候", icon="🚀")

        params_json = json.dumps(phone['params'], ensure_ascii=False, indent=2)
        action_col3.download_button(
            "⬇️ 下载参数 JSON",
            data=params_json,
            file_name=f"{safe_name}_params.json",
            mime="application/json",
            key=f"dl_params_{phone_id}",
            use_container_width=True
        )

        # 后台任务状态与结果（自动轮询刷新）
        task_fragment(f"review_{phone_id}", "📝 专业评测", f"{safe_name}_评测.md")
        task_fragment(f"advice_{phone_id}", "🎯 个性化选购建议（依据左侧用户画像）",
                      f"{safe_name}_个性化建议.md")

        st.markdown("---")

# ================= 对比评测（后台生成） =================
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
        phones_data = [
            {
                "phone_name": p["phone_name"],
                "params": p["params"],
                "knowledge_summary": p["knowledge_summary"]
            }
            for p in st.session_state.phones if p["id"] in selected_ids
        ]
        profile = st.session_state.get("user_profile")
        accepted = submit_generation_task(
            "comparison", f"{len(phones_data)} 部手机对比评测",
            lambda cb: generate_comparison(phones_data, profile, on_text=cb)
        )
        st.toast("对比评测已开始后台生成 🚀" if accepted
                 else "对比评测已在生成中，请稍候", icon="🚀")

    task_fragment("comparison", "🔍 对比评测", "手机对比评测.md")
