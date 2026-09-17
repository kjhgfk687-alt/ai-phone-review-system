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
    generate_personalized_advice, generate_appearance_analysis,
    download_manual_images, resolve_image, save_uploaded_images,
    generation_task_stats, KEY_TRANSLATION, clear_cache,
    MAX_PARALLEL_URLS, EXTRACT_API_KEY,
    submit_generation_task, get_generation_task
)
from model_finder import find_model_candidates, confirm_mapping
from registry import (
    stats as registry_stats, all_phones as registry_phones, upsert as registry_upsert,
    find_by_spec_url, lookup as registry_lookup
)
from resolvers import BRAND_ADAPTERS
from scoring import score_phone
from knowledge_store import load_params as kb_load_params
import os as _os
DEV_PASSWORD = _os.environ.get("DEV_PASSWORD", "shumaceping")

# 页面配置
st.set_page_config(
    page_title="AI手机参数提取与评测系统",
    page_icon="📱",
    layout="wide"
)

st.title("📱 AI手机参数提取与评测系统")
st.markdown("---")

# 紧凑排版样式（参数双栏/知识解读/生成正文统一小字号）
st.markdown("""<style>
.compact-kv p { margin: 0 0 5px 0; font-size: 0.88em; line-height: 1.5; }
.compact-doc h4 { margin: 8px 0 4px 0; font-size: 0.95em; }
.compact-doc ul { margin: 2px 0 10px 0; padding-left: 18px; }
.compact-doc li { font-size: 0.88em; margin: 2px 0; line-height: 1.55; }
.compact-doc p { margin: 4px 0; font-size: 0.88em; }
[data-testid="stMarkdownContainer"] h3 { font-size: 1.1em; margin-top: 0.9em; }
[data-testid="stMarkdownContainer"] h4 { font-size: 0.95em; margin: 0.6em 0 0.3em 0; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
.tag-neg { background-color:#FDE8E8; color:#B3261E; }
.tag-pos { background-color:#E6F4EA; color:#1E7B34; }
span[class^="tag-"] { padding:4px 8px; margin-right:6px; border-radius:4px; font-size:0.88em; display:inline-block; }
</style>""", unsafe_allow_html=True)

# 初始化会话状态
if "phones" not in st.session_state:
    st.session_state.phones = []          # 已提取的手机列表
if "user_profile" not in st.session_state:
    st.session_state.user_profile = None  # 侧边栏用户画像
# 评测/建议/对比结果改由 cshi 后台任务存储管理（get_generation_task），
# 界面轮询渲染，因此不再需要 review_results/advice_results/comparison_result

# ================= 顶部指标行（开发者选项可见） =================
if st.session_state.get("dev_unlocked"):
    _reg_stats = registry_stats()
    _task_stats = generation_task_stats()
    _m1, _m2, _m3, _m4 = st.columns(4)
    _m1.metric("📱 本次已提取", len(st.session_state.phones))
    _m2.metric("🗂️ 注册表收录", _reg_stats["total"], help="phone_registry.yaml 中人工维护与确认过的机型")
    _m3.metric("⚙️ 后台生成中", _task_stats["running"], help="评测/建议/对比正在后台生成的任务数")
    _m4.metric("🧠 RAG 待入库", _reg_stats["rag_pending"], help="未来 RAG 检索库待收录的注册表条目")

# ================= 侧边栏 =================
with st.sidebar:
    st.header("⚙️ 设置")
    # 双视图：访客视图干净无调试元素；开发者选项（密码门）暴露维护工具
    dev_unlocked = st.session_state.get("dev_unlocked", False)
    with st.expander("🛠️ 开发者模式", expanded=dev_unlocked):
        if dev_unlocked:
            st.success("已解锁，维护工具可见")
            if st.button("🔒 锁定并隐藏维护工具", use_container_width=True):
                st.session_state["dev_unlocked"] = False
                st.rerun()
        else:
            _pwd = st.text_input("访问密码", type="password", key="dev_pwd",
                                 placeholder="仅开发者需要")
            if st.button("🔓 解锁", use_container_width=True):
                if _pwd == DEV_PASSWORD:
                    st.session_state["dev_unlocked"] = True
                    st.rerun()
                else:
                    st.error("密码错误")

    if dev_unlocked:
        use_cache = st.checkbox("启用缓存（同一 URL 秒出结果）", value=True)
        st.caption("缓存目录：`cache/`，可在下方手动清空")
        if st.button("🗑️ 清空全部结果", use_container_width=True):
            st.session_state.phones = []
            st.rerun()
        if st.button("🧹 清除缓存", use_container_width=True):
            removed = clear_cache()
            st.toast(f"已清除 {removed} 个缓存文件", icon="🧹")
    else:
        use_cache = True
        if st.button("🗑️ 清空全部结果", use_container_width=True):
            st.session_state.phones = []
            st.rerun()

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
    # API Key 配置状态检查（开发者工具；访客视图不暴露开发细节）
    if st.session_state.get("dev_unlocked"):
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
    """知识解读：紧凑 HTML 排版（小标题+小字号列表），替代大标题渲染"""
    if not summary:
        return

    display_key_map = {
        "resolution_desc": "分辨率", "refresh_desc": "刷新率",
        "screen_type_desc": "屏幕类型", "brightness_desc": "亮度",
        "capacity_desc": "电池容量", "charging_desc": "有线充电",
        "wireless_charging_desc": "无线充电", "sensor_desc": "传感器",
        "aperture_desc": "光圈", "pixel_desc": "像素", "zoom_desc": "变焦",
    }

    import html as _html
    def _h4(t): return f"<h4>{_html.escape(str(t))}</h4>"
    def _li(t): return f"<li>{_html.escape(str(t))}</li>"
    def _ul(items): return "<ul>" + "".join(_li(i) for i in items) + "</ul>"

    parts = ['<div class="compact-doc">']

    if "chipset" in summary:
        info = (summary["chipset"].get("info") or {})
        chipset = summary["chipset"]
        parts.append(_h4(f"🔥 芯片 · {chipset.get('name','')}（{info.get('tier','')} · 评分 {info.get('performance_score','?')}/100）"))
        if info.get("description"):
            parts.append(f"<p>📝 {_html.escape(info['description'])}</p>")
        rows = []
        if info.get("strengths"): rows.append("优势：" + "；".join(info["strengths"]))
        if info.get("weaknesses"): rows.append("注意：" + "；".join(info["weaknesses"]))
        if info.get("typical_price_range"): rows.append(f"常见价格区间：{info['typical_price_range']}")
        if info.get("note"): rows.append(f"备注：{info['note']}")
        parts.append(_ul(rows))

    for sec_key, title in [("display", "🖥️ 屏幕"), ("battery", "🔋 电池与充电"), ("camera", "📷 相机")]:
        if sec_key in summary:
            parts.append(_h4(title))
            parts.append(_ul([f"{display_key_map.get(k, k.replace('_desc',''))}：{v}"
                              for k, v in summary[sec_key].items()]))

    if "brand" in summary:
        brand = summary["brand"]
        parts.append(_h4("🏢 品牌"))
        rows = []
        if brand.get("brand_positioning"): rows.append(f"定位：{brand['brand_positioning']}")
        if brand.get("target_users"): rows.append(f"目标用户：{brand['target_users']}")
        if brand.get("key_technologies"): rows.append("核心技术：" + "、".join(brand["key_technologies"]))
        for t in brand.get("new_technologies_2025_2026") or []:
            rows.append(f"新技术：{t}")
        parts.append(_ul(rows))

    parts.append("</div>")
    container.subheader("📚 专业知识解读")
    container.markdown("".join(parts), unsafe_allow_html=True)

# ---- 后台生成任务的轮询渲染（fragment 自动刷新） ----
@st.fragment(run_every="0.8s")
def task_fragment(task_id: str, header: str, download_name: str = None):
    """轮询渲染后台生成任务：运行中显示耗时与流式正文（0.8s 粒度，观感接近逐字），
    完成显示结果与下载。模型输出的 ### 标题降级为 #### 配合紧凑 CSS。"""
    t = get_generation_task(task_id)
    if not t:
        return
    text = t["text"].replace("\n### ", "\n#### ") if t["text"] else ""
    if t["status"] == "running":
        elapsed = time.time() - t["started"]
        extra = f"，正文已到 {len(text)} 字" if text else ""
        st.info(f"⏳ {t['title']} 生成中… 已用 {elapsed:.0f} 秒{extra}")
        if text:
            st.markdown(text)
    elif t["status"] == "done":
        st.markdown(f"**{header}**")
        st.markdown(text)
        if download_name:
            st.download_button(
                "⬇️ 下载（Markdown）", data=t["text"],
                file_name=download_name, mime="text/markdown",
                key=f"dl_{task_id}"
            )
    else:
        st.error(f"{t['title']} 生成失败：{t['error']}（可重试）")

@st.fragment(run_every="0.8s")
def appearance_fragment(task_id: str, phone_id: str, safe_name: str):
    """外观分析专用轮询：生成中流式展示；完成后渲染注册表缓存的解读文本
    （分析结果由生成任务写回注册表，访客下次打开无需任何操作即可看到）。"""
    t = get_generation_task(task_id)
    entry = None
    for p in st.session_state.get("phones", []):
        if p["id"] == phone_id:
            entry = find_by_spec_url(p['url']) or registry_lookup(p['phone_name'])
            break
    cached = (entry or {}).get("appearance") or {}

    if t and t["status"] == "running":
        elapsed = time.time() - t["started"]
        h3 = chr(10) + "### "
        h4 = chr(10) + "#### "
        text = t["text"].replace(h3, h4) if t["text"] else ""
        st.info(f"⏳ 外观分析生成中… 已用 {elapsed:.0f} 秒" + (f"，正文已到 {len(text)} 字" if text else ""))
        if text:
            st.markdown(text)
    elif t and t["status"] == "error":
        st.error(f"外观分析生成失败：{t['error']}（可重试）")
        if cached.get("analysis"):
            st.markdown("**🎨 外观解读**（历史版本）")
            st.markdown(cached["analysis"])
    elif cached.get("analysis"):
        st.markdown("**🎨 外观解读**（基于官方渲染图，非真机实拍）")
        h3 = chr(10) + "### "
        st.markdown(cached["analysis"].replace(h3, chr(10) + "#### "))
        st.download_button("⬇️ 下载外观解读", data=cached["analysis"],
                           file_name=f"{safe_name}_外观分析.md", mime="text/markdown",
                           key=f"dl_ape_{phone_id}")

# ================= 输入与提取 =================
# ---- 按型号自动查找（V1.4 功能①） ----
st.caption("主航道：优先从本地参数知识库直取完整参数（秒回、离线可用）；未收录机型自动经注册表 / 官网模板 / 搜索定位。")
model_query = st.text_input(
    "手机型号",
    placeholder="如：小米15 / iPhone 17 Pro / vivo X200 / 红米 K90",
    key="model_query"
)
if st.button("🔍 检索", type="primary"):
    q = (model_query or "").strip()
    if not q:
        st.warning("请先输入手机型号")
        st.stop()
    # ① 参数知识库直取（主力路径）
    kb_hit = kb_load_params(q)
    if kb_hit and kb_hit.get("params"):
        params = kb_hit["params"]
        pname = kb_hit.get("model") or q
        if any(p["phone_name"] == pname for p in st.session_state.phones):
            st.info(f"「{pname}」已在本次结果列表中，可直接查看。")
        else:
            st.session_state.phones.append({
                "id": uuid.uuid4().hex,
                "url": kb_hit.get("spec_url") or "",
                "phone_name": pname,
                "params": params,
                "knowledge_summary": params.get("_knowledge_summary"),
                "conflicts": params.get("_conflicts", []),
                "tags": params.get("_tags", []),
            })
        st.success(f"⚡ 知识库命中「{pname}」，已直接载入完整参数")
        st.session_state["model_candidates"] = None
    else:
        # ② 未命中：注册表/模板/搜索 定位管线
        with st.status(f"正在查找「{q}」...", expanded=True) as search_status:
            def s_progress(p2, msg):
                search_status.update(label=msg)
            try:
                cands = find_model_candidates(q, progress=s_progress)
            except Exception as e:
                cands = []
                st.error(f"查找出错：{str(e)[:120]}")
            search_status.update(
                label=f"查找完成：{len(cands)} 个有效候选" if cands else "查找完成：未找到匹配的参数页",
                state="complete" if cands else "error", expanded=cands == []
            )
        st.session_state["model_candidates"] = cands
        st.session_state["model_candidates_for"] = q

    cands = st.session_state.get("model_candidates") or []
    if cands:
        st.caption(f"「{st.session_state.get('model_candidates_for', '')}」的候选（按匹配度排序，确认后写回注册表）：")
        how_badge = {"已收录": "📚 已收录", "模板直配": "⚡ 模板直配", "搜索": "🔎 搜索验证",
                     "直配": "⚡ 模板直配", "缓存": "📚 已收录"}
        for i, c in enumerate(cands):
            cc1, cc2, cc3 = st.columns([4, 2, 1])
            badge = how_badge.get(c["how"], c["how"])
            cc1.markdown(f"**{c['matched_name']}**（{badge}）")
            cc1.caption(c["url"][:90])
            cc2.progress(min(c["confidence"], 1.0), text=f"匹配度 {c['confidence']:.0%}")
            if cc3.button("✅ 用这个", key=f"use_cand_{i}", use_container_width=True):
                confirm_mapping(st.session_state.get("model_candidates_for", ""), c["url"])
                st.session_state["url_input_box"] = c["url"]
                st.session_state["model_candidates"] = None
                st.toast("已填入 URL 并保存到机型注册表 ✅")
                st.rerun()
    elif st.session_state.get("model_candidates") is not None and not cands:
        st.info("没有匹配度足够的候选。可尝试：补上品牌名后重新查找，或展开下方 URL 提取直接粘贴。")

with st.expander("🔗 按参数页 URL 提取（高级，一般无需使用）", expanded=False):
    st.caption("已知道确切参数页地址时的快捷方式。")
    url_input = st.text_area(
        "输入手机参数页URL（每行一个，高级用法）",
        height=100,
        placeholder="https://www.oppo.com/cn/smartphones/series-find-x/find-x9s-pro/specs/",
        key="url_input_box"
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

if not st.session_state.phones:
    st.info("👆 **三步上手**：① 粘贴官网参数页 URL，或在上方按型号自动查找 → ② 点击开始提取 → ③ 生成评测 / 对比 / 个性化建议。侧边栏可开启个性化画像。")

# ================= 提取结果展示 =================
if st.session_state.phones:
    st.markdown("---")
    st.subheader(f"📋 提取结果（共 {len(st.session_state.phones)} 部）")

    for phone in st.session_state.phones:
        with st.container(border=True):
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
                NEG = ("较低", "卡顿", "不支持", "单频", "弱")
                tags_html = " ".join([
                    f"<span class='{'tag-neg' if any(k in str(tag) for k in NEG) else 'tag-pos'}'>"
                    f"{html.escape(str(tag))}</span>"
                    for tag in phone['tags']
                ])
                st.markdown(tags_html, unsafe_allow_html=True)
                st.markdown("")

            # 知识总结
            if phone['knowledge_summary']:
                with st.expander("📚 专业知识解读", expanded=False):
                    display_knowledge_summary(phone['knowledge_summary'], st)

            # 完整参数：分类标签页 + 双栏；冲突明细仅演示者视图展示（访客视图保持干净）
            presenter = st.session_state.get("dev_unlocked", False)
            conflict_count = len(phone.get('conflicts', []))
            if presenter and conflict_count:
                expander_title = f"🔍 完整参数（{conflict_count} 处冲突 ⚠️）"
            else:
                expander_title = "🔍 完整参数"
            with st.expander(expander_title, expanded=False):
                render_params_tabs(st, phone['params'])
                if presenter:
                    render_conflicts_table(st, phone.get('conflicts', []))

            # 操作行：提交后台生成任务（点击即返回，界面不锁，可同时生成多部）
            profile = st.session_state.get("user_profile")
            action_col1, action_col2, action_col3, action_col4 = st.columns(4)

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

            # ---- 外观区：策展图（人工维护）为唯一数据源（AI 自找图已按需求移除） ----
            ape_entry = find_by_spec_url(phone['url']) or registry_lookup(phone['phone_name'])
            curated = (ape_entry or {}).get("images") or []
            ape_cached = (ape_entry or {}).get("appearance") or {}

            if curated:
                st.markdown("**🖼️ 官方外观图**（人工策展）")
                gcols = st.columns(min(len(curated), 4))
                resolved_paths = []
                for gc, s in zip(gcols, curated[:4]):
                    rp = resolve_image(s, referer='/'.join(phone['url'].split('/')[:3]))
                    if rp:
                        resolved_paths.append(rp)
                        try:
                            gc.image(rp, use_container_width=True)
                        except Exception:
                            gc.caption("预览失败")
                    else:
                        gc.caption("图片不可用")
                st.caption("图片来源：品牌官方渲染图，版权归品牌方所有，此处仅作评测参考引用。")
                if ape_cached.get("analysis"):
                    st.markdown("**🎨 外观解读**（基于官方渲染图）")
                    h3 = chr(10) + "### "
                    st.markdown(ape_cached["analysis"].replace(h3, chr(10) + "#### "))

            if action_col3.button("🎨 生成/更新外观分析", key=f"appear_{phone_id}",
                                  help="基于人工策展的官方渲染图进行视觉分析" if curated
                                       else "该机型暂无策展外观图"):
                if not curated:
                    st.warning("该机型暂无策展外观图——请在开发者模式『手动补录 / 编辑机型与外观图』中补充官方渲染图后重试。")
                else:
                    resolved = [p2 for p2 in (resolve_image(s, '/'.join(phone['url'].split('/')[:3])) for s in curated) if p2]
                    if not resolved:
                        st.error("策展图全部下载失败，请检查链接或重新上传")
                    else:
                        def _ape_task(cb, p=phone, ims=resolved, nm=name):
                            res = generate_appearance_analysis(nm, ims, profile, on_text=cb)
                            if res:
                                registry_upsert(p['phone_name'], p['url'],
                                                appearance={"analysis": res})
                            return res
                        accepted = submit_generation_task(
                            f"appear_{phone_id}", f"{name} 的外观分析", _ape_task)
                        st.toast("外观分析已开始后台生成 🚀" if accepted
                                 else "该外观分析已在生成中，请稍候", icon="🚀")

            params_json = json.dumps(phone['params'], ensure_ascii=False, indent=2)
            action_col4.download_button(
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
            appearance_fragment(f"appear_{phone_id}", phone_id, safe_name)


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

    # 五维评分对比（启发式评分，规则见 scoring.py）
    if len(selected_ids) >= 2:
        import pandas as pd
        score_rows = {}
        for p in st.session_state.phones:
            if p["id"] in selected_ids:
                s = score_phone(p["params"], p.get("knowledge_summary"))
                score_rows[p["phone_name"]] = {k: (v or 0) for k, v in s.items()}
        if score_rows:
            st.markdown("**📊 五维评分对比**（启发式评分，仅供参考）")
            st.bar_chart(pd.DataFrame(score_rows))
        st.markdown("")
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

# ================= 机型注册表（演示者视图：可视化 + 手动补录） =================
if st.session_state.get("dev_unlocked"):
    st.markdown("---")
    st.subheader("🗂️ 机型注册表")
    st.caption("数据驱动的核心：确认过的型号→参数页映射都在这里，长尾特例降维成一条数据。"
               "文件：`phone_registry.yaml`（人工可维护），未来 RAG 的语料资产底座。")

    s = registry_stats()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("已收录机型", s["total"])
    m2.metric("覆盖品牌", s["brands"])
    m3.metric("RAG 待入库", s["rag_pending"])
    m4.metric("确认来源数", sum(s["by_source"].values()))

    with st.expander("📋 注册表明细", expanded=False):
        rows = [{
            "型号": p.get("model", ""),
            "品牌": p.get("brand", ""),
            "来源": p.get("source", ""),
            "最近验证": p.get("verified_at") or "未验证",
            "参数页": (p.get("spec_url") or "待补录")[:60],
        } for p in registry_phones()]
        st.dataframe(rows, use_container_width=True, hide_index=True)

    with st.expander("➕ 手动补录 / 编辑机型与外观图", expanded=False):
        def _prefill():
            sel = st.session_state.get("reg_edit_sel")
            if sel and sel != "（录入新机型）":
                e0 = registry_lookup(sel)
                if e0:
                    st.session_state["reg_model_m"] = e0.get("model", "")
                    st.session_state["reg_url_m"] = e0.get("spec_url", "")
                    st.session_state["reg_brand_m"] = e0.get("brand") or "OPPO"
                    st.session_state["reg_alias_m"] = ", ".join(e0.get("aliases") or [])
        brand_names = list(BRAND_ADAPTERS.keys())
        edit_sel = st.selectbox("编辑已收录机型（选它自动带出信息）",
                                ["（录入新机型）"] + [p.get("model", "") for p in registry_phones()],
                                key="reg_edit_sel", on_change=_prefill)
        m_model = st.text_input("型号", key="reg_model_m",
                                placeholder="如：小米17T（官方名或常用叫法）")
        m_url = st.text_input("参数页 URL", key="reg_url_m", placeholder="https://...")
        _cur_brand = st.session_state.get("reg_brand_m", "OPPO")
        m_brand = st.selectbox("品牌", brand_names,
                               index=brand_names.index(_cur_brand) if _cur_brand in brand_names else 0,
                               key="reg_brand_m")
        m_alias = st.text_input("别名（可选，逗号分隔）", key="reg_alias_m",
                                placeholder="如：Xiaomi 17T, 17T")
        m_files = st.file_uploader("上传外观图（可多选，自动压缩为 webp 存入 assets/，随 git 永久保存）",
                                   type=["png", "jpg", "jpeg", "webp"],
                                   accept_multiple_files=True, key="reg_files")
        m_images = st.text_area("或粘贴外观图 URL（可选，空格/逗号/换行分隔）", key="reg_images_m",
                                height=68,
                                placeholder="预约页/无渲染图机型可人工补充官方概念图链接，外观分析将优先使用")
        if st.button("📥 录入 / 更新注册表"):
            if m_model.strip() and m_url.strip():
                aliases = [a.strip() for a in m_alias.split(",") if a.strip()]
                url_imgs = [u.strip() for u in m_images.replace(",", " ").split()
                            if u.strip().startswith("http")]
                saved_paths = save_uploaded_images(m_model.strip(), m_files) if m_files else []
                all_imgs = saved_paths + url_imgs
                registry_upsert(m_model.strip(), m_url.strip(), brand=m_brand,
                                aliases=aliases or None, images=all_imgs or None, source="manual")
                msg = f"已录入「{m_model.strip()}」✅ 此后查找将直接命中注册表"
                detail = []
                if saved_paths: detail.append(f"上传 {len(saved_paths)} 张（assets/ 永久保存）")
                if url_imgs: detail.append(f"URL 图 {len(url_imgs)} 张")
                st.success(msg + ("（" + "，".join(detail) + "）" if detail else ""))
            else:
                st.error("型号和参数页 URL 都不能为空")

# ================= 页脚 =================
st.markdown("---")
st.caption("数据来源：各品牌官网公开参数页 · 外观图片版权归品牌方所有，仅作评测参考引用 · 本项目为个人学习作品")
