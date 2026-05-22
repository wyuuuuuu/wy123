"""Teaching-oriented knowledge graph prototype.

This Streamlit app is a lightweight application layer on top of the existing
knowledge graph. It does not change relation extraction, training, or the
Step0-Step6 pipeline. The app focuses on three demo views:

1. Overview: graph statistics and four course topic clusters.
2. Teacher view: topic graph, key concepts, evidence, and teaching hints.
3. Student view: concept search, graph-based QA entry, and simple recommendations.
"""

from __future__ import annotations

import html
import json
import math
import re
import urllib.error
import urllib.request
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from string import Template

import networkx as nx
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


ROOT_DIR = Path(__file__).resolve().parents[1]
KG_DIR = ROOT_DIR / "outputs" / "kg"
NODES_PATH = KG_DIR / "nodes.csv"
EDGES_PATH = KG_DIR / "edges.csv"
SUMMARY_PATH = KG_DIR / "kg_summary.json"
TOPIC_PATH = KG_DIR / "topic_clusters.json"
UPLOAD_DIR = ROOT_DIR / "inputs" / "uploaded_triples"
RESOURCE_LIBRARY_PATH = ROOT_DIR / "data" / "resource_library.json"
PROGRESS_PATH = KG_DIR / "learning_progress.json"

REL_IS_A = "是一种"
REL_CONTAINS = "包含"
REL_USED_FOR = "用于"
RELATION_OPTIONS = [REL_IS_A, REL_CONTAINS, REL_USED_FOR]

RELATION_COLORS = {
    REL_IS_A: "#7c3aed",
    REL_CONTAINS: "#f97316",
    REL_USED_FOR: "#16a34a",
}
RELATION_LABELS = {
    REL_IS_A: "概念归属",
    REL_CONTAINS: "组成关系",
    REL_USED_FOR: "功能关系",
    "主题连接": "章节关联",
    "章节归属": "章节关联",
}
PAGE_OVERVIEW = "图谱总览"
PAGE_TEACHER = "数据分析"
PAGE_STUDENT = "学习问答"

TOPIC_COLORS = ["#dbeafe", "#dcfce7", "#f3e8ff", "#ffedd5"]
TOPIC_BORDER_COLORS = ["#2563eb", "#16a34a", "#9333ea", "#ea580c"]
OLLAMA_MODEL = "qwen3:1.7b"
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"


def inject_page_style() -> None:
    """Add modern dashboard styling for the teaching knowledge graph."""
    st.markdown(
        """
        <style>
        :root {
            --kg-blue: #2563eb;
            --kg-teal: #14b8a6;
            --kg-bg: #f5f7fb;
            --kg-text: #0f172a;
            --kg-muted: #64748b;
            --kg-border: #e5eaf3;
            --kg-card-shadow: 0 14px 36px rgba(15, 23, 42, 0.07);
        }
        .stApp {
            background: linear-gradient(180deg, #f8fbff 0%, #f3f6fb 100%);
            color: var(--kg-text);
        }
        section[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #e8edf5;
            box-shadow: 8px 0 28px rgba(15, 23, 42, 0.05);
        }
        section[data-testid="stSidebar"] > div {
            padding-top: 1.4rem;
        }
        .block-container {
            padding-top: 1.05rem;
            padding-bottom: 1.2rem;
            max-width: 1760px;
        }
        h1, h2, h3 {
            color: #0f172a;
            letter-spacing: -0.02em;
        }
        .kg-sidebar-brand {
            display: flex;
            gap: 12px;
            align-items: center;
            padding: 12px 10px 18px;
            margin-bottom: 12px;
            border-bottom: 1px solid #edf2f7;
        }
        .kg-logo {
            width: 42px;
            height: 42px;
            display: grid;
            place-items: center;
            border-radius: 14px;
            background: linear-gradient(135deg, #0ea5e9, #14b8a6);
            color: #ffffff;
            font-weight: 900;
            box-shadow: 0 12px 24px rgba(20, 184, 166, 0.22);
        }
        .kg-brand-title {
            font-weight: 900;
            color: #0f172a;
            line-height: 1.25;
            font-size: 1.02rem;
        }
        .kg-brand-subtitle {
            color: #64748b;
            font-size: 0.78rem;
            margin-top: 2px;
        }
        .kg-sidebar-footer {
            margin-top: 28px;
            padding: 12px;
            border: 1px solid #e5eaf3;
            border-radius: 16px;
            background: linear-gradient(135deg, #f8fafc, #eefdfa);
            color: #475569;
            font-size: 0.83rem;
            line-height: 1.55;
        }
        div.stButton > button {
            border-radius: 14px;
            min-height: 46px;
            font-weight: 750;
            border: 1px solid #e2e8f0;
            background: #ffffff;
            color: #334155;
            transition: all 140ms ease;
            box-shadow: 0 5px 14px rgba(15, 23, 42, 0.04);
        }
        div.stButton > button:hover {
            border-color: #14b8a6;
            color: #0f766e;
            background: #f0fdfa;
            transform: translateY(-1px);
        }
        div.stButton > button[kind="primary"],
        div.stButton > button[data-testid="baseButton-primary"] {
            background: linear-gradient(135deg, #2563eb, #14b8a6) !important;
            border-color: transparent !important;
            color: #ffffff !important;
            box-shadow: 0 12px 26px rgba(37, 99, 235, 0.18);
        }
        div.stButton > button[kind="secondary"],
        div.stButton > button[data-testid="baseButton-secondary"] {
            background: #ffffff !important;
            color: #334155 !important;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e5eaf3;
            padding: 12px 14px;
            border-radius: 16px;
            box-shadow: var(--kg-card-shadow);
        }
        .section-caption {
            color: #475569;
            font-size: 0.94rem;
            line-height: 1.65;
        }
        .kg-hero {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 18px;
            margin-bottom: 18px;
        }
        .kg-title {
            font-size: 1.62rem;
            font-weight: 900;
            color: #0f172a;
            margin: 0 0 8px 0;
        }
        .kg-subtitle {
            color: #64748b;
            font-size: 0.92rem;
            line-height: 1.7;
        }
        .kg-topbar {
            display: grid;
            grid-template-columns: 1.1fr 0.9fr;
            gap: 14px;
            align-items: center;
            margin: -4px 0 14px;
        }
        .kg-top-card {
            background: #ffffff;
            border: 1px solid #e5eaf3;
            border-radius: 20px;
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.055);
            padding: 12px 14px;
        }
        .kg-top-actions {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
        }
        .kg-chapter-nav-title {
            margin: 18px 0 10px;
            color: #475569;
            font-weight: 900;
            font-size: 0.9rem;
        }
        .kg-chapter-mini {
            border: 1px solid var(--chapter-border);
            border-radius: 16px;
            padding: 10px 12px;
            margin: 8px 0;
            background: linear-gradient(135deg, var(--chapter-soft), #ffffff);
            color: #0f172a;
        }
        .kg-chapter-mini-title {
            color: var(--chapter);
            font-weight: 900;
            font-size: 0.92rem;
        }
        .kg-chapter-mini-desc {
            color: #64748b;
            font-size: 0.76rem;
            margin-top: 3px;
        }
        .kg-mini-card {
            background: #ffffff;
            border: 1px solid #e5eaf3;
            border-radius: 18px;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.055);
            padding: 14px;
            min-height: 172px;
        }
        .kg-mini-card h4 {
            margin: 0 0 8px;
            color: #0f172a;
            font-size: 1rem;
        }
        .kg-mini-card p {
            color: #64748b;
            font-size: 0.86rem;
            line-height: 1.55;
            margin: 4px 0;
        }
        .kg-top-status {
            display: flex;
            justify-content: flex-end;
            align-items: center;
            gap: 10px;
            height: 42px;
        }
        .kg-user-chip {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            border: 1px solid #e5eaf3;
            border-radius: 999px;
            background: #ffffff;
            color: #334155;
            font-weight: 800;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.045);
        }
        .kg-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            border-radius: 999px;
            color: #0f766e;
            background: #ecfdf5;
            border: 1px solid #ccfbf1;
            font-weight: 800;
            font-size: 0.86rem;
            white-space: nowrap;
        }
        .kg-card {
            background: rgba(255, 255, 255, 0.94);
            border: 1px solid var(--kg-border);
            border-radius: 22px;
            box-shadow: var(--kg-card-shadow);
            padding: 18px;
        }
        .kg-stat-card {
            min-height: 88px;
            background: #ffffff;
            border: 1px solid #e8eef7;
            border-radius: 18px;
            padding: 12px 14px;
            box-shadow: var(--kg-card-shadow);
            position: relative;
            overflow: hidden;
        }
        .kg-stat-card::after {
            content: "";
            position: absolute;
            width: 120px;
            height: 120px;
            right: -50px;
            top: -55px;
            border-radius: 999px;
            background: var(--accent-soft);
        }
        .kg-stat-head {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
            position: relative;
            z-index: 1;
        }
        .kg-stat-icon {
            width: 34px;
            height: 34px;
            border-radius: 15px;
            display: grid;
            place-items: center;
            background: var(--accent-soft);
            color: var(--accent);
            font-weight: 900;
        }
        .kg-stat-label {
            color: #334155;
            font-size: 0.92rem;
            font-weight: 800;
        }
        .kg-stat-value {
            font-size: 1.48rem;
            font-weight: 900;
            color: #0f172a;
            line-height: 1.1;
            margin: 4px 0;
            position: relative;
            z-index: 1;
        }
        .kg-stat-desc {
            color: #64748b;
            font-size: 0.82rem;
            line-height: 1.45;
            position: relative;
            z-index: 1;
        }
        .kg-section-title {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin: 8px 0 12px;
        }
        .kg-section-title h3 {
            margin: 0;
            font-size: 1.15rem;
            font-weight: 900;
        }
        .kg-section-title span {
            color: #64748b;
            font-size: 0.85rem;
        }
        .kg-chapter-card {
            border-radius: 22px;
            border: 1px solid var(--topic-border);
            background: linear-gradient(180deg, var(--topic-soft), #ffffff 56%);
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
            padding: 18px;
            min-height: 150px;
        }
        .kg-chapter-title {
            display: flex;
            gap: 10px;
            align-items: center;
            color: var(--topic);
            font-weight: 900;
            margin-bottom: 14px;
            line-height: 1.45;
        }
        .kg-diamond {
            width: 18px;
            height: 18px;
            border: 4px solid var(--topic);
            transform: rotate(45deg);
            border-radius: 4px;
            flex: 0 0 auto;
        }
        .kg-chapter-metrics {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            color: #334155;
            font-weight: 750;
            margin-bottom: 12px;
        }
        .kg-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }
        .kg-tag {
            display: inline-flex;
            padding: 5px 9px;
            border-radius: 999px;
            background: var(--topic-soft);
            color: var(--topic);
            font-size: 0.78rem;
            font-weight: 800;
        }
        .kg-chapter-overview {
            background: #ffffff;
            border: 1px solid #e5eaf3;
            border-radius: 18px;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.055);
            padding: 14px 16px;
            margin: 12px 0 14px;
        }
        .kg-chapter-overview-head {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 14px;
            margin-bottom: 12px;
        }
        .kg-chapter-overview-title {
            color: var(--topic);
            font-size: 1.2rem;
            font-weight: 900;
            margin-bottom: 4px;
        }
        .kg-chapter-overview-desc {
            color: #64748b;
            font-size: 0.88rem;
            line-height: 1.55;
        }
        .kg-overview-strip {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 10px;
        }
        .kg-sub-card {
            border: 1px solid #e5eaf3;
            border-radius: 14px;
            background: linear-gradient(180deg, #ffffff, #f8fafc);
            padding: 10px 12px;
            min-height: 58px;
        }
        .kg-sub-card strong {
            display: block;
            color: #0f172a;
            font-size: 0.92rem;
            margin-bottom: 3px;
        }
        .kg-sub-card span {
            color: #64748b;
            font-size: 0.78rem;
        }
        .kg-qa-card {
            background: #ffffff;
            border: 1px solid #e5eaf3;
            border-radius: 18px;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.055);
            padding: 14px 16px;
            margin-top: 12px;
        }
        .kg-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 8px;
        }
        .kg-light-chip {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 5px 10px;
            background: #eff6ff;
            color: #2563eb;
            border: 1px solid #dbeafe;
            font-size: 0.78rem;
            font-weight: 800;
        }
        .kg-relation-list {
            display: grid;
            gap: 10px;
            margin-bottom: 14px;
        }
        .kg-relation-item {
            display: grid;
            grid-template-columns: 12px 1fr;
            gap: 10px;
            align-items: start;
            color: #334155;
            line-height: 1.45;
            font-size: 0.9rem;
        }
        .kg-dot {
            width: 10px;
            height: 10px;
            margin-top: 5px;
            border-radius: 999px;
            background: var(--dot);
        }
        .kg-detail-box {
            border: 1px solid #e5eaf3;
            border-radius: 16px;
            background: #f8fafc;
            padding: 13px 14px;
            color: #334155;
            line-height: 1.75;
            font-size: 0.9rem;
        }
        div[data-testid="stDataFrame"] div[role="columnheader"],
        div[data-testid="stDataFrame"] div[role="gridcell"] {
            justify-content: center !important;
            text-align: center !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def button_nav(key: str, options: list[str]) -> str:
    """Render a rectangular button navigation instead of radio circles."""
    st.session_state.setdefault(key, options[0])
    if st.session_state[key] not in options:
        st.session_state[key] = options[0]
    for option in options:
        selected = st.session_state[key] == option
        label = option
        if st.button(label, key=f"{key}_{option}", type="primary" if selected else "secondary", use_container_width=True):
            st.session_state[key] = option
            st.rerun()
    return st.session_state[key]


def horizontal_button_nav(key: str, options: list[str]) -> str:
    """Render a compact horizontal button navigation for in-page views."""
    st.session_state.setdefault(key, options[0])
    if st.session_state[key] not in options:
        st.session_state[key] = options[0]
    columns = st.columns(len(options))
    for column, option in zip(columns, options):
        with column:
            selected = st.session_state[key] == option
            if st.button(option, key=f"{key}_{option}", type="primary" if selected else "secondary", use_container_width=True):
                st.session_state[key] = option
                st.rerun()
    return st.session_state[key]


def display_relation(relation: str) -> str:
    """Map internal relation names to user-facing labels."""
    return RELATION_LABELS.get(str(relation or ""), str(relation or ""))


def centered_dataframe(frame: pd.DataFrame, **kwargs) -> None:
    """Render a table with centered headers and cell text."""
    if frame is None:
        st.dataframe(frame, **kwargs)
        return
    styles = [
        {"selector": "th", "props": [("text-align", "center")]},
        {"selector": "td", "props": [("text-align", "center")]},
    ]
    styled = frame.style.set_properties(**{"text-align": "center"}).set_table_styles(styles)
    st.dataframe(styled, **kwargs)


def render_app_header() -> None:
    """Render the main product title for the dashboard."""
    st.markdown(
        """
        <div class="kg-hero">
          <div>
            <div class="kg-title">智能网联汽车教材知识图谱</div>
            <div class="kg-subtitle">面向教材知识组织、关系展示与教学辅助的可视化系统</div>
          </div>
          <div class="kg-pill">毕业设计展示界面</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_brand() -> None:
    """Render the left navigation brand block."""
    st.sidebar.markdown(
        """
        <div class="kg-sidebar-brand">
          <div class="kg-logo">KG</div>
          <div>
            <div class="kg-brand-title">智能网联汽车<br/>教材知识图谱</div>
            <div class="kg-brand-subtitle">教材知识组织与教学辅助</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_footer() -> None:
    """Render the static data update hint requested for the presentation UI."""
    update_time = "2024-05-12 16:30"
    if SUMMARY_PATH.exists():
        try:
            summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
            update_time = summary.get("updated_at") or update_time
        except Exception:
            pass
    st.sidebar.markdown(
        f"""
        <div class="kg-sidebar-footer">
          <strong>图谱数据更新</strong><br/>
          {html.escape(update_time)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def chapter_display_name(index: int, topic_name: str) -> str:
    """Return concise chapter labels for the left navigation."""
    names = ["第一章 绪论", "第二章 传感器", "第三章 深度学习应用", "第四章 强化学习应用"]
    if 0 <= index < len(names):
        return names[index]
    return short_topic_name(topic_name)


def render_sidebar_chapters(topics: dict) -> None:
    """Render four textbook chapters in the left sidebar."""
    st.sidebar.markdown('<div class="kg-chapter-nav-title">教材章节</div>', unsafe_allow_html=True)
    topic_styles = [
        ("#2563eb", "#eff6ff", "#bfdbfe"),
        ("#16a34a", "#f0fdf4", "#bbf7d0"),
        ("#9333ea", "#faf5ff", "#e9d5ff"),
        ("#ea580c", "#fff7ed", "#fed7aa"),
    ]
    current_topic_id = st.session_state.get("selected_topic_id")
    for index, topic in enumerate(topics.get("topics", [])[:4]):
        color, soft, border = topic_styles[index % len(topic_styles)]
        label = chapter_display_name(index, topic["topic_name"])
        selected = current_topic_id == topic["topic_id"]
        st.sidebar.markdown(
            f"""
            <div class="kg-chapter-mini" style="--chapter:{color};--chapter-soft:{soft};--chapter-border:{border};">
              <div class="kg-chapter-mini-title">{html.escape(label)}</div>
              <div class="kg-chapter-mini-desc">{int(topic.get("node_count", 0))} 个知识点 · {int(topic.get("edge_count", 0))} 条关系</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.sidebar.button("查看本章" if not selected else "当前章节", key=f"sidebar_topic_{topic['topic_id']}", type="primary" if selected else "secondary", use_container_width=True):
            st.session_state["selected_topic_id"] = topic["topic_id"]
            st.session_state["teacher_topic_id"] = topic["topic_id"]
            st.session_state["page"] = PAGE_TEACHER
            st.rerun()
        if selected:
            core_names = [item.get("name", "") for item in topic.get("core_nodes", [])[:4]]
            st.sidebar.caption("核心知识点：" + "、".join(core_names))


def render_sidebar_main_menu() -> str:
    """Render the main navigation in the same left-column style as the reference UI."""
    st.sidebar.markdown("### 功能导航")
    menu_items = [
        ("图谱总览", PAGE_OVERVIEW, None),
        ("教师端", PAGE_TEACHER, None),
        ("学生端", PAGE_STUDENT, "知识点查询"),
    ]
    current_page = st.session_state.get("page", PAGE_OVERVIEW)
    for label, target_page, target_view in menu_items:
        active = current_page == target_page
        if st.sidebar.button(label, key=f"main_menu_{label}", type="primary" if active else "secondary", use_container_width=True):
            st.session_state["page"] = target_page
            if target_view:
                st.session_state["student_view"] = target_view
            st.rerun()
    return st.session_state.get("page", PAGE_OVERVIEW)


def render_top_function_area() -> None:
    """Render the overview-only search area."""
    st.text_input(
        "顶部搜索",
        placeholder="搜索知识点、概念、关系、章节，回车后在总览图中定位...",
        label_visibility="collapsed",
        key="global_search_keyword",
    )


def render_stat_card(label: str, value: int | str, desc: str, accent: str, soft: str, icon: str) -> None:
    """Render one dashboard statistic card."""
    st.markdown(
        f"""
        <div class="kg-stat-card" style="--accent:{accent};--accent-soft:{soft};">
          <div class="kg-stat-head">
            <div class="kg-stat-icon">{html.escape(icon)}</div>
            <div class="kg-stat-label">{html.escape(label)}</div>
          </div>
          <div class="kg-stat-value">{html.escape(str(value))}</div>
          <div class="kg-stat-desc">{html.escape(desc)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_cards(nodes: pd.DataFrame, edges: pd.DataFrame, topics: dict) -> None:
    """Render the top statistic cards in the compact dashboard style."""
    evidence_count = int(edges["evidence_text"].astype(str).str.strip().ne("").sum()) if "evidence_text" in edges.columns else 0
    typed_node_count = int(nodes["entity_type"].astype(str).str.strip().ne("").sum()) if "entity_type" in nodes.columns else len(nodes)
    cards = [
        ("知识点总数", len(nodes), "教材知识点总量", "#2563eb", "#dbeafe", "书"),
        ("实体节点", typed_node_count, "已补充类型的图谱节点", "#14b8a6", "#ccfbf1", "点"),
        ("关系总数", len(edges), "知识点之间的语义关系", "#7c3aed", "#ede9fe", "联"),
        ("章节数", len(topics.get("topics", [])), "四个教材章节集群", "#f97316", "#ffedd5", "章"),
        ("资源证据", evidence_count, "可回溯的教材证据片段", "#06b6d4", "#cffafe", "证"),
    ]
    columns = st.columns(len(cards))
    for column, card in zip(columns, cards):
        with column:
            render_stat_card(*card)


def topic_keywords(topic: dict, limit: int = 5) -> list[str]:
    """Pick compact representative keywords for a chapter card."""
    keywords = [str(item) for item in topic.get("seed_keywords", []) if str(item).strip()]
    if not keywords:
        keywords = [node.get("name", "") for node in topic.get("core_nodes", []) if node.get("name")]
    return [short_text(keyword, 8) for keyword in keywords[:limit]]


def top_knowledge_table(nodes: pd.DataFrame, edges: pd.DataFrame, topics: dict) -> pd.DataFrame:
    """Build the user-facing Top 5 knowledge point table."""
    graph = build_graph(nodes, edges)
    centrality = nx.degree_centrality(graph) if graph.number_of_nodes() > 1 else {}
    degree_map = dict(graph.degree())
    topic_map = node_topic_names(topics)
    table = (
        nodes.assign(
            _score=nodes["node_id"].map(centrality).fillna(0.0),
            关联数=nodes["node_id"].map(degree_map).fillna(0).astype(int),
            所属章节=nodes["node_id"].map(lambda node_id: "、".join(topic_map.get(node_id, [])) or "未归入章节"),
        )
        .sort_values(["_score", "关联数", "name"], ascending=[False, False, True])
        .head(5)
        .reset_index(drop=True)
    )
    return pd.DataFrame(
        {
            "排名": range(1, len(table) + 1),
            "知识点名称": table["name"],
            "类型": table["entity_type"],
            "所属章节": table["所属章节"].map(lambda value: short_text(value, 18)),
            "关联数": table["关联数"],
            "重要度": table["_score"].round(3),
        }
    )


def get_understood_node_ids() -> set[str]:
    """Return knowledge points that the learner has marked as understood."""
    if "understood_node_ids" not in st.session_state:
        if PROGRESS_PATH.exists():
            try:
                progress_data = json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
                st.session_state["understood_node_ids"] = progress_data.get("understood_node_ids", [])
            except Exception:
                st.session_state["understood_node_ids"] = []
        else:
            st.session_state["understood_node_ids"] = []
    return set(st.session_state["understood_node_ids"])


def set_node_understood(node_id: str, understood: bool) -> None:
    """Update the learner's understanding state for one knowledge point."""
    current = get_understood_node_ids()
    if understood:
        current.add(node_id)
    else:
        current.discard(node_id)
    st.session_state["understood_node_ids"] = sorted(current)
    KG_DIR.mkdir(parents=True, exist_ok=True)
    PROGRESS_PATH.write_text(
        json.dumps(
            {
                "understood_node_ids": st.session_state["understood_node_ids"],
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def topic_progress(topic: dict) -> tuple[int, int, int]:
    """Calculate real chapter progress from understood knowledge points."""
    topic_node_ids = set(topic.get("node_ids", []))
    total = len(topic_node_ids)
    understood = len(topic_node_ids & get_understood_node_ids())
    percent = round(understood / total * 100) if total else 0
    return understood, total, percent


def render_chapter_cards(topics: dict) -> None:
    """Render the four chapter cards with soft colors and keyword tags."""
    topic_styles = [
        ("#2563eb", "#eff6ff", "#bfdbfe"),
        ("#16a34a", "#f0fdf4", "#bbf7d0"),
        ("#9333ea", "#faf5ff", "#e9d5ff"),
        ("#ea580c", "#fff7ed", "#fed7aa"),
    ]
    topic_items = topics.get("topics", [])[:4]
    columns = st.columns(4)
    for index, (column, topic) in enumerate(zip(columns, topic_items)):
        color, soft, border = topic_styles[index % len(topic_styles)]
        tags = "".join(f'<span class="kg-tag">{html.escape(keyword)}</span>' for keyword in topic_keywords(topic))
        with column:
            st.markdown(
                f"""
                <div class="kg-chapter-card" style="--topic:{color};--topic-soft:{soft};--topic-border:{border};">
                  <div class="kg-chapter-title"><span class="kg-diamond"></span><span>{html.escape(topic["topic_name"])}</span></div>
                  <div class="kg-chapter-metrics">
                    <div>知识点数&nbsp;&nbsp;<strong>{int(topic.get("node_count", 0))}</strong></div>
                    <div>关系数&nbsp;&nbsp;<strong>{int(topic.get("edge_count", 0))}</strong></div>
                  </div>
                  <div class="kg-tags">{tags}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_quick_entries() -> None:
    """Render a quick guide without duplicating the left-side navigation."""
    st.markdown(
        """
        <div class="kg-section-title">
          <h3>功能入口</h3>
          <span>所有教师和学生功能统一通过左侧功能导航切换</span>
        </div>
        <div class="kg-mini-card">
          <h4>左侧导航说明</h4>
          <p>教师相关内容请进入“数据分析”；学生相关内容请进入“知识点查询、学习路径、知识问答、资源管理”。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def short_text(value: str, limit: int = 12) -> str:
    """Keep labels compact for buttons and graph nodes."""
    text = str(value or "")
    return text if len(text) <= limit else text[:limit] + "…"


def short_topic_name(name: str) -> str:
    """Compress chapter names for rectangular topic buttons."""
    replacements = {
        "第一章 智能网联汽车基础与体系结构": "第一章 体系结构",
        "第一章 绪论与智能网联汽车基础": "第一章 体系结构",
        "第二章 车载传感器与环境感知": "第二章 感知",
        "第三章 人工智能与深度学习方法": "第三章 深度学习",
        "第三章 人工智能与深度学习应用": "第三章 深度学习",
        "第四章 强化学习与车路协同应用": "第四章 强化学习",
    }
    return replacements.get(name, short_text(name, 14))


def topic_button_selector(topics: dict, *, key: str, selected_id: str | None = None) -> str:
    """Render chapter topics as rectangular buttons."""
    topic_items = topics.get("topics", [])
    if not topic_items:
        return ""
    valid_ids = {topic["topic_id"] for topic in topic_items}
    if key not in st.session_state:
        st.session_state[key] = selected_id if selected_id in valid_ids else topic_items[0]["topic_id"]
    if st.session_state[key] not in valid_ids:
        st.session_state[key] = topic_items[0]["topic_id"]
    for topic in topic_items:
        selected = st.session_state[key] == topic["topic_id"]
        prefix = "当前：" if selected else ""
        label = f"{prefix}{short_topic_name(topic['topic_name'])}"
        if st.button(label, key=f"{key}_{topic['topic_id']}", type="primary" if selected else "secondary", use_container_width=True):
            st.session_state[key] = topic["topic_id"]
            st.session_state["selected_topic_id"] = topic["topic_id"]
            st.rerun()
    return st.session_state[key]


def render_selected_chapter_overview(topics: dict) -> None:
    """Render the selected chapter summary above the graph, matching the UI spec."""
    topic_items = topics.get("topics", [])
    if not topic_items:
        return
    topic_map = topic_by_id(topics)
    selected_id = st.session_state.get("selected_topic_id", topic_items[0]["topic_id"])
    topic = topic_map.get(selected_id, topic_items[0])
    topic_index = next((idx for idx, item in enumerate(topic_items) if item["topic_id"] == topic["topic_id"]), 0)
    color = TOPIC_BORDER_COLORS[topic_index % len(TOPIC_BORDER_COLORS)]
    soft = TOPIC_COLORS[topic_index % len(TOPIC_COLORS)]
    core_count = len(topic.get("core_nodes", []))
    cross_count = len(topic.get("cross_topic_edges", []))
    understood_count, total_count, progress_percent = topic_progress(topic)
    core_cards = []
    for node in topic.get("core_nodes", [])[:5]:
        core_cards.append(
            f'<div class="kg-sub-card"><strong>{html.escape(short_text(node.get("name", ""), 12))}</strong><span>{int(node.get("degree", 0))} 条关联关系</span></div>'
        )
    while len(core_cards) < 5:
        fallback_label = ["智能网联汽车概述", "发展历程与趋势", "技术体系构成", "关键技术一览", "标准与法规"][len(core_cards)]
        core_cards.append(
            f'<div class="kg-sub-card"><strong>{html.escape(fallback_label)}</strong><span>章节知识点概览</span></div>'
        )
    chapter_html = (
        f'<div class="kg-chapter-overview" style="--topic:{color};--topic-soft:{soft};">'
        '<div class="kg-chapter-overview-head">'
        '<div>'
        f'<div class="kg-chapter-overview-title">{html.escape(chapter_display_name(topic_index, topic.get("topic_name", "")))}</div>'
        f'<div class="kg-chapter-overview-desc">{html.escape(topic.get("topic_description", "围绕教材章节组织知识点、语义关系和教材证据。"))}</div>'
        '</div>'
        '<div class="kg-tags">'
        f'<span class="kg-tag">知识点 {int(topic.get("node_count", 0))}</span>'
        f'<span class="kg-tag">核心 {core_count}</span>'
        f'<span class="kg-tag">关联章节 {cross_count}</span>'
        '<span class="kg-tag">约45分钟</span>'
        f'<span class="kg-tag">掌握进度 {progress_percent}%</span>'
        f'<span class="kg-tag">已理解 {understood_count}/{total_count}</span>'
        '</div>'
        '</div>'
        f'<div class="kg-overview-strip">{"".join(core_cards)}</div>'
        '</div>'
    )
    st.markdown(chapter_html, unsafe_allow_html=True)


TYPE_COLORS = {
    "技术": "#3b82f6",
    "功能": "#f97316",
    "设备": "#dc2626",
    "场景": "#0891b2",
    "类别": "#16a34a",
    "模块": "#9333ea",
    "未知": "#64748b",
}


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load graph tables and topic clusters."""
    nodes = pd.read_csv(NODES_PATH, encoding="utf-8-sig")
    edges = pd.read_csv(EDGES_PATH, encoding="utf-8-sig")
    topics = json.loads(TOPIC_PATH.read_text(encoding="utf-8")) if TOPIC_PATH.exists() else {"topics": []}

    for column in ["node_id", "name", "entity_type", "raw_name"]:
        if column not in nodes.columns:
            nodes[column] = ""
        nodes[column] = nodes[column].fillna("").astype(str)

    for column in ["source", "target", "relation", "source_name", "target_name", "source_file", "evidence_text", "context_snippet", "block_id"]:
        if column not in edges.columns:
            edges[column] = ""
        edges[column] = edges[column].fillna("").astype(str)

    if "edge_id" not in edges.columns:
        edges["edge_id"] = [f"edge_{idx:05d}" for idx in range(len(edges))]
    edges = edges[edges["relation"].isin(RELATION_OPTIONS)].copy().reset_index(drop=True)
    return nodes, edges, topics


@st.cache_data(show_spinner=False)
def load_resource_library() -> pd.DataFrame:
    """Load the pre-curated real resource link library."""
    columns = ["title", "type", "source", "url", "topics", "keywords", "reason"]
    if not RESOURCE_LIBRARY_PATH.exists():
        return pd.DataFrame(columns=columns)
    records = json.loads(RESOURCE_LIBRARY_PATH.read_text(encoding="utf-8"))
    frame = pd.DataFrame(records)
    for column in columns:
        if column not in frame.columns:
            frame[column] = "" if column not in {"topics", "keywords"} else [[] for _ in range(len(frame))]
    return frame


def rebuild_kg_from_uploaded_file(uploaded_file) -> dict:
    """Import a new triples file and rebuild display-ready graph outputs."""
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in {".json", ".csv"}:
        raise ValueError("仅支持 JSON 或 CSV 三元组文件。")

    from kg import build_knowledge_graph as kg_builder
    from kg import build_topic_clusters as topic_builder

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = re.sub(r"[^0-9A-Za-z_.\-\u4e00-\u9fff]", "_", uploaded_file.name)
    upload_path = UPLOAD_DIR / f"{timestamp}_{safe_name}"
    upload_path.write_bytes(uploaded_file.getvalue())

    dictionary_path = kg_builder.find_dictionary_file(None)
    triples = kg_builder.load_triples(upload_path)
    build_result = kg_builder.build_records(triples, upload_path, dictionary_path)
    analysis = kg_builder.analyze_graph(build_result["nodes"], build_result["edges"])

    nodes_rows = sorted(build_result["nodes"].values(), key=lambda item: item["name"])
    edges_rows = sorted(
        build_result["edges"],
        key=lambda item: (item["relation"], item["source_name"], item["target_name"]),
    )

    KG_DIR.mkdir(parents=True, exist_ok=True)
    kg_builder.write_csv(
        NODES_PATH,
        ["node_id", "name", "entity_type", "raw_name", "alias_count"],
        nodes_rows,
    )
    kg_builder.write_csv(
        EDGES_PATH,
        [
            "source",
            "target",
            "relation",
            "source_name",
            "target_name",
            "source_file",
            "evidence_text",
            "block_id",
            "triple_id",
            "confidence",
            "status",
            "evidence_count",
        ],
        edges_rows,
    )

    summary = {
        "input_file": upload_path.relative_to(ROOT_DIR).as_posix(),
        "dictionary_file": dictionary_path.relative_to(ROOT_DIR).as_posix() if dictionary_path else None,
        "raw_edge_count": build_result["raw_edge_count"],
        "edge_count_after_dedup": analysis["edge_count"],
        "node_count": analysis["node_count"],
        "relation_counts": analysis["relation_counts"],
        "node_type_counts": analysis["node_type_counts"],
        "connected_components": analysis["connected_components"],
        "largest_component_size": analysis["largest_component_size"],
        "duplicate_edge_instances_removed": build_result["duplicate_edge_instances"],
        "review_filtered_count": build_result["review_filtered_count"],
        "skipped_relations": dict(build_result["skipped_relations"]),
        "self_loop_count": analysis["self_loop_count"],
        "multi_relation_conflict_count": analysis["multi_relation_conflict_count"],
        "isa_bidirectional_conflict_count": analysis["isa_bidirectional_conflict_count"],
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    kg_builder.write_summary_json(SUMMARY_PATH, summary)
    kg_builder.write_markdown_report(
        KG_DIR / "kg_analysis_report.md",
        summary,
        analysis,
        build_result["duplicate_edge_instances"],
        build_result["review_filtered_count"],
        dict(build_result["skipped_relations"]),
    )
    kg_builder.write_neo4j_guide(KG_DIR / "neo4j_import_guide.md")
    kg_builder.write_neo4j_cypher(KG_DIR / "neo4j_import.cypher")

    topic_nodes, topic_edges = topic_builder.load_graph_frames(NODES_PATH, EDGES_PATH)
    cluster_data = topic_builder.build_topic_clusters(topic_nodes, topic_edges)
    TOPIC_PATH.write_text(json.dumps(cluster_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    topic_builder.write_summary_markdown(cluster_data, KG_DIR / "topic_cluster_summary.md")

    load_data.clear()
    return {
        "uploaded_file": upload_path.relative_to(ROOT_DIR).as_posix(),
        "node_count": analysis["node_count"],
        "edge_count": analysis["edge_count"],
        "topic_count": cluster_data["metadata"]["topic_count"],
    }


def render_kg_update_panel() -> None:
    """Render a sidebar uploader for replacing the graph with a new triples file."""
    with st.expander("更新知识图谱", expanded=True):
        st.caption("上传新的三元组 JSON/CSV 后，将自动重建当前展示使用的图谱表和章节主题。")
        result = st.session_state.pop("kg_update_result", None)
        if result:
            st.success(
                f"最近更新完成：{result['node_count']} 个知识点，"
                f"{result['edge_count']} 条关系，{result['topic_count']} 个章节主题。"
            )
        uploaded_file = st.file_uploader(
            "导入新的三元组文件",
            type=["json", "csv"],
            accept_multiple_files=False,
            key="kg_triples_uploader",
        )
        if uploaded_file is not None:
            st.caption(f"已选择：{uploaded_file.name}")
        if st.button("更新知识图谱", use_container_width=True, disabled=uploaded_file is None):
            try:
                with st.spinner("正在重建知识图谱，请稍候..."):
                    result = rebuild_kg_from_uploaded_file(uploaded_file)
                st.session_state["kg_update_result"] = result
                st.rerun()
            except Exception as exc:
                st.error(f"更新失败：{exc}")


def build_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.Graph:
    """Build an undirected graph for component filtering and paths."""
    graph = nx.Graph()
    for _, row in nodes.iterrows():
        graph.add_node(row["node_id"], name=row["name"])
    for _, row in edges.iterrows():
        graph.add_edge(row["source"], row["target"], edge_id=row["edge_id"], relation=row["relation"])
    return graph


def relation_summary(edges: pd.DataFrame) -> pd.DataFrame:
    """Return the three relation counts in a stable order."""
    counts = edges["relation"].value_counts().to_dict()
    return pd.DataFrame([{"关系": display_relation(rel), "数量": int(counts.get(rel, 0))} for rel in RELATION_OPTIONS])


def topic_by_id(topics: dict) -> dict[str, dict]:
    """Index topic cluster records by topic id."""
    return {topic["topic_id"]: topic for topic in topics.get("topics", [])}


def node_topic_names(topics: dict) -> dict[str, list[str]]:
    """Collect the topic names that contain each node."""
    mapping: dict[str, list[str]] = {}
    for topic in topics.get("topics", []):
        for node_id in topic.get("node_ids", []):
            mapping.setdefault(node_id, []).append(topic["topic_name"])
    return mapping


def get_topic_edges(topic: dict, edges: pd.DataFrame) -> pd.DataFrame:
    """Get all edges that belong to a topic cluster."""
    edge_ids = set(topic.get("edge_ids", []))
    return edges[edges["edge_id"].isin(edge_ids)].copy()


def get_topic_nodes(topic: dict, nodes: pd.DataFrame, topic_edges: pd.DataFrame) -> pd.DataFrame:
    """Get topic nodes, preferring nodes that appear in the topic edge set."""
    node_ids = set(topic.get("node_ids", []))
    if not topic_edges.empty:
        node_ids.update(topic_edges["source"].tolist())
        node_ids.update(topic_edges["target"].tolist())
    return nodes[nodes["node_id"].isin(node_ids)].copy()


def filter_small_components(nodes: pd.DataFrame, edges: pd.DataFrame, max_component_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hide connected components whose node count is less than or equal to N."""
    if max_component_size <= 0 or nodes.empty:
        return nodes.copy(), edges.copy()

    graph = build_graph(nodes, edges)
    keep_nodes = set()
    for component in nx.connected_components(graph):
        if len(component) > max_component_size:
            keep_nodes.update(component)

    filtered_nodes = nodes[nodes["node_id"].isin(keep_nodes)].copy()
    filtered_edges = edges[
        edges["source"].isin(keep_nodes) & edges["target"].isin(keep_nodes)
    ].copy()
    return filtered_nodes, filtered_edges


def split_main_and_supplemental_edges(
    topic_nodes: pd.DataFrame,
    topic_edges: pd.DataFrame,
    max_component_size: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Move small components into a supplemental list to keep the main canvas readable."""
    if max_component_size <= 0:
        return topic_edges.copy(), topic_edges.iloc[0:0].copy()

    graph = build_graph(topic_nodes, topic_edges)
    supplemental_nodes = set()
    for component in nx.connected_components(graph):
        if len(component) <= max_component_size:
            supplemental_nodes.update(component)
    supplemental = topic_edges[
        topic_edges["source"].isin(supplemental_nodes) & topic_edges["target"].isin(supplemental_nodes)
    ].copy()
    main = topic_edges.drop(index=supplemental.index).copy()
    return main, supplemental


def get_cross_topic_edges(topic: dict) -> pd.DataFrame:
    """Convert the cross-topic edge records stored in JSON into a DataFrame."""
    records = topic.get("cross_topic_edges", [])
    if not records:
        return pd.DataFrame(columns=["edge_id", "source", "target", "source_name", "relation", "target_name", "source_file", "block_id", "evidence_text"])
    frame = pd.DataFrame(records)
    frame["is_cross_topic"] = True
    return frame


def ensure_edge_flags(edges: pd.DataFrame, is_cross: bool = False) -> pd.DataFrame:
    """Add display flags used by the graph renderer."""
    output = edges.copy()
    output["is_cross_topic"] = bool(is_cross)
    return output


def filter_edges_by_relations(edges: pd.DataFrame, relations: list[str]) -> pd.DataFrame:
    """Filter graph edges by the three allowed relation types."""
    if edges.empty:
        return edges.copy()
    if not relations:
        return edges.iloc[0:0].copy()
    return edges[edges["relation"].isin(relations)].copy()


def nodes_used_by_edges(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Keep only nodes that still have visible edges after filtering."""
    if edges.empty:
        return nodes.iloc[0:0].copy()
    visible_ids = set(edges["source"]) | set(edges["target"])
    return nodes[nodes["node_id"].isin(visible_ids)].copy()


def apply_search_highlight(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    search_node_id: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Highlight a searched node and its direct visible edges without changing layout rules."""
    if not search_node_id:
        return nodes.copy(), edges.copy(), ""
    output_nodes = nodes.copy()
    output_edges = edges.copy()
    if search_node_id in set(output_nodes["node_id"]):
        output_nodes.loc[output_nodes["node_id"] == search_node_id, "display_role"] = "search_hit"
        if not output_edges.empty:
            search_edge_mask = (output_edges["source"] == search_node_id) | (output_edges["target"] == search_node_id)
            output_edges.loc[search_edge_mask, "display_role"] = "search_edge"
        return output_nodes, output_edges, search_node_id
    return output_nodes, output_edges, ""


def make_network_html(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    title: str,
    *,
    height: int = 680,
    physics_enabled: bool = True,
    region_boxes: list[dict] | None = None,
    focus_node_id: str | None = None,
    spring_length: int = 220,
    gravitational_constant: int = -150,
    central_gravity: float = 0.012,
    avoid_overlap: float = 1.8,
    stabilization_iterations: int = 260,
) -> str:
    """Render a clickable vis-network graph with an embedded detail panel."""
    def script_json(value: object) -> str:
        """Safely embed JSON in a script tag without breaking the iframe HTML."""
        return (
            json.dumps(value, ensure_ascii=False)
            .replace("<", "\\u003c")
            .replace(">", "\\u003e")
            .replace("&", "\\u0026")
            .replace("\u2028", "\\u2028")
            .replace("\u2029", "\\u2029")
        )

    node_lookup = nodes.set_index("node_id").to_dict("index") if not nodes.empty else {}
    graph = build_graph(nodes, edges)
    degree_map = dict(graph.degree())
    relation_map: dict[str, set[str]] = {}
    for _, edge_row in edges.iterrows():
        label = display_relation(edge_row.get("relation", ""))
        relation_map.setdefault(edge_row.get("source", ""), set()).add(label)
        relation_map.setdefault(edge_row.get("target", ""), set()).add(label)

    node_items = []
    for _, row in nodes.iterrows():
        entity_type = row.get("entity_type", "未知") or "未知"
        display_role = row.get("display_role", "")
        display_color = row.get("display_color", "")
        topic_id_value = row.get("overview_topic_id", "")
        if pd.isna(topic_id_value):
            topic_id_value = ""
        chapter_value = row.get("chapter_name", "")
        if pd.isna(chapter_value) or not str(chapter_value).strip():
            chapter_value = "可在章节导航中查看"
        relation_text = "、".join(sorted(relation_map.get(row["node_id"], set()))) or "暂无直接关系"
        degree_count = degree_map.get(row["node_id"], 0)
        node_shape = "dot"
        node_size = 22 + min(degree_count, 10) * 2
        node_color = {
            "background": TYPE_COLORS.get(entity_type, TYPE_COLORS["未知"]),
            "border": "#1f2937",
            "highlight": {"background": "#fef3c7", "border": "#f59e0b"},
        }
        border_width = 1.4
        font_size = 15
        if display_role == "overview_center":
            node_shape = "square"
            node_size = 54
            node_color = {
                "background": "#fde047",
                "border": "#be123c",
                "highlight": {"background": "#fef08a", "border": "#e11d48"},
            }
            border_width = 4.5
            font_size = 20
        elif display_role == "topic_center":
            node_shape = "diamond"
            node_size = 50
            node_color = {
                "background": "#ffffff",
                "border": display_color or "#7c3aed",
                "highlight": {"background": "#fef3c7", "border": "#e11d48"},
            }
            border_width = 4.6
            font_size = 18
        elif display_role == "overview_member":
            node_size = 30 + min(degree_count, 8)
            border_width = 2.2
            font_size = 16
        elif display_role == "search_hit":
            node_size = max(node_size, 52)
            node_color = {
                "background": "#fef08a",
                "border": "#dc2626",
                "highlight": {"background": "#fde68a", "border": "#b91c1c"},
            }
            border_width = 5
            font_size = 19
        node_items.append(
            {
                "id": row["node_id"],
                "label": row["name"],
                "displayRole": display_role,
                "topicId": str(topic_id_value),
                "shape": node_shape,
                "size": node_size,
                "color": node_color,
                "borderWidth": border_width,
                "font": {"color": "#111827", "size": font_size, "face": "Microsoft YaHei", "strokeWidth": 4, "strokeColor": "#ffffff"},
                "title": html.escape(f"{row['name']}｜{entity_type}"),
                "detail": {
                    "名称": row["name"],
                    "类型": entity_type,
                    "所属章节": str(chapter_value),
                    "简要说明": f"该知识点与 {degree_count} 个相关知识点存在教材语义关系，可用于查看概念归属、组成结构或功能场景。",
                    "关联知识点数": str(degree_count),
                    "相关关系": relation_text,
                    "来源信息": row.get("raw_name", row["name"]),
                },
            }
        )
        if "x" in row and "y" in row and str(row.get("x", "")).strip() and str(row.get("y", "")).strip():
            try:
                node_items[-1]["x"] = float(row.get("x"))
                node_items[-1]["y"] = float(row.get("y"))
            except (TypeError, ValueError):
                pass

    edge_items = []
    for _, row in edges.iterrows():
        relation = row["relation"]
        is_cross = bool(row.get("is_cross_topic", False))
        is_topic_link = relation == "主题连接"
        is_membership_link = relation == "章节归属"
        edge_display_role = row.get("display_role", "")
        if pd.isna(edge_display_role):
            edge_display_role = ""
        is_search_edge = edge_display_role == "search_edge"
        edge_color = "#ef4444" if is_search_edge else ("#be123c" if is_topic_link else ("#94a3b8" if is_membership_link else ("#cbd5e1" if is_cross else RELATION_COLORS.get(relation, "#64748b"))))
        edge_width = 4.2 if is_search_edge else (4.4 if is_topic_link else (1.05 if is_membership_link else (1.2 if is_cross else 2.2)))
        edge_dashes = [14, 8] if is_topic_link else ([4, 8] if is_membership_link else bool(is_cross))
        edge_length = 680 if is_topic_link else (270 if is_cross else 220)
        edge_items.append(
            {
                "id": row["edge_id"],
                "from": row["source"],
                "to": row["target"],
                "relation": relation,
                "isThemeLink": is_topic_link or is_membership_link,
                "length": edge_length,
                "arrows": "" if is_topic_link else "to",
                "width": edge_width,
                "dashes": edge_dashes,
                "color": {"color": edge_color},
                "smooth": False,
                "title": html.escape(f"{row['source_name']} -[{display_relation(relation)}]-> {row['target_name']}"),
                "detail": {
                    "名称": f"{row.get('source_name', '')} → {row.get('target_name', '')}",
                    "类型": display_relation(relation),
                    "所属章节": "跨章节关联" if is_cross else "当前章节或当前图谱视图",
                    "简要说明": f"{row.get('source_name', '')} 与 {row.get('target_name', '')} 之间存在“{display_relation(relation)}”。",
                    "关联知识点数": "2",
                    "相关关系": display_relation(relation),
                    "来源信息": f"{row.get('source_file', '')} {row.get('block_id', '')}".strip() or "暂无来源信息",
                    "教材证据": row.get("evidence_text", "") or "无",
                },
            }
        )

    default_detail = {
        "名称": "教材知识体系",
        "类型": "核心节点",
        "所属章节": "全书",
        "简要说明": "用于组织智能网联汽车教材中的章节主题、知识点和语义关系。",
        "关联知识点数": str(max(len(nodes) - 1, 0)),
        "相关关系": "概念归属、组成关系、功能关系、章节关联",
        "来源信息": title,
    }
    region_boxes = region_boxes or []

    template = Template(
        r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    body { margin: 0; font-family: "Microsoft YaHei", "Segoe UI", sans-serif; background: #ffffff; }
    .wrap { display: grid; grid-template-columns: 1fr 330px; height: ${height}px; border: 1px solid #e5eaf3; border-radius: 22px; overflow: hidden; background: #fff; box-shadow: 0 12px 30px rgba(15, 23, 42, 0.05); }
    #network { height: ${height}px; min-width: 0; }
    .side { border-left: 1px solid #e5eaf3; padding: 16px; overflow: auto; background: linear-gradient(180deg, #ffffff, #f8fafc); }
    .side-card { border: 1px solid #e5eaf3; border-radius: 18px; background: #ffffff; padding: 14px; margin-bottom: 14px; box-shadow: 0 8px 18px rgba(15, 23, 42, 0.045); }
    .title { font-weight: 900; color: #0f172a; margin-bottom: 10px; font-size: 16px; }
    .legend { display: grid; gap: 9px; margin: 8px 0 14px; font-size: 13px; color: #334155; }
    .legend span { display: inline-block; width: 30px; height: 4px; border-radius: 999px; vertical-align: middle; margin-right: 8px; }
    .relation-note { color: #64748b; font-size: 12px; line-height: 1.6; }
    .toolbar { display: grid; grid-template-columns: 1fr; gap: 8px; margin: 10px 0 12px; }
    .toolbar button { border: 1px solid #cbd5e1; background: #f8fafc; color: #0f172a; border-radius: 12px; padding: 9px 10px; cursor: pointer; font-weight: 800; }
    .toolbar button:hover { border-color: #14b8a6; color: #0f766e; background: #f0fdfa; }
    .graph-panel { position: relative; min-width: 0; }
    .graph-controls {
      position: absolute;
      top: 12px;
      right: 12px;
      z-index: 5;
      display: flex;
      gap: 8px;
      align-items: center;
    }
    .graph-controls button {
      border: 1px solid #93c5fd;
      background: #eff6ff;
      color: #2563eb;
      border-radius: 10px;
      min-width: 34px;
      height: 34px;
      padding: 0 10px;
      font-weight: 800;
      cursor: pointer;
      box-shadow: 0 8px 22px rgba(15, 23, 42, 0.14);
    }
    .graph-controls button:hover { background: #dbeafe; }
    .bottom-legend {
      position: absolute;
      left: 20px;
      right: 20px;
      bottom: 14px;
      z-index: 4;
      display: flex;
      justify-content: center;
      gap: 18px;
      align-items: center;
      padding: 9px 12px;
      border: 1px solid #e5eaf3;
      border-radius: 14px;
      background: rgba(255, 255, 255, 0.92);
      box-shadow: 0 8px 20px rgba(15, 23, 42, 0.08);
      color: #475569;
      font-size: 12px;
      pointer-events: none;
    }
    .bottom-legend i {
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 999px;
      margin-right: 6px;
      vertical-align: -1px;
    }
    .bottom-legend .dash {
      width: 26px;
      height: 0;
      border-top: 2px dashed #94a3b8;
      border-radius: 0;
    }
    .kv { margin: 8px 0; border-bottom: 1px solid #f1f5f9; padding-bottom: 7px; }
    .k { font-size: 12px; color: #64748b; }
    .v { font-size: 13px; color: #111827; word-break: break-word; white-space: pre-wrap; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="graph-panel">
      <div class="graph-controls">
        <button id="zoomInGraph" type="button" title="放大">+</button>
        <button id="zoomOutGraph" type="button" title="缩小">-</button>
        <button id="fitGraph" type="button" title="适应画布">适应</button>
        <button id="resetGraphFloat" type="button" title="重置视图">重置</button>
      </div>
      <div id="network"></div>
      <div class="bottom-legend">
        <span><i style="background:#2563eb"></i>第一章 绪论</span>
        <span><i style="background:#16a34a"></i>第二章 传感器</span>
        <span><i style="background:#9333ea"></i>第三章 深度学习应用</span>
        <span><i style="background:#ea580c"></i>第四章 强化学习应用</span>
        <span><i class="dash"></i>跨章节关系</span>
      </div>
    </div>
    <div class="side">
      <div class="side-card">
        <div class="title">关系说明</div>
        <div class="legend">
          <div><span style="background:#7c3aed"></span>概念归属：具体知识点 → 上位类别</div>
          <div><span style="background:#f97316"></span>组成关系：整体系统 → 部件/模块</div>
          <div><span style="background:#16a34a"></span>功能关系：技术/设备/模块 → 功能/场景</div>
          <div><span style="background:#94a3b8;border-top:1px dashed #64748b"></span>章节关联：知识点与教材章节之间的关联</div>
        </div>
        <div class="toolbar">
          <button id="resetGraph" type="button">重置视图</button>
        </div>
      </div>
      <div class="side-card">
        <div class="title">知识点详情</div>
        <div id="detail"></div>
      </div>
    </div>
  </div>
  <script>
    const fullNodes = ${nodes_json};
    const fullEdges = ${edges_json};
    const nodes = new vis.DataSet(fullNodes);
    const edges = new vis.DataSet(fullEdges);
    const nodeDetails = ${node_details_json};
    const edgeDetails = ${edge_details_json};
    const defaultDetail = ${default_detail_json};
    const regionBoxes = ${region_boxes_json};
    const initialFocusNodeId = ${focus_node_id_json};
    const basePhysicsEnabled = ${physics_enabled_json};
    let isFocused = false;

    function renderDetail(data) {
      const target = document.getElementById("detail");
      target.innerHTML = Object.entries(data).map(([k, v]) =>
        `<div class="kv"><div class="k">$${k}</div><div class="v">$${String(v || "无")}</div></div>`
      ).join("");
    }

    function fitSoon(useInitialFocus = false) {
      setTimeout(() => {
        if (useInitialFocus && initialFocusNodeId && nodes.get(initialFocusNodeId)) {
          network.selectNodes([initialFocusNodeId]);
          network.focus(initialFocusNodeId, {
            scale: 1.15,
            animation: { duration: 260, easingFunction: "easeInOutQuad" }
          });
          return;
        }
        network.fit({ animation: { duration: 240, easingFunction: "easeInOutQuad" } });
      }, 80);
    }

    function resetGraph() {
      isFocused = false;
      nodes.clear();
      edges.clear();
      nodes.add(fullNodes);
      edges.add(fullEdges);
      renderDetail(defaultDetail);
      network.setOptions({ physics: { enabled: basePhysicsEnabled } });
      if (basePhysicsEnabled) network.stabilize(120);
      fitSoon();
    }

    function focusTopic(topicId, nodeId) {
      isFocused = true;
      const topicNodes = fullNodes.filter(node => node.topicId === topicId);
      const visibleIds = new Set(topicNodes.map(node => node.id));
      const topicEdges = fullEdges.filter(edge => visibleIds.has(edge.from) && visibleIds.has(edge.to));
      nodes.clear();
      edges.clear();
      nodes.add(topicNodes);
      edges.add(topicEdges);
      const detail = Object.assign({}, nodeDetails[nodeId] || {});
      detail["章节子图"] = "已显示该章节全部节点 " + topicNodes.length + " 个，关系 " + topicEdges.length + " 条";
      renderDetail(detail);
      network.setOptions({ physics: { enabled: basePhysicsEnabled } });
      if (basePhysicsEnabled) network.stabilize(120);
      fitSoon();
    }

    function focusNode(nodeId) {
      if (nodeId === "__course_center__") {
        resetGraph();
        return;
      }
      const clickedNode = fullNodes.find(node => node.id === nodeId);
      if (clickedNode && clickedNode.displayRole === "topic_center" && clickedNode.topicId) {
        focusTopic(clickedNode.topicId, nodeId);
        return;
      }
      isFocused = true;
      const graphEdges = fullEdges.filter(edge => !edge.isThemeLink);
      const adjacency = new Map();
      fullNodes.forEach(node => adjacency.set(node.id, new Set()));
      graphEdges.forEach(edge => {
        if (!adjacency.has(edge.from)) adjacency.set(edge.from, new Set());
        if (!adjacency.has(edge.to)) adjacency.set(edge.to, new Set());
        adjacency.get(edge.from).add(edge.to);
        adjacency.get(edge.to).add(edge.from);
      });
      const visibleIds = new Set();
      const queue = [nodeId];
      while (queue.length) {
        const current = queue.shift();
        if (visibleIds.has(current)) continue;
        visibleIds.add(current);
        (adjacency.get(current) || []).forEach(next => {
          if (!visibleIds.has(next)) queue.push(next);
        });
      }
      const relatedEdges = fullEdges.filter(edge =>
        !edge.isThemeLink && visibleIds.has(edge.from) && visibleIds.has(edge.to)
      );
      const focusedNodes = fullNodes.filter(node => visibleIds.has(node.id));
      nodes.clear();
      edges.clear();
      nodes.add(focusedNodes);
      edges.add(relatedEdges);
      const detail = Object.assign({}, nodeDetails[nodeId] || {});
      detail["聚焦显示"] = "相关子图 " + focusedNodes.length + " 个节点，" + relatedEdges.length + " 条关系";
      renderDetail(detail);
      network.setOptions({ physics: { enabled: basePhysicsEnabled } });
      if (basePhysicsEnabled) network.stabilize(120);
      fitSoon();
    }

    const network = new vis.Network(
      document.getElementById("network"),
      { nodes, edges },
      {
        layout: { improvedLayout: true, randomSeed: 8 },
        physics: {
          enabled: basePhysicsEnabled,
          solver: "forceAtlas2Based",
          stabilization: { iterations: ${stabilization_iterations}, fit: true },
          forceAtlas2Based: {
            gravitationalConstant: ${gravitational_constant},
            centralGravity: ${central_gravity},
            springLength: ${spring_length},
            springConstant: 0.055,
            damping: 0.45,
            avoidOverlap: ${avoid_overlap}
          }
        },
        interaction: { hover: true, navigationButtons: true, keyboard: true },
        nodes: { borderWidth: 1.4 },
        edges: { smooth: false }
      }
    );
    network.once("stabilizationIterationsDone", () => {
      network.setOptions({ physics: { enabled: false } });
    });
    network.on("beforeDrawing", ctx => {
      if (!regionBoxes.length) return;
      ctx.save();
      regionBoxes.forEach(box => {
        ctx.fillStyle = box.stroke;
        ctx.font = 'bold 28px "Microsoft YaHei", sans-serif';
        ctx.fillText(box.label, box.x + 34, box.y + 58);
      });
      ctx.restore();
    });
    network.on("click", params => {
      if (params.nodes.length) renderDetail(nodeDetails[params.nodes[0]] || defaultDetail);
      else if (params.edges.length) renderDetail(edgeDetails[params.edges[0]] || defaultDetail);
      else renderDetail(defaultDetail);
    });
    network.on("doubleClick", params => {
      if (params.nodes.length) focusNode(params.nodes[0]);
      else if (isFocused) resetGraph();
    });
    document.getElementById("resetGraph").addEventListener("click", resetGraph);
    document.getElementById("resetGraphFloat").addEventListener("click", resetGraph);
    document.getElementById("fitGraph").addEventListener("click", () => {
      network.fit({ animation: { duration: 220, easingFunction: "easeInOutQuad" } });
    });
    document.getElementById("zoomInGraph").addEventListener("click", () => {
      const scale = Math.min(network.getScale() * 1.18, 3.5);
      network.moveTo({ scale, animation: { duration: 180, easingFunction: "easeInOutQuad" } });
    });
    document.getElementById("zoomOutGraph").addEventListener("click", () => {
      const scale = Math.max(network.getScale() / 1.18, 0.18);
      network.moveTo({ scale, animation: { duration: 180, easingFunction: "easeInOutQuad" } });
    });
    renderDetail(defaultDetail);
    fitSoon(true);
  </script>
</body>
</html>
"""
    )

    node_details = {item["id"]: item["detail"] for item in node_items}
    edge_details = {item["id"]: item["detail"] for item in edge_items}
    return template.substitute(
        height=height,
        safe_title=html.escape(title),
        nodes_json=script_json(node_items),
        edges_json=script_json(edge_items),
        node_details_json=script_json(node_details),
        edge_details_json=script_json(edge_details),
        default_detail_json=script_json(default_detail),
        region_boxes_json=script_json(region_boxes),
        focus_node_id_json=script_json(focus_node_id or ""),
        physics_enabled_json="true" if physics_enabled else "false",
        spring_length=spring_length,
        gravitational_constant=gravitational_constant,
        central_gravity=central_gravity,
        avoid_overlap=avoid_overlap,
        stabilization_iterations=stabilization_iterations,
    )


def make_structure_html(center_id: str, nodes: pd.DataFrame, edges: pd.DataFrame, title: str, *, height: int = 620) -> str:
    """Render a center-node sector layout for one-hop relation structure."""
    if center_id not in set(nodes["node_id"]):
        return "<div>当前节点不在主题图中。</div>"

    center_row = nodes[nodes["node_id"] == center_id].iloc[0]
    related_edges = edges[(edges["source"] == center_id) | (edges["target"] == center_id)].copy()
    neighbor_ids = set(related_edges["source"].tolist()) | set(related_edges["target"].tolist())
    neighbor_ids.discard(center_id)
    structure_nodes = nodes[nodes["node_id"].isin(neighbor_ids | {center_id})].copy()

    relation_angles = {REL_IS_A: -90, REL_CONTAINS: 160, REL_USED_FOR: 20}
    positions = {center_id: {"x": 0, "y": 0}}
    for relation, angle in relation_angles.items():
        rel_edges = related_edges[related_edges["relation"] == relation]
        rel_neighbors = []
        for _, row in rel_edges.iterrows():
            rel_neighbors.append(row["target"] if row["source"] == center_id else row["source"])
        rel_neighbors = sorted(set(rel_neighbors))
        count = max(len(rel_neighbors), 1)
        for idx, node_id in enumerate(rel_neighbors):
            offset = (idx - (count - 1) / 2) * 24
            current_angle = angle + offset
            radius = 260 + (idx // 6) * 100
            radians = current_angle * 3.1415926 / 180
            positions[node_id] = {"x": round(radius * __import__("math").cos(radians), 2), "y": round(radius * __import__("math").sin(radians), 2)}

    html_text = make_network_html(structure_nodes, related_edges, title, height=height)
    position_script = f"""
<script>
setTimeout(() => {{
  const fixedPositions = {json.dumps(positions, ensure_ascii=False)};
  Object.entries(fixedPositions).forEach(([id, pos]) => {{
    try {{ network.moveNode(id, pos.x, pos.y); }} catch(e) {{}}
  }});
  network.setOptions({{ physics: {{ enabled: false }} }});
}}, 500);
</script>
"""
    return html_text.replace("</body>", position_script + "\n</body>")


def make_overview_cluster_html(nodes: pd.DataFrame, edges: pd.DataFrame, topics: dict, *, height: int = 720) -> str:
    """Render four course clusters in one fixed, region-separated overview graph."""
    node_lookup = nodes.set_index("node_id").to_dict("index") if not nodes.empty else {}
    edge_lookup = edges.set_index("edge_id").to_dict("index") if "edge_id" in edges.columns else {}
    centers = [(-520, -260), (520, -260), (-520, 260), (520, 260)]
    region_boxes = []
    node_items = []
    edge_items = []

    for topic_index, topic in enumerate(topics.get("topics", [])[:4]):
        cx, cy = centers[topic_index]
        topic_color = TOPIC_COLORS[topic_index % len(TOPIC_COLORS)]
        border_color = TOPIC_BORDER_COLORS[topic_index % len(TOPIC_BORDER_COLORS)]
        region_boxes.append(
            {
                "x": cx - 430,
                "y": cy - 215,
                "w": 860,
                "h": 430,
                "label": short_topic_name(topic["topic_name"]),
                "fill": topic_color,
                "stroke": border_color,
            }
        )

        core_ids = [item["node_id"] for item in topic.get("core_nodes", [])[:8] if item.get("node_id") in node_lookup]
        topic_node_ids = [node_id for node_id in topic.get("node_ids", []) if node_id in node_lookup]
        selected_ids = []
        for node_id in core_ids + topic_node_ids:
            if node_id not in selected_ids:
                selected_ids.append(node_id)
            if len(selected_ids) >= 16:
                break

        for idx, node_id in enumerate(selected_ids):
            row = node_lookup[node_id]
            is_core = node_id in core_ids[:5]
            if idx == 0:
                x, y = cx, cy
            else:
                angle = 2 * math.pi * (idx - 1) / max(len(selected_ids) - 1, 1)
                radius = 135 if idx <= 8 else 185
                x = cx + radius * math.cos(angle)
                y = cy + radius * math.sin(angle)
            node_items.append(
                {
                    "id": f"{topic['topic_id']}::{node_id}",
                    "label": short_text(row.get("name", ""), 10),
                    "title": html.escape(str(row.get("name", ""))),
                    "x": round(x, 2),
                    "y": round(y, 2),
                    "fixed": True,
                    "shape": "dot",
                    "size": 28 if is_core else 20,
                    "color": {
                        "background": "#ffffff" if is_core else TYPE_COLORS.get(row.get("entity_type", "未知"), TYPE_COLORS["未知"]),
                        "border": border_color,
                        "highlight": {"background": "#fef3c7", "border": "#f59e0b"},
                    },
                    "font": {"color": "#111827", "size": 14 if is_core else 12, "face": "Microsoft YaHei", "strokeWidth": 4, "strokeColor": "#ffffff"},
                    "detail": {
                        "名称": row.get("name", ""),
                        "类型": row.get("entity_type", "未知"),
                        "所属主题": topic["topic_name"],
                    },
                }
            )

        selected_set = set(selected_ids)
        for edge_id in topic.get("edge_ids", [])[:40]:
            row = edge_lookup.get(edge_id)
            if not row or row.get("source") not in selected_set or row.get("target") not in selected_set:
                continue
            relation = row.get("relation", "")
            edge_items.append(
                {
                    "id": f"{topic['topic_id']}::{edge_id}",
                    "from": f"{topic['topic_id']}::{row.get('source')}",
                    "to": f"{topic['topic_id']}::{row.get('target')}",
                    "arrows": "to",
                    "width": 1.8,
                    "color": {"color": RELATION_COLORS.get(relation, "#64748b")},
                    "smooth": False,
                    "detail": {
                        "head": row.get("source_name", ""),
                        "relation": relation,
                        "tail": row.get("target_name", ""),
                        "source_file": row.get("source_file", ""),
                        "block_id": row.get("block_id", ""),
                        "evidence_text": row.get("evidence_text", "") or "无",
                    },
                }
            )

    template = Template(
        r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    body { margin: 0; font-family: "Microsoft YaHei", "Segoe UI", sans-serif; background: #f8fafc; }
    .wrap { display: grid; grid-template-columns: 1fr 300px; height: ${height}px; border: 1px solid #e5e7eb; background: #fff; }
    #network { height: ${height}px; min-width: 0; }
    .side { border-left: 1px solid #e5e7eb; padding: 14px; overflow: auto; background: #ffffff; }
    .title { font-weight: 800; color: #111827; margin-bottom: 10px; }
    .hint { color: #64748b; font-size: 13px; line-height: 1.7; margin-bottom: 14px; }
    .legend { display: grid; gap: 7px; margin: 10px 0 14px; font-size: 13px; color: #374151; }
    .legend span { display: inline-block; width: 28px; height: 3px; vertical-align: middle; margin-right: 8px; }
    .kv { margin: 8px 0; border-bottom: 1px solid #f1f5f9; padding-bottom: 7px; }
    .k { font-size: 12px; color: #64748b; }
    .v { font-size: 13px; color: #111827; word-break: break-word; white-space: pre-wrap; }
  </style>
</head>
<body>
  <div class="wrap">
    <div id="network"></div>
    <div class="side">
      <div class="title">四大主题总览</div>
      <div class="hint">四个区域对应课程章节主题。白底大节点表示主题核心知识点，线条颜色表示关系类型。</div>
      <div class="legend">
        <div><span style="background:#7c3aed"></span>概念归属</div>
        <div><span style="background:#f97316"></span>组成关系</div>
        <div><span style="background:#16a34a"></span>功能关系</div>
      </div>
      <div id="detail"></div>
    </div>
  </div>
  <script>
    const nodes = new vis.DataSet(${nodes_json});
    const edges = new vis.DataSet(${edges_json});
    const regions = ${regions_json};
    const nodeDetails = ${node_details_json};
    const edgeDetails = ${edge_details_json};
    const defaultDetail = {"说明": "点击节点或边查看详情。"};

    function renderDetail(data) {
      const target = document.getElementById("detail");
      target.innerHTML = Object.entries(data).map(function(entry) {
        return '<div class="kv"><div class="k">' + entry[0] + '</div><div class="v">' + String(entry[1] || "无") + '</div></div>';
      }).join("");
    }

    const network = new vis.Network(
      document.getElementById("network"),
      { nodes, edges },
      {
        layout: { improvedLayout: false },
        physics: false,
        interaction: { hover: true, navigationButtons: true, keyboard: true, dragNodes: false },
        nodes: { borderWidth: 2 },
        edges: { smooth: false }
      }
    );

    network.on("beforeDrawing", function(ctx) {
      ctx.save();
      regions.forEach(function(region) {
        ctx.fillStyle = region.stroke;
        ctx.font = 'bold 30px "Microsoft YaHei", sans-serif';
        ctx.fillText(region.label, region.x + 24, region.y + 42);
      });
      ctx.restore();
    });

    network.once("afterDrawing", function() {
      network.fit({ animation: false });
    });

    network.on("click", function(params) {
      if (params.nodes.length) renderDetail(nodeDetails[params.nodes[0]] || defaultDetail);
      else if (params.edges.length) renderDetail(edgeDetails[params.edges[0]] || defaultDetail);
      else renderDetail(defaultDetail);
    });
    renderDetail(defaultDetail);
  </script>
</body>
</html>
"""
    )
    node_details = {item["id"]: item["detail"] for item in node_items}
    edge_details = {item["id"]: item["detail"] for item in edge_items}
    return template.substitute(
        height=height,
        nodes_json=json.dumps(node_items, ensure_ascii=False),
        edges_json=json.dumps(edge_items, ensure_ascii=False),
        regions_json=json.dumps(region_boxes, ensure_ascii=False),
        node_details_json=json.dumps(node_details, ensure_ascii=False),
        edge_details_json=json.dumps(edge_details, ensure_ascii=False),
    )


def make_overview_radial_html(nodes: pd.DataFrame, edges: pd.DataFrame, topics: dict, *, height: int = 760) -> str:
    """Render the overview as one circle split into four topic subviews.

    The overview is deliberately rendered as plain SVG instead of vis-network.
    It is a static presentation view, so SVG is more stable and keeps the four
    quarter-circle regions visible even when external JavaScript is unavailable.
    """
    node_lookup = nodes.set_index("node_id").to_dict("index") if not nodes.empty else {}
    edge_lookup = edges.set_index("edge_id").to_dict("index") if "edge_id" in edges.columns else {}
    topic_angles = [-135, -45, 135, 45]
    topic_colors = ["#dbeafe", "#dcfce7", "#f3e8ff", "#ffedd5"]
    topic_strokes = ["#2563eb", "#16a34a", "#9333ea", "#ea580c"]
    svg_nodes = []
    svg_edges = []
    svg_links = []
    svg_labels = []

    def polar(radius: float, angle_deg: float) -> tuple[float, float]:
        angle = math.radians(angle_deg)
        return radius * math.cos(angle), radius * math.sin(angle)

    def svg_escape(value: object) -> str:
        return html.escape(str(value or ""), quote=True)

    def wedge_path(start_deg: float, end_deg: float, radius: float = 780) -> str:
        x1, y1 = polar(radius, start_deg)
        x2, y2 = polar(radius, end_deg)
        large_arc = 1 if abs(end_deg - start_deg) > 180 else 0
        return f"M 0 0 L {x1:.1f} {y1:.1f} A {radius:.1f} {radius:.1f} 0 {large_arc} 1 {x2:.1f} {y2:.1f} Z"

    for topic_index, topic in enumerate(topics.get("topics", [])[:4]):
        angle_deg = topic_angles[topic_index]
        stroke = topic_strokes[topic_index % len(topic_strokes)]
        center_x, center_y = polar(390, angle_deg)
        topic_label = short_topic_name(topic["topic_name"])
        label_x, label_y = polar(650, angle_deg)
        svg_labels.append(
            f'<text x="{label_x - 90:.1f}" y="{label_y - 20:.1f}" class="topic-label" fill="{stroke}">{svg_escape(topic_label)}</text>'
        )

        core_ids = [item["node_id"] for item in topic.get("core_nodes", []) if item.get("node_id") in node_lookup]
        center_id = core_ids[0] if core_ids else next((node_id for node_id in topic.get("node_ids", []) if node_id in node_lookup), "")
        if not center_id:
            continue

        selected_ids = [center_id]
        topic_edge_records = [edge_lookup[edge_id] for edge_id in topic.get("edge_ids", []) if edge_lookup.get(edge_id)]
        neighbor_ids = []
        for row in topic_edge_records:
            if row.get("source") == center_id and row.get("target") in node_lookup:
                neighbor_ids.append(row["target"])
            elif row.get("target") == center_id and row.get("source") in node_lookup:
                neighbor_ids.append(row["source"])
        for node_id in neighbor_ids + core_ids + [node_id for node_id in topic.get("node_ids", []) if node_id in node_lookup]:
            if node_id not in selected_ids:
                selected_ids.append(node_id)
            if len(selected_ids) >= 18:
                break
        selected_set = set(selected_ids)

        position_map = {}
        for idx, node_id in enumerate(selected_ids):
            row = node_lookup[node_id]
            if idx == 0:
                x, y = center_x, center_y
                size = 34
                label = topic_label
                bg = "#ffffff"
            else:
                arc_count = max(len(selected_ids) - 1, 1)
                offset = -42 + 84 * (idx - 1) / max(arc_count - 1, 1)
                radius = 155 if idx <= 9 else 230
                node_angle = math.radians(angle_deg + offset)
                x = center_x + radius * math.cos(node_angle)
                y = center_y + radius * math.sin(node_angle)
                size = 23 if node_id in core_ids else 18
                label = short_text(row.get("name", ""), 9)
                bg = TYPE_COLORS.get(row.get("entity_type", "未知"), TYPE_COLORS["未知"])
            position_map[node_id] = (x, y)
            tooltip = f"{row.get('name', '')}｜{row.get('entity_type', '未知')}｜{topic['topic_name']}"
            svg_nodes.append(
                "\n".join(
                    [
                        f'<g class="node" transform="translate({x:.1f},{y:.1f})">',
                        f"  <title>{svg_escape(tooltip)}</title>",
                        f'  <circle r="{size}" fill="{svg_escape(bg)}" stroke="{stroke}" stroke-width="2.4" />',
                        f'  <text y="{size + 18}" text-anchor="middle" class="node-text">{svg_escape(label)}</text>',
                        "</g>",
                    ]
                )
            )

        svg_links.append(
            f'<line class="topic-link" x1="0" y1="0" x2="{center_x:.1f}" y2="{center_y:.1f}" stroke="{stroke}">'
            f"<title>{svg_escape('主题连接线：' + topic['topic_name'])}</title></line>"
        )

        added_edges = 0
        for row in topic_edge_records:
            if row.get("source") not in selected_set or row.get("target") not in selected_set:
                continue
            if row.get("source") not in position_map or row.get("target") not in position_map:
                continue
            relation = row.get("relation", "")
            x1, y1 = position_map[row.get("source")]
            x2, y2 = position_map[row.get("target")]
            color = RELATION_COLORS.get(relation, "#64748b")
            marker_id = {REL_IS_A: "arrow-purple", REL_CONTAINS: "arrow-orange", REL_USED_FOR: "arrow-green"}.get(relation, "arrow-gray")
            tooltip = f"{row.get('source_name', '')} -[{relation}]-> {row.get('target_name', '')}"
            svg_edges.append(
                f'<line class="rel-edge" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                f'stroke="{color}" marker-end="url(#{marker_id})"><title>{svg_escape(tooltip)}</title></line>'
            )
            added_edges += 1
            if added_edges >= 34:
                break

    wedge_svg = "\n".join(
        [
            f'<path d="{wedge_path(-180, -90)}" fill="{topic_colors[0]}" stroke="{topic_strokes[0]}" />',
            f'<path d="{wedge_path(-90, 0)}" fill="{topic_colors[1]}" stroke="{topic_strokes[1]}" />',
            f'<path d="{wedge_path(90, 180)}" fill="{topic_colors[2]}" stroke="{topic_strokes[2]}" />',
            f'<path d="{wedge_path(0, 90)}" fill="{topic_colors[3]}" stroke="{topic_strokes[3]}" />',
        ]
    )

    return Template(
        """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <style>
    body { margin: 0; font-family: "Microsoft YaHei", "Segoe UI", sans-serif; background: #f8fafc; }
    .wrap { display: grid; grid-template-columns: 1fr 300px; height: ${height}px; border: 1px solid #e5e7eb; background: #fff; }
    .canvas { height: ${height}px; min-width: 0; background: #f8fafc; }
    .side { border-left: 1px solid #e5e7eb; padding: 14px; overflow: auto; background: #ffffff; }
    .title { font-weight: 800; color: #111827; margin-bottom: 10px; }
    .hint { color: #64748b; font-size: 13px; line-height: 1.7; margin-bottom: 14px; }
    .legend { display: grid; gap: 7px; margin: 10px 0 14px; font-size: 13px; color: #374151; }
    .legend span { display: inline-block; width: 28px; height: 3px; vertical-align: middle; margin-right: 8px; }
    svg { width: 100%; height: 100%; display: block; }
    .wedge path { opacity: 0.68; stroke-width: 2.2; }
    .axis { stroke: #94a3b8; stroke-width: 1.4; }
    .topic-link { stroke-width: 2.6; stroke-dasharray: 10 8; opacity: 0.78; }
    .rel-edge { stroke-width: 2; opacity: 0.88; }
    .node-text, .topic-label, .center-text {
      font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
      fill: #111827;
      paint-order: stroke;
      stroke: #ffffff;
      stroke-width: 5px;
      stroke-linejoin: round;
    }
    .node-text { font-size: 24px; font-weight: 700; }
    .topic-label { font-size: 34px; font-weight: 800; }
    .center-text { font-size: 30px; font-weight: 900; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="canvas">
      <svg viewBox="-920 -820 1840 1640" preserveAspectRatio="xMidYMid meet" role="img" aria-label="课程知识图谱总览">
        <defs>
          <marker id="arrow-purple" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#7c3aed"/></marker>
          <marker id="arrow-green" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#16a34a"/></marker>
          <marker id="arrow-orange" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#f97316"/></marker>
          <marker id="arrow-gray" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b"/></marker>
        </defs>
        <g class="wedge">
          ${wedge_svg}
        </g>
        <line class="axis" x1="-780" y1="0" x2="780" y2="0"/>
        <line class="axis" x1="0" y1="-780" x2="0" y2="780"/>
        <g class="topic-links">
          ${link_svg}
        </g>
        <g class="edges">
          ${edge_svg}
        </g>
        <g class="center-node">
          <circle cx="0" cy="0" r="58" fill="#0f766e" stroke="#064e3b" stroke-width="4"/>
          <text x="0" y="8" text-anchor="middle" class="center-text">课程图谱</text>
        </g>
        <g class="nodes">
          ${node_svg}
        </g>
        <g class="labels">
          ${label_svg}
        </g>
      </svg>
    </div>
    <div class="side">
      <div class="title">首页概览</div>
      <div class="hint">中间为课程图谱，四个 1/4 圆区域分别展示一个主题子图。虚线只表示主题组织，不新增图谱关系。</div>
      <div class="legend">
        <div><span style="background:#7c3aed"></span>概念归属</div>
        <div><span style="background:#f97316"></span>组成关系</div>
        <div><span style="background:#16a34a"></span>功能关系</div>
        <div><span style="background:#94a3b8;border-top:1px dashed #64748b"></span>章节关联</div>
      </div>
      <div class="hint">鼠标悬停节点或连线可查看名称与关系。</div>
    </div>
  </div>
</body>
</html>
"""
    ).substitute(
        height=height,
        wedge_svg=wedge_svg,
        link_svg="\n".join(svg_links),
        edge_svg="\n".join(svg_edges),
        node_svg="\n".join(svg_nodes),
        label_svg="\n".join(svg_labels),
    )


def topic_bundle(topic: dict, nodes: pd.DataFrame, edges: pd.DataFrame, max_component_size: int = 2) -> dict:
    """Prepare graph tables for one topic page."""
    topic_edges = get_topic_edges(topic, edges)
    topic_nodes = get_topic_nodes(topic, nodes, topic_edges)
    main_edges, supplemental_edges = split_main_and_supplemental_edges(topic_nodes, topic_edges, max_component_size)
    main_nodes = get_topic_nodes(topic, topic_nodes, main_edges)
    cross_edges = get_cross_topic_edges(topic)
    visible_cross = cross_edges[
        cross_edges["source"].isin(set(main_nodes["node_id"])) | cross_edges["target"].isin(set(main_nodes["node_id"]))
    ].copy()
    cross_nodes = nodes[nodes["node_id"].isin(set(visible_cross["source"]) | set(visible_cross["target"]))].copy()
    graph_nodes = pd.concat([main_nodes, cross_nodes], ignore_index=True).drop_duplicates("node_id")
    graph_edges = pd.concat([ensure_edge_flags(main_edges, False), ensure_edge_flags(visible_cross, True)], ignore_index=True)
    return {
        "topic_nodes": topic_nodes,
        "topic_edges": topic_edges,
        "main_nodes": graph_nodes,
        "main_edges": graph_edges,
        "supplemental_edges": supplemental_edges,
    }


def build_overview_network(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    topics: dict,
    max_component_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Build a draggable overview graph from four topic subviews."""
    node_lookup = nodes.set_index("node_id").to_dict("index") if not nodes.empty else {}
    overview_nodes = [
        {
            "node_id": "__course_center__",
            "name": "教材知识体系",
            "entity_type": "模块",
            "raw_name": "教材知识体系",
            "display_role": "overview_center",
            "display_color": "#be123c",
            "overview_topic_id": "",
            "chapter_name": "全书",
            "x": 0,
            "y": 0,
        }
    ]
    overview_edges = []
    region_boxes = []
    topic_centers = [(-1080, -760), (1080, -760), (-1080, 760), (1080, 760)]

    for topic_index, topic in enumerate(topics.get("topics", [])[:4]):
        base_x, base_y = topic_centers[topic_index]
        topic_edges = get_topic_edges(topic, edges)
        topic_nodes = get_topic_nodes(topic, nodes, topic_edges)
        filtered_nodes, filtered_edges = filter_small_components(topic_nodes, topic_edges, max_component_size)
        region_boxes.append(
            {
                "x": base_x - 520,
                "y": base_y - 390,
                "w": 1040,
                "h": 780,
                "label": chapter_display_name(topic_index, topic["topic_name"]),
                "fill": TOPIC_COLORS[topic_index % len(TOPIC_COLORS)],
                "stroke": TOPIC_BORDER_COLORS[topic_index % len(TOPIC_BORDER_COLORS)],
            }
        )

        if filtered_nodes.empty:
            filtered_nodes = topic_nodes.head(1).copy()
            filtered_edges = topic_edges.iloc[0:0].copy()

        filtered_node_ids = set(filtered_nodes["node_id"])
        core_ids = [item["node_id"] for item in topic.get("core_nodes", []) if item.get("node_id") in filtered_node_ids]
        if core_ids:
            center_id = core_ids[0]
        else:
            graph = build_graph(filtered_nodes, filtered_edges)
            center_id = max(dict(graph.degree()).items(), key=lambda item: item[1])[0] if graph.number_of_nodes() else filtered_nodes.iloc[0]["node_id"]

        selected_ids = []
        for node_id in [center_id] + core_ids + filtered_nodes["node_id"].tolist():
            if node_id in filtered_node_ids and node_id not in selected_ids:
                selected_ids.append(node_id)
        selected_set = set(selected_ids)

        for idx, node_id in enumerate(selected_ids):
            source_row = node_lookup.get(node_id, {})
            if idx == 0:
                x, y = base_x, base_y
                name = short_topic_name(topic["topic_name"])
                entity_type = "模块"
                display_role = "topic_center"
            else:
                member_index = idx - 1
                if member_index < 12:
                    ring_start = 0
                    ring_size = 12
                    radius = 240
                elif member_index < 32:
                    ring_start = 12
                    ring_size = 20
                    radius = 390
                else:
                    ring_start = 32
                    ring_size = max(len(selected_ids) - 33, 1)
                    radius = 540
                angle = 2 * math.pi * (member_index - ring_start) / max(ring_size, 1)
                x = base_x + radius * math.cos(angle)
                y = base_y + radius * math.sin(angle)
                name = source_row.get("name", node_id)
                entity_type = source_row.get("entity_type", "未知")
                display_role = "overview_member"
            overview_nodes.append(
                {
                    "node_id": f"{topic['topic_id']}::{node_id}",
                    "name": name,
                    "entity_type": entity_type,
                    "raw_name": source_row.get("name", name),
                    "display_role": display_role,
                    "display_color": TOPIC_BORDER_COLORS[topic_index % len(TOPIC_BORDER_COLORS)],
                    "overview_topic_id": topic["topic_id"],
                    "chapter_name": topic["topic_name"],
                    "x": round(x, 2),
                    "y": round(y, 2),
                }
            )

        overview_edges.append(
            {
                "edge_id": f"topic_link::{topic['topic_id']}",
                "source": "__course_center__",
                "target": f"{topic['topic_id']}::{center_id}",
                "source_name": "教材知识体系",
                "target_name": short_topic_name(topic["topic_name"]),
                "relation": "主题连接",
                "source_file": "",
                "block_id": "",
                "evidence_text": "展示组织线，不属于三元组关系。",
                "context_snippet": "",
                "is_cross_topic": True,
            }
        )

        for node_id in selected_ids[1:]:
            overview_edges.append(
                {
                    "edge_id": f"membership::{topic['topic_id']}::{node_id}",
                    "source": f"{topic['topic_id']}::{center_id}",
                    "target": f"{topic['topic_id']}::{node_id}",
                    "source_name": short_topic_name(topic["topic_name"]),
                    "target_name": node_lookup.get(node_id, {}).get("name", node_id),
                    "relation": "章节归属",
                    "source_file": "",
                    "block_id": "",
                    "evidence_text": "展示组织线，不属于三元组关系。",
                    "context_snippet": "",
                    "is_cross_topic": True,
                }
            )

        added_edges = 0
        for _, row in filtered_edges.iterrows():
            if row["source"] not in selected_set or row["target"] not in selected_set:
                continue
            overview_edges.append(
                {
                    "edge_id": f"{topic['topic_id']}::{row['edge_id']}",
                    "source": f"{topic['topic_id']}::{row['source']}",
                    "target": f"{topic['topic_id']}::{row['target']}",
                    "source_name": row.get("source_name", ""),
                    "target_name": row.get("target_name", ""),
                    "relation": row.get("relation", ""),
                    "source_file": row.get("source_file", ""),
                    "block_id": row.get("block_id", ""),
                    "evidence_text": row.get("evidence_text", ""),
                    "context_snippet": row.get("context_snippet", ""),
                    "is_cross_topic": False,
                }
            )
            added_edges += 1

    return pd.DataFrame(overview_nodes), pd.DataFrame(overview_edges), region_boxes


def apply_overview_search(
    overview_nodes: pd.DataFrame,
    overview_edges: pd.DataFrame,
    search_keyword: str,
) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    """Highlight and focus an overview graph node from the top search box."""
    keyword = str(search_keyword or "").strip().casefold()
    if not keyword or overview_nodes.empty:
        return overview_nodes, overview_edges, "", ""

    def match_score(row: pd.Series) -> int:
        fields = [
            str(row.get("raw_name", "")),
            str(row.get("name", "")),
            str(row.get("chapter_name", "")),
        ]
        folded = [field.casefold() for field in fields]
        if any(field == keyword for field in folded):
            return 4
        if any(keyword in field for field in folded):
            return 3
        if any(part and part in " ".join(folded) for part in keyword.split()):
            return 2
        return 0

    scored_nodes = overview_nodes.copy()
    scored_nodes["_match_score"] = scored_nodes.apply(match_score, axis=1)
    matches = scored_nodes[scored_nodes["_match_score"] > 0].copy()
    if matches.empty:
        return overview_nodes, overview_edges, "", ""

    matches["_role_rank"] = matches["display_role"].map({"overview_member": 0, "topic_center": 1, "overview_center": 2}).fillna(3)
    match = matches.sort_values(["_match_score", "_role_rank"], ascending=[False, True]).iloc[0]
    focus_node_id = match["node_id"]
    output_nodes = overview_nodes.copy()
    output_edges = overview_edges.copy()
    if str(match.get("display_role", "")) != "topic_center":
        output_nodes.loc[output_nodes["node_id"] == focus_node_id, "display_role"] = "search_hit"
    incident_mask = (output_edges["source"] == focus_node_id) | (output_edges["target"] == focus_node_id)
    output_edges.loc[incident_mask, "display_role"] = "search_edge"
    return output_nodes, output_edges, focus_node_id, str(match.get("raw_name") or match.get("name") or "")


def teaching_hints(topic: dict, topic_edges: pd.DataFrame) -> list[str]:
    """Generate simple teaching hints from core nodes and relation distribution."""
    relation_counts = topic_edges["relation"].value_counts().to_dict()
    core_names = [node["name"] for node in topic.get("core_nodes", [])[:5]]
    dominant_relation = max(relation_counts.items(), key=lambda item: item[1])[0] if relation_counts else "暂无"
    return [
        f"本主题建议优先讲解：{'、'.join(core_names) if core_names else '核心节点待补充'}。",
        f"主题内主要关系类型是“{display_relation(dominant_relation)}”，可据此组织知识点之间的逻辑顺序。",
        "讲解时可从核心节点出发，结合边详情中的教材证据进行概念追溯。",
    ]


def render_home(nodes: pd.DataFrame, edges: pd.DataFrame, topics: dict, max_component_size: int) -> None:
    """Render the modern overview dashboard."""
    render_metric_cards(nodes, edges, topics)

    st.markdown(
        """
        <div class="kg-section-title">
          <h3>知识图谱总览</h3>
          <span>中心节点为“课程图谱”，四周为四个章节主题簇</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    overview_nodes, overview_edges, overview_regions = build_overview_network(nodes, edges, topics, max_component_size)
    search_keyword = st.session_state.get("global_search_keyword", "")
    overview_nodes, overview_edges, focus_node_id, matched_name = apply_overview_search(overview_nodes, overview_edges, search_keyword)
    if search_keyword.strip():
        if focus_node_id:
            st.caption(f"已定位知识点：{matched_name}")
        else:
            st.warning("总览图中没有找到匹配知识点，请换一个关键词。")
    components.html(
        make_network_html(
            overview_nodes,
            overview_edges,
            "知识图谱总览",
            height=560,
            physics_enabled=False,
            region_boxes=overview_regions,
            focus_node_id=focus_node_id,
            stabilization_iterations=220,
        ),
        height=600,
        scrolling=False,
    )

    st.markdown(
        """
        <div class="kg-section-title">
          <h3>重点知识点 Top 5</h3>
          <span>根据知识点关联情况识别教材重点</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    centered_dataframe(top_knowledge_table(nodes, edges, topics), use_container_width=True, hide_index=True)

    with st.expander("章节知识概览", expanded=True):
        render_chapter_cards(topics)


def render_teacher(nodes: pd.DataFrame, edges: pd.DataFrame, topics: dict, max_component_size: int) -> None:
    """Render the teacher-facing teaching analysis page."""
    topic_map = topic_by_id(topics)
    selected_id = st.session_state.get("selected_topic_id")
    if selected_id not in topic_map:
        selected_id = topics["topics"][0]["topic_id"]
    st.session_state["teacher_topic_id"] = selected_id
    st.markdown(
        """
        <div class="kg-section-title">
          <h3>教学分析</h3>
          <span>从章节、关系和重点知识点角度分析教材知识结构</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.sidebar:
        st.markdown("### 当前视图工具")
        relation_filter = st.multiselect(
            "关系类型筛选",
            RELATION_OPTIONS,
            default=RELATION_OPTIONS,
            format_func=display_relation,
            key=f"teacher_rel_filter_{selected_id}",
        )
        search_keyword = st.text_input("知识点搜索", key=f"teacher_search_{selected_id}", placeholder="搜索知识点或关键词")
    st.session_state["selected_topic_id"] = selected_id
    topic = topic_map[selected_id]
    bundle = topic_bundle(topic, nodes, edges, max_component_size)
    filtered_topic_edges = filter_edges_by_relations(bundle["topic_edges"], relation_filter)
    filtered_main_edges = filter_edges_by_relations(bundle["main_edges"], relation_filter)
    filtered_supplemental_edges = filter_edges_by_relations(bundle["supplemental_edges"], relation_filter)
    display_main_nodes = nodes_used_by_edges(bundle["main_nodes"], filtered_main_edges)

    search_node_id = ""
    search_matches = pd.DataFrame()
    if search_keyword.strip():
        search_matches = bundle["topic_nodes"][
            bundle["topic_nodes"]["name"].str.contains(search_keyword.strip(), case=False, na=False, regex=False)
        ].copy()
        with st.sidebar:
            if search_matches.empty:
                st.caption("当前主题内没有匹配知识点。")
            else:
                search_name_map = search_matches.set_index("node_id")["name"].to_dict()
                search_node_id = st.selectbox(
                    "匹配结果",
                    search_matches["node_id"].tolist(),
                    format_func=lambda node_id: search_name_map.get(node_id, node_id),
                )

    display_main_nodes, display_main_edges, focus_node_id = apply_search_highlight(display_main_nodes, filtered_main_edges, search_node_id)

    st.markdown(f"### {topic['topic_name']}")
    st.caption(topic["topic_description"])
    metric_cols = st.columns(4)
    metric_cols[0].metric("知识点数", topic["node_count"])
    metric_cols[1].metric("关系数", topic["edge_count"])
    metric_cols[2].metric("补充关系", len(filtered_supplemental_edges))
    metric_cols[3].metric("跨章节关联", len(topic.get("cross_topic_edges", [])))
    relation_filter_text = "、".join(display_relation(relation) for relation in relation_filter) if relation_filter else "未选择关系"
    st.caption(f"当前关系筛选：{relation_filter_text}；当前可见关系 {len(display_main_edges)} 条。")
    if search_node_id and not focus_node_id:
        st.warning("命中知识点当前不在主题主画布中，可能被关系类型筛选或小组件过滤隐藏。可以恢复全部关系类型或调小左侧 N 值。")

    topic_node_ids = bundle["topic_nodes"]["node_id"].tolist()
    topic_node_name_map = bundle["topic_nodes"].set_index("node_id")["name"].to_dict()
    if topic_node_ids:
        default_detail_index = topic_node_ids.index(search_node_id) if search_node_id in topic_node_ids else 0
        detail_node_id = st.selectbox(
            "选择知识点查看详情",
            topic_node_ids,
            index=default_detail_index,
            format_func=lambda node_id: topic_node_name_map.get(node_id, node_id),
            key=f"teacher_detail_node_{selected_id}",
        )
        detail_node = bundle["topic_nodes"][bundle["topic_nodes"]["node_id"] == detail_node_id].iloc[0]
        detail_edges = find_related_edges(detail_node_id, filtered_topic_edges)
        detail_relations = "、".join(display_relation(relation) for relation in sorted(set(detail_edges["relation"]))) if not detail_edges.empty else "暂无直接关系"
        detail_cols = st.columns(4)
        detail_cols[0].metric("知识点名称", detail_node["name"])
        detail_cols[1].metric("类型", detail_node.get("entity_type", "未知"))
        detail_cols[2].metric("关联知识点数", len(detail_edges))
        detail_cols[3].metric("相关关系", detail_relations)

    triple_table = filtered_topic_edges[["source_name", "relation", "target_name", "source_file", "block_id", "evidence_text"]].copy()
    triple_table["relation"] = triple_table["relation"].map(display_relation)
    triple_table = triple_table.rename(
        columns={
            "source_name": "知识点",
            "relation": "关系类型",
            "target_name": "相关知识点",
            "source_file": "来源教材",
            "block_id": "片段编号",
            "evidence_text": "教材证据",
        }
    )

    teacher_view = horizontal_button_nav("teacher_view", ["章节知识结构", "关系统计与重点", "三元组与证据"])
    if teacher_view == "章节知识结构":
        if display_main_nodes.empty or display_main_edges.empty:
            st.info("当前过滤条件下，主题主画布没有可显示的关系。可以调小左侧 N 值。")
        else:
            components.html(
                make_network_html(display_main_nodes, display_main_edges, topic["topic_name"], height=620, focus_node_id=focus_node_id),
                height=680,
                scrolling=False,
            )
        with st.expander("查看中心知识点局部结构", expanded=True):
            core_options = [item["node_id"] for item in topic["core_nodes"] if item["node_id"] in set(bundle["topic_nodes"]["node_id"])]
            if core_options:
                name_map = bundle["topic_nodes"].set_index("node_id")["name"].to_dict()
                center_id = st.selectbox("中心知识点", core_options, format_func=lambda node_id: name_map.get(node_id, node_id))
                components.html(
                    make_structure_html(center_id, bundle["topic_nodes"], filtered_topic_edges, f"{topic['topic_name']}：局部结构"),
                    height=620,
                    scrolling=False,
                )

    elif teacher_view == "关系统计与重点":
        left_col, right_col = st.columns([1.1, 1])
        with left_col:
            st.markdown("**重点知识点列表**")
            core_frame = pd.DataFrame(topic["core_nodes"])[["name", "degree", "topic_score"]].rename(
                columns={"name": "知识点名称", "degree": "关联数", "topic_score": "重要度"}
            )
            centered_dataframe(core_frame, use_container_width=True, hide_index=True)
        with right_col:
            st.markdown("**关系类型统计**")
            centered_dataframe(relation_summary(filtered_topic_edges), use_container_width=True, hide_index=True)
            st.markdown("**教学要点提示**")
            for hint in teaching_hints(topic, filtered_topic_edges):
                st.write(f"- {hint}")
        st.markdown("**教学资料库推荐**")
        st.caption("以下资料来自预先整理的真实链接库，不由大模型临时生成网址。")
        teacher_resources = match_resource_library(
            resource_context_keywords(topic.get("topic_name", ""), [topic.get("topic_name", "")], filtered_topic_edges),
            [topic.get("topic_name", "")],
            limit=6,
        )
        render_resource_links(teacher_resources)
        teacher_hint_key = f"teacher_curated_hints_{selected_id}"
        if st.button("大模型生成教学要点", key=f"teacher_curated_hints_btn_{selected_id}", use_container_width=True):
            with st.spinner("正在基于知识图谱和资料库生成教学要点..."):
                hints, llm_error = llm_teacher_resource_hints(topic, filtered_topic_edges, teacher_resources)
            st.session_state[teacher_hint_key] = hints or fallback_teacher_resource_hints(topic, filtered_topic_edges, teacher_resources, llm_error)
        if st.session_state.get(teacher_hint_key):
            st.write(st.session_state[teacher_hint_key])

    else:
        st.markdown("**知识关系三元组表**")
        centered_dataframe(triple_table.head(80), use_container_width=True, hide_index=True)
        if not filtered_supplemental_edges.empty:
            st.markdown("**补充关系 / 边缘关系**")
            supplemental_table = filtered_supplemental_edges[["source_name", "relation", "target_name", "source_file", "block_id", "evidence_text"]].copy()
            supplemental_table["relation"] = supplemental_table["relation"].map(display_relation)
            supplemental_table = supplemental_table.rename(
                columns={
                    "source_name": "知识点",
                    "relation": "关系类型",
                    "target_name": "相关知识点",
                    "source_file": "来源教材",
                    "block_id": "片段编号",
                    "evidence_text": "教材证据",
                }
            )
            centered_dataframe(supplemental_table, use_container_width=True, hide_index=True)


def find_related_edges(node_id: str, edges: pd.DataFrame) -> pd.DataFrame:
    """Find direct edges around one knowledge point."""
    return edges[(edges["source"] == node_id) | (edges["target"] == node_id)].copy()


def simple_answer(question: str, nodes: pd.DataFrame, edges: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """A lightweight graph-retrieval QA entry without adding a new QA system."""
    question_norm = question.strip().lower()
    matched_nodes = nodes[nodes["name"].str.lower().apply(lambda name: name in question_norm if name else False)].copy()
    relation_intent = None
    if any(key in question for key in ["用于", "作用", "用来"]):
        relation_intent = REL_USED_FOR
    elif any(key in question for key in ["包含", "组成", "包括"]):
        relation_intent = REL_CONTAINS
    elif any(key in question for key in ["是什么", "属于", "一种", "类型"]):
        relation_intent = REL_IS_A

    if matched_nodes.empty:
        return "当前问题未命中明确知识点，请换一个更具体的实体名称。", edges.iloc[0:0].copy()

    node_ids = set(matched_nodes["node_id"])
    hits = edges[edges["source"].isin(node_ids) | edges["target"].isin(node_ids)].copy()
    if relation_intent:
        relation_hits = hits[hits["relation"] == relation_intent].copy()
        if not relation_hits.empty:
            hits = relation_hits
    hits = hits.head(8)
    if hits.empty:
        return "图谱中暂未检索到该知识点的直接关系。", hits
    triples = [f"{row['source_name']} -[{display_relation(row['relation'])}]-> {row['target_name']}" for _, row in hits.head(3).iterrows()]
    return "根据图谱检索，相关关系包括：" + "；".join(triples) + "。下方可查看教材证据。", hits


def evidence_to_text(evidence: pd.DataFrame, limit: int = 8) -> str:
    """Convert graph evidence rows into compact text for local LLM grounding."""
    if evidence.empty:
        return "未检索到直接证据。"

    lines = []
    for idx, row in evidence.head(limit).iterrows():
        source_file = row.get("source_file", "")
        block_id = row.get("block_id", "")
        evidence_text = row.get("evidence_text", "") or row.get("context_snippet", "")
        lines.append(
            f"{len(lines) + 1}. {row.get('source_name', '')} -[{display_relation(row.get('relation', ''))}]-> {row.get('target_name', '')}\n"
            f"   来源：{source_file} {block_id}\n"
            f"   证据：{evidence_text[:180]}"
        )
    return "\n".join(lines)


def clean_llm_response(text: str) -> str:
    """Remove optional Qwen thinking tags when present."""
    cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    return cleaned or (text or "").strip()


def call_ollama(prompt: str, *, model: str = OLLAMA_MODEL, timeout: int = 180, num_predict: int = 640) -> tuple[str, str]:
    """Call local Ollama and return response text plus an error message if any."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "top_p": 0.85,
            "num_predict": num_predict,
        },
        "keep_alive": "5m",
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_GENERATE_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        return "", f"无法连接本地 Ollama：{exc.reason}"
    except TimeoutError:
        return "", "本地模型响应超时。"
    except Exception as exc:
        return "", f"本地模型调用失败：{exc}"
    return clean_llm_response(result.get("response", "")), ""


def llm_answer_question(question: str, graph_answer: str, evidence: pd.DataFrame) -> tuple[str, str]:
    """Generate a student-facing answer grounded in graph retrieval results."""
    prompt = f"""/no_think
你是智能网联汽车课程的学习助手。请只依据给定的知识图谱关系和教材证据回答，不要编造未给出的资料。
回答要求：
1. 先用 2-4 句话直接回答学生问题。
2. 再列出 2-3 条相关知识点。
3. 如果证据不足，请明确说明“图谱中暂未提供充分证据”。

学生问题：
{question}

图谱检索摘要：
{graph_answer}

教材证据：
{evidence_to_text(evidence)}
"""
    return call_ollama(prompt, num_predict=420)


def llm_recommendation(node_name: str, topic_names: list[str], related_edges: pd.DataFrame, resources: pd.DataFrame) -> tuple[str, str]:
    """Generate a concise learning recommendation explanation with local Ollama."""
    prompt = f"""/no_think
你是智能网联汽车课程的学习推荐助手。请基于知识图谱邻接关系，为学生生成一段适合页面展示的学习推荐说明。
要求：
1. 只可以引用下方“预置真实资料库”中给出的资料名称，不要编造新网址。
2. 内容要比普通提示更完整，建议 500-800 字。
3. 结构包含：学习目标、图谱依据、推荐资源、学习顺序、复习检查问题。
4. 语言面向学生，解释为什么推荐这些内容，不要写成开发说明。
5. 最后一段必须用一句完整总结收尾，不要以省略号、顿号或半句话结束。

当前知识点：{node_name}
所属主题：{'、'.join(topic_names) if topic_names else '未归入主题'}
相关关系：
{evidence_to_text(related_edges, limit=6)}

预置真实资料库：
{curated_resources_to_text(resources)}
"""
    return call_ollama(prompt, timeout=240, num_predict=1800)


def llm_teacher_resource_hints(topic: dict, topic_edges: pd.DataFrame, resources: pd.DataFrame) -> tuple[str, str]:
    """Generate teacher-facing teaching hints grounded in curated resource links."""
    core_names = "、".join(item.get("name", "") for item in topic.get("core_nodes", [])[:5])
    prompt = f"""/no_think
你是智能网联汽车课程教师的备课助手。请结合章节知识图谱和预置真实资料库，生成适合课堂讲解和答辩展示的教学要点说明。
要求：
1. 建议 700-1000 字，不要过短。
2. 用 4 个小标题组织：教学重点、讲解顺序、图谱证据使用、资源补充建议。
3. 每个小标题下写 2-3 句话，重点围绕教材章节、核心知识点、关系类型和可补充的学习资料。
4. 只引用下方资料库中已有的资料名称，不要编造链接。
5. 语言面向教师备课，不要写成后台管理说明。
6. 最后一段必须完整收尾，不要以省略号、顿号或半句话结束。

章节：{topic.get('topic_name', '')}
章节说明：{topic.get('topic_description', '')}
核心知识点：{core_names}
关系分布：
{relation_summary(topic_edges).to_string(index=False)}

预置真实资料库：
{curated_resources_to_text(resources)}
"""
    return call_ollama(prompt, timeout=300, num_predict=2200)


def fallback_teacher_resource_hints(topic: dict, topic_edges: pd.DataFrame, resources: pd.DataFrame, error: str = "") -> str:
    """Build a fuller teacher-facing explanation when the local LLM is unavailable."""
    relation_counts = relation_summary(topic_edges)
    core_names = [item.get("name", "") for item in topic.get("core_nodes", [])[:5] if item.get("name")]
    resource_names = [row.get("资料名称", "") for _, row in resources.head(3).iterrows() if row.get("资料名称", "")]
    return (
        f"**教学重点**\n\n"
        f"本章可围绕“{topic.get('topic_name', '')}”展开，优先讲清 {'、'.join(core_names) if core_names else '章节核心知识点'}。"
        f"这些知识点在图谱中关联度较高，适合作为课堂讲解的主线入口。教师可以先说明章节主题，再把核心节点与周边概念、组成结构和功能应用联系起来。\n\n"
        f"**讲解顺序**\n\n"
        f"建议先从核心概念定义进入，再按照“概念归属、组成关系、功能关系”的顺序展开。当前关系分布为：{relation_counts.to_string(index=False)}。"
        f"这种顺序能帮助学生先建立概念边界，再理解系统组成，最后落到技术用途和应用场景。\n\n"
        f"**图谱证据使用**\n\n"
        f"课堂展示时可以点击图谱中的核心节点和关键边，利用右侧的来源文件、片段编号和教材证据说明该关系来自教材抽取结果。"
        f"这样既能展示知识图谱的可追溯性，也能避免学生只看到抽象节点而不知道依据来自哪里。\n\n"
        f"**资源补充建议**\n\n"
        f"可补充使用 {'、'.join(resource_names) if resource_names else '预置资料库中匹配到的相关资料'}，作为课后阅读或案例拓展。"
        f"这些资源不替代教材主线，而是帮助学生把图谱中的概念和真实技术材料联系起来。{('本地模型暂不可用：' + error) if error else ''}"
    )


def fallback_recommendation_text(node_name: str, topic_names: list[str], related_edges: pd.DataFrame, resources: pd.DataFrame) -> str:
    """Build a deterministic recommendation explanation when the local LLM is unavailable."""
    topic_text = "、".join(topic_names) if topic_names else "当前主题"
    adjacent = []
    for _, row in related_edges.head(8).iterrows():
        adjacent.append(row.get("target_name", "") if row.get("source_name", "") == node_name else row.get("source_name", ""))
    adjacent = [item for item in adjacent if item and item != node_name]
    resource_lines = []
    for _, row in resources.head(3).iterrows():
        resource_lines.append(f"“{row.get('资料名称', '')}”适合作为补充资料，推荐理由是：{row.get('推荐理由', '')}")
    if related_edges.empty:
        resource_lines.append("当前知识点直接关系较少，建议先结合所属主题补足基础概念，再回到教材原文确认定义。")
    adjacent_text = "、".join(adjacent[:6]) if adjacent else "主题核心知识点"
    return (
        f"围绕“{node_name}”，建议先回到“{topic_text}”建立章节背景，明确它在教材知识结构中的位置。"
        f"随后沿图谱相邻知识点继续学习，例如 {adjacent_text}，这样可以把单个概念放到上下位分类、组成结构或功能应用中理解。\n\n"
        "学习顺序上，可以先阅读教材证据片段确认概念定义，再查看相邻关系中的“概念归属、组成关系、功能关系”，最后结合推荐资料做扩展阅读。"
        "完成后建议用两个问题自测：这个知识点属于哪一类？它和哪些系统、模块或应用场景有关？\n\n"
        + "\n".join(resource_lines)
    )


def recommend_resources(node_row: pd.Series, related_edges: pd.DataFrame, topic_names: list[str]) -> pd.DataFrame:
    """Generate simple resource recommendations from topic and neighbor relations."""
    relation_counts = related_edges["relation"].value_counts().to_dict()
    adjacent = []
    for _, row in related_edges.head(6).iterrows():
        adjacent.append(row["target_name"] if row["source"] == node_row["node_id"] else row["source_name"])
    reasons = [
        {
            "资料": f"{node_row['name']} 概念复习",
            "推荐理由": f"属于 {'、'.join(topic_names) if topic_names else '当前主题'}，适合先建立基本概念。",
        },
        {
            "资料": "相邻知识点联学",
            "推荐理由": f"建议联学：{'、'.join(adjacent[:5]) if adjacent else '先结合所属主题复习核心概念'}。",
        },
        {
            "资料": "教材证据片段",
            "推荐理由": f"关联 {len(related_edges)} 条，其中“用于”{relation_counts.get(REL_USED_FOR, 0)} 条、“包含”{relation_counts.get(REL_CONTAINS, 0)} 条。",
        },
    ]
    return pd.DataFrame(reasons)


def normalize_keywords(values: list[str] | tuple[str, ...] | str) -> list[str]:
    """Normalize keyword fields from resource records."""
    if isinstance(values, str):
        return [values] if values.strip() else []
    return [str(item).strip() for item in values if str(item).strip()]


def resource_context_keywords(node_name: str = "", topic_names: list[str] | None = None, related_edges: pd.DataFrame | None = None) -> list[str]:
    """Collect matching keywords from a selected knowledge point and graph context."""
    keywords = [node_name]
    keywords.extend(topic_names or [])
    if related_edges is not None and not related_edges.empty:
        for _, row in related_edges.head(8).iterrows():
            keywords.extend([row.get("source_name", ""), row.get("target_name", ""), row.get("relation", "")])
    return [str(item).strip().lower() for item in keywords if str(item).strip()]


def match_resource_library(context_keywords: list[str], topic_names: list[str] | None = None, limit: int = 6) -> pd.DataFrame:
    """Match curated real links by topic and keyword overlap."""
    library = load_resource_library()
    if library.empty:
        return pd.DataFrame(columns=["资料名称", "类型", "来源", "适用章节", "推荐理由", "链接"])

    topic_names = topic_names or []
    scored_rows = []
    for _, row in library.iterrows():
        row_topics = normalize_keywords(row.get("topics", []))
        row_keywords = normalize_keywords(row.get("keywords", []))
        haystack = " ".join([row.get("title", ""), row.get("type", ""), row.get("source", ""), row.get("reason", "")] + row_topics + row_keywords).lower()
        score = 0
        for keyword in context_keywords:
            if keyword and keyword in haystack:
                score += 3 if len(keyword) >= 3 else 1
        for topic in topic_names:
            if topic in row_topics:
                score += 5
        if score <= 0 and not topic_names:
            score = 1
        if score > 0:
            scored_rows.append((score, row))

    scored_rows.sort(key=lambda item: (-item[0], str(item[1].get("title", ""))))
    records = []
    for _, row in scored_rows[:limit]:
        records.append(
            {
                "资料名称": row.get("title", ""),
                "类型": row.get("type", ""),
                "来源": row.get("source", ""),
                "适用章节": "、".join(normalize_keywords(row.get("topics", []))),
                "推荐理由": row.get("reason", ""),
                "链接": row.get("url", ""),
            }
        )
    return pd.DataFrame(records)


def curated_resources_to_text(resources: pd.DataFrame, limit: int = 5) -> str:
    """Convert curated resource rows into LLM prompt text."""
    if resources.empty:
        return "预置资料库中暂未匹配到资源。"
    lines = []
    for _, row in resources.head(limit).iterrows():
        lines.append(
            f"- {row.get('资料名称', '')}（{row.get('类型', '')}，{row.get('来源', '')}）："
            f"{row.get('推荐理由', '')}\n  链接：{row.get('链接', '')}"
        )
    return "\n".join(lines)


def render_resource_links(resources: pd.DataFrame) -> None:
    """Render curated resource links with a clickable URL column."""
    if resources.empty:
        st.info("预置资料库中暂未匹配到相关资料，可以先依据教材证据和相邻知识点学习。")
        return
    centered_dataframe(
        resources,
        use_container_width=True,
        hide_index=True,
        column_config={"链接": st.column_config.LinkColumn("链接")},
    )


def learning_reason(relation: str) -> str:
    """Explain why a relation helps form a learning path."""
    if relation == REL_IS_A:
        return "先厘清上下位分类关系，帮助建立概念边界。"
    if relation == REL_CONTAINS:
        return "继续查看组成关系，理解整体与部分。"
    if relation == REL_USED_FOR:
        return "最后联系用途或场景，理解知识点如何应用。"
    return "沿图谱相邻关系继续扩展学习。"


def edge_path_text(row: pd.Series) -> str:
    """Format one directed graph edge as a readable path segment."""
    return f"{row.get('source_name', '')} -[{display_relation(row.get('relation', ''))}]-> {row.get('target_name', '')}"


def neighbor_from_edge(row: pd.Series | dict, current_id: str) -> str:
    """Return the node reached from current_id through one edge."""
    source = row.get("source", "")
    target = row.get("target", "")
    return target if source == current_id else source


def build_learning_chain(node_row: pd.Series, edges: pd.DataFrame, max_steps: int | None = None) -> list[dict]:
    """Find one continuous learning chain and keep walking until no new neighbor remains."""
    node_id = node_row["node_id"]
    if edges.empty:
        return []
    if max_steps is None:
        max_steps = max(len(set(edges["source"]) | set(edges["target"])) - 1, 1)
    relation_rank = {REL_IS_A: 0, REL_CONTAINS: 1, REL_USED_FOR: 2}
    degree_counter = Counter(edges["source"].tolist() + edges["target"].tolist())
    adjacency: dict[str, list[dict]] = {}
    for _, row in edges.iterrows():
        record = row.to_dict()
        adjacency.setdefault(record["source"], []).append({**record, "_neighbor": record["target"], "_walk_direction": "out"})
        adjacency.setdefault(record["target"], []).append({**record, "_neighbor": record["source"], "_walk_direction": "in"})

    for node_edges in adjacency.values():
        node_edges.sort(
            key=lambda item: (
                0 if item.get("_walk_direction") == "out" else 1,
                relation_rank.get(item.get("relation", ""), 9),
                -degree_counter.get(item.get("_neighbor", ""), 0),
                item.get("target_name", "") or item.get("source_name", ""),
            )
        )

    def estimate_future_depth(current_id: str, visited: set[str], depth: int = 4) -> int:
        """Small look-ahead so the chain tends to choose a longer continuation."""
        if depth <= 0:
            return 0
        best = 0
        for edge_record in adjacency.get(current_id, [])[:8]:
            neighbor_id = edge_record["_neighbor"]
            if neighbor_id in visited:
                continue
            best = max(best, 1 + estimate_future_depth(neighbor_id, visited | {neighbor_id}, depth - 1))
        return best

    chain: list[dict] = []
    current_id = node_id
    visited = {node_id}
    used_edges = set()
    while len(chain) < max_steps:
        candidates = [
            edge_record
            for edge_record in adjacency.get(current_id, [])
            if edge_record["_neighbor"] not in visited and edge_record.get("edge_id") not in used_edges
        ]
        if not candidates:
            break

        def candidate_key(edge_record: dict) -> tuple:
            neighbor_id = edge_record["_neighbor"]
            future_depth = estimate_future_depth(neighbor_id, visited | {neighbor_id})
            return (
                -future_depth,
                0 if edge_record.get("_walk_direction") == "out" else 1,
                relation_rank.get(edge_record.get("relation", ""), 9),
                -degree_counter.get(neighbor_id, 0),
                edge_record.get("target_name", "") or edge_record.get("source_name", ""),
            )

        next_edge = min(candidates, key=candidate_key)
        chain.append(next_edge.copy())
        used_edges.add(next_edge.get("edge_id"))
        current_id = next_edge["_neighbor"]
        visited.add(current_id)
    return chain


def learning_path_recommendations(node_row: pd.Series, nodes: pd.DataFrame, edges: pd.DataFrame, max_steps: int | None = None) -> pd.DataFrame:
    """Recommend a continuous graph learning chain from one selected knowledge point."""
    node_id = node_row["node_id"]
    node_name = node_row["name"]
    name_map = nodes.set_index("node_id")["name"].to_dict()
    chain_edges = build_learning_chain(node_row, edges, max_steps=max_steps)
    records = [
        {
            "阶段": "起点",
            "推荐知识点": node_name,
            "关系路径": node_name,
            "推荐理由": "先确认当前知识点的基本含义，再沿相邻关系扩展。",
        }
    ]
    current_id = node_id
    chain_names = [node_name]
    for step_index, edge_record in enumerate(chain_edges, start=1):
        neighbor_id = neighbor_from_edge(edge_record, current_id)
        chain_names.append(name_map.get(neighbor_id, neighbor_id))
        records.append(
            {
                "阶段": f"第{step_index}步",
                "推荐知识点": name_map.get(neighbor_id, neighbor_id),
                "关系路径": edge_path_text(edge_record),
                "推荐理由": learning_reason(edge_record.get("relation", "")),
            }
        )
        current_id = neighbor_id
    return pd.DataFrame(records)


def learning_path_graph_tables(node_row: pd.Series, nodes: pd.DataFrame, edges: pd.DataFrame, max_steps: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a fixed left-to-right graph for one continuous learning chain."""
    node_id = node_row["node_id"]
    chain_edges = build_learning_chain(node_row, edges, max_steps=max_steps)
    chain_node_ids = [node_id]
    current_id = node_id
    selected_edge_ids = []
    for edge_record in chain_edges:
        selected_edge_ids.append(edge_record["edge_id"])
        current_id = neighbor_from_edge(edge_record, current_id)
        chain_node_ids.append(current_id)

    graph_nodes = nodes[nodes["node_id"].isin(chain_node_ids)].copy()
    order_map = {current_node_id: index for index, current_node_id in enumerate(chain_node_ids)}
    graph_nodes["_order"] = graph_nodes["node_id"].map(order_map)
    graph_nodes = graph_nodes.sort_values("_order").drop(columns=["_order"])
    center_offset = (len(chain_node_ids) - 1) * 150
    for current_node_id, index in order_map.items():
        graph_nodes.loc[graph_nodes["node_id"] == current_node_id, "x"] = index * 300 - center_offset
        graph_nodes.loc[graph_nodes["node_id"] == current_node_id, "y"] = 0
    graph_nodes.loc[graph_nodes["node_id"] == node_id, "display_role"] = "search_hit"
    graph_edges = edges[edges["edge_id"].isin(selected_edge_ids)].copy()
    return graph_nodes, graph_edges


def render_student(nodes: pd.DataFrame, edges: pd.DataFrame, topics: dict) -> None:
    """Render the student-facing learning assistant page."""
    st.markdown(
        """
        <div class="kg-section-title">
          <h3>学习助手</h3>
          <span>基于教材知识图谱，支持知识点查询、关联学习和基础问答</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("当前问答基于已构建的教材知识图谱进行查询，回答来源于结构化三元组，具有可解释性。")
    keyword = st.text_input("知识点搜索", placeholder="搜索知识点或关键词")
    filtered_nodes = nodes[nodes["name"].str.contains(keyword, case=False, na=False, regex=False)] if keyword else nodes.head(40)
    if filtered_nodes.empty:
        st.info("没有匹配的知识点。")
        return
    selected_name = st.selectbox("知识点", filtered_nodes["name"].tolist())
    selected = nodes[nodes["name"] == selected_name].iloc[0]
    related = find_related_edges(selected["node_id"], edges)
    topic_names = node_topic_names(topics).get(selected["node_id"], [])
    recommendation_state_key = f"llm_recommendation_{selected['node_id']}"

    valid_student_views = ["知识点查询", "学习问答", "资源推荐", "学习路径"]
    st.session_state.setdefault("student_view", "知识点查询")
    if st.session_state["student_view"] not in valid_student_views:
        st.session_state["student_view"] = "知识点查询"
    student_view = horizontal_button_nav("student_view", valid_student_views)
    if student_view == "知识点查询":
        cols = st.columns([1, 1])
        with cols[0]:
            st.markdown("**知识点详情**")
            centered_dataframe(
                pd.DataFrame(
                    [
                        {
                            "名称": selected["name"],
                            "类型": selected.get("entity_type", "未知"),
                            "所属主题": "、".join(topic_names) if topic_names else "未归入主题",
                            "直接关系数": len(related),
                        }
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
            understood = selected["node_id"] in get_understood_node_ids()
            button_label = "标记为已理解" if not understood else "取消已理解"
            if st.button(button_label, key=f"understood_{selected['node_id']}", use_container_width=True):
                set_node_understood(selected["node_id"], not understood)
                st.rerun()
            related_topics = [topic for topic in topics.get("topics", []) if selected["node_id"] in set(topic.get("node_ids", []))]
            if related_topics:
                progress_topic = related_topics[0]
                understood_count, total_count, progress_percent = topic_progress(progress_topic)
                st.progress(progress_percent / 100, text=f"{short_topic_name(progress_topic['topic_name'])} 掌握进度：{progress_percent}%（{understood_count}/{total_count}）")
        with cols[1]:
            st.markdown("**相关知识推荐**")
            adjacent_names = []
            for _, row in related.head(8).iterrows():
                adjacent_names.append(row["target_name"] if row["source"] == selected["node_id"] else row["source_name"])
            st.markdown("、".join(adjacent_names) if adjacent_names else "暂无直接相邻知识点")

        related_display = related[["source_name", "relation", "target_name", "source_file", "block_id", "evidence_text"]].head(12).copy()
        related_display["relation"] = related_display["relation"].map(display_relation)
        related_display = related_display.rename(
            columns={
                "source_name": "知识点",
                "relation": "关系类型",
                "target_name": "相关知识点",
                "source_file": "来源教材",
                "block_id": "片段编号",
                "evidence_text": "教材证据",
            }
        )
        centered_dataframe(related_display, use_container_width=True, hide_index=True)
        related_nodes = nodes[nodes["node_id"].isin(set(related["source"]) | set(related["target"]))].copy()
        if not related.empty:
            components.html(
                make_network_html(related_nodes, ensure_edge_flags(related, False), f"{selected['name']} 关联图", height=500),
                height=580,
                scrolling=False,
            )

    elif student_view == "学习问答":
        st.caption(f"大模型：{OLLAMA_MODEL}。回答会先检索图谱关系和教材证据，再由本地模型组织语言。")
        st.session_state.setdefault("student_question", "")
        example_questions = [
            "感知系统包含什么？",
            "毫米波雷达是什么？",
            "GPS 用于什么？",
            "车载传感器有哪些？",
            "哪些技术用于环境感知？",
        ]
        example_cols = st.columns(5)
        for index, (column, example) in enumerate(zip(example_cols, example_questions)):
            with column:
                if st.button(example, key=f"example_question_{index}", use_container_width=True):
                    st.session_state["student_question"] = example
        question = st.text_area(
            "问题",
            height=90,
            key="student_question",
            placeholder="请输入你想查询的教材知识问题，例如：毫米波雷达用于什么？",
        )
        if st.button("查询", use_container_width=True):
            if not question.strip():
                st.warning("请输入问题。")
            else:
                graph_answer, evidence = simple_answer(question, nodes, edges)
                with st.spinner("本地模型回答中..."):
                    llm_answer, error = llm_answer_question(question, graph_answer, evidence)
                if llm_answer:
                    st.markdown("**本地模型回答**")
                    st.write(llm_answer)
                else:
                    st.warning(error or "本地模型暂未返回回答，以下为图谱检索结果。")
                    st.write(graph_answer)
                if not evidence.empty:
                    st.markdown("**命中证据**")
                    evidence_display = evidence[["source_name", "relation", "target_name", "source_file", "block_id", "evidence_text"]].copy()
                    evidence_display["relation"] = evidence_display["relation"].map(display_relation)
                    evidence_display = evidence_display.rename(
                        columns={
                            "source_name": "知识点",
                            "relation": "关系类型",
                            "target_name": "相关知识点",
                            "source_file": "来源教材",
                            "block_id": "片段编号",
                            "evidence_text": "教材证据",
                        }
                    )
                    centered_dataframe(evidence_display, use_container_width=True, hide_index=True)

    elif student_view == "资源推荐":
        st.markdown("**学习资源推荐**")
        resources = match_resource_library(
            resource_context_keywords(selected["name"], topic_names, related),
            topic_names,
            limit=6,
        )
        st.caption("以下资料来自预先整理的真实链接库，不由大模型临时生成网址。")
        render_resource_links(resources)
        if st.button("大模型生成推荐说明", use_container_width=True):
            with st.spinner("本地模型生成中..."):
                recommendation, error = llm_recommendation(selected["name"], topic_names, related, resources)
            if recommendation:
                st.session_state[recommendation_state_key] = recommendation
            else:
                st.session_state[recommendation_state_key] = fallback_recommendation_text(selected["name"], topic_names, related, resources)
                if error:
                    st.caption(f"本地大模型暂不可用，已显示图谱规则推荐：{error}")
        if st.session_state.get(recommendation_state_key):
            st.write(st.session_state[recommendation_state_key])

    else:
        st.markdown("**学习路径推荐**")
        path_options = filtered_nodes["node_id"].tolist()
        path_name_map = filtered_nodes.set_index("node_id")["name"].to_dict()
        default_path_index = path_options.index(selected["node_id"]) if selected["node_id"] in path_options else 0
        path_node_id = st.selectbox(
            "路径起点",
            path_options,
            index=default_path_index,
            format_func=lambda node_id: path_name_map.get(node_id, node_id),
            key="student_path_node",
        )
        if st.button("生成学习路径", use_container_width=True):
            path_node = nodes[nodes["node_id"] == path_node_id].iloc[0]
            path_graph_nodes, path_graph_edges = learning_path_graph_tables(path_node, nodes, edges)
            path_table = learning_path_recommendations(path_node, nodes, edges)
            st.session_state["student_learning_path_result"] = {
                "node_id": path_node_id,
                "node_name": path_node["name"],
                "table": path_table,
                "graph_nodes": path_graph_nodes,
                "graph_edges": path_graph_edges,
            }
        path_result = st.session_state.get("student_learning_path_result")
        if path_result and path_result.get("node_id") == path_node_id:
            st.caption(f"已生成连续链路：{len(path_result['graph_nodes'])} 个节点，{len(path_result['graph_edges'])} 条关系。")
            centered_dataframe(path_result["table"], use_container_width=True, hide_index=True)
            if not path_result["graph_nodes"].empty and not path_result["graph_edges"].empty:
                components.html(
                    make_network_html(
                        path_result["graph_nodes"],
                        ensure_edge_flags(path_result["graph_edges"], False),
                        f"{path_result['node_name']} 学习路径图",
                        height=520,
                        physics_enabled=False,
                    ),
                    height=600,
                    scrolling=False,
                )
            else:
                st.info("当前知识点可形成文字推荐，但图谱中暂无可展开的路径关系。")


def main() -> None:
    """Streamlit app entry."""
    st.set_page_config(page_title="智能网联汽车教材知识图谱", layout="wide")
    inject_page_style()
    nodes, edges, topics = load_data()
    if not topics.get("topics"):
        st.error("未找到 topic_clusters.json，请先运行 kg/build_topic_clusters.py。")
        return

    st.session_state.setdefault("page", PAGE_OVERVIEW)
    st.session_state.setdefault("selected_topic_id", topics["topics"][0]["topic_id"])
    render_app_header()
    if st.session_state.get("page", PAGE_OVERVIEW) == PAGE_OVERVIEW:
        render_top_function_area()

    with st.sidebar:
        render_sidebar_brand()
        page = render_sidebar_main_menu()
        render_sidebar_chapters(topics)
        st.markdown("### 图谱显示设置")
        max_component_size = st.slider("隐藏知识点数 <= N 的小组件", min_value=0, max_value=8, value=0, step=1)
        st.caption("N=0 表示不过滤；该设置会按当前视图重新计算连通分量。")
        render_kg_update_panel()
        render_sidebar_footer()

    if page == PAGE_OVERVIEW:
        render_home(nodes, edges, topics, max_component_size)
    elif page == PAGE_TEACHER:
        render_teacher(nodes, edges, topics, max_component_size)
    else:
        render_student(nodes, edges, topics)


if __name__ == "__main__":
    main()
