"""知识图谱展示底座。

这个模块负责 QA 平台复用的展示能力，包括数据读取、图谱筛选、
结构质检、主题抽取、自由图渲染、结构图渲染和证据回显。
"""

import hashlib
import json
import math
import uuid
from collections import Counter
from pathlib import Path
from string import Template

import networkx as nx
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components


ROOT_DIR = Path(__file__).resolve().parents[1]
KG_DIR = ROOT_DIR / "outputs" / "kg"
NODES_PATH = KG_DIR / "nodes.csv"
EDGES_PATH = KG_DIR / "edges.csv"
SUMMARY_PATH = KG_DIR / "kg_summary.json"
REPORT_PATH = KG_DIR / "kg_analysis_report.md"

NEO4J_HTTP = "http://localhost:7474/db/neo4j/tx/commit"
NEO4J_AUTH = ("neo4j", "neo4j123")

REL_IS_A = "是一种"
REL_CONTAINS = "包含"
REL_USED_FOR = "用于"
UNKNOWN_TYPE = "未知"
RELATION_OPTIONS = [REL_IS_A, REL_CONTAINS, REL_USED_FOR]

RELATION_COLORS = {
    REL_IS_A: "#2563eb",
    REL_CONTAINS: "#16a34a",
    REL_USED_FOR: "#ea580c",
}
DEFAULT_NODE_SIZE_SCALE = 1.6
DEFAULT_EDGE_WIDTH_SCALE = 1.3
DEFAULT_SPACING_SCALE = 0.72
DEFAULT_INITIAL_SCALE = 1.2
LAYOUT_DISTANCE_BOOST = 1.5
MAX_EDGE_TO_NODE_RATIO = 5
STRUCTURE_SECTOR_CONFIG = {
    REL_IS_A: {"angle_deg": -90, "span_deg": 88, "radius": 260, "ring_gap": 110, "per_ring": 5},
    REL_CONTAINS: {"angle_deg": 150, "span_deg": 88, "radius": 272, "ring_gap": 110, "per_ring": 5},
    REL_USED_FOR: {"angle_deg": 30, "span_deg": 88, "radius": 272, "ring_gap": 110, "per_ring": 5},
}
STRUCTURE_CENTER_ROLE_RULES = {
    REL_IS_A: {"upper": "target", "lower": "source"},
    REL_CONTAINS: {"upper": "source", "lower": "target"},
    REL_USED_FOR: {"upper": "source", "lower": "target"},
}
TYPE_COLORS = {
    "技术": "#355c7d",
    "功能": "#f08a24",
    "设备": "#d1495b",
    "场景": "#00798c",
    "类别": "#4f772d",
    "模块": "#8f5db7",
    UNKNOWN_TYPE: "#6b7280",
}
EDGE_ISSUE_COLORS = {
    "duplicate": "#f59e0b",
    "conflict": "#dc2626",
    "short_cycle": "#7c3aed",
}
ABNORMAL_NODE_COLOR = "#dc2626"

THEME_PRESETS = [
    {
        "name": "传感器与感知",
        "keywords": ["传感器", "感知", "摄像头", "激光雷达", "毫米波雷达", "雷达", "GPS", "IMU", "目标检测"],
        "radius": 1,
        "preferred_relation": REL_CONTAINS,
        "description": "围绕传感器、感知对象和检测任务组织，适合展示整体到部件的结构。",
    },
    {
        "name": "神经网络与算法类别",
        "keywords": ["神经网络", "卷积", "CNN", "RNN", "LSTM", "深度学习", "强化学习", "算法", "分类", "检测"],
        "radius": 1,
        "preferred_relation": REL_IS_A,
        "description": "围绕神经网络和算法类别展开，优先体现上下位层级结构。",
    },
    {
        "name": "平台与模块",
        "keywords": ["平台", "模块", "ROS", "感知模块", "定位模块", "控制模块", "通信模块", "云控平台", "执行模块"],
        "radius": 1,
        "preferred_relation": REL_CONTAINS,
        "description": "围绕平台、模块和系统组成展开，适合树状和中心扩散式展示。",
    },
    {
        "name": "定位与融合",
        "keywords": ["定位", "融合", "卡尔曼", "SLAM", "惯性导航", "多传感器融合", "组合定位", "GPS", "IMU"],
        "radius": 1,
        "preferred_relation": REL_USED_FOR,
        "description": "围绕定位方法和融合策略展开，适合技术到功能的讲解方式。",
    },
]
FEATURED_THEME_SEQUENCE = [THEME_PRESETS[0]["name"], THEME_PRESETS[1]["name"], THEME_PRESETS[3]["name"]]


@st.cache_data(show_spinner=False)
def load_local_data():
    """从本地 outputs/kg 目录读取节点、边和摘要文件。"""
    nodes_df = pd.read_csv(NODES_PATH, encoding="utf-8-sig")
    edges_df = pd.read_csv(EDGES_PATH, encoding="utf-8-sig")
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8")) if SUMMARY_PATH.exists() else {}
    report_text = REPORT_PATH.read_text(encoding="utf-8") if REPORT_PATH.exists() else ""
    return nodes_df, edges_df, summary, report_text


def run_neo4j(statement):
    """通过 Neo4j HTTP API 执行一条查询语句。"""
    response = requests.post(
        NEO4J_HTTP,
        json={"statements": [{"statement": statement}]},
        auth=NEO4J_AUTH,
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("errors"):
        raise RuntimeError(payload["errors"])
    result = payload["results"][0]
    return result["columns"], [item["row"] for item in result["data"]]


@st.cache_data(show_spinner=False)
def load_neo4j_data():
    """从 Neo4j 读取节点和边，并转成与本地 CSV 一致的结构。"""
    node_stmt = """
    MATCH (n:Entity)
    RETURN n.node_id AS node_id,
           n.name AS name,
           coalesce(n.entity_type, '未知') AS entity_type,
           coalesce(n.raw_name, n.name) AS raw_name,
           coalesce(n.alias_count, 1) AS alias_count
    ORDER BY n.name
    """
    edge_stmt = """
    MATCH (s:Entity)-[r:RELATED]->(t:Entity)
    RETURN s.node_id AS source,
           t.node_id AS target,
           coalesce(r.relation, '') AS relation,
           coalesce(r.source_name, s.name) AS source_name,
           coalesce(r.target_name, t.name) AS target_name,
           coalesce(r.source_file, '') AS source_file,
           coalesce(r.evidence_text, '') AS evidence_text,
           coalesce(r.context_snippet, '') AS context_snippet,
           coalesce(r.block_id, '') AS block_id,
           coalesce(r.triple_id, '') AS triple_id,
           coalesce(r.confidence, '') AS confidence,
           coalesce(r.status, 'normal') AS status,
           coalesce(r.evidence_count, 1) AS evidence_count
    ORDER BY relation, source_name, target_name
    """
    node_columns, node_rows = run_neo4j(node_stmt)
    edge_columns, edge_rows = run_neo4j(edge_stmt)
    nodes_df = pd.DataFrame(node_rows, columns=node_columns)
    edges_df = pd.DataFrame(edge_rows, columns=edge_columns)
    summary = {
        "input_file": "neo4j://localhost:7474",
        "node_count": int(len(nodes_df)),
        "edge_count": int(len(edges_df)),
        "relation_counts": edges_df["relation"].value_counts().to_dict(),
        "node_type_counts": nodes_df["entity_type"].value_counts().to_dict(),
    }
    report_text = "当前页面从 Neo4j 直接读取节点和边，再做本地展示层组织。"
    return nodes_df, edges_df, summary, report_text


def choose_source(source_name):
    """根据数据源选项决定读取本地 CSV 还是 Neo4j。"""
    if source_name == "Neo4j":
        return load_neo4j_data()
    return load_local_data()


def theme_page_name(theme_name):
    """统一主题页名称格式。"""
    return f"主题：{theme_name}"


def init_state():
    """初始化展示平台在 Streamlit 中依赖的默认状态。"""
    defaults = {
        "kg_v5_page": "总览",
        "kg_v5_theme_name": THEME_PRESETS[0]["name"],
        "kg_v5_defense_mode": True,
        "kg_v5_prev_defense_mode": True,
        "kg_v5_hide_two_node_components": False,
        "kg_v5_min_component_size": 3,
        "kg_v5_hide_issue_edges": True,
        "kg_v5_only_issue_graph": False,
        "kg_v5_high_label_only": True,
        "kg_v5_focus_largest_component": False,
        "kg_v5_max_nodes": 180,
        "kg_v5_display_graph_mode": "自由图模式",
        "kg_v5_focus_depth": 2,
        "kg_v5_window_height": 720,
        "kg_v5_node_size_scale": 1.6,
        "kg_v5_edge_width_scale": 1.3,
        "kg_v5_spacing_scale": 0.72,
        "kg_v5_initial_scale": 1.2,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def jump_to_page(page_name):
    """切换到指定页面并刷新界面。"""
    st.session_state["kg_v5_page"] = page_name
    st.rerun()


def jump_to_theme(theme_name):
    """切换到指定主题页并记录当前主题。"""
    st.session_state["kg_v5_theme_name"] = theme_name
    st.session_state["kg_v5_page"] = theme_page_name(theme_name)
    st.rerun()


def normalize_frames(nodes_df, edges_df):
    """补齐字段、统一类型，并规范节点边表的基础格式。"""
    nodes = nodes_df.copy()
    edges = edges_df.copy()

    for column, default in {
        "node_id": "",
        "name": "",
        "entity_type": UNKNOWN_TYPE,
        "raw_name": "",
        "alias_count": 1,
    }.items():
        if column not in nodes.columns:
            nodes[column] = default
    nodes["node_id"] = nodes["node_id"].astype(str)
    nodes["name"] = nodes["name"].astype(str).str.strip()
    nodes["entity_type"] = nodes["entity_type"].fillna(UNKNOWN_TYPE).replace("", UNKNOWN_TYPE)
    nodes["raw_name"] = nodes["raw_name"].fillna(nodes["name"])
    nodes.loc[nodes["raw_name"].astype(str).eq(""), "raw_name"] = nodes["name"]
    nodes["alias_count"] = pd.to_numeric(nodes["alias_count"], errors="coerce").fillna(1).astype(int)
    nodes = nodes.drop_duplicates(subset=["node_id"]).reset_index(drop=True)

    for column, default in {
        "source": "",
        "target": "",
        "relation": "",
        "source_name": "",
        "target_name": "",
        "source_file": "",
        "evidence_text": "",
        "context_snippet": "",
        "block_id": "",
        "triple_id": "",
        "confidence": "",
        "status": "normal",
        "evidence_count": 1,
    }.items():
        if column not in edges.columns:
            edges[column] = default
    edges["source"] = edges["source"].astype(str)
    edges["target"] = edges["target"].astype(str)
    edges["relation"] = edges["relation"].astype(str).str.strip()
    edges = edges[edges["relation"].isin(RELATION_OPTIONS)].copy()
    edges["source_file"] = edges["source_file"].fillna("").astype(str)
    edges["evidence_text"] = edges["evidence_text"].fillna("").astype(str)
    edges["context_snippet"] = edges["context_snippet"].fillna("").astype(str)
    edges["block_id"] = edges["block_id"].fillna("").astype(str)
    edges["triple_id"] = edges["triple_id"].fillna("").astype(str)
    edges["status"] = edges["status"].fillna("normal").astype(str)
    edges["confidence"] = edges["confidence"].fillna("").astype(str)
    edges["evidence_count"] = pd.to_numeric(edges["evidence_count"], errors="coerce").fillna(1).astype(int)

    name_map = nodes.set_index("node_id")["name"].to_dict()
    edges["source_name"] = edges["source_name"].fillna("").astype(str)
    edges["target_name"] = edges["target_name"].fillna("").astype(str)
    edges.loc[edges["source_name"].eq(""), "source_name"] = edges["source"].map(name_map).fillna(edges["source"])
    edges.loc[edges["target_name"].eq(""), "target_name"] = edges["target"].map(name_map).fillna(edges["target"])
    edges = edges.reset_index(drop=True)
    edges["edge_uid"] = [f"edge_{idx:05d}" for idx in range(len(edges))]
    return nodes, edges

def sync_nodes_to_edges(nodes_df, edge_view):
    """根据当前边集合反推可见节点，并补齐缺失元信息。"""
    columns = ["node_id", "name", "entity_type", "raw_name", "alias_count"]
    if edge_view.empty:
        return pd.DataFrame(columns=columns)

    visible_ids = pd.unique(pd.concat([edge_view["source"], edge_view["target"]], ignore_index=True))
    base_nodes = nodes_df[nodes_df["node_id"].isin(visible_ids)].drop_duplicates(subset=["node_id"]).copy()
    existing_ids = set(base_nodes["node_id"])
    missing_ids = [node_id for node_id in visible_ids if node_id not in existing_ids]
    if missing_ids:
        edge_name_map = {}
        for _, row in edge_view.iterrows():
            edge_name_map.setdefault(row["source"], row.get("source_name", row["source"]))
            edge_name_map.setdefault(row["target"], row.get("target_name", row["target"]))
        extra = pd.DataFrame(
            [
                {
                    "node_id": node_id,
                    "name": edge_name_map.get(node_id, node_id),
                    "entity_type": UNKNOWN_TYPE,
                    "raw_name": edge_name_map.get(node_id, node_id),
                    "alias_count": 1,
                }
                for node_id in missing_ids
            ]
        )
        base_nodes = pd.concat([base_nodes, extra], ignore_index=True)

    order_map = {node_id: idx for idx, node_id in enumerate(visible_ids)}
    base_nodes["_order"] = base_nodes["node_id"].map(order_map)
    base_nodes = base_nodes.sort_values("_order").drop(columns="_order")
    base_nodes["entity_type"] = base_nodes["entity_type"].fillna(UNKNOWN_TYPE).replace("", UNKNOWN_TYPE)
    base_nodes["raw_name"] = base_nodes["raw_name"].fillna(base_nodes["name"])
    base_nodes.loc[base_nodes["raw_name"].astype(str).eq(""), "raw_name"] = base_nodes["name"]
    base_nodes["alias_count"] = pd.to_numeric(base_nodes["alias_count"], errors="coerce").fillna(1).astype(int)
    return base_nodes[columns].reset_index(drop=True)


def filter_frames(nodes_df, edges_df, relations, node_types, keyword):
    """按关系、类型和关键词筛选当前展示用的数据。"""
    edge_view = edges_df[edges_df["relation"].isin(relations)].copy()
    type_map = nodes_df.set_index("node_id")["entity_type"].to_dict()
    edge_view["source_type"] = edge_view["source"].map(type_map).fillna(UNKNOWN_TYPE)
    edge_view["target_type"] = edge_view["target"].map(type_map).fillna(UNKNOWN_TYPE)
    edge_view = edge_view[
        edge_view["source_type"].isin(node_types) & edge_view["target_type"].isin(node_types)
    ].copy()

    if keyword:
        pattern = str(keyword).strip()
        mask = (
            edge_view["source_name"].str.contains(pattern, case=False, na=False)
            | edge_view["target_name"].str.contains(pattern, case=False, na=False)
            | edge_view["evidence_text"].str.contains(pattern, case=False, na=False)
            | edge_view["context_snippet"].str.contains(pattern, case=False, na=False)
            | edge_view["relation"].str.contains(pattern, case=False, na=False)
        )
        edge_view = edge_view[mask].copy()

    node_view = sync_nodes_to_edges(nodes_df, edge_view)
    return node_view, edge_view.reset_index(drop=True)


def apply_qc_flags(edge_view):
    """给边打上重复边、冲突边和短环等结构质检标记。"""
    flagged = edge_view.copy().reset_index(drop=True)
    if flagged.empty:
        for column in ["is_duplicate", "is_conflict", "is_short_cycle", "has_issue"]:
            flagged[column] = pd.Series(dtype=bool)
        return flagged

    flagged["is_duplicate"] = flagged.duplicated(subset=["source", "relation", "target"], keep=False)
    relation_count = flagged.groupby(["source", "target"])["relation"].transform("nunique")
    flagged["is_conflict"] = relation_count > 1
    isa_edges = flagged[flagged["relation"] == REL_IS_A]
    isa_pairs = set(zip(isa_edges["source"], isa_edges["target"]))
    short_cycle_pairs = {(a, b) for a, b in isa_pairs if a != b and (b, a) in isa_pairs}
    flagged["is_short_cycle"] = [
        row["relation"] == REL_IS_A and (row["source"], row["target"]) in short_cycle_pairs for _, row in flagged.iterrows()
    ]
    flagged["has_issue"] = flagged[["is_duplicate", "is_conflict", "is_short_cycle"]].any(axis=1)
    return flagged


def build_graph(node_view, edge_view):
    """把当前视图的节点边表转换为 NetworkX 有向图。"""
    graph = nx.DiGraph()
    for _, row in node_view.iterrows():
        graph.add_node(
            row["node_id"],
            name=row["name"],
            entity_type=row["entity_type"],
            raw_name=row.get("raw_name", row["name"]),
            alias_count=int(row.get("alias_count", 1)),
        )
    valid_ids = set(node_view["node_id"])
    for _, row in edge_view.iterrows():
        if row["source"] in valid_ids and row["target"] in valid_ids:
            graph.add_edge(row["source"], row["target"], relation=row["relation"], edge_uid=row.get("edge_uid", ""))
    return graph


def enrich_node_metrics(node_view, edge_view):
    """补充节点的中心性、组件信息和局部质检统计。"""
    node_view = node_view.copy().reset_index(drop=True)
    if node_view.empty or edge_view.empty:
        for column, default in {
            "degree": 0,
            "degree_centrality": 0.0,
            "pagerank": 0.0,
            "betweenness": 0.0,
            "component_id": "",
            "component_size": 0,
            "neighbor_relation_stats": "",
            "qc_flags": "",
            "is_abnormal": False,
        }.items():
            node_view[column] = default
        return node_view, build_graph(node_view, edge_view)

    graph = build_graph(node_view, edge_view)
    undirected = graph.to_undirected()
    degree_map = dict(undirected.degree())
    if undirected.number_of_nodes() > 1:
        degree_centrality = nx.degree_centrality(undirected)
    else:
        degree_centrality = {node_id: 0.0 for node_id in graph.nodes()}
    if graph.number_of_edges() > 0 and graph.number_of_nodes() > 0:
        pagerank = nx.pagerank(graph, alpha=0.85)
    else:
        uniform = 1.0 / max(graph.number_of_nodes(), 1)
        pagerank = {node_id: uniform for node_id in graph.nodes()}
    if undirected.number_of_nodes() > 2 and undirected.number_of_edges() > 0:
        betweenness = nx.betweenness_centrality(undirected, normalized=True)
    else:
        betweenness = {node_id: 0.0 for node_id in graph.nodes()}

    component_id_map = {}
    component_size_map = {}
    for idx, component in enumerate(sorted(nx.connected_components(undirected), key=lambda item: (-len(item), sorted(item))), start=1):
        component_id = f"C{idx:02d}"
        for node_id in component:
            component_id_map[node_id] = component_id
            component_size_map[node_id] = len(component)

    issue_nodes = set(edge_view.loc[edge_view["has_issue"], "source"]) | set(edge_view.loc[edge_view["has_issue"], "target"])
    relation_stats = {}
    for node_id in graph.nodes():
        incident = edge_view[(edge_view["source"] == node_id) | (edge_view["target"] == node_id)]
        counter = Counter(incident["relation"].tolist())
        relation_stats[node_id] = "、".join(f"{rel}:{count}" for rel, count in sorted(counter.items())) if counter else "无"

    node_view["degree"] = node_view["node_id"].map(degree_map).fillna(0).astype(int)
    node_view["degree_centrality"] = node_view["node_id"].map(degree_centrality).fillna(0.0)
    node_view["pagerank"] = node_view["node_id"].map(pagerank).fillna(0.0)
    node_view["betweenness"] = node_view["node_id"].map(betweenness).fillna(0.0)
    node_view["component_id"] = node_view["node_id"].map(component_id_map).fillna("")
    node_view["component_size"] = node_view["node_id"].map(component_size_map).fillna(0).astype(int)
    node_view["neighbor_relation_stats"] = node_view["node_id"].map(relation_stats).fillna("无")
    node_view["is_abnormal"] = node_view["node_id"].isin(issue_nodes)

    flags = []
    for _, row in node_view.iterrows():
        parts = []
        if row["node_id"] in issue_nodes:
            parts.append("关联结构问题边")
        if row["entity_type"] == UNKNOWN_TYPE:
            parts.append("类型待补充")
        flags.append(" / ".join(parts) if parts else "正常")
    node_view["qc_flags"] = flags
    return node_view, graph


def build_component_records(graph, node_view, edge_view):
    """把连通分量整理成可展示、可隐藏的组件记录。"""
    if graph.number_of_nodes() == 0:
        return []
    name_map = node_view.set_index("node_id")["name"].to_dict()
    relation_map = edge_view[["source", "target", "relation", "source_name", "target_name"]].copy()
    records = []
    for idx, component in enumerate(sorted(nx.connected_components(graph.to_undirected()), key=lambda item: (-len(item), sorted(item))), start=1):
        comp_ids = set(component)
        comp_edges = relation_map[relation_map["source"].isin(comp_ids) & relation_map["target"].isin(comp_ids)].copy()
        node_names = [name_map.get(node_id, node_id) for node_id in comp_ids]
        relation_counter = Counter(comp_edges["relation"].tolist())
        relation_summary = "、".join(f"{relation}:{count}" for relation, count in sorted(relation_counter.items())) if relation_counter else "无"
        edge_preview = "；".join(
            f"{row['source_name']} -[{row['relation']}]-> {row['target_name']}" for _, row in comp_edges.head(4).iterrows()
        )
        records.append(
            {
                "component_id": f"C{idx:02d}",
                "size": len(comp_ids),
                "edge_count": int(len(comp_edges)),
                "node_ids": sorted(comp_ids),
                "node_names": sorted(node_names),
                "nodes_preview": "、".join(sorted(node_names)[:8]) + (" ..." if len(node_names) > 8 else ""),
                "relation_summary": relation_summary,
                "edge_preview": edge_preview,
            }
        )
    return records


def summarize_stats(graph, edge_view):
    """汇总当前视图的节点、边和连通性统计。"""
    if graph.number_of_nodes() == 0:
        return {
            "visible_nodes": 0,
            "visible_edges": 0,
            "components": 0,
            "one_node_components": 0,
            "two_node_components": 0,
            "largest_component": 0,
        }
    components = list(nx.connected_components(graph.to_undirected()))
    sizes = [len(comp) for comp in components]
    return {
        "visible_nodes": int(graph.number_of_nodes()),
        "visible_edges": int(len(edge_view)),
        "components": len(components),
        "one_node_components": sum(1 for size in sizes if size == 1),
        "two_node_components": sum(1 for size in sizes if size == 2),
        "largest_component": max(sizes, default=0),
    }


def apply_component_filters(nodes_df, edge_view, hide_two_node_components, min_component_size, focus_largest_component):
    """按组件大小和主连通子图策略过滤当前视图。"""
    candidate_nodes = sync_nodes_to_edges(nodes_df, edge_view)
    candidate_nodes, candidate_graph = enrich_node_metrics(candidate_nodes, edge_view)
    component_records = build_component_records(candidate_graph, candidate_nodes, edge_view)

    keep_ids = set(candidate_graph.nodes())
    hidden_components = []
    removed = {
        "isolated_removed": 0,
        "two_node_components_removed": 0,
        "small_components_removed": 0,
        "non_largest_removed": 0,
    }

    if focus_largest_component and candidate_graph.number_of_nodes() > 0:
        largest = max(nx.connected_components(candidate_graph.to_undirected()), key=len)
        largest_ids = set(largest)
        for record in component_records:
            comp_ids = set(record["node_ids"])
            if not comp_ids.issubset(largest_ids):
                hidden_components.append({**record, "reason": "非最大连通子图"})
                removed["non_largest_removed"] += 1
        keep_ids &= largest_ids

    for record in component_records:
        comp_ids = set(record["node_ids"])
        if not comp_ids.issubset(keep_ids):
            continue
        reason = ""
        if record["size"] == 1:
            reason = "1 节点组件"
            removed["isolated_removed"] += 1
        elif hide_two_node_components and record["size"] == 2:
            reason = "2 节点小组件"
            removed["two_node_components_removed"] += 1
        elif record["size"] < min_component_size:
            reason = f"小于 {min_component_size} 节点的组件"
            removed["small_components_removed"] += 1
        if reason:
            hidden_components.append({**record, "reason": reason})
            keep_ids -= comp_ids

    filtered_edges = edge_view[edge_view["source"].isin(keep_ids) & edge_view["target"].isin(keep_ids)].copy()
    return filtered_edges, candidate_nodes, hidden_components, removed

def build_display_view(
    nodes_df,
    edges_df,
    *,
    max_nodes,
    hide_two_node_components,
    min_component_size,
    focus_largest_component,
    deduplicate,
    hide_issue_edges,
    only_issue_edges,
):
    """构建页面真正使用的可见图视图。"""
    working_edges = apply_qc_flags(edges_df)
    if deduplicate:
        working_edges = working_edges.drop_duplicates(subset=["source", "relation", "target"], keep="first").copy()
    if only_issue_edges:
        working_edges = working_edges[working_edges["has_issue"]].copy()
    elif hide_issue_edges:
        working_edges = working_edges[~working_edges["has_issue"]].copy()

    visible_edges, candidate_nodes, hidden_components, removed = apply_component_filters(
        nodes_df,
        working_edges,
        hide_two_node_components=hide_two_node_components,
        min_component_size=min_component_size,
        focus_largest_component=focus_largest_component,
    )
    removed["max_nodes_trimmed"] = 0
    visible_nodes = sync_nodes_to_edges(nodes_df, visible_edges)
    visible_nodes, visible_graph = enrich_node_metrics(visible_nodes, visible_edges)

    if max_nodes and len(visible_nodes) > max_nodes:
        ranked_ids = (
            visible_nodes.sort_values(["pagerank", "degree_centrality", "degree", "name"], ascending=[False, False, False, True])
            .head(max_nodes)["node_id"]
            .tolist()
        )
        keep_ranked = set(ranked_ids)
        removed["max_nodes_trimmed"] = len(visible_nodes) - len(keep_ranked)
        visible_edges = visible_edges[
            visible_edges["source"].isin(keep_ranked) & visible_edges["target"].isin(keep_ranked)
        ].copy()
        visible_edges, _, post_hidden_components, post_removed = apply_component_filters(
            nodes_df,
            visible_edges,
            hide_two_node_components=hide_two_node_components,
            min_component_size=min_component_size,
            focus_largest_component=focus_largest_component,
        )
        hidden_components.extend(post_hidden_components)
        for key in ["isolated_removed", "two_node_components_removed", "small_components_removed", "non_largest_removed"]:
            removed[key] += post_removed[key]
        visible_nodes = sync_nodes_to_edges(nodes_df, visible_edges)
        visible_nodes, visible_graph = enrich_node_metrics(visible_nodes, visible_edges)

    stats = summarize_stats(visible_graph, visible_edges)
    return {
        "graph": visible_graph,
        "nodes": visible_nodes,
        "edges": visible_edges,
        "stats": stats,
        "hidden_components": hidden_components,
        "removed": removed,
        "candidate_nodes": candidate_nodes,
        "candidate_edges": working_edges,
    }


def compute_qc(nodes_df, edges_df):
    """生成结构质检页所需的摘要、问题边和问题图。"""
    edge_view = apply_qc_flags(edges_df)
    qc_duplicates = edge_view[edge_view["is_duplicate"]].copy()
    qc_short_cycles = edge_view[edge_view["is_short_cycle"]].copy()
    qc_conflicts = edge_view[edge_view["is_conflict"]].copy()
    abnormal_node_ids = set(edge_view.loc[edge_view["has_issue"], "source"]) | set(edge_view.loc[edge_view["has_issue"], "target"])
    qc_nodes = sync_nodes_to_edges(nodes_df, edge_view[edge_view["has_issue"]])
    qc_nodes["is_abnormal"] = qc_nodes["node_id"].isin(abnormal_node_ids)
    summary = pd.DataFrame(
        [
            {
                "重复边实例": int(qc_duplicates.shape[0]),
                "冲突边实例": int(qc_conflicts.shape[0]),
                "是一种短环边": int(qc_short_cycles.shape[0]),
                "异常节点": int(len(abnormal_node_ids)),
            }
        ]
    )
    qc_graph = build_graph(sync_nodes_to_edges(nodes_df, edge_view[edge_view["has_issue"]]), edge_view[edge_view["has_issue"]])
    return summary, qc_duplicates, qc_short_cycles, qc_conflicts, qc_nodes, qc_graph


def extract_theme(base_graph, base_nodes, base_edges, preset, radius):
    """按主题种子词和跳数抽取局部主题子图。"""
    if base_graph.number_of_nodes() == 0 or base_edges.empty:
        return pd.DataFrame(), pd.DataFrame(), []

    keywords = preset["keywords"]
    seed_mask = base_nodes["name"].apply(lambda value: any(keyword.lower() in str(value).lower() for keyword in keywords))
    seed_mask |= base_nodes["raw_name"].apply(lambda value: any(keyword.lower() in str(value).lower() for keyword in keywords))
    seed_nodes = base_nodes[seed_mask].copy()
    if seed_nodes.empty:
        return pd.DataFrame(), pd.DataFrame(), []

    undirected = base_graph.to_undirected()
    selected_ids = set()
    for node_id in seed_nodes["node_id"].tolist():
        selected_ids.update(nx.single_source_shortest_path_length(undirected, node_id, cutoff=radius).keys())

    theme_edges = base_edges[base_edges["source"].isin(selected_ids) & base_edges["target"].isin(selected_ids)].copy()
    theme_nodes = sync_nodes_to_edges(base_nodes, theme_edges)
    return theme_nodes, theme_edges, seed_nodes["name"].drop_duplicates().tolist()


def choose_layout_relation(edge_view, preferred_relation=None):
    """为当前视图选择更适合的主布局关系。"""
    if edge_view.empty:
        return None
    counts = edge_view["relation"].value_counts().to_dict()
    total = max(len(edge_view), 1)
    if preferred_relation and counts.get(preferred_relation, 0) >= max(2, int(total * 0.25)):
        return preferred_relation
    top_relation, top_count = max(counts.items(), key=lambda item: item[1])
    if top_count >= max(2, int(total * 0.45)):
        return top_relation
    return None


def build_layout_options(layout_relation, physics_enabled, spacing_scale, node_size_scale=DEFAULT_NODE_SIZE_SCALE):
    """生成自由图模式下的 vis-network 布局参数。"""
    spacing_scale = max(0.55, float(spacing_scale)) * LAYOUT_DISTANCE_BOOST
    min_node_size = max(34, int(22 * float(node_size_scale)))
    edge_length_cap = max(96, min_node_size * MAX_EDGE_TO_NODE_RATIO)
    dense_spacing = min(edge_length_cap, max(84, int(min_node_size * 2.5 * spacing_scale)))
    level_spacing = min(edge_length_cap, max(112, int(min_node_size * 3.1 * spacing_scale)))
    tree_spacing = min(edge_length_cap, max(134, int(min_node_size * 3.7 * spacing_scale)))

    def clamp_length(base_value):
        return min(int(base_value * spacing_scale), edge_length_cap)

    if layout_relation == REL_IS_A:
        return {
            "layout": {
                "hierarchical": {
                    "enabled": True,
                    "direction": "UD",
                    "sortMethod": "directed",
                    "levelSeparation": level_spacing,
                    "nodeSpacing": dense_spacing,
                    "treeSpacing": tree_spacing,
                }
            },
            "physics": {"enabled": False},
            "edges": {"smooth": False},
            "label": "层级布局（上下位结构）",
        }
    if layout_relation == REL_CONTAINS:
        return {
            "layout": {
                "hierarchical": {
                    "enabled": True,
                    "direction": "LR",
                    "sortMethod": "directed",
                    "levelSeparation": level_spacing,
                    "nodeSpacing": dense_spacing,
                    "treeSpacing": tree_spacing,
                }
            },
            "physics": {"enabled": False},
            "edges": {"smooth": False},
            "label": "树状布局（整体到部件）",
        }
    if layout_relation == REL_USED_FOR:
        return {
            "layout": {
                "hierarchical": {
                    "enabled": True,
                    "direction": "LR",
                    "sortMethod": "directed",
                    "levelSeparation": level_spacing,
                    "nodeSpacing": min(edge_length_cap, max(52, int(min_node_size * 1.8 * spacing_scale))),
                    "treeSpacing": min(edge_length_cap, max(68, int(min_node_size * 2.4 * spacing_scale))),
                }
            },
            "physics": {"enabled": False},
            "edges": {"smooth": False},
            "label": "左右分层布局（技术到功能）",
        }
    return {
        "layout": {"improvedLayout": True},
        "physics": {
            "enabled": physics_enabled,
            "solver": "barnesHut",
            "stabilization": {"iterations": 420, "fit": True, "updateInterval": 20},
            "barnesHut": {
                "gravitationalConstant": -7600,
                "centralGravity": 0.04,
                "springLength": clamp_length(150),
                "springConstant": 0.04,
                "damping": 0.34,
                "avoidOverlap": 1.1,
            },
            "minVelocity": 0.45,
        },
        "edges": {"smooth": False},
        "label": "均衡布局",
    }


def build_neo4j_layout_options(physics_enabled, spacing_scale, node_size_scale=DEFAULT_NODE_SIZE_SCALE):
    """生成更接近 Neo4j Browser 风格的自由图布局参数。"""
    spacing_scale = max(0.6, float(spacing_scale)) * LAYOUT_DISTANCE_BOOST
    min_node_size = max(34, int(24 * float(node_size_scale)))
    edge_length_cap = max(110, min_node_size * MAX_EDGE_TO_NODE_RATIO)

    def clamp_length(base_value):
        return min(int(base_value * spacing_scale), edge_length_cap)

    return {
        "layout": {"improvedLayout": True, "randomSeed": 7},
        "physics": {
            "enabled": physics_enabled,
            "solver": "forceAtlas2Based",
            "stabilization": {"iterations": 360, "fit": True, "updateInterval": 20},
            "forceAtlas2Based": {
                "gravitationalConstant": -72,
                "centralGravity": 0.018,
                "springLength": clamp_length(132),
                "springConstant": 0.08,
                "damping": 0.48,
                "avoidOverlap": 1.18,
            },
            "minVelocity": 0.55,
            "maxVelocity": 24,
            "timestep": 0.38,
        },
        "edges": {"smooth": False},
        "label": "Neo4j 风格自由布局",
    }


def select_labeled_nodes(node_view, high_centrality_only):
    """决定哪些节点标签常显，哪些节点仅在 hover 时显示。"""
    if node_view.empty:
        return set()
    ranked = node_view.sort_values(["pagerank", "degree_centrality", "degree", "name"], ascending=[False, False, False, True])
    if not high_centrality_only:
        return set(ranked["node_id"].tolist())
    limit = min(max(8, int(len(node_view) * 0.18)), 16)
    return set(ranked.head(limit)["node_id"].tolist())


def select_theme_hub_ids(node_view, limit=1):
    """为主题图选择少量高中心性节点作为重点节点。"""
    if node_view.empty:
        return []
    ranked = node_view[node_view["degree"] >= 2].sort_values(
        ["pagerank", "degree_centrality", "degree", "component_size", "name"],
        ascending=[False, False, False, False, True],
    )
    if ranked.empty:
        ranked = node_view.sort_values(["pagerank", "degree_centrality", "degree", "name"], ascending=[False, False, False, True])
    return ranked.head(limit)["node_id"].tolist()


def format_neo4j_caption(text, max_chars=8, max_lines=2):
    """把节点标题压缩成适合圆节点内显示的短标签。"""
    value = str(text or "").strip()
    if not value:
        return ""
    chunks = [value[idx : idx + max_chars] for idx in range(0, len(value), max_chars)]
    if len(chunks) > max_lines:
        kept = chunks[: max_lines]
        kept[-1] = kept[-1][: max(1, max_chars - 1)] + "…"
        chunks = kept
    return "\n".join(chunks)


def format_float(value, digits=6):
    """把浮点数格式化为适合表格展示的字符串。"""
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "0.000000"


def build_layout_storage_key(title, node_view, edge_view):
    """为每个视图生成布局持久化键。"""
    node_ids = sorted(node_view["node_id"].astype(str).tolist()) if not node_view.empty else []
    edge_triplets = (
        edge_view[["source", "relation", "target"]].astype(str).agg("->".join, axis=1).sort_values().tolist()
        if not edge_view.empty
        else []
    )
    digest = hashlib.md5(("|".join(node_ids) + "||" + "|".join(edge_triplets)).encode("utf-8")).hexdigest()[:12]
    return f"kg_v5_layout::{title}::{digest}"

def make_clickable_network_html(
    graph,
    node_view,
    edge_view,
    title,
    *,
    height=700,
    physics_enabled=True,
    high_centrality_labels=True,
    layout_relation=None,
    neighborhood_depth=2,
    show_edge_labels=False,
    node_size_scale=DEFAULT_NODE_SIZE_SCALE,
    edge_width_scale=DEFAULT_EDGE_WIDTH_SCALE,
    spacing_scale=DEFAULT_SPACING_SCALE,
    initial_scale=DEFAULT_INITIAL_SCALE,
    cluster_hub_ids=None,
    enable_hub_clusters=False,
    layout_storage_key=None,
    neo4j_style=False,
):
    """生成自由图模式的交互 HTML。"""
    if graph.number_of_nodes() == 0:
        return f"<div style='padding:24px;font-family:Microsoft YaHei,sans-serif'>{title}：当前没有可展示的节点。</div>"

    node_size_scale = max(0.8, float(node_size_scale))
    edge_width_scale = max(0.8, float(edge_width_scale))
    spacing_scale = max(0.55, float(spacing_scale))
    initial_scale = max(0.7, float(initial_scale))
    neighborhood_depth = max(1, int(st.session_state.get("kg_v5_focus_depth", neighborhood_depth)))
    cluster_hub_ids = cluster_hub_ids or []
    layout_storage_key = layout_storage_key or build_layout_storage_key(title, node_view, edge_view)

    resolved_relation = choose_layout_relation(edge_view, preferred_relation=layout_relation)
    layout_bundle = (
        build_neo4j_layout_options(physics_enabled, spacing_scale, node_size_scale=node_size_scale)
        if neo4j_style
        else build_layout_options(resolved_relation, physics_enabled, spacing_scale, node_size_scale=node_size_scale)
    )
    node_rows = []
    node_info = {}
    node_value_map = {}
    node_degree_map = {}
    node_count = max(1, len(node_view))
    if neo4j_style:
        if node_count >= 90:
            base_font_size = 11
        elif node_count >= 55:
            base_font_size = 12
        else:
            base_font_size = 13
        auto_initial_scale = 1.02 if node_count <= 28 else 0.92 if node_count <= 60 else 0.84 if node_count <= 96 else 0.78
    else:
        if node_count >= 90:
            base_font_size = 12
        elif node_count >= 55:
            base_font_size = 13
        else:
            base_font_size = 14
        auto_initial_scale = 1.05 if node_count <= 28 else 0.96 if node_count <= 60 else 0.88 if node_count <= 96 else 0.8
    effective_initial_scale = min(initial_scale, auto_initial_scale)
    for _, row in node_view.iterrows():
        node_id = row["node_id"]
        is_abnormal = bool(row.get("is_abnormal", False))
        bg_color = TYPE_COLORS.get(row["entity_type"], TYPE_COLORS[UNKNOWN_TYPE])
        border_color = ABNORMAL_NODE_COLOR if is_abnormal else ("#1f2937" if neo4j_style else "#d7dee9")
        value = max(
            30 if neo4j_style else 24,
            int(((22 if neo4j_style else 20) + row.get("degree", 0) * (2.6 if neo4j_style else 2.3) + row.get("pagerank", 0) * (250 if neo4j_style else 230)) * node_size_scale),
        )
        font_size = max(base_font_size, int(base_font_size * (1.0 if neo4j_style else node_size_scale)))
        label_offset = 0 if neo4j_style else max(20, int(value * 0.62))
        display_label = format_neo4j_caption(row["name"]) if neo4j_style else row["name"]
        node_value_map[node_id] = value
        node_degree_map[node_id] = int(row.get("degree", 0))
        node_rows.append(
            {
                "id": node_id,
                "label": display_label,
                "baseLabel": display_label,
                "fullLabel": display_label,
                "rawLabel": row["name"],
                "title": f"{row['name']}<br>类型: {row['entity_type']}<br>度数: {int(row.get('degree', 0))}",
                "shape": "circle" if neo4j_style else "dot",
                "color": {
                    "background": bg_color,
                    "border": border_color,
                    "highlight": {"background": bg_color, "border": border_color},
                },
                "font": {
                    "size": font_size,
                    "color": "#111111" if neo4j_style else "#172033",
                    "face": "Microsoft YaHei",
                    "strokeWidth": 3 if neo4j_style else 4,
                    "strokeColor": "#ffffff",
                    "vadjust": label_offset,
                },
                "borderWidth": 3.6 if is_abnormal else (2.2 if neo4j_style else 1.5),
                "size": value,
                "nodeSize": value,
                "shadow": (
                    {"enabled": True, "color": "rgba(220,38,38,0.28)", "size": 20}
                    if is_abnormal
                    else {"enabled": True, "color": "rgba(15,23,42,0.18)", "size": 12, "x": 0, "y": 4}
                    if neo4j_style
                    else False
                ),
            }
        )
        node_info[node_id] = {
            "名称": row["name"],
            "类型": row["entity_type"],
            "度数": int(row.get("degree", 0)),
            "PageRank": format_float(row.get("pagerank", 0.0)),
            "度中心性": format_float(row.get("degree_centrality", 0.0)),
            "桥梁中心性": format_float(row.get("betweenness", 0.0)),
            "所属连通分量": f"{row.get('component_id', '')} / 大小 {int(row.get('component_size', 0))}",
            "质检标记": row.get("qc_flags", "正常"),
            "相邻节点分关系统计": row.get("neighbor_relation_stats", "无"),
            "原始名称": row.get("raw_name", row["name"]),
            "别名数": int(row.get("alias_count", 1)),
        }

    edge_rows = []
    edge_info = {}
    for _, row in edge_view.iterrows():
        edge_id = row.get("edge_uid") or f"edge_{uuid.uuid4().hex}"
        color = RELATION_COLORS.get(row["relation"], "#94a3b8")
        dashes = False
        width = 1.7 if neo4j_style else 1.9
        source_size = node_value_map.get(row["source"], max(18, int(18 * node_size_scale)))
        target_size = node_value_map.get(row["target"], max(18, int(18 * node_size_scale)))
        endpoint_size = max(source_size, target_size)
        endpoint_degree = max(node_degree_map.get(row["source"], 1), node_degree_map.get(row["target"], 1), 1)
        length_ratio = (
            min(MAX_EDGE_TO_NODE_RATIO, 2.25 + max(0, endpoint_degree - 1) * 0.24)
            if neo4j_style
            else min(MAX_EDGE_TO_NODE_RATIO, 2.7 + max(0, endpoint_degree - 1) * 0.3)
        )
        min_length = int(endpoint_size * (2.15 if neo4j_style else 2.8))
        edge_length = min(int(endpoint_size * MAX_EDGE_TO_NODE_RATIO), max(min_length, int(endpoint_size * length_ratio)))
        if bool(row.get("is_short_cycle", False)):
            color = EDGE_ISSUE_COLORS["short_cycle"]
            width = 2.9
        elif bool(row.get("is_conflict", False)):
            color = EDGE_ISSUE_COLORS["conflict"]
            width = 2.7
        elif bool(row.get("is_duplicate", False)):
            color = EDGE_ISSUE_COLORS["duplicate"]
            width = 2.7
            dashes = [8, 6]
        edge_rows.append(
            {
                "id": edge_id,
                "from": row["source"],
                "to": row["target"],
                "label": "",
                "baseLabel": "",
                "fullLabel": row["relation"],
                "title": f"{row['source_name']} -[{row['relation']}]-> {row['target_name']}",
                "color": {"color": color, "highlight": color},
                "arrows": {"to": {"enabled": True, "scaleFactor": 0.82}},
                "width": round(width * edge_width_scale, 2),
                "length": edge_length,
                "dashes": dashes,
                "font": {"size": max(10, int(10 * node_size_scale)), "align": "middle", "face": "Microsoft YaHei"},
            }
        )
        edge_info[edge_id] = {
            "head": row["source_name"],
            "relation": row["relation"],
            "tail": row["target_name"],
            "source_file": row.get("source_file", ""),
            "block_id": row.get("block_id", ""),
            "evidence_text": row.get("evidence_text", "") or "无",
            "上下文片段": row.get("context_snippet", "") or "无",
            "是否重复边": "是" if bool(row.get("is_duplicate", False)) else "否",
            "是否冲突边": "是" if bool(row.get("is_conflict", False)) else "否",
            "是否短环": "是" if bool(row.get("is_short_cycle", False)) else "否",
            "状态": row.get("status", "normal"),
        }

    dom_id = f"net_{uuid.uuid4().hex}"
    detail_id = f"detail_{uuid.uuid4().hex}"
    template = Template(
        """
<html>
<head>
  <script src="https://unpkg.com/vis-network@9.1.2/dist/vis-network.min.js"></script>
  <style>
    body { margin: 0; background: #edf2f7; font-family: 'Microsoft YaHei', sans-serif; }
    .title { padding: 8px 2px 10px; color: #334155; font-size: 13px; }
    .legend { display:flex; flex-wrap:wrap; gap:10px; margin: 0 0 10px 0; color:#334155; font-size:12px; }
    .legend-title { font-weight:700; margin-right:6px; }
    .legend-item { display:flex; align-items:center; gap:6px; }
    .legend-item i { display:inline-block; width:12px; height:12px; border-radius:999px; }
    .wrap { display:grid; grid-template-columns: minmax(0, 1fr) 290px; gap:14px; }
    #$dom_id { width:100%; height:${height}px; background:#ffffff; border:1px solid #d9e2ec; border-radius:14px; }
    #$detail_id { height:${height}px; background:#ffffff; border:1px solid #d9e2ec; border-radius:14px; padding:14px; overflow-y:auto; }
    .panel-title { font-weight:700; color:#0f172a; margin-bottom:10px; }
    .panel-text { color:#475569; line-height:1.5; }
    .kv { margin-bottom:10px; }
    .k { font-weight:700; color:#0f172a; margin-bottom:4px; }
    .v { color:#334155; word-break:break-word; line-height:1.45; }
    .graph-panel { position: relative; }
    .layout-tools { position:absolute; top:12px; left:12px; z-index:10; display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
    .layout-btn { border:none; border-radius:999px; padding:7px 12px; background:#0f172a; color:#fff; font-size:12px; cursor:pointer; }
    .layout-btn.alt { background:#475569; }
    .layout-status { background:rgba(255,255,255,0.92); border:1px solid #d9e2ec; color:#334155; border-radius:999px; padding:6px 10px; font-size:12px; }
    .selection-box { position:absolute; border:2px dashed #2563eb; background:rgba(37,99,235,0.12); border-radius:12px; display:none; z-index:9; pointer-events:none; }
  </style>
</head>
<body>
  <div class="title">$title。功能介绍：关系浏览、节点详情、证据回溯、布局保存。</div>
  <div class="legend">
    <span class="legend-title">图例</span>
    <span class="legend-item"><i style="background:#2563eb;"></i>是一种</span>
    <span class="legend-item"><i style="background:#16a34a;"></i>包含</span>
    <span class="legend-item"><i style="background:#ea580c;"></i>用于</span>
    <span class="legend-item"><i style="background:#f59e0b;"></i>重复边</span>
    <span class="legend-item"><i style="background:#dc2626;"></i>冲突边 / 异常节点描边</span>
    <span class="legend-item"><i style="background:#7c3aed;"></i>短环边</span>
  </div>
  <div class="wrap">
    <div class="graph-panel">
      <div class="layout-tools">
        <button class="layout-btn" id="${dom_id}_save">保存布局</button>
        <button class="layout-btn alt" id="${dom_id}_reset">恢复默认</button>
        <span class="layout-status" id="${dom_id}_status">功能：查看关系方向、节点信息与教材证据</span>
      </div>
      <div class="selection-box" id="${dom_id}_selection_box"></div>
      <div id="$dom_id"></div>
    </div>
    <div id="$detail_id">$default_detail</div>
  </div>
  <script>
    const baseNodes = $node_rows;
    const baseEdges = $edge_rows;
    const nodeInfo = $node_info;
    const edgeInfo = $edge_info;
    const nodes = new vis.DataSet(baseNodes);
    const edges = new vis.DataSet(baseEdges);
    const detail = document.getElementById('$detail_id');
    const defaultDetail = $default_detail_json;
    const neighborhoodDepth = $neighborhood_depth;
    const enableHubClusters = $enable_hub_clusters;
    const clusterHubIds = $cluster_hub_ids;
    const usePhysics = $use_physics;
    const hasRelationLayout = $has_relation_layout;
    const initialScale = $initial_scale;
    const layoutStorageKey = $layout_storage_key;
    const adjacency = {};
    const clusterMembers = {};
    const clusterHubMap = {};

    function escapeHtml(value) {
      return String(value === undefined || value === null ? '' : value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }

    function addLink(a, b) {
      if (!adjacency[a]) adjacency[a] = new Set();
      adjacency[a].add(b);
    }

    baseEdges.forEach((edge) => {
      addLink(edge.from, edge.to);
      addLink(edge.to, edge.from);
    });

    function renderMap(titleText, obj) {
      let html = '<div class="panel-title">' + escapeHtml(titleText) + '</div>';
      Object.entries(obj).forEach(([k, v]) => {
        html += '<div class="kv"><div class="k">' + escapeHtml(k) + '</div><div class="v">' + escapeHtml(v) + '</div></div>';
      });
      detail.innerHTML = html;
    }

    function renderClusterDetail(clusterId) {
      const memberIds = network.getNodesInCluster(clusterId) || [];
      const memberNames = memberIds
        .map((nodeId) => {
          const matched = baseNodes.find((node) => node.id === nodeId);
          return matched ? (matched.rawLabel || matched.fullLabel) : nodeId;
        })
        .filter(Boolean);
      const clusterNode = network.body.nodes[clusterId];
      renderMap('聚合节点', {
        '名称': clusterNode && clusterNode.options ? clusterNode.options.rawLabel || clusterNode.options.fullLabel || clusterId : clusterId,
        '包含节点数': memberNames.length,
        '包含节点': memberNames.join('、'),
        '说明': '这是按中心节点折叠后的大节点，点击后已展开其包含的小节点。',
      });
    }

    function getHubMembers(hubId) {
        const members = new Set([hubId]);
        baseEdges.forEach((edge) => {
          if (edge.from === hubId && !clusterHubIds.includes(edge.to)) members.add(edge.to);
          if (edge.to === hubId && !clusterHubIds.includes(edge.from)) members.add(edge.from);
        });
        return Array.from(members);
    }

    function getNodeRadiusById(nodeId) {
      const datasetNode = nodes.get(nodeId) || baseNodes.find((node) => node.id === nodeId);
      const rawSize = datasetNode && Number.isFinite(datasetNode.nodeSize)
        ? datasetNode.nodeSize
        : datasetNode && Number.isFinite(datasetNode.size)
        ? datasetNode.size
        : 28;
      return Math.max(18, rawSize * 0.7);
    }

    function collectOccupiedNodes(excludeIds) {
      const excludeSet = new Set(excludeIds || []);
      return nodes.getIds()
        .filter((nodeId) => !excludeSet.has(nodeId))
        .map((nodeId) => {
          const datasetNode = nodes.get(nodeId);
          const bodyNode = network.body.nodes[nodeId];
          if (!datasetNode || datasetNode.hidden || !bodyNode) return null;
          const pos = bodyNode.getPosition();
          return { x: pos.x, y: pos.y, radius: getNodeRadiusById(nodeId), nodeId };
        })
        .filter(Boolean);
    }

    function findCollisionFreePosition(candidate, occupied, minDistance, preferredAngle) {
      let x = candidate.x;
      let y = candidate.y;
      const baseAngle = Number.isFinite(preferredAngle) ? preferredAngle : 0;
      for (let iteration = 0; iteration < 24; iteration += 1) {
        let adjusted = false;
        occupied.forEach((item, index) => {
          const limit = Math.max(minDistance, (item.radius || 0) + minDistance * 0.72);
          let dx = x - item.x;
          let dy = y - item.y;
          let dist = Math.hypot(dx, dy);
          if (dist < limit) {
            adjusted = true;
            if (dist < 1e-6) {
              const angle = baseAngle + (index + iteration + 1) * 0.7;
              dx = Math.cos(angle);
              dy = Math.sin(angle);
              dist = 1;
            }
            const push = limit - dist + 8;
            x += (dx / dist) * push;
            y += (dy / dist) * push;
          }
        });
        if (!adjusted) break;
      }
      return { x, y };
    }

    function spreadNodesAround(center, hubId, memberIds) {
      if (!memberIds.length) return;
      if (hubId && network.body.nodes[hubId]) {
        network.moveNode(hubId, center.x, center.y);
      }
      const childIds = memberIds.filter((nodeId) => nodeId !== hubId && network.body.nodes[nodeId]);
      if (!childIds.length) return;
      const hubNode = baseNodes.find((node) => node.id === hubId);
      const hubSize = hubNode && Number.isFinite(hubNode.nodeSize) ? hubNode.nodeSize : 28;
      const baseRadius = Math.max(hubSize * 2.1, 88);
      const occupied = collectOccupiedNodes(memberIds);
      childIds.forEach((nodeId, index) => {
        const ring = Math.floor(index / 8);
        const ringRadius = baseRadius + ring * Math.max(42, hubSize * 1.1);
        const slotCount = Math.min(childIds.length - ring * 8, 8);
        const angle = slotCount <= 1 ? 0 : ((index % 8) / slotCount) * Math.PI * 2;
        const candidate = {
          x: center.x + Math.cos(angle) * ringRadius,
          y: center.y + Math.sin(angle) * ringRadius,
        };
        const collisionFree = findCollisionFreePosition(
          candidate,
          occupied,
          Math.max(72, getNodeRadiusById(nodeId) * 2.6),
          angle
        );
        network.moveNode(nodeId, collisionFree.x, collisionFree.y);
        occupied.push({ x: collisionFree.x, y: collisionFree.y, radius: getNodeRadiusById(nodeId), nodeId });
      });
    }

    function expandClusterNode(clusterId, focusAfterOpen) {
      const center = network.getPositions([clusterId])[clusterId] || network.getViewPosition();
      const hubId = clusterHubMap[clusterId] || String(clusterId).replace('cluster_', '');
      const memberIds = (clusterMembers[clusterId] || []).slice();
      network.openCluster(clusterId);
      setTimeout(() => {
        spreadNodesAround(center, hubId, memberIds);
        unlockManualDrag();
        if (focusAfterOpen && network.body.nodes[hubId]) {
          focusNeighborhood(hubId, neighborhoodDepth);
          renderMap('节点详情', nodeInfo[hubId] || { '名称': hubId, '说明': '已聚焦该节点的局部邻域。' });
        }
      }, 120);
    }

    const network = new vis.Network(document.getElementById('$dom_id'), { nodes, edges }, $options_json);
    const graphDom = document.getElementById('$dom_id');
    const statusEl = document.getElementById('${dom_id}_status');
    const saveBtn = document.getElementById('${dom_id}_save');
    const resetBtn = document.getElementById('${dom_id}_reset');
    const selectionBox = document.getElementById('${dom_id}_selection_box');
    let initialized = false;
    let selectionDrag = null;
    let selectionIntent = null;
    let longPressTimer = null;
    let groupDragState = null;
    let suppressNextClick = false;
    const longPressMs = 260;
    const dragCancelThreshold = 8;

    function setStatus(text) {
      if (statusEl) statusEl.textContent = text;
    }

    function hideSelectionBox() {
      if (!selectionBox) return;
      selectionBox.style.display = 'none';
      selectionBox.style.left = '0px';
      selectionBox.style.top = '0px';
      selectionBox.style.width = '0px';
      selectionBox.style.height = '0px';
    }

    function updateSelectionBox(startX, startY, currentX, currentY) {
      if (!selectionBox) return;
      const left = Math.min(startX, currentX);
      const top = Math.min(startY, currentY);
      const width = Math.abs(currentX - startX);
      const height = Math.abs(currentY - startY);
      selectionBox.style.display = 'block';
      selectionBox.style.left = left + 'px';
      selectionBox.style.top = top + 'px';
      selectionBox.style.width = width + 'px';
      selectionBox.style.height = height + 'px';
    }

    function getNodesInSelectionRect(rect) {
      return nodes.getIds().filter((nodeId) => {
        const bodyNode = network.body.nodes[nodeId];
        if (!bodyNode || bodyNode.options.hidden) return false;
        const domPoint = network.canvasToDOM(bodyNode.getPosition());
        return domPoint.x >= rect.left && domPoint.x <= rect.right && domPoint.y >= rect.top && domPoint.y <= rect.bottom;
      });
    }

    function cancelLongPressIntent() {
      if (longPressTimer) {
        clearTimeout(longPressTimer);
        longPressTimer = null;
      }
      selectionIntent = null;
    }

    function collectCurrentPositions() {
      const ids = Object.keys(network.body.nodes || {});
      if (!ids.length) return {};
      return network.getPositions(ids);
    }

    function saveCurrentLayout() {
      try {
        const positions = collectCurrentPositions();
        localStorage.setItem(layoutStorageKey, JSON.stringify(positions));
        setStatus('已保存当前视图布局');
      } catch (error) {
        setStatus('保存失败，请重试');
      }
    }

    function applySavedLayout() {
      try {
        const raw = localStorage.getItem(layoutStorageKey);
        if (!raw) return false;
        const saved = JSON.parse(raw);
        let applied = 0;
        Object.entries(saved).forEach(([nodeId, pos]) => {
          if (network.body.nodes[nodeId] && Number.isFinite(pos.x) && Number.isFinite(pos.y)) {
            network.moveNode(nodeId, pos.x, pos.y);
            applied += 1;
          }
        });
        if (applied > 0) {
          network.setOptions({ physics: false });
          setStatus('已恢复你上次保存的布局');
          return true;
        }
      } catch (error) {
        setStatus('读取已保存布局失败');
      }
      return false;
    }

    function resetSavedLayout() {
      localStorage.removeItem(layoutStorageKey);
      setStatus('已恢复默认布局');
      restoreGraph();
      setTimeout(() => {
        network.fit({ animation: false });
        const position = network.getViewPosition();
        network.moveTo({ position, scale: initialScale, animation: false });
        unlockManualDrag();
      }, 120);
    }

    function unlockManualDrag() {
      network.setOptions({
        physics: false,
        layout: { hierarchical: { enabled: false } },
        edges: { smooth: false },
      });
    }

    function applyHubClusters() {
      if (!enableHubClusters || !clusterHubIds.length) return;
      const used = new Set();
      clusterHubIds.forEach((hubId) => {
        const hubNode = baseNodes.find((node) => node.id === hubId);
        if (!hubNode) return;
        const members = getHubMembers(hubId).filter((nodeId) => !used.has(nodeId));
        if (members.length < 3) return;
        members.forEach((nodeId) => used.add(nodeId));
        clusterMembers['cluster_' + hubId] = members.slice();
        clusterHubMap['cluster_' + hubId] = hubId;
        network.cluster({
          joinCondition(nodeOptions) {
            return members.includes(nodeOptions.id);
          },
          processProperties(clusterOptions, childNodes) {
            const childCount = childNodes.length;
            return {
              id: 'cluster_' + hubId,
              label: hubNode.baseLabel + '\\n(' + childCount + ')',
              baseLabel: hubNode.baseLabel + '\\n(' + childCount + ')',
              fullLabel: hubNode.fullLabel,
              rawLabel: hubNode.rawLabel || hubNode.fullLabel,
              title: (hubNode.rawLabel || hubNode.fullLabel) + '<br>包含 ' + childCount + ' 个节点，点击后展开',
              shape: hubNode.shape || 'dot',
              size: Math.max(hubNode.nodeSize * 1.45, 36),
              nodeSize: Math.max(hubNode.nodeSize * 1.45, 36),
              color: hubNode.color,
              borderWidth: 4,
              font: {
                size: Math.max(hubNode.font.size + 2, 18),
                color: '#0f172a',
                face: 'Microsoft YaHei',
              },
              shadow: hubNode.shadow,
            };
          },
        });
      });
    }

    function restoreGraph() {
      nodes.clear();
      edges.clear();
      nodes.add(baseNodes.map((node) => ({ ...node, hidden: false, label: node.baseLabel })));
      edges.add(baseEdges.map((edge) => ({ ...edge, hidden: false, label: edge.baseLabel })));
      network.setData({ nodes, edges });
      applyHubClusters();
    }

    function restoreCurrentWindowState() {
      const currentPositions = collectCurrentPositions();
      const existingNodeIds = new Set(nodes.getIds());
      const existingEdgeIds = new Set(edges.getIds());
      nodes.update(
        baseNodes
          .filter((node) => existingNodeIds.has(node.id))
          .map((node) => ({ ...node, hidden: false, label: node.baseLabel }))
      );
      edges.update(
        baseEdges
          .filter((edge) => existingEdgeIds.has(edge.id))
          .map((edge) => ({ ...edge, hidden: false, label: edge.baseLabel }))
      );
      Object.entries(currentPositions).forEach(([nodeId, pos]) => {
        if (network.body.nodes[nodeId] && Number.isFinite(pos.x) && Number.isFinite(pos.y)) {
          network.moveNode(nodeId, pos.x, pos.y);
        }
      });
      network.redraw();
    }

    function finishSelectionDrag(clientX, clientY) {
      if (!selectionDrag) return;
      const rect = graphDom.getBoundingClientRect();
      const endX = clientX - rect.left;
      const endY = clientY - rect.top;
      const selectionRect = {
        left: Math.min(selectionDrag.startX, endX),
        top: Math.min(selectionDrag.startY, endY),
        right: Math.max(selectionDrag.startX, endX),
        bottom: Math.max(selectionDrag.startY, endY),
      };
      const selectedIds = getNodesInSelectionRect(selectionRect);
      network.setSelection({ nodes: selectedIds, edges: [] }, { unselectAll: true, highlightEdges: false });
      network.setOptions({ interaction: { dragView: true, multiselect: true, selectConnectedEdges: false } });
      hideSelectionBox();
      selectionDrag = null;
      suppressNextClick = true;
      if (selectedIds.length) {
        setStatus('已框选 ' + selectedIds.length + ' 个节点，可拖动任一已选节点整体平移');
      } else {
        setStatus('框选区域内没有节点');
      }
    }

    graphDom.addEventListener('mousedown', function(event) {
      if (event.button !== 0) return;
      const rect = graphDom.getBoundingClientRect();
      const startX = event.clientX - rect.left;
      const startY = event.clientY - rect.top;
      const nodeAt = network.getNodeAt({ x: startX, y: startY });
      if (nodeAt || event.ctrlKey || event.metaKey) {
        cancelLongPressIntent();
        return;
      }
      selectionIntent = { startX, startY, clientX: event.clientX, clientY: event.clientY };
      longPressTimer = window.setTimeout(() => {
        if (!selectionIntent) return;
        selectionDrag = { startX: selectionIntent.startX, startY: selectionIntent.startY };
        network.setOptions({ interaction: { dragView: false, multiselect: true, selectConnectedEdges: false } });
        updateSelectionBox(selectionIntent.startX, selectionIntent.startY, selectionIntent.startX, selectionIntent.startY);
        setStatus('框选中：松开鼠标完成多选');
        suppressNextClick = true;
      }, longPressMs);
    });

    window.addEventListener('mousemove', function(event) {
      const rect = graphDom.getBoundingClientRect();
      if (selectionDrag) {
        updateSelectionBox(selectionDrag.startX, selectionDrag.startY, event.clientX - rect.left, event.clientY - rect.top);
        return;
      }
      if (!selectionIntent) return;
      const dx = event.clientX - selectionIntent.clientX;
      const dy = event.clientY - selectionIntent.clientY;
      if (Math.abs(dx) > dragCancelThreshold || Math.abs(dy) > dragCancelThreshold) {
        cancelLongPressIntent();
      }
    });

    window.addEventListener('mouseup', function(event) {
      cancelLongPressIntent();
      finishSelectionDrag(event.clientX, event.clientY);
    });

    graphDom.addEventListener('mouseleave', function() {
      cancelLongPressIntent();
      if (selectionDrag) {
        hideSelectionBox();
        selectionDrag = null;
        network.setOptions({ interaction: { dragView: true, multiselect: true, selectConnectedEdges: false } });
      }
    });

    window.addEventListener('keydown', function(event) {
      if (event.key === 'Escape') {
        cancelLongPressIntent();
        hideSelectionBox();
        selectionDrag = null;
        network.unselectAll();
        network.setOptions({ interaction: { dragView: true, multiselect: true, selectConnectedEdges: false } });
        setStatus('已取消当前多选');
      }
    });

    function getNeighborhood(startId, depth) {
      let frontier = [startId];
      const visited = new Set([startId]);
      for (let step = 0; step < depth; step += 1) {
        const next = [];
        frontier.forEach((nodeId) => {
          (adjacency[nodeId] || []).forEach((neighborId) => {
            if (!visited.has(neighborId)) {
              visited.add(neighborId);
              next.push(neighborId);
            }
          });
        });
        frontier = next;
      }
      return visited;
    }

    function focusNeighborhood(nodeId, depthOverride) {
      const effectiveDepth = Number.isFinite(depthOverride) ? depthOverride : neighborhoodDepth;
      const focusNodes = getNeighborhood(nodeId, effectiveDepth);
      const focusEdges = new Set(baseEdges.filter((edge) => focusNodes.has(edge.from) && focusNodes.has(edge.to)).map((edge) => edge.id));
      nodes.update(baseNodes.map((node) => {
        if (focusNodes.has(node.id)) {
          const focusSize = node.id === nodeId ? Math.max(node.nodeSize + 8, node.nodeSize * 1.35) : node.nodeSize;
          return {
            ...node,
            hidden: false,
            label: node.fullLabel,
            size: focusSize,
            nodeSize: focusSize,
          };
        }
        return { ...node, hidden: true };
      }));
      edges.update(baseEdges.map((edge) => {
        if (focusEdges.has(edge.id)) {
          return { ...edge, hidden: false, label: '', width: edge.width + 0.4 };
        }
        return { ...edge, hidden: true };
      }));
    }

    function clearDetail() {
      detail.innerHTML = defaultDetail;
      restoreCurrentWindowState();
      network.unselectAll();
      setTimeout(() => network.redraw(), 0);
    }

    function initializeView() {
      if (initialized) return;
      initialized = true;
      const restored = applySavedLayout();
      if (!restored) {
        network.fit({ animation: false });
        const position = network.getViewPosition();
        network.moveTo({ position, scale: initialScale, animation: false });
      }
      unlockManualDrag();
      if (!restored) {
        setStatus('可拖拽节点后点击“保存布局”');
      }
    }

    applyHubClusters();
    if (saveBtn) saveBtn.addEventListener('click', saveCurrentLayout);
    if (resetBtn) resetBtn.addEventListener('click', resetSavedLayout);

    network.once('stabilizationIterationsDone', initializeView);
    setTimeout(initializeView, 1200);

    network.on('dragStart', function(params) {
      const selectedIds = network.getSelectedNodes();
      if (params.nodes.length > 0 && selectedIds.length > 1 && selectedIds.includes(params.nodes[0])) {
        groupDragState = {
          selectedIds: selectedIds.slice(),
          startPositions: network.getPositions(selectedIds),
          startPointer: (params.pointer && params.pointer.canvas)
            ? { x: params.pointer.canvas.x, y: params.pointer.canvas.y }
            : network.DOMtoCanvas(params.pointer.DOM),
        };
      } else {
        groupDragState = null;
      }
    });

    network.on('dragging', function(params) {
      if (!groupDragState) return;
      const pointer = (params.pointer && params.pointer.canvas)
        ? { x: params.pointer.canvas.x, y: params.pointer.canvas.y }
        : network.DOMtoCanvas(params.pointer.DOM);
      const dx = pointer.x - groupDragState.startPointer.x;
      const dy = pointer.y - groupDragState.startPointer.y;
      groupDragState.selectedIds.forEach((nodeId) => {
        const startPos = groupDragState.startPositions[nodeId];
        if (!startPos) return;
        network.moveNode(nodeId, startPos.x + dx, startPos.y + dy);
      });
    });

    network.on('dragEnd', function() {
      if (groupDragState && groupDragState.selectedIds.length > 1) {
        setStatus('已整体平移 ' + groupDragState.selectedIds.length + ' 个节点');
      }
      groupDragState = null;
    });

    network.on('click', function(params) {
      if (suppressNextClick) {
        suppressNextClick = false;
        return;
      }
      if (params.nodes.length > 0 && String(params.nodes[0]).startsWith('cluster_')) {
        const clusterId = params.nodes[0];
        renderClusterDetail(clusterId);
        expandClusterNode(clusterId, false);
      } else if (params.nodes.length > 0 && nodeInfo[params.nodes[0]]) {
        renderMap('节点详情', nodeInfo[params.nodes[0]]);
        const clickEvent = params.event && params.event.srcEvent ? params.event.srcEvent : null;
        if (clickEvent && (clickEvent.ctrlKey || clickEvent.metaKey)) {
          const selectedIds = network.getSelectedNodes();
          setStatus('已多选 ' + selectedIds.length + ' 个节点，可直接整体拖动');
        }
      } else if (params.edges.length > 0 && edgeInfo[params.edges[0]]) {
        renderMap('边详情', edgeInfo[params.edges[0]]);
      } else {
        clearDetail();
      }
    });

    network.on('doubleClick', function(params) {
      if (params.nodes.length > 0 && String(params.nodes[0]).startsWith('cluster_')) {
        const clusterId = params.nodes[0];
        renderClusterDetail(clusterId);
        expandClusterNode(clusterId, true);
      } else if (params.nodes.length > 0 && nodeInfo[params.nodes[0]]) {
        focusNeighborhood(params.nodes[0], neighborhoodDepth);
        renderMap('节点详情', nodeInfo[params.nodes[0]]);
        setStatus('已聚焦该节点的 ' + neighborhoodDepth + ' 跳邻域');
      }
    });
  </script>
</body>
</html>
        """
    )
    options = {
        "autoResize": True,
        "interaction": {
            "hover": True,
            "dragNodes": True,
            "dragView": True,
            "zoomView": True,
            "navigationButtons": True,
            "multiselect": True,
            "selectConnectedEdges": False,
        },
        "layout": layout_bundle["layout"],
        "physics": layout_bundle["physics"],
        "nodes": {"shape": "circle" if neo4j_style else "dot"},
        "edges": {
            "selectionWidth": 4,
            "arrowStrikethrough": False,
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.82}},
            **layout_bundle["edges"],
        },
    }
    default_detail = "<div class='panel-title'>功能介绍</div><div class='panel-text'>支持节点详情查看、关系方向辨识、教材证据回溯与布局保存。</div>"
    return template.substitute(
        dom_id=dom_id,
        detail_id=detail_id,
        title=title,
        layout_label=layout_bundle["label"],
        height=int(height),
        node_rows=json.dumps(node_rows, ensure_ascii=False),
        edge_rows=json.dumps(edge_rows, ensure_ascii=False),
        node_info=json.dumps(node_info, ensure_ascii=False),
        edge_info=json.dumps(edge_info, ensure_ascii=False),
        default_detail=default_detail,
        default_detail_json=json.dumps(default_detail, ensure_ascii=False),
        neighborhood_depth=int(neighborhood_depth),
        enable_hub_clusters=json.dumps(bool(enable_hub_clusters)),
        cluster_hub_ids=json.dumps(cluster_hub_ids, ensure_ascii=False),
        use_physics=json.dumps(bool(physics_enabled)),
        has_relation_layout=json.dumps(bool(resolved_relation)),
        initial_scale=json.dumps(effective_initial_scale),
        layout_storage_key=json.dumps(layout_storage_key, ensure_ascii=False),
        options_json=json.dumps(options, ensure_ascii=False),
    )


def metrics_table(metrics):
    """把统计字典整理成适合页面展示的单行表格。"""
    return pd.DataFrame(
        [
            {
                "可见节点数": metrics["visible_nodes"],
                "可见边数": metrics["visible_edges"],
                "连通分量数": metrics["components"],
                "1 节点组件数": metrics["one_node_components"],
                "2 节点组件数": metrics["two_node_components"],
                "最大连通子图": metrics["largest_component"],
            }
        ]
    )


def get_view_heights():
    """集中维护各类视图的默认画布高度。"""
    base_height = int(st.session_state.get("kg_v5_window_height", 720))
    preview_height = max(320, int(base_height * 0.5))
    theme_window_height = max(300, int(base_height * 0.47))
    compact_height = max(420, int(base_height * 0.72))
    return {
        "base": base_height,
        "base_wrapper": base_height + 80,
        "preview": preview_height,
        "preview_wrapper": preview_height + 70,
        "theme_window": theme_window_height,
        "theme_window_wrapper": theme_window_height + 70,
        "compact": compact_height,
        "compact_wrapper": compact_height + 80,
    }


def adaptive_canvas_height(node_count, base_height, *, threshold=18, growth_per_node=6, max_extra=360, minimum=560, maximum=1260):
    """按节点数自适应放大画布，避免图谱过密。"""
    base = max(minimum, int(base_height))
    extra = max(0, int(node_count) - threshold) * growth_per_node
    return min(maximum, base + min(max_extra, extra))


def adaptive_wrapper_height(canvas_height, extra=80):
    """计算嵌入 HTML 外层容器的实际高度。"""
    return int(canvas_height) + extra


def pick_structure_default_center(node_view, preferred_names=None):
    """为结构图模式选择默认中心节点。"""
    if node_view.empty:
        return None
    ranked = node_view.sort_values(["pagerank", "degree_centrality", "degree", "name"], ascending=[False, False, False, True]).copy()
    preferred_names = [name for name in (preferred_names or []) if str(name).strip()]
    if preferred_names:
        preferred = ranked[ranked["name"].isin(preferred_names)]
        if not preferred.empty:
            return preferred.iloc[0]["node_id"]
    return ranked.iloc[0]["node_id"]


def get_valid_structure_center_ids(edge_view):
    """筛选符合“纯上位节点”规则的结构图中心候选。"""
    if edge_view.empty:
        return set()
    working = edge_view[edge_view["relation"].isin(RELATION_OPTIONS)].copy()
    upper_ids = set()
    lower_ids = set()
    for relation, role_rule in STRUCTURE_CENTER_ROLE_RULES.items():
        rel_edges = working[working["relation"] == relation]
        if rel_edges.empty:
            continue
        upper_ids.update(rel_edges[role_rule["upper"]].astype(str).tolist())
        lower_ids.update(rel_edges[role_rule["lower"]].astype(str).tolist())
    return upper_ids - lower_ids


def compute_arc_positions(node_ids, *, angle_deg, span_deg, radius, ring_gap=92, per_ring=5):
    """把一跳邻居按扇区圆弧规则摆放。"""
    positions = {}
    ordered = list(node_ids)
    if not ordered:
        return positions
    for ring_index, start in enumerate(range(0, len(ordered), per_ring)):
        chunk = ordered[start : start + per_ring]
        ring_radius = radius + ring_index * ring_gap
        if len(chunk) == 1:
            angles = [angle_deg]
        else:
            start_angle = angle_deg - span_deg / 2
            step = span_deg / len(chunk)
            angles = [start_angle + step * (idx + 0.5) for idx in range(len(chunk))]
        for node_id, current_angle in zip(chunk, angles):
            radians = math.radians(current_angle)
            positions[node_id] = {
                "x": round(math.cos(radians) * ring_radius, 2),
                "y": round(math.sin(radians) * ring_radius, 2),
            }
    return positions


def compute_branch_positions(parent_position, node_ids, *, anchor_angle_deg, base_radius=152, ring_gap=84, per_ring=3, span_deg=46):
    """围绕一跳节点继续摆放二跳分支节点。"""
    positions = {}
    ordered = list(node_ids)
    if not ordered:
        return positions
    for ring_index, start in enumerate(range(0, len(ordered), per_ring)):
        chunk = ordered[start : start + per_ring]
        ring_radius = base_radius + ring_index * ring_gap
        if len(chunk) == 1:
            angles = [anchor_angle_deg]
        else:
            start_angle = anchor_angle_deg - span_deg / 2
            step = span_deg / len(chunk)
            angles = [start_angle + step * (idx + 0.5) for idx in range(len(chunk))]
        for node_id, current_angle in zip(chunk, angles):
            radians = math.radians(current_angle)
            positions[node_id] = {
                "x": round(parent_position["x"] + math.cos(radians) * ring_radius, 2),
                "y": round(parent_position["y"] + math.sin(radians) * ring_radius, 2),
            }
    return positions


def build_structure_bundle(node_view, edge_view, center_node_id):
    """围绕中心节点构建结构图模式的数据包。"""
    if node_view.empty or center_node_id not in set(node_view["node_id"]):
        return None

    nodes = node_view.copy().reset_index(drop=True)
    edges = apply_qc_flags(edge_view).copy().reset_index(drop=True)
    center_row = nodes[nodes["node_id"] == center_node_id].iloc[0]
    incident_edges = edges[(edges["source"] == center_node_id) | (edges["target"] == center_node_id)].copy()

    if incident_edges.empty:
        return {
            "center_id": center_node_id,
            "center_name": center_row["name"],
            "nodes": [
                {
                    "id": center_node_id,
                    "name": center_row["name"],
                    "entity_type": center_row["entity_type"],
                    "x": 0.0,
                    "y": 0.0,
                    "value": 54,
                    "label": center_row["name"],
                    "base_label": center_row["name"],
                    "hidden": False,
                    "level": 0,
                    "sector": "中心",
                }
            ],
            "edges": [],
            "children_map": {},
            "stats": {"一跳邻居数": 0, "二跳候选数": 0, REL_IS_A: 0, REL_CONTAINS: 0, REL_USED_FOR: 0},
        }

    relation_priority = {REL_IS_A: 0, REL_CONTAINS: 1, REL_USED_FOR: 2}
    node_lookup = nodes.set_index("node_id").to_dict("index")
    neighbor_groups = {}
    for _, row in incident_edges.iterrows():
        neighbor_id = row["target"] if row["source"] == center_node_id else row["source"]
        neighbor_groups.setdefault(neighbor_id, []).append(row.to_dict())

    one_hop_relations = {}
    relation_buckets = {relation: [] for relation in RELATION_OPTIONS}
    for neighbor_id, records in neighbor_groups.items():
        primary = sorted(records, key=lambda item: (relation_priority.get(item["relation"], 99), item.get("target_name", "")))[0]
        one_hop_relations[neighbor_id] = primary
        relation_buckets[primary["relation"]].append(neighbor_id)

    for relation in RELATION_OPTIONS:
        relation_buckets[relation] = sorted(
            relation_buckets[relation],
            key=lambda node_id: (
                -float(node_lookup.get(node_id, {}).get("pagerank", 0.0)),
                -float(node_lookup.get(node_id, {}).get("degree_centrality", 0.0)),
                node_lookup.get(node_id, {}).get("name", node_id),
            ),
        )

    one_hop_positions = {}
    for relation, config in STRUCTURE_SECTOR_CONFIG.items():
        one_hop_positions.update(
            compute_arc_positions(
                relation_buckets[relation],
                angle_deg=config["angle_deg"],
                span_deg=config["span_deg"],
                radius=int(config["radius"] * LAYOUT_DISTANCE_BOOST),
                ring_gap=int(config.get("ring_gap", 92) * LAYOUT_DISTANCE_BOOST),
                per_ring=config.get("per_ring", 5),
            )
        )

    ranked_neighbors = sorted(
        one_hop_relations.keys(),
        key=lambda node_id: (
            relation_priority.get(one_hop_relations[node_id]["relation"], 99),
            -float(node_lookup.get(node_id, {}).get("pagerank", 0.0)),
            -float(node_lookup.get(node_id, {}).get("degree_centrality", 0.0)),
            node_lookup.get(node_id, {}).get("name", node_id),
        ),
    )

    node_records = []
    edge_records = []
    children_map = {}
    used_second_hop = set()
    second_hop_count = 0

    node_records.append(
        {
            "id": center_node_id,
            "name": center_row["name"],
            "entity_type": center_row["entity_type"],
            "x": 0.0,
            "y": 0.0,
            "value": max(56, int(44 + center_row.get("degree", 0) * 2.2 + center_row.get("pagerank", 0.0) * 280)),
            "label": center_row["name"],
            "base_label": center_row["name"],
            "hidden": False,
            "level": 0,
            "sector": "中心",
        }
    )

    for neighbor_id in ranked_neighbors:
        row = node_lookup.get(neighbor_id, {})
        primary_edge = one_hop_relations[neighbor_id]
        position = one_hop_positions.get(neighbor_id, {"x": 0.0, "y": 0.0})
        label = row.get("name", neighbor_id)
        node_records.append(
            {
                "id": neighbor_id,
                "name": row.get("name", neighbor_id),
                "entity_type": row.get("entity_type", UNKNOWN_TYPE),
                "x": position["x"],
                "y": position["y"],
                "value": max(34, int(28 + row.get("degree", 0) * 1.6 + row.get("pagerank", 0.0) * 220)),
                "label": label,
                "base_label": label,
                "hidden": False,
                "level": 1,
                "sector": primary_edge["relation"],
            }
        )
        edge_records.append(
            {
                "id": f"struct_{center_node_id}_{neighbor_id}_{primary_edge['relation']}",
                "from": center_node_id,
                "to": neighbor_id,
                "relation": primary_edge["relation"],
                "source_name": primary_edge["source_name"],
                "target_name": primary_edge["target_name"],
                "source_file": primary_edge.get("source_file", ""),
                "block_id": primary_edge.get("block_id", ""),
                "evidence_text": primary_edge.get("evidence_text", ""),
                "context_snippet": primary_edge.get("context_snippet", ""),
                "is_duplicate": bool(primary_edge.get("is_duplicate", False)),
                "is_conflict": bool(primary_edge.get("is_conflict", False)),
                "is_short_cycle": bool(primary_edge.get("is_short_cycle", False)),
                "hidden": False,
                "level": 1,
            }
        )

    for parent_id in ranked_neighbors:
        parent_position = one_hop_positions.get(parent_id, {"x": 0.0, "y": 0.0})
        anchor_angle = math.degrees(math.atan2(parent_position["y"], parent_position["x"])) if parent_position["x"] or parent_position["y"] else 0.0
        parent_edges = edges[(edges["source"] == parent_id) | (edges["target"] == parent_id)].copy()
        parent_edges = parent_edges[
            ~(
                ((parent_edges["source"] == center_node_id) & (parent_edges["target"] == parent_id))
                | ((parent_edges["source"] == parent_id) & (parent_edges["target"] == center_node_id))
            )
        ].copy()
        parent_candidates = []
        for _, row in parent_edges.iterrows():
            other_id = row["target"] if row["source"] == parent_id else row["source"]
            if other_id == center_node_id or other_id in ranked_neighbors or other_id in used_second_hop:
                continue
            parent_candidates.append((other_id, row.to_dict()))
        if not parent_candidates:
            continue
        dedup_parent = {}
        for other_id, row in parent_candidates:
            current = dedup_parent.get(other_id)
            if current is None or relation_priority.get(row["relation"], 99) < relation_priority.get(current["relation"], 99):
                dedup_parent[other_id] = row
        child_ids = sorted(
            dedup_parent.keys(),
            key=lambda node_id: (
                -float(node_lookup.get(node_id, {}).get("pagerank", 0.0)),
                -float(node_lookup.get(node_id, {}).get("degree_centrality", 0.0)),
                node_lookup.get(node_id, {}).get("name", node_id),
            ),
        )[:6]
        child_positions = compute_branch_positions(
            parent_position,
            child_ids,
            anchor_angle_deg=anchor_angle,
            base_radius=int(152 * LAYOUT_DISTANCE_BOOST),
            ring_gap=int(84 * LAYOUT_DISTANCE_BOOST),
        )
        children_map[parent_id] = {"nodes": [], "edges": []}
        for child_id in child_ids:
            used_second_hop.add(child_id)
            second_hop_count += 1
            row = node_lookup.get(child_id, {})
            position = child_positions.get(child_id, parent_position)
            child_edge = dedup_parent[child_id]
            show_label = float(row.get("pagerank", 0.0)) >= 0.01 and len(child_ids) <= 3
            node_records.append(
                {
                    "id": child_id,
                    "name": row.get("name", child_id),
                    "entity_type": row.get("entity_type", UNKNOWN_TYPE),
                    "x": position["x"],
                    "y": position["y"],
                    "value": max(24, int(22 + row.get("degree", 0) * 1.1 + row.get("pagerank", 0.0) * 160)),
                    "label": row.get("name", child_id) if show_label else "",
                    "base_label": row.get("name", child_id) if show_label else "",
                    "hidden": True,
                    "level": 2,
                    "sector": child_edge["relation"],
                }
            )
            edge_id = f"struct_{parent_id}_{child_id}_{child_edge['relation']}"
            edge_records.append(
                {
                    "id": edge_id,
                    "from": parent_id,
                    "to": child_id,
                    "relation": child_edge["relation"],
                    "source_name": child_edge["source_name"],
                    "target_name": child_edge["target_name"],
                    "source_file": child_edge.get("source_file", ""),
                    "block_id": child_edge.get("block_id", ""),
                    "evidence_text": child_edge.get("evidence_text", ""),
                    "context_snippet": child_edge.get("context_snippet", ""),
                    "is_duplicate": bool(child_edge.get("is_duplicate", False)),
                    "is_conflict": bool(child_edge.get("is_conflict", False)),
                    "is_short_cycle": bool(child_edge.get("is_short_cycle", False)),
                    "hidden": True,
                    "level": 2,
                }
            )
            children_map[parent_id]["nodes"].append(child_id)
            children_map[parent_id]["edges"].append(edge_id)

    return {
        "center_id": center_node_id,
        "center_name": center_row["name"],
        "nodes": node_records,
        "edges": edge_records,
        "children_map": children_map,
        "stats": {
            "一跳邻居数": len(ranked_neighbors),
            "二跳候选数": second_hop_count,
            REL_IS_A: len(relation_buckets[REL_IS_A]),
            REL_CONTAINS: len(relation_buckets[REL_CONTAINS]),
            REL_USED_FOR: len(relation_buckets[REL_USED_FOR]),
        },
    }


def make_structure_network_html(structure_bundle, node_view, edge_view, title, *, height=700):
    """生成局部知识结构图的交互 HTML。"""
    if not structure_bundle:
        return f"<div style='padding:24px;font-family:Microsoft YaHei,sans-serif'>{title}：当前节点没有可展示的局部结构。</div>"

    node_lookup = node_view.set_index("node_id").to_dict("index") if not node_view.empty else {}
    node_rows = []
    node_info = {}
    structure_node_count = max(1, len(structure_bundle["nodes"]))
    for item in structure_bundle["nodes"]:
        node_row = node_lookup.get(item["id"], {})
        is_center = item["level"] == 0
        bg_color = TYPE_COLORS.get(item["entity_type"], TYPE_COLORS[UNKNOWN_TYPE])
        font_size = 18 if is_center else 14 if item["level"] == 1 else 12
        label_offset = 0 if is_center else max(18, int(item["value"] * 0.58))
        node_rows.append(
            {
                "id": item["id"],
                "label": item["label"],
                "baseLabel": item["base_label"],
                "fullLabel": item["name"],
                "title": f"{item['name']}<br>类型: {item['entity_type']}<br>层级: {item['level']}",
                "x": item["x"],
                "y": item["y"],
                "physics": False,
                "fixed": {"x": is_center, "y": is_center},
                "color": {
                    "background": bg_color,
                    "border": "#0f172a" if is_center else "#d7dee9",
                    "highlight": {"background": bg_color, "border": "#0f172a" if is_center else "#d7dee9"},
                },
                "font": {
                    "size": font_size,
                    "color": "#172033",
                    "face": "Microsoft YaHei",
                    "strokeWidth": 4,
                    "strokeColor": "#ffffff",
                    "vadjust": label_offset,
                },
                "borderWidth": 4 if is_center else 2,
                "value": item["value"],
                "hidden": bool(item["hidden"]),
                "level": item["level"],
                "sector": item["sector"],
                "shadow": {"enabled": True, "color": "rgba(15,23,42,0.16)", "size": 18} if is_center else False,
            }
        )
        node_info[item["id"]] = {
            "名称": item["name"],
            "类型": item["entity_type"],
            "结构层级": "中心节点" if item["level"] == 0 else f"{item['level']} 跳节点",
            "扇区": item["sector"],
            "度数": int(node_row.get("degree", 0)),
            "PageRank": format_float(node_row.get("pagerank", 0.0)),
            "度中心性": format_float(node_row.get("degree_centrality", 0.0)),
            "组件大小": int(node_row.get("component_size", 0)),
            "质检标记": node_row.get("qc_flags", "正常"),
        }

    edge_rows = []
    edge_info = {}
    for item in structure_bundle["edges"]:
        color = RELATION_COLORS.get(item["relation"], "#94a3b8")
        if item["is_short_cycle"]:
            color = EDGE_ISSUE_COLORS["short_cycle"]
        elif item["is_conflict"]:
            color = EDGE_ISSUE_COLORS["conflict"]
        elif item["is_duplicate"]:
            color = EDGE_ISSUE_COLORS["duplicate"]
        edge_rows.append(
            {
                "id": item["id"],
                "from": item["from"],
                "to": item["to"],
                "label": "",
                "baseLabel": "",
                "color": {"color": color, "highlight": color},
                "arrows": {"to": {"enabled": True, "scaleFactor": 0.82}},
                "width": 2.8 if item["level"] == 1 else 2.0,
                "hidden": bool(item["hidden"]),
                "dashes": [8, 6] if item["is_duplicate"] else False,
            }
        )
        edge_info[item["id"]] = {
            "head": item["source_name"],
            "relation": item["relation"],
            "tail": item["target_name"],
            "source_file": item.get("source_file", ""),
            "block_id": item.get("block_id", ""),
            "evidence_text": item.get("evidence_text", "") or "无",
            "上下文片段": item.get("context_snippet", "") or "无",
            "是否重复边": "是" if item["is_duplicate"] else "否",
            "是否冲突边": "是" if item["is_conflict"] else "否",
            "是否短环": "是" if item["is_short_cycle"] else "否",
        }

    full_node_meta = {}
    for _, row in node_view.iterrows():
        full_node_meta[str(row["node_id"])] = {
            "name": row.get("name", row["node_id"]),
            "entity_type": row.get("entity_type", UNKNOWN_TYPE),
            "degree": int(row.get("degree", 0) or 0),
            "pagerank": float(row.get("pagerank", 0.0) or 0.0),
            "degree_centrality": float(row.get("degree_centrality", 0.0) or 0.0),
            "component_size": int(row.get("component_size", 0) or 0),
            "qc_flags": row.get("qc_flags", "正常") or "正常",
        }
    full_edge_rows = []
    qc_edge_view = apply_qc_flags(edge_view).copy().reset_index(drop=True) if not edge_view.empty else edge_view.copy()
    for idx, row in qc_edge_view.iterrows():
        source_id = str(row.get("source", ""))
        target_id = str(row.get("target", ""))
        source_name = row.get("source_name") or full_node_meta.get(source_id, {}).get("name", source_id)
        target_name = row.get("target_name") or full_node_meta.get(target_id, {}).get("name", target_id)
        full_edge_rows.append(
            {
                "id": f"full_edge_{idx}",
                "source": source_id,
                "target": target_id,
                "relation": row.get("relation", ""),
                "source_name": source_name,
                "target_name": target_name,
                "source_file": row.get("source_file", "") or "",
                "block_id": row.get("block_id", "") or "",
                "evidence_text": row.get("evidence_text", "") or "",
                "context_snippet": row.get("context_snippet", "") or "",
                "is_duplicate": bool(row.get("is_duplicate", False)),
                "is_conflict": bool(row.get("is_conflict", False)),
                "is_short_cycle": bool(row.get("is_short_cycle", False)),
            }
        )

    dom_id = f"struct_{uuid.uuid4().hex}"
    detail_id = f"struct_detail_{uuid.uuid4().hex}"
    storage_nodes = pd.DataFrame([{"node_id": n["id"]} for n in structure_bundle["nodes"]], columns=["node_id"])
    storage_edges = pd.DataFrame(
        [{"source": e["from"], "relation": e["relation"], "target": e["to"]} for e in structure_bundle["edges"]],
        columns=["source", "relation", "target"],
    )
    storage_key = f"{build_layout_storage_key(title, storage_nodes, storage_edges)}::structure"
    detail_panel_width = 280 if structure_node_count >= 18 else 300
    template = Template(
        """
<html>
<head>
  <script src="https://unpkg.com/vis-network@9.1.2/dist/vis-network.min.js"></script>
  <style>
    body { margin: 0; background: #edf2f7; font-family: 'Microsoft YaHei', sans-serif; }
    .title { padding: 8px 2px 10px; color: #334155; font-size: 13px; }
    .legend { display:flex; flex-wrap:wrap; gap:10px; margin: 0 0 10px 0; color:#334155; font-size:12px; }
    .legend-title { font-weight:700; margin-right:6px; }
    .legend-item { display:flex; align-items:center; gap:6px; }
    .legend-item i { display:inline-block; width:12px; height:12px; border-radius:999px; }
    .wrap { display:grid; grid-template-columns: minmax(0, 1fr) ${detail_panel_width}px; gap:14px; }
    #$dom_id { width:100%; height:${height}px; background:#ffffff; border:1px solid #d9e2ec; border-radius:14px; }
    #$detail_id { height:${height}px; background:#ffffff; border:1px solid #d9e2ec; border-radius:14px; padding:14px; overflow-y:auto; }
    .panel-title { font-weight:700; color:#0f172a; margin-bottom:10px; }
    .panel-text { color:#475569; line-height:1.5; }
    .kv { margin-bottom:10px; }
    .k { font-weight:700; color:#0f172a; margin-bottom:4px; }
    .v { color:#334155; word-break:break-word; line-height:1.45; }
    .graph-panel { position: relative; }
    .layout-tools { position:absolute; top:12px; left:12px; z-index:10; display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
    .layout-btn { border:none; border-radius:999px; padding:7px 12px; background:#0f172a; color:#fff; font-size:12px; cursor:pointer; }
    .layout-btn.alt { background:#475569; }
    .layout-status { background:rgba(255,255,255,0.92); border:1px solid #d9e2ec; color:#334155; border-radius:999px; padding:6px 10px; font-size:12px; }
  </style>
</head>
<body>
  <div class="title">$title。功能介绍：以中心节点组织局部知识结构，按关系类型分区展示并支持证据查看。</div>
  <div class="legend">
    <span class="legend-title">扇区说明</span>
    <span class="legend-item"><i style="background:#2563eb;"></i>上方扇区：是一种</span>
    <span class="legend-item"><i style="background:#16a34a;"></i>左下扇区：包含</span>
    <span class="legend-item"><i style="background:#ea580c;"></i>右下扇区：用于</span>
  </div>
  <div class="wrap">
    <div class="graph-panel">
      <div class="layout-tools">
        <button class="layout-btn" id="${dom_id}_save">保存布局</button>
        <button class="layout-btn alt" id="${dom_id}_reset">恢复结构图</button>
        <span class="layout-status" id="${dom_id}_status">功能：查看局部结构、关系方向、节点详情与教材证据</span>
      </div>
      <div id="$dom_id"></div>
    </div>
    <div id="$detail_id">$default_detail</div>
  </div>
  <script>
    const baseNodes = $node_rows;
    const baseEdges = $edge_rows;
    const nodeInfo = $node_info;
    const edgeInfo = $edge_info;
    const childMap = JSON.parse(JSON.stringify($children_map));
    const initialChildMap = JSON.parse(JSON.stringify($children_map));
    const fullNodeMeta = $full_node_meta;
    const fullEdgeRows = $full_edge_rows;
    const structureCenterId = $structure_center_id;
    const relationPriority = { '是一种': 0, '包含': 1, '用于': 2 };
    const relationAngles = { '是一种': -90, '包含': 150, '用于': 30 };
    const persistentVisibleNodeIds = new Set(baseNodes.filter((node) => !node.hidden).map((node) => node.id));
    const persistentVisibleEdgeIds = new Set(baseEdges.filter((edge) => !edge.hidden).map((edge) => edge.id));
    const nodeRevealOwners = {};
    const edgeRevealOwners = {};
    const detail = document.getElementById('$detail_id');
    const defaultDetail = $default_detail_json;
    const statusEl = document.getElementById('${dom_id}_status');
    const storageKey = $storage_key;
    const nodes = new vis.DataSet(baseNodes);
    const edges = new vis.DataSet(baseEdges);
    const network = new vis.Network(
      document.getElementById('$dom_id'),
      { nodes, edges },
      {
        autoResize: true,
        physics: false,
        interaction: { hover: true, dragNodes: true, dragView: true, zoomView: true, navigationButtons: true },
        nodes: { shape: 'dot' },
        edges: {
          smooth: false,
          selectionWidth: 4,
          arrowStrikethrough: false,
          arrows: { to: { enabled: true, scaleFactor: 0.82 } },
        },
      }
    );

    function setStatus(text) {
      if (statusEl) statusEl.textContent = text;
    }

    function escapeHtml(value) {
      return String(value === undefined || value === null ? '' : value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }

    function renderMap(titleText, obj) {
      let html = '<div class="panel-title">' + escapeHtml(titleText) + '</div>';
      Object.entries(obj).forEach(([k, v]) => {
        html += '<div class="kv"><div class="k">' + escapeHtml(k) + '</div><div class="v">' + escapeHtml(v) + '</div></div>';
      });
      detail.innerHTML = html;
    }

    function ensureOwnerBucket(store, id) {
      if (!store[id]) store[id] = new Set();
      return store[id];
    }

    function addRevealOwner(store, id, ownerId) {
      ensureOwnerBucket(store, id).add(ownerId);
    }

    function removeRevealOwner(store, id, ownerId) {
      if (!store[id]) return;
      store[id].delete(ownerId);
      if (store[id].size === 0) delete store[id];
    }

    function hasRevealOwner(store, id) {
      return !!(store[id] && store[id].size > 0);
    }

    function getNodeMeta(nodeId) {
      return fullNodeMeta[nodeId] || { name: nodeId, entity_type: '未知', degree: 0, pagerank: 0, degree_centrality: 0, component_size: 0, qc_flags: '正常' };
    }

    function buildNodeDetail(nodeId, level, sector) {
      const meta = getNodeMeta(nodeId);
      return {
        '名称': meta.name || nodeId,
        '类型': meta.entity_type || '未知',
        '结构层级': level === 0 ? '中心节点' : level + ' 跳节点',
        '扇区': sector || '扩展节点',
        '度数': Number(meta.degree || 0),
        'PageRank': Number(meta.pagerank || 0).toFixed(6),
        '度中心性': Number(meta.degree_centrality || 0).toFixed(6),
        '组件大小': Number(meta.component_size || 0),
        '质检标记': meta.qc_flags || '正常',
      };
    }

    function buildDynamicNode(nodeId, level, sector, position) {
      const meta = getNodeMeta(nodeId);
      const isCenter = level === 0;
      const pagerank = Number(meta.pagerank || 0);
      const degree = Number(meta.degree || 0);
      const size = level <= 1
        ? Math.max(34, Math.round(28 + degree * 1.6 + pagerank * 220))
        : Math.max(24, Math.round(22 + degree * 1.1 + pagerank * 160));
      const showLabel = level <= 1 || pagerank >= 0.01;
      return {
        id: nodeId,
        label: showLabel ? (meta.name || nodeId) : '',
        baseLabel: showLabel ? (meta.name || nodeId) : '',
        fullLabel: meta.name || nodeId,
        title: (meta.name || nodeId) + '<br>类型: ' + (meta.entity_type || '未知') + '<br>层级: ' + level,
        x: position.x,
        y: position.y,
        physics: false,
        fixed: { x: isCenter, y: isCenter },
        color: {
          background: $type_colors[meta.entity_type || '未知'] || $type_colors['未知'],
          border: isCenter ? '#0f172a' : '#d7dee9',
          highlight: { background: $type_colors[meta.entity_type || '未知'] || $type_colors['未知'], border: isCenter ? '#0f172a' : '#d7dee9' },
        },
        font: {
          size: level === 1 ? 14 : 12,
          color: '#172033',
          face: 'Microsoft YaHei',
          strokeWidth: 4,
          strokeColor: '#ffffff',
          vadjust: Math.max(18, Math.round(size * 0.58)),
        },
        borderWidth: isCenter ? 4 : 2,
        value: size,
        size: size,
        hidden: true,
        level: level,
        sector: sector,
      };
    }

    function buildDynamicEdge(edgeRow, parentId, childId, level) {
      let color = $relation_colors[edgeRow.relation] || '#94a3b8';
      if (edgeRow.is_short_cycle) {
        color = $issue_colors.short_cycle;
      } else if (edgeRow.is_conflict) {
        color = $issue_colors.conflict;
      } else if (edgeRow.is_duplicate) {
        color = $issue_colors.duplicate;
      }
      return {
        id: edgeRow.id + '__' + parentId + '__' + childId,
        from: parentId,
        to: childId,
        label: '',
        baseLabel: '',
        relation: edgeRow.relation,
        color: { color: color, highlight: color },
        arrows: { to: { enabled: true, scaleFactor: 0.82 } },
        width: level <= 1 ? 2.8 : 2.0,
        hidden: true,
        dashes: edgeRow.is_duplicate ? [8, 6] : false,
      };
    }

    function rankDynamicCandidates(candidateMap) {
      return Array.from(candidateMap.keys()).sort((a, b) => {
        const aMeta = getNodeMeta(a);
        const bMeta = getNodeMeta(b);
        const aPagerank = Number(aMeta.pagerank || 0);
        const bPagerank = Number(bMeta.pagerank || 0);
        if (bPagerank !== aPagerank) return bPagerank - aPagerank;
        const aDegree = Number(aMeta.degree_centrality || 0);
        const bDegree = Number(bMeta.degree_centrality || 0);
        if (bDegree !== aDegree) return bDegree - aDegree;
        return String(aMeta.name || a).localeCompare(String(bMeta.name || b), 'zh-Hans-CN');
      });
    }

    function ensureChildPayload(parentId) {
      if (childMap[parentId] && Array.isArray(childMap[parentId].nodes) && childMap[parentId].nodes.length > 0) {
        if (childMap[parentId].expanded === undefined) childMap[parentId].expanded = false;
        return childMap[parentId];
      }
      const parentNode = nodes.get(parentId);
      if (!parentNode) return null;
      const parentLevel = Number.isFinite(parentNode.level) ? parentNode.level : 1;
      const knownNodeIds = new Set(nodes.getIds());
      const candidateMap = new Map();
      fullEdgeRows.forEach((edgeRow) => {
        if (edgeRow.source !== parentId && edgeRow.target !== parentId) return;
        const otherId = edgeRow.source === parentId ? edgeRow.target : edgeRow.source;
        if (!otherId || otherId === structureCenterId || otherId === parentId) return;
        const existingNode = nodes.get(otherId);
        if (persistentVisibleNodeIds.has(otherId) || (existingNode && !existingNode.hidden)) return;
        const existing = candidateMap.get(otherId);
        if (!existing || (relationPriority[edgeRow.relation] ?? 99) < (relationPriority[existing.relation] ?? 99)) {
          candidateMap.set(otherId, edgeRow);
        }
      });
      if (!candidateMap.size) return null;
      const rankedIds = rankDynamicCandidates(candidateMap).slice(0, 6);
      const parentPosition = network.getPositions([parentId])[parentId] || { x: 0, y: 0 };
      const occupied = collectVisibleStructureNodes([]);
      const payload = { nodes: [], edges: [], expanded: false };
      const groupedIds = { '是一种': [], '包含': [], '用于': [] };
      rankedIds.forEach((nodeId) => {
        const row = candidateMap.get(nodeId);
        if (row && groupedIds[row.relation]) groupedIds[row.relation].push(nodeId);
      });
      Object.keys(groupedIds).forEach((relationKey) => {
        const group = groupedIds[relationKey];
        const baseAngle = relationAngles[relationKey] ?? 0;
        group.forEach((childId, index) => {
          const edgeRow = candidateMap.get(childId);
          const ring = Math.floor(index / 3);
          const slotIndex = index % 3;
          const slotCount = Math.min(group.length - ring * 3, 3);
          const span = slotCount <= 1 ? 0 : 42;
          const offset = slotCount <= 1 ? 0 : (-span / 2) + (span / (slotCount - 1)) * slotIndex;
          const angle = (baseAngle + offset) * Math.PI / 180;
          const baseRadius = 156 + ring * 92;
          const candidate = {
            x: parentPosition.x + Math.cos(angle) * baseRadius,
            y: parentPosition.y + Math.sin(angle) * baseRadius,
          };
          const childRadius = getStructureNodeRadius(childId);
          const collisionFree = findStructureCollisionFreePosition(
            candidate,
            occupied,
            childRadius,
            Math.max(128, childRadius * 3.8),
            angle
          );
          const level = Math.min(parentLevel + 1, 3);
          if (!knownNodeIds.has(childId)) {
            const dynamicNode = buildDynamicNode(childId, level, edgeRow.relation, collisionFree);
            nodes.add(dynamicNode);
            knownNodeIds.add(childId);
          } else {
            const currentNode = nodes.get(childId);
            if (currentNode) {
              nodes.update({ id: childId, x: collisionFree.x, y: collisionFree.y, hidden: true, level: currentNode.level || level, sector: currentNode.sector || edgeRow.relation });
            }
          }
          const dynamicEdge = buildDynamicEdge(edgeRow, parentId, childId, level);
          if (!edges.get(dynamicEdge.id)) {
            edges.add(dynamicEdge);
          }
          nodeInfo[childId] = buildNodeDetail(childId, level, edgeRow.relation);
          edgeInfo[dynamicEdge.id] = {
            head: edgeRow.source_name,
            relation: edgeRow.relation,
            tail: edgeRow.target_name,
            source_file: edgeRow.source_file || '',
            block_id: edgeRow.block_id || '',
            evidence_text: edgeRow.evidence_text || '无',
            '上下文片段': edgeRow.context_snippet || '无',
            '是否重复边': edgeRow.is_duplicate ? '是' : '否',
            '是否冲突边': edgeRow.is_conflict ? '是' : '否',
            '是否短环': edgeRow.is_short_cycle ? '是' : '否',
          };
          payload.nodes.push(childId);
          payload.edges.push(dynamicEdge.id);
          occupied.push({ x: collisionFree.x, y: collisionFree.y, radius: getStructureNodeRadius(childId), nodeId: childId });
        });
      });
      if (!payload.nodes.length) return null;
      childMap[parentId] = payload;
      return payload;
    }

    function collectCurrentPositions() {
      const ids = Object.keys(network.body.nodes || {});
      if (!ids.length) return {};
      return network.getPositions(ids);
    }

    function saveCurrentLayout() {
      try {
        localStorage.setItem(storageKey, JSON.stringify(collectCurrentPositions()));
        setStatus('已保存当前结构图布局');
      } catch (error) {
        setStatus('保存失败，请重试');
      }
    }

    function applySavedLayout() {
      try {
        const raw = localStorage.getItem(storageKey);
        if (!raw) return false;
        const saved = JSON.parse(raw);
        let applied = 0;
        Object.entries(saved).forEach(([nodeId, pos]) => {
          if (network.body.nodes[nodeId] && Number.isFinite(pos.x) && Number.isFinite(pos.y)) {
            network.moveNode(nodeId, pos.x, pos.y);
            applied += 1;
          }
        });
        if (applied > 0) {
          setStatus('已恢复上次保存的结构图布局');
          return true;
        }
      } catch (error) {
        setStatus('读取已保存布局失败');
      }
      return false;
    }

    function restoreStructureView() {
      nodes.clear();
      edges.clear();
      nodes.add(baseNodes.map((node) => ({ ...node })));
      edges.add(baseEdges.map((edge) => ({ ...edge })));
      Object.keys(nodeRevealOwners).forEach((key) => delete nodeRevealOwners[key]);
      Object.keys(edgeRevealOwners).forEach((key) => delete edgeRevealOwners[key]);
      Object.keys(childMap).forEach((key) => delete childMap[key]);
      Object.entries(initialChildMap).forEach(([key, value]) => {
        childMap[key] = JSON.parse(JSON.stringify(value));
      });
      network.setData({ nodes, edges });
    }

    function getStructureNodeRadius(nodeId) {
      const datasetNode = nodes.get(nodeId) || baseNodes.find((node) => node.id === nodeId);
      const rawSize = datasetNode && Number.isFinite(datasetNode.value)
        ? datasetNode.value
        : datasetNode && Number.isFinite(datasetNode.size)
        ? datasetNode.size
        : 28;
      return Math.max(18, rawSize * 0.68);
    }

    function collectVisibleStructureNodes(excludeIds) {
      const excludeSet = new Set(excludeIds || []);
      return nodes.getIds()
        .filter((nodeId) => !excludeSet.has(nodeId))
        .map((nodeId) => {
          const datasetNode = nodes.get(nodeId);
          const bodyNode = network.body.nodes[nodeId];
          if (!datasetNode || datasetNode.hidden || !bodyNode) return null;
          const pos = bodyNode.getPosition();
          return { x: pos.x, y: pos.y, radius: getStructureNodeRadius(nodeId), nodeId };
        })
        .filter(Boolean);
    }

    function isStructurePositionSafe(candidate, occupied, candidateRadius, minDistance) {
      return occupied.every((item) => {
        const itemRadius = item.radius || 0;
        const limit = Math.max(minDistance, candidateRadius + itemRadius + 56);
        const dx = candidate.x - item.x;
        const dy = candidate.y - item.y;
        return Math.hypot(dx, dy) >= limit;
      });
    }

    function findStructureCollisionFreePosition(candidate, occupied, candidateRadius, minDistance, preferredAngle) {
      const baseAngle = Number.isFinite(preferredAngle) ? preferredAngle : 0;
      if (isStructurePositionSafe(candidate, occupied, candidateRadius, minDistance)) {
        return candidate;
      }

      const baseRadius = Math.max(minDistance, candidateRadius * 3.2, 132);
      for (let ring = 0; ring < 8; ring += 1) {
        const ringRadius = baseRadius + ring * Math.max(candidateRadius * 1.8, 72);
        const sampleCount = 10 + ring * 4;
        for (let index = 0; index < sampleCount; index += 1) {
          const angle = baseAngle + (Math.PI * 2 * index) / sampleCount;
          const probe = {
            x: candidate.x + Math.cos(angle) * ringRadius,
            y: candidate.y + Math.sin(angle) * ringRadius,
          };
          if (isStructurePositionSafe(probe, occupied, candidateRadius, minDistance)) {
            return probe;
          }
        }
      }

      let x = candidate.x;
      let y = candidate.y;
      for (let iteration = 0; iteration < 32; iteration += 1) {
        let adjusted = false;
        occupied.forEach((item, index) => {
          const itemRadius = item.radius || 0;
          const limit = Math.max(minDistance, candidateRadius + itemRadius + 56);
          let dx = x - item.x;
          let dy = y - item.y;
          let dist = Math.hypot(dx, dy);
          if (dist < limit) {
            adjusted = true;
            if (dist < 1e-6) {
              const angle = baseAngle + (index + iteration + 1) * 0.72;
              dx = Math.cos(angle);
              dy = Math.sin(angle);
              dist = 1;
            }
            const push = limit - dist + 12;
            x += (dx / dist) * push;
            y += (dy / dist) * push;
          }
        });
        if (!adjusted) break;
      }
      return { x, y };
    }

    function resetStructureLayout() {
      localStorage.removeItem(storageKey);
      restoreStructureView();
      setTimeout(() => {
        expandAllChildren();
        network.fit({ animation: false });
        setStatus('已恢复结构图默认视图');
      }, 80);
    }

    function expandAllChildren() {
      Object.keys(childMap).forEach((parentId) => {
        const payload = ensureChildPayload(parentId);
        if (payload && !payload.expanded) {
          toggleChildren(parentId, true);
        }
      });
    }

    function toggleChildren(parentId, silentMode) {
      const payload = ensureChildPayload(parentId);
      if (!payload) return;
      const willShow = !payload.expanded;
      if (willShow) {
        const parentPosition = network.getPositions([parentId])[parentId] || { x: 0, y: 0 };
        const occupied = collectVisibleStructureNodes(payload.nodes);
        payload.nodes.forEach((nodeId, index) => {
          const current = nodes.get(nodeId);
          if (!current) return;
          const rawAngle = Math.atan2((current.y || parentPosition.y) - parentPosition.y, (current.x || parentPosition.x) - parentPosition.x);
          const preferredAngle = Number.isFinite(rawAngle) ? rawAngle : index * 0.8;
          const candidate = {
            x: Number.isFinite(current.x) ? current.x : parentPosition.x + Math.cos(preferredAngle) * 120,
            y: Number.isFinite(current.y) ? current.y : parentPosition.y + Math.sin(preferredAngle) * 120,
          };
          const nodeRadius = getStructureNodeRadius(nodeId);
          const collisionFree = findStructureCollisionFreePosition(
            candidate,
            occupied,
            nodeRadius,
            Math.max(124, nodeRadius * 3.6),
            preferredAngle
          );
          addRevealOwner(nodeRevealOwners, nodeId, parentId);
          nodes.update({ id: nodeId, hidden: false, x: collisionFree.x, y: collisionFree.y });
          occupied.push({ x: collisionFree.x, y: collisionFree.y, radius: getStructureNodeRadius(nodeId), nodeId });
        });
      } else {
        payload.nodes.forEach((nodeId) => {
          const current = nodes.get(nodeId);
          removeRevealOwner(nodeRevealOwners, nodeId, parentId);
          if (current && !persistentVisibleNodeIds.has(nodeId) && !hasRevealOwner(nodeRevealOwners, nodeId)) {
            nodes.update({ id: nodeId, hidden: true });
          }
        });
      }
      payload.edges.forEach((edgeId) => {
        const current = edges.get(edgeId);
        if (!current) return;
        if (willShow) {
          addRevealOwner(edgeRevealOwners, edgeId, parentId);
          edges.update({ id: edgeId, hidden: false });
        } else {
          removeRevealOwner(edgeRevealOwners, edgeId, parentId);
          if (!persistentVisibleEdgeIds.has(edgeId) && !hasRevealOwner(edgeRevealOwners, edgeId)) {
            edges.update({ id: edgeId, hidden: true });
          }
        }
      });
      payload.expanded = willShow;
      if (!silentMode) {
        setStatus(willShow ? '已展开该节点的下一层结构' : '已收起该节点的下一层结构');
      }
    }

    document.getElementById('${dom_id}_save').addEventListener('click', saveCurrentLayout);
    document.getElementById('${dom_id}_reset').addEventListener('click', resetStructureLayout);

    setTimeout(() => {
      expandAllChildren();
      const restored = applySavedLayout();
      if (!restored) network.fit({ animation: false });
    }, 120);

    network.on('click', function(params) {
      if (params.nodes.length > 0 && ensureChildPayload(params.nodes[0])) {
        toggleChildren(params.nodes[0]);
      }
      if (params.nodes.length > 0 && nodeInfo[params.nodes[0]]) {
        renderMap('节点详情', nodeInfo[params.nodes[0]]);
      } else if (params.edges.length > 0 && edgeInfo[params.edges[0]]) {
        renderMap('边详情', edgeInfo[params.edges[0]]);
      } else {
        detail.innerHTML = defaultDetail;
      }
    });
  </script>
</body>
</html>
        """
    )
    default_detail = (
        "<div class='panel-title'>功能介绍</div>"
        "<div class='panel-text'>该视图用于展示中心节点周围的局部知识结构，可直接查看三类关系分布、节点信息与教材证据。</div>"
    )
    return template.substitute(
        dom_id=dom_id,
        detail_id=detail_id,
        title=title,
        height=int(height),
        detail_panel_width=int(detail_panel_width),
        node_rows=json.dumps(node_rows, ensure_ascii=False),
        edge_rows=json.dumps(edge_rows, ensure_ascii=False),
        node_info=json.dumps(node_info, ensure_ascii=False),
        edge_info=json.dumps(edge_info, ensure_ascii=False),
        children_map=json.dumps(structure_bundle["children_map"], ensure_ascii=False),
        full_node_meta=json.dumps(full_node_meta, ensure_ascii=False),
        full_edge_rows=json.dumps(full_edge_rows, ensure_ascii=False),
        structure_center_id=json.dumps(structure_bundle["center_id"], ensure_ascii=False),
        relation_colors=json.dumps(RELATION_COLORS, ensure_ascii=False),
        issue_colors=json.dumps(EDGE_ISSUE_COLORS, ensure_ascii=False),
        type_colors=json.dumps(TYPE_COLORS, ensure_ascii=False),
        default_detail=default_detail,
        default_detail_json=json.dumps(default_detail, ensure_ascii=False),
        storage_key=json.dumps(storage_key, ensure_ascii=False),
    )


def relation_summary_table(edge_view):
    """生成三类关系数量统计表。"""
    counts = edge_view["relation"].value_counts().to_dict() if not edge_view.empty else {}
    return pd.DataFrame(
        [
            {"关系": REL_IS_A, "数量": int(counts.get(REL_IS_A, 0))},
            {"关系": REL_CONTAINS, "数量": int(counts.get(REL_CONTAINS, 0))},
            {"关系": REL_USED_FOR, "数量": int(counts.get(REL_USED_FOR, 0))},
        ]
    )


def core_top5_table(node_view):
    """生成核心节点 Top 5 表，用于首页概览。"""
    if node_view.empty:
        return pd.DataFrame(columns=["节点", "类型", "PageRank", "度中心性", "度数", "组件大小"])
    return (
        node_view[["name", "entity_type", "pagerank", "degree_centrality", "degree", "component_size"]]
        .rename(
            columns={
                "name": "节点",
                "entity_type": "类型",
                "pagerank": "PageRank",
                "degree_centrality": "度中心性",
                "degree": "度数",
                "component_size": "组件大小",
            }
        )
        .sort_values(["PageRank", "度中心性", "度数", "节点"], ascending=[False, False, False, True])
        .head(5)
        .assign(PageRank=lambda df: df["PageRank"].round(6), 度中心性=lambda df: df["度中心性"].round(6))
        .reset_index(drop=True)
    )


def build_component_preview(view_bundle, selected_record):
    """把某个被隐藏组件还原成单独预览视图。"""
    comp_ids = set(selected_record["node_ids"])
    preview_edges = view_bundle["candidate_edges"][
        view_bundle["candidate_edges"]["source"].isin(comp_ids) & view_bundle["candidate_edges"]["target"].isin(comp_ids)
    ].copy()
    preview_nodes = sync_nodes_to_edges(view_bundle["candidate_nodes"], preview_edges)
    preview_nodes, preview_graph = enrich_node_metrics(preview_nodes, preview_edges)
    return {"graph": preview_graph, "nodes": preview_nodes, "edges": preview_edges, "record": selected_record}


def render_hidden_component_panel(view_name, view_bundle):
    """在侧边栏展示被隐藏组件的数量、原因和预览入口。"""
    preview_bundle = None
    hidden_components = view_bundle["hidden_components"]
    with st.sidebar.expander(f"{view_name}隐藏组件面板", expanded=False):
        summary_row = {
            "被隐藏的 2 节点组件": view_bundle["removed"]["two_node_components_removed"],
            "被隐藏的小连通分量": view_bundle["removed"]["small_components_removed"],
            "被隐藏的 1 节点组件": view_bundle["removed"]["isolated_removed"],
            "非最大连通子图": view_bundle["removed"]["non_largest_removed"],
            "隐藏组件总数": len(hidden_components),
        }
        st.dataframe(pd.DataFrame([summary_row]), use_container_width=True, hide_index=True)
        if not hidden_components:
            st.caption("当前没有被隐藏的小组件。")
            return None

        hidden_df = pd.DataFrame(hidden_components)[
            ["component_id", "reason", "size", "edge_count", "relation_summary", "nodes_preview"]
        ].rename(
            columns={
                "component_id": "组件 ID",
                "reason": "隐藏原因",
                "size": "节点数",
                "edge_count": "边数",
                "relation_summary": "关系组成",
                "nodes_preview": "节点预览",
            }
        )
        st.dataframe(hidden_df, use_container_width=True, hide_index=True, height=min(320, 42 * len(hidden_df) + 80))
        labels = [
            f"{record['component_id']} | {record['reason']} | {record['relation_summary']} | {record['nodes_preview']}"
            for record in hidden_components
        ]
        selected_label = st.selectbox(
            "选择一个隐藏组件单独查看",
            options=labels,
            key=f"{view_name}_hidden_component_selector",
        )
        selected_record = hidden_components[labels.index(selected_label)]
        st.caption(f"节点：{'、'.join(selected_record['node_names'])}")
        st.caption(f"关系：{selected_record['relation_summary']}")
        if selected_record.get("edge_preview"):
            st.caption(f"边预览：{selected_record['edge_preview']}")
        if st.toggle("单独弹出该隐藏组件", value=False, key=f"{view_name}_hidden_component_toggle"):
            preview_bundle = build_component_preview(view_bundle, selected_record)
    return preview_bundle

def render_component_preview(preview_bundle, title, *, physics_enabled, high_centrality_labels, node_size_scale, edge_width_scale, spacing_scale, initial_scale):
    """渲染被隐藏小组件的单独预览图。"""
    if not preview_bundle:
        return
    heights = get_view_heights()
    canvas_height = adaptive_canvas_height(
        len(preview_bundle["nodes"]),
        heights["preview"],
        threshold=8,
        growth_per_node=10,
        max_extra=220,
        minimum=340,
        maximum=880,
    )
    record = preview_bundle["record"]
    st.markdown(f"**{title}**")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "组件 ID": record["component_id"],
                    "隐藏原因": record["reason"],
                    "节点数": record["size"],
                    "边数": record["edge_count"],
                    "关系组成": record["relation_summary"],
                }
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
    components.html(
        make_clickable_network_html(
            preview_bundle["graph"],
            preview_bundle["nodes"],
            preview_bundle["edges"],
            f"{title}：{record['component_id']}",
            height=canvas_height,
            physics_enabled=physics_enabled,
            high_centrality_labels=high_centrality_labels,
            node_size_scale=node_size_scale,
            edge_width_scale=edge_width_scale,
            spacing_scale=spacing_scale,
            initial_scale=initial_scale,
        ),
        height=adaptive_wrapper_height(canvas_height, 70),
        scrolling=False,
    )


def render_structure_mode(view_bundle, *, title, key_prefix, preferred_names=None):
    """渲染中心节点驱动的局部知识结构图页面。"""
    heights = get_view_heights()
    node_view = view_bundle["nodes"]
    edge_view = view_bundle["edges"]
    if node_view.empty or edge_view.empty:
        st.warning("当前视图没有可生成的局部知识结构图。")
        return

    name_map = node_view.set_index("node_id")["name"].to_dict()
    type_map = node_view.set_index("node_id")["entity_type"].to_dict()
    valid_center_ids = get_valid_structure_center_ids(edge_view)
    option_rows = (
        node_view[node_view["node_id"].isin(valid_center_ids)]
        .sort_values(["pagerank", "degree_centrality", "degree", "name"], ascending=[False, False, False, True])
        .copy()
    )
    if option_rows.empty:
        st.warning("当前视图中没有满足条件的中心节点。结构图中心节点只允许选择纯上位节点：是一种取上位类，包含/用于取源节点；若某节点同时作为任一关系的下位节点出现，则不纳入中心节点候选。")
        return
    option_ids = option_rows["node_id"].tolist()
    center_key = f"{key_prefix}_structure_center_id"
    default_center = pick_structure_default_center(option_rows, preferred_names=preferred_names)
    if st.session_state.get(center_key) not in option_ids:
        st.session_state[center_key] = default_center

    selected_center = st.selectbox(
        "选择中心节点",
        options=option_ids,
        key=center_key,
        format_func=lambda node_id: f"{name_map.get(node_id, node_id)} [{type_map.get(node_id, UNKNOWN_TYPE)}]",
    )
    structure_bundle = build_structure_bundle(node_view, edge_view, selected_center)
    if not structure_bundle:
        st.warning("当前节点暂无可展示结构。")
        return

    stats_row = {
        "中心节点": structure_bundle["center_name"],
        "一跳邻居数": structure_bundle["stats"]["一跳邻居数"],
        "二跳候选数": structure_bundle["stats"]["二跳候选数"],
        REL_IS_A: structure_bundle["stats"][REL_IS_A],
        REL_CONTAINS: structure_bundle["stats"][REL_CONTAINS],
        REL_USED_FOR: structure_bundle["stats"][REL_USED_FOR],
    }
    st.caption("功能介绍：围绕中心节点展示局部知识结构，适合用于主题讲解、关系展示与证据回看。")
    st.dataframe(pd.DataFrame([stats_row]), use_container_width=True, hide_index=True)
    canvas_height = adaptive_canvas_height(
        len(structure_bundle["nodes"]),
        heights["base"],
        threshold=8,
        growth_per_node=12,
        max_extra=360,
        minimum=660,
        maximum=1240,
    )
    components.html(
        make_structure_network_html(
            structure_bundle,
            node_view,
            edge_view,
            title=f"{title}：{structure_bundle['center_name']}",
            height=canvas_height,
        ),
        height=adaptive_wrapper_height(canvas_height),
        scrolling=False,
    )


def render_presentation_order():
    """渲染答辩讲解顺序的快捷导航按钮。"""
    steps = [
        ("第一步：总体统计", ("page", "总览")),
        ("第二步：主题图 1", ("theme", FEATURED_THEME_SEQUENCE[0])),
        ("第三步：主题图 2", ("theme", FEATURED_THEME_SEQUENCE[1])),
        ("第四步：主题图 3", ("theme", FEATURED_THEME_SEQUENCE[2])),
    ]
    st.markdown("**答辩讲解顺序**")
    cols = st.columns(len(steps))
    for idx, ((label, target), col) in enumerate(zip(steps, cols), start=1):
        if col.button(label, key=f"kg_v5_step_{idx}", use_container_width=True):
            if target[0] == "theme":
                jump_to_theme(target[1])
            else:
                jump_to_page(target[1])


def render_overview(full_view, display_view, relation_table, core_table, theme_rows, source_name, source_file_count, theme_views, physics_enabled, high_label_only, node_size_scale, edge_width_scale, spacing_scale, initial_scale):
    """渲染首页概览，只展示统计和核心节点信息。"""
    overview_cols = st.columns(6)
    relation_counts = {row["关系"]: row["数量"] for row in relation_table.to_dict("records")}
    overview_cols[0].metric("节点数", full_view["stats"]["visible_nodes"])
    overview_cols[1].metric("边数", full_view["stats"]["visible_edges"])
    overview_cols[2].metric("是一种", relation_counts.get(REL_IS_A, 0))
    overview_cols[3].metric("包含", relation_counts.get(REL_CONTAINS, 0))
    overview_cols[4].metric("用于", relation_counts.get(REL_USED_FOR, 0))
    overview_cols[5].metric("来源教材文件数", int(source_file_count))
    st.info(f"功能介绍：展示教材关系抽取结果的统计概览、主题关系结构与证据回溯。当前数据源：{source_name}。")

    cols = st.columns([1, 1])
    with cols[0]:
        st.subheader("图谱总体统计")
        st.dataframe(metrics_table(full_view["stats"]), use_container_width=True, hide_index=True)
        st.subheader("三类关系数量")
        st.dataframe(relation_table, use_container_width=True, hide_index=True)
    with cols[1]:
        st.subheader("核心节点 Top 5")
        st.dataframe(core_table, use_container_width=True, hide_index=True)


def render_display_page(display_view, physics_enabled, high_label_only, node_size_scale, edge_width_scale, spacing_scale, initial_scale):
    """渲染主展示页，支持自由图和结构图两种方式。"""
    heights = get_view_heights()
    if st.button("返回首页", key="kg_v5_back_home_from_display"):
        jump_to_page("总览")
    preview_bundle = render_hidden_component_panel("展示视图", display_view)
    st.subheader("答辩展示视图")
    st.dataframe(metrics_table(display_view["stats"]), use_container_width=True, hide_index=True)
    canvas_height = adaptive_canvas_height(
        display_view["stats"]["visible_nodes"],
        heights["base"],
        threshold=18,
        growth_per_node=6,
        max_extra=360,
        minimum=660,
        maximum=1220,
    )
    mode_key = "kg_v5_display_graph_mode"
    if mode_key not in st.session_state:
        st.session_state[mode_key] = "自由图模式"
    graph_mode = st.radio("展示方式", options=["结构图模式", "自由图模式"], horizontal=True, key=mode_key)
    if graph_mode == "结构图模式":
        render_structure_mode(display_view, title="节点局部知识结构图", key_prefix="kg_v5_display")
    else:
        components.html(
            make_clickable_network_html(
                display_view["graph"],
                display_view["nodes"],
                display_view["edges"],
                "答辩展示视图",
                height=canvas_height,
                physics_enabled=physics_enabled,
                high_centrality_labels=high_label_only,
                layout_relation=choose_layout_relation(display_view["edges"]),
                node_size_scale=node_size_scale,
                edge_width_scale=edge_width_scale,
                spacing_scale=spacing_scale,
                initial_scale=initial_scale,
                neo4j_style=True,
            ),
            height=adaptive_wrapper_height(canvas_height),
            scrolling=False,
        )
    if preview_bundle:
        render_component_preview(
            preview_bundle,
            "主图中被隐藏的小组件",
            physics_enabled=physics_enabled,
            high_centrality_labels=high_label_only,
            node_size_scale=node_size_scale,
            edge_width_scale=edge_width_scale,
            spacing_scale=spacing_scale,
            initial_scale=initial_scale,
        )
    render_embedded_evidence_panel(display_view, "kg_v5_display", title="当前展示视图证据")


def render_theme_page(selected_theme, theme_views, theme_hops, physics_enabled, high_label_only, node_size_scale, edge_width_scale, spacing_scale, initial_scale):
    """渲染单个主题页，展示主题子图、结构图和证据。"""
    heights = get_view_heights()
    if st.button("返回主概览", key=f"kg_v5_back_home_{selected_theme}"):
        jump_to_page("总览")
    nav_cols = st.columns(len(THEME_PRESETS))
    for idx, preset in enumerate(THEME_PRESETS):
        theme_view = theme_views[preset["name"]]["view"]
        label = f"{preset['name']} ({theme_view['stats']['visible_nodes']} 节点)"
        if nav_cols[idx].button(label, key=f"kg_v5_theme_nav_{idx}", use_container_width=True):
            jump_to_theme(preset["name"])

    selected_bundle = theme_views[selected_theme]
    selected_view = selected_bundle["view"]
    theme_preview = render_hidden_component_panel(selected_theme, selected_view)
    st.subheader(selected_theme)
    st.caption(f"功能介绍：围绕“{selected_theme}”展示主题关系结构、自由图谱视图与教材证据。")
    seed_text = "、".join(selected_bundle["matched_seeds"][:8]) if selected_bundle["matched_seeds"] else "无"
    summary_row = {
        "主题": selected_theme,
        "匹配种子": seed_text,
        "扩展跳数": theme_hops,
        "布局风格": build_layout_options(selected_bundle["layout_relation"], False, spacing_scale)["label"],
    }
    summary_row.update(metrics_table(selected_view["stats"]).iloc[0].to_dict())
    st.dataframe(pd.DataFrame([summary_row]), use_container_width=True, hide_index=True)
    canvas_height = adaptive_canvas_height(
        selected_view["stats"]["visible_nodes"],
        heights["base"],
        threshold=14,
        growth_per_node=7,
        max_extra=360,
        minimum=660,
        maximum=1220,
    )
    mode_key = f"kg_v5_theme_graph_mode_{selected_theme}"
    if mode_key not in st.session_state:
        st.session_state[mode_key] = "自由图模式"
    graph_mode = st.radio("展示方式", options=["结构图模式", "自由图模式"], horizontal=True, key=mode_key)
    if graph_mode == "结构图模式":
        render_structure_mode(
            selected_view,
            title=f"主题结构图：{selected_theme}",
            key_prefix=f"kg_v5_theme_{selected_theme}",
            preferred_names=selected_bundle["matched_seeds"],
        )
    else:
        components.html(
            make_clickable_network_html(
                selected_view["graph"],
                selected_view["nodes"],
                selected_view["edges"],
                f"主题图谱：{selected_theme}",
                height=canvas_height,
                physics_enabled=physics_enabled,
                high_centrality_labels=high_label_only,
                layout_relation=selected_bundle["preset"].get("preferred_relation"),
                cluster_hub_ids=[],
                enable_hub_clusters=False,
                node_size_scale=node_size_scale,
                edge_width_scale=edge_width_scale,
                spacing_scale=spacing_scale,
                initial_scale=initial_scale,
                neo4j_style=True,
            ),
            height=adaptive_wrapper_height(canvas_height),
            scrolling=False,
        )
    if theme_preview:
        render_component_preview(
            theme_preview,
            "当前主题页被隐藏的小组件",
            physics_enabled=physics_enabled,
            high_centrality_labels=high_label_only,
            node_size_scale=node_size_scale,
            edge_width_scale=edge_width_scale,
            spacing_scale=spacing_scale,
            initial_scale=initial_scale,
        )
    render_embedded_evidence_panel(selected_view, f"kg_v5_theme_{selected_theme}", title=f"{selected_theme}相关证据")

def render_qc_page(qc_summary, qc_duplicates, qc_short_cycles, qc_conflicts, qc_nodes, qc_graph, physics_enabled, node_size_scale, edge_width_scale, spacing_scale, initial_scale):
    """渲染结构质检页，集中展示问题边和问题图。"""
    heights = get_view_heights()
    if st.button("返回首页", key="kg_v5_back_home_from_qc"):
        jump_to_page("总览")
    st.subheader("结构质检")
    st.dataframe(qc_summary, use_container_width=True, hide_index=True)
    if qc_graph.number_of_edges() > 0:
        issue_edges = apply_qc_flags(pd.concat([qc_duplicates, qc_short_cycles, qc_conflicts], ignore_index=True))
        issue_nodes = sync_nodes_to_edges(qc_nodes, issue_edges)
        issue_nodes, _ = enrich_node_metrics(issue_nodes, issue_edges)
        canvas_height = adaptive_canvas_height(
            len(issue_nodes),
            heights["compact"],
            threshold=8,
            growth_per_node=9,
            max_extra=240,
            minimum=480,
            maximum=940,
        )
        components.html(
            make_clickable_network_html(
                qc_graph,
                issue_nodes,
                issue_edges,
                "结构问题视图",
                height=canvas_height,
                physics_enabled=physics_enabled,
                high_centrality_labels=False,
                node_size_scale=node_size_scale,
                edge_width_scale=edge_width_scale,
                spacing_scale=spacing_scale,
                initial_scale=initial_scale,
            ),
            height=adaptive_wrapper_height(canvas_height),
            scrolling=False,
        )
    else:
        st.success("当前三元组整体质量较高，未发现需要在答辩中重点强调的结构性问题。")

    for label, frame in [
        ("重复边", qc_duplicates),
        ("是一种短环", qc_short_cycles),
        ("同一实体对多关系冲突", qc_conflicts),
    ]:
        st.markdown(f"**{label}**")
        if frame.empty:
            st.caption("无")
        else:
            st.dataframe(
                frame[["source_name", "relation", "target_name", "source_file", "block_id", "evidence_text"]],
                use_container_width=True,
                hide_index=True,
            )


def render_embedded_evidence_panel(view_bundle, key_prefix, title="教材原文证据"):
    """在当前视图下方内嵌显示教材原文证据。"""
    evidence_edges = view_bundle["edges"].copy()
    evidence_edges = evidence_edges[
        evidence_edges["evidence_text"].str.strip().ne("") | evidence_edges["context_snippet"].str.strip().ne("")
    ].copy()
    if evidence_edges.empty:
        return

    st.subheader(title)
    labels = [
        f"{row['source_name']} -[{row['relation']}]-> {row['target_name']} | {row.get('block_id', '')}" for _, row in evidence_edges.iterrows()
    ]
    selected_label = st.selectbox("选择一条边查看教材原文证据", options=labels, key=f"{key_prefix}_evidence_selector")
    selected_row = evidence_edges.iloc[labels.index(selected_label)]

    detail_row = {
        "head": selected_row["source_name"],
        "relation": selected_row["relation"],
        "tail": selected_row["target_name"],
        "source_file": selected_row.get("source_file", ""),
        "block_id": selected_row.get("block_id", ""),
        "evidence_text": selected_row.get("evidence_text", "") or "无",
        "上下文片段": selected_row.get("context_snippet", "") or "无",
        "是否重复边": "是" if bool(selected_row.get("is_duplicate", False)) else "否",
        "是否冲突边": "是" if bool(selected_row.get("is_conflict", False)) else "否",
        "是否短环": "是" if bool(selected_row.get("is_short_cycle", False)) else "否",
    }
    st.dataframe(pd.DataFrame([detail_row]), use_container_width=True, hide_index=True)


def main():
    """通用展示平台入口。"""
    st.set_page_config(page_title="知识图谱展示 V5", layout="wide")
    init_state()
    st.title("知识图谱展示 V5")
    st.caption("功能介绍：统计概览、主题图谱、结构质检与教材证据回溯。")

    with st.sidebar:
        source_name = st.radio("数据源", ["本地 CSV", "Neo4j"])
        physics_enabled = st.toggle("启用自动布局", value=True)
        node_size_scale = DEFAULT_NODE_SIZE_SCALE
        edge_width_scale = DEFAULT_EDGE_WIDTH_SCALE
        spacing_scale = DEFAULT_SPACING_SCALE
        initial_scale = DEFAULT_INITIAL_SCALE

    defense_mode = True
    st.session_state["kg_v5_defense_mode"] = True
    st.session_state["kg_v5_prev_defense_mode"] = True
    st.session_state["kg_v5_window_height"] = 720

    with st.sidebar:
        min_component_size = st.slider("只显示节点数 >= N 的连通子图", min_value=1, max_value=10, key="kg_v5_min_component_size")
        hide_issue_edges = st.toggle("隐藏结构异常边", key="kg_v5_hide_issue_edges")
        only_issue_graph = st.toggle("只看结构问题", key="kg_v5_only_issue_graph")
        high_label_only = st.toggle("只显示高中心性节点标签", key="kg_v5_high_label_only")
        focus_largest_component = st.toggle("展示视图聚焦最大连通子图", key="kg_v5_focus_largest_component")
        max_nodes = st.slider("每个视图最多节点数", min_value=20, max_value=180, step=10, key="kg_v5_max_nodes")
        focus_depth = st.select_slider("双击聚焦跳数", options=[1, 2], key="kg_v5_focus_depth")
        theme_hops = st.select_slider("主题扩展跳数", options=[1, 2], value=1 if defense_mode else 2)
        keyword = st.text_input("关键词过滤")

    try:
        nodes_df, edges_df, summary, report_text = choose_source(source_name)
    except Exception as exc:
        st.error(f"{source_name} 读取失败：{exc}")
        return

    nodes_df, edges_df = normalize_frames(nodes_df, edges_df)
    all_relations = [relation for relation in RELATION_OPTIONS if relation in set(edges_df["relation"].unique())]
    all_types = sorted(nodes_df["entity_type"].dropna().unique().tolist())
    with st.sidebar:
        relations = st.multiselect("关系类型", options=all_relations, default=all_relations)
        node_types = st.multiselect("节点类型", options=all_types, default=all_types)

    filtered_nodes, filtered_edges = filter_frames(nodes_df, edges_df, relations, node_types, keyword)
    filtered_edges = apply_qc_flags(filtered_edges)
    qc_summary, qc_duplicates, qc_short_cycles, qc_conflicts, qc_nodes, qc_graph = compute_qc(filtered_nodes, filtered_edges)

    full_view = build_display_view(
        filtered_nodes,
        filtered_edges,
        max_nodes=max(260, max_nodes),
        hide_two_node_components=False,
        min_component_size=1,
        focus_largest_component=False,
        deduplicate=True,
        hide_issue_edges=False,
        only_issue_edges=False,
    )
    display_view = build_display_view(
        filtered_nodes,
        filtered_edges,
        max_nodes=max_nodes,
        hide_two_node_components=False,
        min_component_size=min_component_size,
        focus_largest_component=focus_largest_component,
        deduplicate=True,
        hide_issue_edges=hide_issue_edges,
        only_issue_edges=only_issue_graph,
    )

    theme_views = {}
    theme_rows = []
    theme_source_graph = full_view["graph"]
    for preset in THEME_PRESETS:
        theme_nodes, theme_edges, matched_seeds = extract_theme(theme_source_graph, full_view["nodes"], full_view["edges"], preset, theme_hops)
        theme_view = build_display_view(
            theme_nodes,
            theme_edges,
            max_nodes=min(max_nodes, 70),
            hide_two_node_components=False,
            min_component_size=max(2, min_component_size) if defense_mode else min_component_size,
            focus_largest_component=False,
            deduplicate=False,
            hide_issue_edges=False,
            only_issue_edges=False,
        )
        layout_relation = choose_layout_relation(theme_view["edges"], preferred_relation=preset.get("preferred_relation"))
        theme_views[preset["name"]] = {
            "view": theme_view,
            "matched_seeds": matched_seeds,
            "layout_relation": layout_relation,
            "preset": preset,
        }
        theme_rows.append(
            {
                "主题": preset["name"],
                "匹配种子": "、".join(matched_seeds[:6]) if matched_seeds else "无",
                "节点数": theme_view["stats"]["visible_nodes"],
                "边数": theme_view["stats"]["visible_edges"],
                "连通分量": theme_view["stats"]["components"],
                "最大连通子图": theme_view["stats"]["largest_component"],
                "布局风格": build_layout_options(layout_relation, False, spacing_scale)["label"],
            }
        )

    relation_table = relation_summary_table(full_view["edges"])
    core_table = core_top5_table(full_view["nodes"])
    source_file_count = full_view["edges"]["source_file"].astype(str).str.strip().replace("", pd.NA).nunique(dropna=True)
    theme_page_map = {theme_page_name(preset["name"]): preset["name"] for preset in THEME_PRESETS}

    pages = ["总览", "展示视图", *theme_page_map.keys(), "结构质检"]
    page = st.radio("页面", options=pages, horizontal=True, key="kg_v5_page")

    if page == "总览":
        render_overview(
            full_view,
            display_view,
            relation_table,
            core_table,
            theme_rows,
            source_name,
            source_file_count,
            theme_views,
            physics_enabled,
            high_label_only,
            node_size_scale,
            edge_width_scale,
            spacing_scale,
            initial_scale,
        )
    elif page == "展示视图":
        render_display_page(display_view, physics_enabled, high_label_only, node_size_scale, edge_width_scale, spacing_scale, initial_scale)
    elif page in theme_page_map:
        render_theme_page(theme_page_map[page], theme_views, theme_hops, physics_enabled, high_label_only, node_size_scale, edge_width_scale, spacing_scale, initial_scale)
    elif page == "结构质检":
        render_qc_page(qc_summary, qc_duplicates, qc_short_cycles, qc_conflicts, qc_nodes, qc_graph, physics_enabled, node_size_scale, edge_width_scale, spacing_scale, initial_scale)

if __name__ == "__main__":
    main()
