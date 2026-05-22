"""Build course-oriented topic clusters from the existing knowledge graph.

The clustering here is intentionally lightweight. It uses course keywords as
seeds, expands by graph neighborhood, then ranks nodes with simple centrality.
This keeps the method explainable for a teaching demo and avoids changing the
relation-extraction research line.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

try:
    import networkx as nx
    import pandas as pd
except ModuleNotFoundError as exc:
    missing_name = exc.name or "required package"
    raise SystemExit(
        f"[ERROR] Missing dependency: {missing_name}\n"
        "Please install project requirements first:\n"
        "  python -m pip install -r requirements.txt\n"
        "Or double-click 01_install_dependencies.bat."
    ) from exc


ROOT_DIR = Path(__file__).resolve().parents[1]
KG_DIR = ROOT_DIR / "outputs" / "kg"
NODES_PATH = KG_DIR / "nodes.csv"
EDGES_PATH = KG_DIR / "edges.csv"
TOPIC_JSON_PATH = KG_DIR / "topic_clusters.json"
TOPIC_MD_PATH = KG_DIR / "topic_cluster_summary.md"

REL_IS_A = "是一种"
REL_CONTAINS = "包含"
REL_USED_FOR = "用于"
RELATION_OPTIONS = [REL_IS_A, REL_CONTAINS, REL_USED_FOR]


TOPIC_DEFINITIONS = [
    {
        "topic_id": "topic_01",
        "topic_name": "第一章 智能网联汽车基础与体系结构",
        "topic_description": "围绕智能网联汽车基本概念、自动驾驶系统、平台架构、控制平台和课程基础知识组织。",
        "seed_keywords": [
            "绪论",
            "智能网联",
            "智能网联汽车",
            "自动驾驶",
            "平台",
            "系统",
            "模块",
            "车辆",
            "控制",
            "云控",
            "架构",
            "通信模块",
        ],
    },
    {
        "topic_id": "topic_02",
        "topic_name": "第二章 车载传感器与环境感知",
        "topic_description": "围绕车载传感器、摄像头、激光雷达、毫米波雷达和环境感知任务组织。",
        "seed_keywords": [
            "车载传感器",
            "传感器",
            "感知",
            "摄像头",
            "激光雷达",
            "毫米波雷达",
            "雷达",
            "目标检测",
            "障碍物检测",
            "物体识别",
        ],
    },
    {
        "topic_id": "topic_03",
        "topic_name": "第三章 人工智能与深度学习方法",
        "topic_description": "围绕人工智能、神经网络、深度学习、卷积网络、循环网络和典型算法方法组织。",
        "seed_keywords": [
            "人工智能",
            "神经网络",
            "深度学习",
            "机器学习",
            "卷积",
            "CNN",
            "RNN",
            "LSTM",
            "算法",
            "分类",
        ],
    },
    {
        "topic_id": "topic_04",
        "topic_name": "第四章 强化学习与车路协同应用",
        "topic_description": "围绕强化学习、车路协同、V2X、定位融合、路径规划和应用场景组织。",
        "seed_keywords": [
            "强化学习",
            "车路协同",
            "V2X",
            "通信",
            "协同",
            "定位",
            "融合",
            "路径规划",
            "决策",
            "SLAM",
            "卡尔曼",
        ],
    },
]


def normalize_text(value: object) -> str:
    """Normalize values for keyword matching."""
    return str(value or "").strip().lower()


def load_graph_frames(nodes_path: Path, edges_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load nodes and edges with stable helper columns."""
    nodes = pd.read_csv(nodes_path, encoding="utf-8-sig")
    edges = pd.read_csv(edges_path, encoding="utf-8-sig")
    for column in ["node_id", "name", "entity_type", "raw_name"]:
        if column not in nodes.columns:
            nodes[column] = ""
        nodes[column] = nodes[column].fillna("").astype(str)
    for column in ["source", "target", "relation", "source_name", "target_name", "source_file", "evidence_text", "block_id"]:
        if column not in edges.columns:
            edges[column] = ""
        edges[column] = edges[column].fillna("").astype(str)
    edges = edges[edges["relation"].isin(RELATION_OPTIONS)].copy().reset_index(drop=True)
    edges["edge_id"] = [f"edge_{idx:05d}" for idx in range(len(edges))]
    return nodes, edges


def build_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.Graph:
    """Build an undirected graph for neighborhood and centrality analysis."""
    graph = nx.Graph()
    for _, row in nodes.iterrows():
        graph.add_node(row["node_id"], name=row["name"])
    for _, row in edges.iterrows():
        graph.add_edge(row["source"], row["target"], relation=row["relation"], edge_id=row["edge_id"])
    return graph


def build_node_text(nodes: pd.DataFrame, edges: pd.DataFrame) -> dict[str, str]:
    """Collect searchable text for each node from names, types, and evidence."""
    text_map: dict[str, list[str]] = defaultdict(list)
    for _, row in nodes.iterrows():
        text_map[row["node_id"]].extend([row["name"], row.get("raw_name", ""), row.get("entity_type", "")])
    for _, row in edges.iterrows():
        evidence = row.get("evidence_text", "")
        text_map[row["source"]].extend([row["source_name"], row["relation"], row["target_name"], evidence])
        text_map[row["target"]].extend([row["target_name"], row["relation"], row["source_name"], evidence])
    return {node_id: normalize_text(" ".join(parts)) for node_id, parts in text_map.items()}


def score_nodes_for_topic(topic: dict, node_text: dict[str, str], graph: nx.Graph) -> dict[str, float]:
    """Score each node against one topic by keyword hits and graph degree."""
    scores = {}
    for node_id, text in node_text.items():
        score = 0.0
        for keyword in topic["seed_keywords"]:
            key = normalize_text(keyword)
            if key and key in text:
                score += 3.0 + min(len(key), 8) / 10
        if score > 0:
            score += min(graph.degree(node_id), 10) * 0.12
            scores[node_id] = round(score, 4)
    return scores


def expand_topic_nodes(topic_scores: dict[str, float], graph: nx.Graph, max_nodes: int = 55, min_nodes: int = 12) -> set[str]:
    """Expand keyword seed nodes by one-hop neighbors and keep a readable size."""
    if not topic_scores:
        return set()

    seed_nodes = set(topic_scores)
    candidate_nodes = set(seed_nodes)
    for node_id in seed_nodes:
        candidate_nodes.update(graph.neighbors(node_id))

    if len(candidate_nodes) < min_nodes:
        for node_id in list(candidate_nodes):
            candidate_nodes.update(graph.neighbors(node_id))

    centrality = nx.degree_centrality(graph) if graph.number_of_nodes() > 1 else {}

    def rank_key(node_id: str) -> tuple[float, float, int, str]:
        return (
            topic_scores.get(node_id, 0.0),
            centrality.get(node_id, 0.0),
            graph.degree(node_id),
            node_id,
        )

    ranked = sorted(candidate_nodes, key=rank_key, reverse=True)
    return set(ranked[:max_nodes])


def assign_primary_topics(topic_node_sets: dict[str, set[str]], topic_scores: dict[str, dict[str, float]], graph: nx.Graph) -> dict[str, str]:
    """Assign each clustered node to the most likely primary topic."""
    assignments = {}
    all_nodes = set().union(*topic_node_sets.values()) if topic_node_sets else set()
    for node_id in all_nodes:
        ranked_topics = []
        for topic_id, node_set in topic_node_sets.items():
            if node_id not in node_set:
                continue
            score = topic_scores[topic_id].get(node_id, 0.0)
            neighbor_bonus = sum(1 for nbr in graph.neighbors(node_id) if nbr in node_set) * 0.08
            ranked_topics.append((score + neighbor_bonus, topic_id))
        if ranked_topics:
            assignments[node_id] = sorted(ranked_topics, reverse=True)[0][1]
    return assignments


def edge_records(edges: pd.DataFrame, edge_ids: set[str]) -> list[dict]:
    """Return compact edge records for JSON summaries."""
    selected = edges[edges["edge_id"].isin(edge_ids)].copy()
    records = []
    for _, row in selected.iterrows():
        records.append(
            {
                "edge_id": row["edge_id"],
                "source": row["source"],
                "target": row["target"],
                "source_name": row["source_name"],
                "relation": row["relation"],
                "target_name": row["target_name"],
                "source_file": row.get("source_file", ""),
                "block_id": row.get("block_id", ""),
                "evidence_text": row.get("evidence_text", ""),
            }
        )
    return records


def build_topic_clusters(nodes: pd.DataFrame, edges: pd.DataFrame) -> dict:
    """Build all topic clusters and a node-to-topic map."""
    graph = build_graph(nodes, edges)
    node_text = build_node_text(nodes, edges)
    topic_scores = {
        topic["topic_id"]: score_nodes_for_topic(topic, node_text, graph)
        for topic in TOPIC_DEFINITIONS
    }
    topic_node_sets = {
        topic["topic_id"]: expand_topic_nodes(topic_scores[topic["topic_id"]], graph)
        for topic in TOPIC_DEFINITIONS
    }
    primary_topic_by_node = assign_primary_topics(topic_node_sets, topic_scores, graph)
    centrality = nx.degree_centrality(graph) if graph.number_of_nodes() > 1 else {}
    node_name = nodes.set_index("node_id")["name"].to_dict()

    topics = []
    for topic in TOPIC_DEFINITIONS:
        topic_id = topic["topic_id"]
        node_ids = topic_node_sets[topic_id]
        internal_edges = edges[edges["source"].isin(node_ids) & edges["target"].isin(node_ids)].copy()
        internal_edge_ids = set(internal_edges["edge_id"])
        boundary_edges = edges[
            (
                edges["source"].isin(node_ids)
                & ~edges["target"].isin(node_ids)
                & edges["target"].isin(primary_topic_by_node)
            )
            | (
                edges["target"].isin(node_ids)
                & ~edges["source"].isin(node_ids)
                & edges["source"].isin(primary_topic_by_node)
            )
        ].copy()
        related_files = sorted({value for value in internal_edges["source_file"].astype(str) if value.strip()})
        core_nodes = sorted(
            node_ids,
            key=lambda node_id: (
                topic_scores[topic_id].get(node_id, 0.0),
                centrality.get(node_id, 0.0),
                graph.degree(node_id),
                node_name.get(node_id, ""),
            ),
            reverse=True,
        )[:8]

        topics.append(
            {
                **topic,
                "core_nodes": [
                    {
                        "node_id": node_id,
                        "name": node_name.get(node_id, node_id),
                        "topic_score": topic_scores[topic_id].get(node_id, 0.0),
                        "degree": int(graph.degree(node_id)),
                    }
                    for node_id in core_nodes
                ],
                "node_count": int(len(node_ids)),
                "edge_count": int(len(internal_edges)),
                "node_ids": sorted(node_ids),
                "edge_ids": sorted(internal_edge_ids),
                "related_source_files": related_files,
                "cross_topic_edges": edge_records(edges, set(boundary_edges["edge_id"].head(20))),
            }
        )

    return {
        "metadata": {
            "topic_count": len(topics),
            "node_count": int(len(nodes)),
            "edge_count": int(len(edges)),
            "method": "keyword_seed_plus_graph_neighborhood",
            "relations": RELATION_OPTIONS,
        },
        "node_primary_topic": primary_topic_by_node,
        "topics": topics,
    }


def write_summary_markdown(cluster_data: dict, path: Path) -> None:
    """Write a human-readable summary for thesis and defense use."""
    lines = [
        "# 主题集群组织说明",
        "",
        "本文件基于已有知识图谱结果生成，用于展示层主题化组织，不改变关系抽取结果。",
        "",
        f"- 主题数量：{cluster_data['metadata']['topic_count']}",
        f"- 图谱节点数：{cluster_data['metadata']['node_count']}",
        f"- 图谱边数：{cluster_data['metadata']['edge_count']}",
        f"- 方法：{cluster_data['metadata']['method']}",
        "",
    ]
    for topic in cluster_data["topics"]:
        lines.extend(
            [
                f"## {topic['topic_name']}",
                "",
                topic["topic_description"],
                "",
                f"- topic_id：`{topic['topic_id']}`",
                f"- 节点数：{topic['node_count']}",
                f"- 边数：{topic['edge_count']}",
                f"- 种子关键词：{'、'.join(topic['seed_keywords'])}",
                f"- 核心节点：{'、'.join(item['name'] for item in topic['core_nodes'][:5])}",
                f"- 相关来源文件数：{len(topic['related_source_files'])}",
                f"- 跨主题边示例数：{len(topic['cross_topic_edges'])}",
                "",
            ]
        )
    lines.extend(
        [
            "## 说明",
            "",
            "- 主题集群服务于答辩展示和教学应用原型。",
            "- 主题之间允许存在少量跨主题边，用于说明课程知识点之间的联系。",
            "- 小组件和边缘关系适合放在补充关系列表中，不作为主题主画布的视觉中心。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """Command-line entry."""
    parser = argparse.ArgumentParser(description="Build teaching topic clusters from KG nodes and edges.")
    parser.add_argument("--nodes", default=str(NODES_PATH), help="Path to nodes.csv")
    parser.add_argument("--edges", default=str(EDGES_PATH), help="Path to edges.csv")
    parser.add_argument("--output-json", default=str(TOPIC_JSON_PATH), help="Output topic_clusters.json")
    parser.add_argument("--output-md", default=str(TOPIC_MD_PATH), help="Output topic_cluster_summary.md")
    args = parser.parse_args()

    nodes, edges = load_graph_frames(Path(args.nodes), Path(args.edges))
    cluster_data = build_topic_clusters(nodes, edges)

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(cluster_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_summary_markdown(cluster_data, output_md)

    print(f"topics={cluster_data['metadata']['topic_count']}")
    for topic in cluster_data["topics"]:
        print(f"{topic['topic_id']} nodes={topic['node_count']} edges={topic['edge_count']} name={topic['topic_name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
