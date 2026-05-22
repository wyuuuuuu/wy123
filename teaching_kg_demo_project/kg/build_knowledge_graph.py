"""从最终三元组构建轻量知识图谱，并导出展示/分析文件。

这个脚本只负责第二步的图谱组织层，不改动关系抽取主线。
输入可以是审核后的 JSON 或最终 CSV，输出包括节点表、边表、
统计摘要、结构分析报告和 Neo4j 导入说明。
"""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx


ALLOWED_RELATIONS = ("是一种", "包含", "用于")
INPUT_CANDIDATES = (
    "triples_all_reviewed.json",
    "triples_final.csv",
    "triples_all.json",
)
ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "outputs" / "kg"
DEFAULT_TRIPLE_SEARCH_DIRS = (
    ROOT_DIR / "outputs",
    ROOT_DIR / "data",
    ROOT_DIR,
)
DEFAULT_DICTIONARY_CANDIDATES = (
    ROOT_DIR / "data" / "dictionaries" / "dictionary.csv",
    ROOT_DIR / "data" / "final_versions" / "final_v1" / "dictionaries" / "dictionary.csv",
    ROOT_DIR / "data" / "v2" / "dictionaries" / "dictionary.csv",
    ROOT_DIR / "data" / "v1" / "dictionaries" / "dictionary.csv",
)


def normalize_text(value: Optional[str]) -> str:
    """统一清洗文本字段，减少全角符号和空白差异。"""
    if value is None:
        return ""
    text = str(value).strip()
    for old, new in (
        ("\u3000", " "),
        ("\xa0", " "),
        ("（", "("),
        ("）", ")"),
        ("，", ","),
    ):
        text = text.replace(old, new)
    return " ".join(text.split())


def stable_node_id(name: str) -> str:
    """基于实体名生成稳定节点 ID，便于重复构图时保持一致。"""
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:12]
    return f"node_{digest}"


def find_best_triple_file(explicit_path: Optional[str] = None) -> Path:
    """按优先级定位最终三元组文件。"""
    if explicit_path:
        path = Path(explicit_path)
        if not path.is_absolute():
            path = (ROOT_DIR / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"指定的三元组文件不存在: {path}")
        return path

    found: Dict[str, List[Path]] = {name: [] for name in INPUT_CANDIDATES}
    for search_dir in DEFAULT_TRIPLE_SEARCH_DIRS:
        if not search_dir.exists():
            continue
        for candidate_name in INPUT_CANDIDATES:
            for path in search_dir.rglob(candidate_name):
                if "backups" in path.parts:
                    continue
                found[candidate_name].append(path)

    for candidate_name in INPUT_CANDIDATES:
        candidates = sorted(found[candidate_name], key=lambda p: (len(p.parts), str(p)))
        if candidates:
            return candidates[0]
    raise FileNotFoundError("未找到 triples_all_reviewed.json / triples_final.csv / triples_all.json")


def find_dictionary_file(explicit_path: Optional[str] = None) -> Optional[Path]:
    """定位实体规范化词典；如果不存在则允许无词典运行。"""
    if explicit_path:
        path = Path(explicit_path)
        if not path.is_absolute():
            path = (ROOT_DIR / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"指定的词典文件不存在: {path}")
        return path
    for path in DEFAULT_DICTIONARY_CANDIDATES:
        if path.exists():
            return path
    return None


def load_dictionary(path: Optional[Path]) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, set]]:
    """读取词典，返回别名映射、类型映射和别名集合。"""
    alias_to_canonical: Dict[str, str] = {}
    canonical_to_type: Dict[str, str] = {}
    canonical_aliases: Dict[str, set] = defaultdict(set)
    if path is None:
        return alias_to_canonical, canonical_to_type, canonical_aliases

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            canonical = normalize_text(row.get("canonical_name"))
            alias = normalize_text(row.get("alias"))
            entity_type = normalize_text(row.get("entity_type"))
            if not canonical:
                continue
            alias_to_canonical[canonical] = canonical
            canonical_aliases[canonical].add(canonical)
            if alias:
                alias_to_canonical[alias] = canonical
                canonical_aliases[canonical].add(alias)
            if entity_type and canonical not in canonical_to_type:
                canonical_to_type[canonical] = entity_type
    return alias_to_canonical, canonical_to_type, canonical_aliases


def read_json_triples(path: Path) -> List[dict]:
    """读取 JSON 格式三元组。"""
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_triples(path: Path) -> List[dict]:
    """读取 CSV 格式三元组。"""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_triples(path: Path) -> List[dict]:
    """根据后缀自动选择三元组读取方式。"""
    if path.suffix.lower() == ".json":
        return read_json_triples(path)
    if path.suffix.lower() == ".csv":
        return read_csv_triples(path)
    raise ValueError(f"不支持的三元组文件格式: {path}")


def canonicalize_entity(
    name: str,
    alias_to_canonical: Dict[str, str],
    canonical_aliases: Dict[str, set],
) -> Tuple[str, str]:
    """把原始实体名规范化为词典中的标准名，同时保留原名。"""
    raw_name = normalize_text(name)
    canonical = alias_to_canonical.get(raw_name, raw_name)
    canonical_aliases[canonical].add(raw_name)
    return raw_name, canonical


def build_records(triples: Iterable[dict], triple_path: Path, dictionary_path: Optional[Path]) -> dict:
    """把三元组转换为节点表和边表的中间记录。

    这里会完成：
    1. 审核结果过滤
    2. 关系类型约束
    3. 实体规范化
    4. 完全重复边去重
    5. 证据字段合并
    """
    alias_to_canonical, canonical_to_type, canonical_aliases = load_dictionary(dictionary_path)
    source_file = triple_path.relative_to(ROOT_DIR).as_posix()

    nodes: Dict[str, dict] = {}
    dedup_edges: Dict[Tuple[str, str, str], dict] = {}
    relation_counter = Counter()
    skipped_relations = Counter()
    raw_edge_count = 0
    review_filtered_count = 0
    duplicate_edge_instances = 0

    for index, item in enumerate(triples, start=1):
        review_decision = normalize_text(item.get("review_decision"))
        if review_decision and review_decision != "accept":
            review_filtered_count += 1
            continue

        relation = normalize_text(item.get("relation"))
        if relation not in ALLOWED_RELATIONS:
            skipped_relations[relation or "EMPTY"] += 1
            continue

        raw_head, canonical_head = canonicalize_entity(
            item.get("head", ""), alias_to_canonical, canonical_aliases
        )
        raw_tail, canonical_tail = canonicalize_entity(
            item.get("tail", ""), alias_to_canonical, canonical_aliases
        )
        if not canonical_head or not canonical_tail:
            continue

        for raw_name, canonical_name in ((raw_head, canonical_head), (raw_tail, canonical_tail)):
            if canonical_name not in nodes:
                nodes[canonical_name] = {
                    "node_id": stable_node_id(canonical_name),
                    "name": canonical_name,
                    "entity_type": canonical_to_type.get(canonical_name, "未知"),
                    "raw_name": raw_name or canonical_name,
                }

        edge_key = (canonical_head, relation, canonical_tail)
        raw_edge_count += 1
        relation_counter[relation] += 1
        record = {
            "source_name": canonical_head,
            "target_name": canonical_tail,
            "source": nodes[canonical_head]["node_id"],
            "target": nodes[canonical_tail]["node_id"],
            "relation": relation,
            "source_file": source_file,
            "evidence_text": normalize_text(item.get("text")),
            "block_id": normalize_text(item.get("block_id") or item.get("chunk_id")),
            "triple_id": normalize_text(item.get("triple_id")) or f"triple_{index:06d}",
            "confidence": item.get("confidence", ""),
            "status": "normal",
            "evidence_count": 1,
        }

        if edge_key in dedup_edges:
            duplicate_edge_instances += 1
            existing = dedup_edges[edge_key]
            existing["evidence_count"] += 1
            if not existing["evidence_text"] and record["evidence_text"]:
                existing["evidence_text"] = record["evidence_text"]
            if not existing["block_id"] and record["block_id"]:
                existing["block_id"] = record["block_id"]
            if not existing["triple_id"] and record["triple_id"]:
                existing["triple_id"] = record["triple_id"]
            if record["confidence"] not in ("", None):
                try:
                    existing_conf = float(existing["confidence"]) if existing["confidence"] != "" else None
                    new_conf = float(record["confidence"])
                    if existing_conf is None or new_conf > existing_conf:
                        existing["confidence"] = new_conf
                except (TypeError, ValueError):
                    pass
            continue

        dedup_edges[edge_key] = record

    for canonical_name, node in nodes.items():
        node["alias_count"] = len(canonical_aliases.get(canonical_name, {canonical_name}))

    return {
        "nodes": nodes,
        "edges": list(dedup_edges.values()),
        "relation_counter": relation_counter,
        "skipped_relations": skipped_relations,
        "raw_edge_count": raw_edge_count,
        "review_filtered_count": review_filtered_count,
        "duplicate_edge_instances": duplicate_edge_instances,
        "canonical_aliases": canonical_aliases,
    }


def analyze_graph(nodes: Dict[str, dict], edges: List[dict]) -> dict:
    """对构好的图谱做轻量图论分析和结构质检。"""
    graph = nx.DiGraph()
    undirected = nx.Graph()

    for node in nodes.values():
        graph.add_node(node["name"], **node)
        undirected.add_node(node["name"], **node)

    relation_counts = Counter()
    type_counts = Counter(node["entity_type"] for node in nodes.values())
    self_loops: List[dict] = []
    pair_relations: Dict[Tuple[str, str], set] = defaultdict(set)

    for edge in edges:
        src = edge["source_name"]
        dst = edge["target_name"]
        relation = edge["relation"]
        graph.add_edge(src, dst, relation=relation)
        undirected.add_edge(src, dst, relation=relation)
        relation_counts[relation] += 1
        pair_relations[(src, dst)].add(relation)
        if src == dst:
            edge["status"] = append_status(edge["status"], "self_loop")
            self_loops.append(edge)

    multi_relation_conflicts = []
    for (src, dst), relations in sorted(pair_relations.items()):
        if len(relations) > 1:
            relation_list = sorted(relations)
            multi_relation_conflicts.append(
                {"source_name": src, "target_name": dst, "relations": relation_list}
            )
            for edge in edges:
                if edge["source_name"] == src and edge["target_name"] == dst:
                    edge["status"] = append_status(edge["status"], "conflict_multi_relation")

    components = list(nx.connected_components(undirected))
    largest_component_size = max((len(component) for component in components), default=0)

    degree_centrality = nx.degree_centrality(undirected) if undirected.number_of_nodes() else {}
    pagerank = nx.pagerank(graph) if graph.number_of_nodes() else {}

    top_degree = rank_metric(degree_centrality)
    top_pagerank = rank_metric(pagerank)

    isa_edges = [(edge["source_name"], edge["target_name"]) for edge in edges if edge["relation"] == "是一种"]
    isa_edge_set = set(isa_edges)
    isa_bidirectional = []
    for src, dst in sorted(isa_edge_set):
        if src != dst and (dst, src) in isa_edge_set and src < dst:
            isa_bidirectional.append({"a": src, "b": dst})
            for edge in edges:
                if edge["relation"] == "是一种" and {
                    edge["source_name"],
                    edge["target_name"],
                } == {src, dst}:
                    edge["status"] = append_status(edge["status"], "isa_bidirectional_conflict")

    analysis = {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "relation_counts": dict(relation_counts),
        "node_type_counts": dict(type_counts),
        "connected_components": len(components),
        "largest_component_size": largest_component_size,
        "self_loop_count": len(self_loops),
        "self_loops": compact_edge_list(self_loops),
        "multi_relation_conflict_count": len(multi_relation_conflicts),
        "multi_relation_conflicts": multi_relation_conflicts,
        "isa_bidirectional_conflict_count": len(isa_bidirectional),
        "isa_bidirectional_conflicts": isa_bidirectional,
        "top_degree_centrality": top_degree,
        "top_pagerank": top_pagerank,
    }
    return analysis


def append_status(current: str, new_status: str) -> str:
    """把新的结构标记追加到边状态字段里。"""
    statuses = [part for part in current.split(";") if part and part != "normal"]
    if new_status not in statuses:
        statuses.append(new_status)
    return ";".join(statuses) if statuses else "normal"


def rank_metric(metric: Dict[str, float], top_n: int = 10) -> List[dict]:
    """把中心性指标排序并整理成便于导出的结构。"""
    return [
        {"name": name, "score": round(score, 6)}
        for name, score in sorted(metric.items(), key=lambda item: (-item[1], item[0]))[:top_n]
    ]


def compact_edge_list(edges: Iterable[dict], max_items: int = 20) -> List[dict]:
    """压缩边样例，避免分析报告中过长。"""
    compact = []
    for edge in list(edges)[:max_items]:
        compact.append(
            {
                "source_name": edge["source_name"],
                "relation": edge["relation"],
                "target_name": edge["target_name"],
                "status": edge["status"],
            }
        )
    return compact


def write_csv(path: Path, fieldnames: List[str], rows: Iterable[dict]) -> None:
    """写出通用 CSV 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_markdown_report(
    path: Path,
    summary: dict,
    analysis: dict,
    duplicate_edge_instances: int,
    review_filtered_count: int,
    skipped_relations: Dict[str, int],
) -> None:
    """生成图谱统计与结构质检的 Markdown 报告。"""
    lines = [
        "# 知识图谱辅助分析报告",
        "",
        "## 1. 构建概况",
        "",
        f"- 主输入文件：`{summary['input_file']}`",
        f"- 词典文件：`{summary['dictionary_file'] or '未使用'}`",
        f"- 节点数：{analysis['node_count']}",
        f"- 边数（去重后）：{analysis['edge_count']}",
        f"- 原始边实例数：{summary['raw_edge_count']}",
        f"- 完全重复边实例数：{duplicate_edge_instances}",
        f"- 审核过滤数：{review_filtered_count}",
        "",
        "## 2. 基本统计",
        "",
        "### 关系数量",
        "",
    ]

    for relation, count in sorted(analysis["relation_counts"].items()):
        lines.append(f"- `{relation}`：{count}")

    lines.extend(["", "### 节点类型数量", ""])
    for entity_type, count in sorted(analysis["node_type_counts"].items()):
        lines.append(f"- `{entity_type}`：{count}")

    lines.extend(
        [
            "",
            "## 3. 连通性分析",
            "",
            f"- 连通分量数量：{analysis['connected_components']}",
            f"- 最大连通子图规模：{analysis['largest_component_size']}",
            "",
            "## 4. 节点重要性",
            "",
            "### 度中心性 Top 10",
            "",
        ]
    )
    for item in analysis["top_degree_centrality"]:
        lines.append(f"- `{item['name']}`：{item['score']}")

    lines.extend(["", "### PageRank Top 10", ""])
    for item in analysis["top_pagerank"]:
        lines.append(f"- `{item['name']}`：{item['score']}")

    lines.extend(
        [
            "",
            "## 5. 结构检查",
            "",
            f"- 自环数量：{analysis['self_loop_count']}",
            f"- 同一实体对多关系冲突数量：{analysis['multi_relation_conflict_count']}",
            f"- `是一种` 双向冲突数量：{analysis['isa_bidirectional_conflict_count']}",
        ]
    )

    if skipped_relations:
        lines.extend(["", "### 被跳过的非目标关系", ""])
        for relation, count in sorted(skipped_relations.items()):
            lines.append(f"- `{relation}`：{count}")

    lines.extend(["", "### 自环样例", ""])
    if analysis["self_loops"]:
        for item in analysis["self_loops"]:
            lines.append(
                f"- `{item['source_name']} -[{item['relation']}]-> {item['target_name']}`，状态：`{item['status']}`"
            )
    else:
        lines.append("- 未发现自环")

    lines.extend(["", "### 同一实体对多关系冲突", ""])
    if analysis["multi_relation_conflicts"]:
        for item in analysis["multi_relation_conflicts"]:
            relation_text = " / ".join(item["relations"])
            lines.append(f"- `{item['source_name']} -> {item['target_name']}`：{relation_text}")
    else:
        lines.append("- 未发现同一实体对多关系冲突")

    lines.extend(["", "### `是一种` 双向冲突", ""])
    if analysis["isa_bidirectional_conflicts"]:
        for item in analysis["isa_bidirectional_conflicts"]:
            lines.append(f"- `{item['a']}` 与 `{item['b']}` 互为 `是一种`")
    else:
        lines.append("- 未发现明显双向短环")

    lines.extend(
        [
            "",
            "## 6. 低风险处理说明",
            "",
            "- 已执行完全重复边去重。",
            "- 已对结构风险边进行状态标记，未进行大规模自动删除。",
            "- 本报告中的结构问题用于辅助复核，不替代关系抽取主线结论。",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_neo4j_guide(path: Path) -> None:
    """生成面向使用者的 Neo4j 导入说明。"""
    content = """# Neo4j 最小导入说明

## 1. 准备文件

将以下文件复制到 Neo4j `import` 目录，或使用 Neo4j Desktop / Aura 可访问的位置：

- `nodes.csv`
- `edges.csv`

## 2. 节点导入

```cypher
LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
MERGE (n:Entity {node_id: row.node_id})
SET n.name = row.name,
    n.entity_type = row.entity_type,
    n.raw_name = row.raw_name,
    n.alias_count = toInteger(row.alias_count);
```

## 3. 关系导入

本项目关系类型固定为 `是一种`、`包含`、`用于`。为兼容 Neo4j 关系类型命名，建议统一导入为 `RELATED`，并把原关系保存在属性 `relation` 中。

```cypher
LOAD CSV WITH HEADERS FROM 'file:///edges.csv' AS row
MATCH (s:Entity {node_id: row.source})
MATCH (t:Entity {node_id: row.target})
MERGE (s)-[r:RELATED {relation: row.relation, source_file: row.source_file, block_id: row.block_id, triple_id: row.triple_id}]->(t)
SET r.evidence_text = row.evidence_text,
    r.status = row.status,
    r.confidence = CASE WHEN row.confidence = '' THEN null ELSE toFloat(row.confidence) END,
    r.evidence_count = toInteger(row.evidence_count);
```

## 4. 常用查看语句

```cypher
MATCH (n:Entity) RETURN count(n) AS node_count;
MATCH ()-[r:RELATED]->() RETURN count(r) AS edge_count;
MATCH ()-[r:RELATED]->() RETURN r.relation, count(*) AS cnt ORDER BY cnt DESC;
MATCH (a:Entity)-[r:RELATED]->(b:Entity) RETURN a, r, b LIMIT 50;
```

## 5. 说明

- 当前导出结果已经做了完全重复边去重。
- `status` 字段保留了自环、多关系冲突、`是一种` 双向冲突等结构标记，便于在 Neo4j 中继续筛查。
- 本导入方案仅用于结果展示与分析，不改变原始关系抽取结论。
"""
    path.write_text(content, encoding="utf-8")


def write_neo4j_cypher(path: Path) -> None:
    """生成最小 Neo4j 导入 Cypher 脚本。"""
    content = """// Neo4j minimal import script for outputs/kg/nodes.csv and outputs/kg/edges.csv
// Place CSV files in Neo4j import directory before executing.

LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
MERGE (n:Entity {node_id: row.node_id})
SET n.name = row.name,
    n.entity_type = row.entity_type,
    n.raw_name = row.raw_name,
    n.alias_count = toInteger(row.alias_count);

LOAD CSV WITH HEADERS FROM 'file:///edges.csv' AS row
MATCH (s:Entity {node_id: row.source})
MATCH (t:Entity {node_id: row.target})
MERGE (s)-[r:RELATED {relation: row.relation, source_file: row.source_file, block_id: row.block_id, triple_id: row.triple_id}]->(t)
SET r.evidence_text = row.evidence_text,
    r.status = row.status,
    r.confidence = CASE WHEN row.confidence = '' THEN null ELSE toFloat(row.confidence) END,
    r.evidence_count = toInteger(row.evidence_count);
"""
    path.write_text(content, encoding="utf-8")


def write_summary_json(path: Path, summary: dict) -> None:
    """写出结构化统计摘要，供平台和外部脚本复用。"""
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    """命令行入口：从最终三元组生成图谱及配套导出物。"""
    parser = argparse.ArgumentParser(description="Build a lightweight knowledge graph from final triples.")
    parser.add_argument("--triples", help="Path to the final triple file.")
    parser.add_argument("--dictionary", help="Path to dictionary.csv.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Directory for KG outputs.")
    args = parser.parse_args()

    triple_path = find_best_triple_file(args.triples)
    dictionary_path = find_dictionary_file(args.dictionary)
    output_dir = Path(args.output_dir)

    triples = load_triples(triple_path)
    build_result = build_records(triples, triple_path, dictionary_path)
    analysis = analyze_graph(build_result["nodes"], build_result["edges"])

    nodes_rows = sorted(build_result["nodes"].values(), key=lambda item: item["name"])
    edges_rows = sorted(
        build_result["edges"],
        key=lambda item: (item["relation"], item["source_name"], item["target_name"]),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "nodes.csv",
        ["node_id", "name", "entity_type", "raw_name", "alias_count"],
        nodes_rows,
    )
    write_csv(
        output_dir / "edges.csv",
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
        "input_file": triple_path.relative_to(ROOT_DIR).as_posix(),
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
        "centrality": {
            "degree_centrality_top10": analysis["top_degree_centrality"],
            "pagerank_top10": analysis["top_pagerank"],
        },
    }

    write_summary_json(output_dir / "kg_summary.json", summary)
    write_markdown_report(
        output_dir / "kg_analysis_report.md",
        summary,
        analysis,
        build_result["duplicate_edge_instances"],
        build_result["review_filtered_count"],
        dict(build_result["skipped_relations"]),
    )
    write_neo4j_guide(output_dir / "neo4j_import_guide.md")
    write_neo4j_cypher(output_dir / "neo4j_import.cypher")

    print(f"input={summary['input_file']}")
    print(f"dictionary={summary['dictionary_file'] or 'None'}")
    print(f"nodes={summary['node_count']}")
    print(f"edges={summary['edge_count_after_dedup']}")
    print(f"duplicates_removed={summary['duplicate_edge_instances_removed']}")
    print(f"self_loops={summary['self_loop_count']}")
    print(f"multi_relation_conflicts={summary['multi_relation_conflict_count']}")
    print(f"isa_bidirectional_conflicts={summary['isa_bidirectional_conflict_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
