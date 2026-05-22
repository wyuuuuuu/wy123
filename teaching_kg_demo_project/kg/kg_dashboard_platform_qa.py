"""QA-first Streamlit entry for the knowledge-graph package.

This module intentionally keeps the platform lightweight:
1. Reuse graph rendering and data-loading helpers from `kg_dashboard_platform_v5`.
2. Put graph retrieval + evidence traceability + Ollama answering on the main path.
3. Keep the UI surface smaller than the generic display platform.

The goal is not to replace the original display module, but to provide a cleaner
entry focused on "knowledge graph + local QA".
"""

from pathlib import Path
import sys

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components


# Allow `streamlit run kg_dashboard_platform_qa.py` to import the sibling module.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

import kg_dashboard_platform_v5 as base


OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "qwen3:1.7b"
QA_MAX_EDGES = 10
FULL_VIEW_MAX_NODES = 260
THEME_VIEW_MAX_NODES = 70

PAGE_QA = "知识问答"
PAGE_OVERVIEW = "总览"
PAGE_DISPLAY = "展示视图"

RELATION_KEYWORDS = {
    base.REL_USED_FOR: ["用于", "用来", "作用", "做什么", "应用于"],
    base.REL_CONTAINS: ["包含", "包括", "组成", "由什么构成", "有哪些部分", "模块"],
    base.REL_IS_A: ["是什么", "属于什么", "是一种", "哪一类", "什么类型"],
}


def init_qa_state() -> None:
    """Initialize session state shared by the QA page and the base renderer."""
    base.init_state()

    qa_defaults = {
        "kg_qa_model": DEFAULT_OLLAMA_MODEL,
        "kg_qa_question": "",
        "kg_qa_result": None,
        "kg_qa_bootstrapped": False,
    }
    for key, value in qa_defaults.items():
        st.session_state.setdefault(key, value)

    if st.session_state["kg_qa_bootstrapped"]:
        return

    # The QA platform still reuses base renderers, so we seed only the states
    # that those renderers rely on. Everything else stays hidden from the UI.
    st.session_state["kg_v5_page"] = PAGE_QA
    st.session_state["kg_v5_display_graph_mode"] = "自由图模式"
    st.session_state["kg_v5_max_nodes"] = 180
    st.session_state["kg_v5_min_component_size"] = 2
    st.session_state["kg_v5_hide_issue_edges"] = False
    st.session_state["kg_v5_only_issue_graph"] = False
    st.session_state["kg_v5_high_label_only"] = True
    st.session_state["kg_v5_focus_largest_component"] = False
    st.session_state["kg_v5_focus_depth"] = 1
    st.session_state["kg_v5_defense_mode"] = False
    st.session_state["kg_v5_prev_defense_mode"] = False
    st.session_state["kg_v5_window_height"] = 720

    for preset in base.THEME_PRESETS:
        st.session_state[f"kg_v5_theme_graph_mode_{preset['name']}"] = "自由图模式"

    st.session_state["kg_qa_bootstrapped"] = True


def normalize_text(text: object) -> str:
    """Normalize text for rough matching without changing semantics too much."""
    return str(text or "").strip().lower()


def detect_relation_intent(question: str) -> str | None:
    """Infer which of the three allowed relations the user is asking about."""
    question_norm = normalize_text(question)
    for relation, keywords in RELATION_KEYWORDS.items():
        if any(keyword in question_norm for keyword in keywords):
            return relation
    return None


def extract_aliases(row: pd.Series) -> list[str]:
    """Collect entity aliases used for lightweight graph-side matching."""
    aliases: list[str] = []
    for value in [row.get("name", ""), row.get("raw_name", "")]:
        alias = str(value or "").strip()
        if alias and alias not in aliases:
            aliases.append(alias)
    return aliases


def match_entities(question: str, nodes_df: pd.DataFrame, limit: int = 4) -> list[dict]:
    """Match question mentions against graph entities by simple alias inclusion.

    This is intentionally lightweight: the project already has a curated graph,
    and the QA layer should stay as a small retrieval wrapper instead of turning
    into a full NER/retrieval system.
    """
    question_norm = normalize_text(question)
    matches: list[dict] = []

    for _, row in nodes_df.iterrows():
        best_alias = None
        best_score = -1
        for alias in extract_aliases(row):
            alias_norm = normalize_text(alias)
            if len(alias_norm) < 2 or alias_norm not in question_norm:
                continue

            score = len(alias_norm) * 10 + int(row.get("degree", 0))
            if alias == row.get("name", ""):
                score += 5
            if score > best_score:
                best_score = score
                best_alias = alias

        if best_alias:
            matches.append(
                {
                    "node_id": row["node_id"],
                    "name": row["name"],
                    "entity_type": row.get("entity_type", base.UNKNOWN_TYPE),
                    "matched_alias": best_alias,
                    "score": best_score,
                }
            )

    matches.sort(key=lambda item: (-item["score"], item["name"]))

    deduped: list[dict] = []
    seen: set[str] = set()
    for item in matches:
        if item["node_id"] in seen:
            continue
        deduped.append(item)
        seen.add(item["node_id"])
        if len(deduped) >= limit:
            break
    return deduped


def retrieve_answer_edges(
    question: str,
    relation_intent: str | None,
    matched_entities: list[dict],
    edge_view: pd.DataFrame,
) -> pd.DataFrame:
    """Rank candidate triples for the current question.

    The retrieval order is:
    1. Prefer edges touching matched entities.
    2. Prefer the explicitly asked relation type.
    3. Prefer rows with evidence/context fields.
    """
    if edge_view.empty:
        return edge_view.copy()

    question_norm = normalize_text(question)
    entity_ids = {item["node_id"] for item in matched_entities}
    working = edge_view.copy()

    if entity_ids:
        working = working[
            working["source"].isin(entity_ids) | working["target"].isin(entity_ids)
        ].copy()
    if working.empty:
        working = edge_view.copy()

    if relation_intent:
        relation_slice = working[working["relation"] == relation_intent].copy()
        if not relation_slice.empty:
            working = relation_slice

    def score_row(row: pd.Series) -> int:
        score = 0
        if row["source"] in entity_ids:
            score += 25
        if row["target"] in entity_ids:
            score += 25
        if relation_intent and row["relation"] == relation_intent:
            score += 20
        if normalize_text(row.get("source_name", "")) in question_norm:
            score += 8
        if normalize_text(row.get("target_name", "")) in question_norm:
            score += 8
        if str(row.get("evidence_text", "")).strip():
            score += 6
        if str(row.get("context_snippet", "")).strip():
            score += 4
        if len(entity_ids) >= 2 and row["source"] in entity_ids and row["target"] in entity_ids:
            score += 15
        return score

    ranked = working.copy()
    ranked["qa_score"] = ranked.apply(score_row, axis=1)
    ranked = ranked.sort_values(
        ["qa_score", "relation", "source_name", "target_name"],
        ascending=[False, True, True, True],
    ).head(QA_MAX_EDGES)
    return ranked.drop(columns=["qa_score"])


def build_rule_based_answer(
    matched_entities: list[dict],
    relation_intent: str | None,
    answer_edges: pd.DataFrame,
) -> str:
    """Provide a deterministic fallback answer when the local model is unavailable."""
    if answer_edges.empty:
        return "当前图谱范围内没有检索到足够证据，建议换一个更具体的教材概念后再提问。"

    relation_label = relation_intent or "相关"
    entity_text = "、".join(item["name"] for item in matched_entities) if matched_entities else "当前问题中的概念"
    triples = [
        f"{row['source_name']} -[{row['relation']}]-> {row['target_name']}"
        for _, row in answer_edges.head(3).iterrows()
    ]
    triple_text = "；".join(triples)
    return f"围绕 {entity_text} 的问题，图谱检索到的核心 {relation_label} 关系包括：{triple_text}。下方给出教材证据支持。"


def build_prompt(
    question: str,
    matched_entities: list[dict],
    relation_intent: str | None,
    answer_edges: pd.DataFrame,
) -> str:
    """Assemble a grounded prompt for the local Ollama model."""
    entity_lines = [
        f"- {item['name']}（类型：{item['entity_type']}，命中词：{item['matched_alias']}）"
        for item in matched_entities
    ]

    triple_lines: list[str] = []
    for idx, (_, row) in enumerate(answer_edges.iterrows(), start=1):
        triple_lines.append(
            f"{idx}. 三元组：{row['source_name']} -[{row['relation']}]-> {row['target_name']}\n"
            f"   source_file: {row.get('source_file', '')}\n"
            f"   block_id: {row.get('block_id', '')}\n"
            f"   evidence_text: {row.get('evidence_text', '') or '无'}\n"
            f"   context_snippet: {row.get('context_snippet', '') or '无'}"
        )

    relation_text = relation_intent or "未显式限定"
    return (
        "你是教材知识图谱问答助手。请严格依据给定三元组和教材证据回答，不能编造。\n"
        "如果证据不足，要直接说明“当前证据不足”。\n\n"
        f"用户问题：{question}\n"
        f"问题意图关系：{relation_text}\n"
        "命中的实体：\n"
        f"{chr(10).join(entity_lines) if entity_lines else '- 无明确命中实体'}\n\n"
        "检索到的图谱证据：\n"
        f"{chr(10).join(triple_lines) if triple_lines else '- 无可用三元组'}\n\n"
        "请按以下格式输出：\n"
        "1. 直接答案\n"
        "2. 图谱依据\n"
        "3. 教材证据\n"
        "4. 不确定性说明（若无可写“无”）"
    )


def call_ollama(
    question: str,
    matched_entities: list[dict],
    relation_intent: str | None,
    answer_edges: pd.DataFrame,
    model_name: str,
) -> str:
    """Query the local Ollama service with grounded graph evidence."""
    prompt = build_prompt(question, matched_entities, relation_intent, answer_edges)
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2},
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()


def build_answer_subgraph(full_view: dict, answer_edges: pd.DataFrame) -> dict | None:
    """Build a small answer graph for the currently retrieved evidence."""
    if answer_edges.empty:
        return None

    answer_nodes = base.sync_nodes_to_edges(full_view["nodes"], answer_edges)
    answer_nodes, answer_graph = base.enrich_node_metrics(answer_nodes, answer_edges)
    return {
        "nodes": answer_nodes,
        "edges": answer_edges,
        "graph": answer_graph,
        "stats": {
            "visible_nodes": int(len(answer_nodes)),
            "visible_edges": int(len(answer_edges)),
        },
    }


def run_qa(question: str, full_view: dict, model_name: str) -> dict:
    """Run the full QA chain: match -> retrieve -> LLM answer / fallback."""
    matched_entities = match_entities(question, full_view["nodes"])
    relation_intent = detect_relation_intent(question)
    answer_edges = retrieve_answer_edges(question, relation_intent, matched_entities, full_view["edges"])
    fallback_answer = build_rule_based_answer(matched_entities, relation_intent, answer_edges)

    llm_answer = ""
    llm_error = ""
    if not answer_edges.empty:
        try:
            llm_answer = call_ollama(question, matched_entities, relation_intent, answer_edges, model_name)
        except Exception as exc:  # noqa: BLE001 - error should be shown to the user.
            llm_error = str(exc)

    return {
        "question": question,
        "relation_intent": relation_intent or "未显式限定",
        "matched_entities": matched_entities,
        "answer_edges": answer_edges.to_dict("records"),
        "llm_answer": llm_answer,
        "fallback_answer": fallback_answer,
        "llm_error": llm_error,
    }


def render_qa_page(full_view: dict, display_options: dict) -> None:
    """Render the QA page and keep evidence traceability visible."""
    st.subheader(PAGE_QA)
    st.caption("流程：先在本地图谱中检索实体、关系和教材证据，再交给本地 Ollama 的 qwen3:1.7b 组织回答。")

    with st.form("kg_qa_form"):
        model_name = st.text_input(
            "Ollama 模型",
            value=st.session_state.get("kg_qa_model", DEFAULT_OLLAMA_MODEL),
        )
        question = st.text_area(
            "输入问题",
            value=st.session_state.get("kg_qa_question", ""),
            height=110,
        )
        submitted = st.form_submit_button("开始问答", use_container_width=True)

    if submitted:
        st.session_state["kg_qa_question"] = question
        st.session_state["kg_qa_model"] = model_name
        st.session_state["kg_qa_result"] = run_qa(question, full_view, model_name)

    result = st.session_state.get("kg_qa_result")
    if not result:
        st.info("在上方输入问题后，平台会返回图谱命中结果、教材证据和本地模型回答。")
        return

    match_df = pd.DataFrame(result["matched_entities"])
    edge_df = pd.DataFrame(result["answer_edges"])

    cols = st.columns([1.4, 1])
    with cols[0]:
        st.markdown("**回答结果**")
        if result["llm_answer"]:
            st.markdown(result["llm_answer"])
        else:
            st.markdown(result["fallback_answer"])
            if result["llm_error"]:
                st.caption(f"Ollama 未成功返回结果，已回退到本地规则答案：{result['llm_error']}")

    with cols[1]:
        st.markdown("**检索摘要**")
        summary_df = pd.DataFrame(
            [
                {
                    "问题关系意图": result["relation_intent"],
                    "命中实体数": int(len(result["matched_entities"])),
                    "命中三元组数": int(len(result["answer_edges"])),
                }
            ]
        )
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.markdown("**命中实体**")
    if match_df.empty:
        st.caption("未命中明确实体。")
    else:
        st.dataframe(match_df[["name", "entity_type", "matched_alias"]], use_container_width=True, hide_index=True)

    st.markdown("**图谱依据**")
    if edge_df.empty:
        st.warning("当前问题在图谱中没有检索到足够的三元组证据。")
        return
    st.dataframe(
        edge_df[["source_name", "relation", "target_name", "source_file", "block_id"]],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("**教材证据**")
    st.dataframe(
        edge_df[
            [
                "source_name",
                "relation",
                "target_name",
                "evidence_text",
                "context_snippet",
                "source_file",
                "block_id",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    answer_bundle = build_answer_subgraph(full_view, edge_df)
    if answer_bundle and answer_bundle["stats"]["visible_edges"] > 0:
        heights = base.get_view_heights()
        canvas_height = base.adaptive_canvas_height(
            answer_bundle["stats"]["visible_nodes"],
            heights["compact"],
            threshold=8,
            growth_per_node=9,
            max_extra=240,
            minimum=520,
            maximum=980,
        )
        st.markdown("**问答子图**")
        components.html(
            base.make_clickable_network_html(
                answer_bundle["graph"],
                answer_bundle["nodes"],
                answer_bundle["edges"],
                "问答检索子图",
                height=canvas_height,
                physics_enabled=display_options["physics_enabled"],
                high_centrality_labels=display_options["high_label_only"],
                layout_relation=base.choose_layout_relation(answer_bundle["edges"]),
                node_size_scale=display_options["node_size_scale"],
                edge_width_scale=display_options["edge_width_scale"],
                spacing_scale=display_options["spacing_scale"],
                initial_scale=display_options["initial_scale"],
                neo4j_style=True,
            ),
            height=base.adaptive_wrapper_height(canvas_height),
            scrolling=False,
        )


def build_theme_views(full_view: dict, max_nodes: int, min_component_size: int, theme_hops: int) -> tuple[dict, list[dict]]:
    """Prepare the fixed theme views used by the overview and theme pages."""
    theme_views: dict = {}
    theme_rows: list[dict] = []
    theme_source_graph = full_view["graph"]

    for preset in base.THEME_PRESETS:
        theme_nodes, theme_edges, matched_seeds = base.extract_theme(
            theme_source_graph,
            full_view["nodes"],
            full_view["edges"],
            preset,
            theme_hops,
        )
        theme_view = base.build_display_view(
            theme_nodes,
            theme_edges,
            max_nodes=min(max_nodes, THEME_VIEW_MAX_NODES),
            hide_two_node_components=False,
            min_component_size=max(2, min_component_size),
            focus_largest_component=False,
            deduplicate=False,
            hide_issue_edges=False,
            only_issue_edges=False,
        )
        layout_relation = base.choose_layout_relation(
            theme_view["edges"],
            preferred_relation=preset.get("preferred_relation"),
        )
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
                "布局风格": base.build_layout_options(layout_relation, False, base.DEFAULT_SPACING_SCALE)["label"],
            }
        )

    return theme_views, theme_rows


def prepare_views(
    source_name: str,
    max_nodes: int,
    min_component_size: int,
    focus_largest_component: bool,
    theme_hops: int,
    keyword: str,
    relations: list[str],
    node_types: list[str],
) -> dict:
    """Load graph data once and derive all page-level views from the same slice."""
    nodes_df, edges_df, _summary, _report_text = base.choose_source(source_name)
    nodes_df, edges_df = base.normalize_frames(nodes_df, edges_df)

    filtered_nodes, filtered_edges = base.filter_frames(nodes_df, edges_df, relations, node_types, keyword)
    filtered_edges = base.apply_qc_flags(filtered_edges)

    full_view = base.build_display_view(
        filtered_nodes,
        filtered_edges,
        max_nodes=max(FULL_VIEW_MAX_NODES, max_nodes),
        hide_two_node_components=False,
        min_component_size=1,
        focus_largest_component=False,
        deduplicate=True,
        hide_issue_edges=False,
        only_issue_edges=False,
    )
    display_view = base.build_display_view(
        filtered_nodes,
        filtered_edges,
        max_nodes=max_nodes,
        hide_two_node_components=False,
        min_component_size=min_component_size,
        focus_largest_component=focus_largest_component,
        deduplicate=True,
        hide_issue_edges=False,
        only_issue_edges=False,
    )

    theme_views, theme_rows = build_theme_views(full_view, max_nodes, min_component_size, theme_hops)
    relation_table = base.relation_summary_table(full_view["edges"])
    core_table = base.core_top5_table(full_view["nodes"])
    source_file_count = (
        full_view["edges"]["source_file"].astype(str).str.strip().replace("", pd.NA).nunique(dropna=True)
    )

    return {
        "full_view": full_view,
        "display_view": display_view,
        "theme_views": theme_views,
        "theme_rows": theme_rows,
        "relation_table": relation_table,
        "core_table": core_table,
        "source_file_count": int(source_file_count),
    }


def render_sidebar() -> dict:
    """Collect the small set of controls still useful for the QA-first platform."""
    with st.sidebar:
        st.markdown("### 平台设置")
        source_name = st.radio("数据源", ["本地 CSV", "Neo4j"])
        physics_enabled = st.toggle("启用自动布局", value=True)
        max_nodes = st.slider("每个视图最多节点数", min_value=40, max_value=180, step=20, key="kg_v5_max_nodes")
        min_component_size = st.slider("只显示节点数 >= N 的连通子图", min_value=1, max_value=10, key="kg_v5_min_component_size")
        focus_largest_component = st.toggle("展示视图聚焦最大连通子图", key="kg_v5_focus_largest_component")
        theme_hops = st.select_slider("主题扩展跳数", options=[1, 2], value=1)
        keyword = st.text_input("关键词过滤")

    return {
        "source_name": source_name,
        "physics_enabled": physics_enabled,
        "max_nodes": max_nodes,
        "min_component_size": min_component_size,
        "focus_largest_component": focus_largest_component,
        "theme_hops": theme_hops,
        "keyword": keyword,
        "high_label_only": True,
        "node_size_scale": base.DEFAULT_NODE_SIZE_SCALE,
        "edge_width_scale": base.DEFAULT_EDGE_WIDTH_SCALE,
        "spacing_scale": base.DEFAULT_SPACING_SCALE,
        "initial_scale": base.DEFAULT_INITIAL_SCALE,
    }


def main() -> None:
    """Streamlit app entry."""
    st.set_page_config(page_title="知识图谱问答平台", layout="wide")
    init_qa_state()

    st.title("知识图谱问答平台")
    st.caption("本地图谱检索 + Ollama 大模型问答")

    ui = render_sidebar()

    try:
        nodes_df, edges_df, _summary, _report_text = base.choose_source(ui["source_name"])
    except Exception as exc:  # noqa: BLE001 - should be visible in the page.
        st.error(f"{ui['source_name']} 读取失败：{exc}")
        return

    nodes_df, edges_df = base.normalize_frames(nodes_df, edges_df)
    all_relations = [relation for relation in base.RELATION_OPTIONS if relation in set(edges_df["relation"].unique())]
    all_types = sorted(nodes_df["entity_type"].dropna().unique().tolist())

    with st.sidebar:
        st.markdown("### 图谱筛选")
        relations = st.multiselect("关系类型", options=all_relations, default=all_relations)
        node_types = st.multiselect("节点类型", options=all_types, default=all_types)

    try:
        prepared = prepare_views(
            source_name=ui["source_name"],
            max_nodes=ui["max_nodes"],
            min_component_size=ui["min_component_size"],
            focus_largest_component=ui["focus_largest_component"],
            theme_hops=ui["theme_hops"],
            keyword=ui["keyword"],
            relations=relations,
            node_types=node_types,
        )
    except Exception as exc:  # noqa: BLE001 - should be visible in the page.
        st.error(f"图谱视图构建失败：{exc}")
        return

    theme_page_map = {base.theme_page_name(preset["name"]): preset["name"] for preset in base.THEME_PRESETS}
    pages = [PAGE_QA, PAGE_OVERVIEW, PAGE_DISPLAY, *theme_page_map.keys()]
    page = st.radio("页面", options=pages, horizontal=True, key="kg_v5_page")

    if page == PAGE_QA:
        render_qa_page(prepared["full_view"], ui)
        return

    if page == PAGE_OVERVIEW:
        base.render_overview(
            prepared["full_view"],
            prepared["display_view"],
            prepared["relation_table"],
            prepared["core_table"],
            prepared["theme_rows"],
            ui["source_name"],
            prepared["source_file_count"],
            prepared["theme_views"],
            ui["physics_enabled"],
            ui["high_label_only"],
            ui["node_size_scale"],
            ui["edge_width_scale"],
            ui["spacing_scale"],
            ui["initial_scale"],
        )
        return

    if page == PAGE_DISPLAY:
        base.render_display_page(
            prepared["display_view"],
            ui["physics_enabled"],
            ui["high_label_only"],
            ui["node_size_scale"],
            ui["edge_width_scale"],
            ui["spacing_scale"],
            ui["initial_scale"],
        )
        return

    if page in theme_page_map:
        base.render_theme_page(
            theme_page_map[page],
            prepared["theme_views"],
            ui["theme_hops"],
            ui["physics_enabled"],
            ui["high_label_only"],
            ui["node_size_scale"],
            ui["edge_width_scale"],
            ui["spacing_scale"],
            ui["initial_scale"],
        )


if __name__ == "__main__":
    main()
