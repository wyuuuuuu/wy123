# 智能网联汽车教材知识图谱

这是一个基于 Streamlit 的教材知识图谱展示与学习辅助原型。系统读取已经构建好的知识点、关系和章节主题数据，展示教材知识结构、教师教学分析和学生学习问答入口。

## 云端部署入口

Streamlit Cloud 部署时请填写：

```text
kg/teaching_kg_app.py
```

依赖文件：

```text
requirements.txt
```

核心数据文件：

```text
inputs/triples_all_reviewed.json
outputs/kg/nodes.csv
outputs/kg/edges.csv
outputs/kg/kg_summary.json
outputs/kg/topic_clusters.json
```

## 本地运行

```powershell
pip install -r requirements.txt
streamlit run kg/teaching_kg_app.py
```

## 说明

- 本项目只展示已有知识图谱结果，不重新训练模型。
- 云端部署后，图谱展示、章节分析、知识点查询和规则问答可直接使用。
- 本地 Ollama 大模型功能依赖 `http://localhost:11434`，部署到云端后默认无法访问本机 Ollama。如需云端大模型能力，需要改接在线模型 API 或公网可访问的 Ollama 服务。

