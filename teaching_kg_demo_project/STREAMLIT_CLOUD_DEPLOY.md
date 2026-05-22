# Streamlit Cloud 部署说明

## 1. 部署目标

把当前演示项目部署到 Streamlit Community Cloud 后，用户只需要打开网页链接即可使用界面，不需要安装 Python、依赖库或本地运行脚本。

## 2. 推荐上传范围

建议把当前文件夹作为一个独立 GitHub 仓库上传：

```text
deliverables/teaching_kg_demo_project
```

仓库根目录建议保持如下结构：

```text
teaching_kg_demo_project/
├─ kg/
│  ├─ teaching_kg_app.py
│  ├─ build_knowledge_graph.py
│  └─ build_topic_clusters.py
├─ inputs/
│  └─ triples_all_reviewed.json
├─ outputs/
│  └─ kg/
│     ├─ nodes.csv
│     ├─ edges.csv
│     ├─ kg_summary.json
│     └─ topic_clusters.json
├─ requirements.txt
├─ README.md
└─ STREAMLIT_CLOUD_DEPLOY.md
```

不要上传：

```text
.venv/
__pycache__/
inputs/uploaded_triples/
outputs/kg/learning_progress.json
```

这些内容已经写入 `.gitignore`。

## 3. 本地检查

在上传前，先在项目根目录执行：

```powershell
cd D:\graduation_design_v2\smart_classroom\deliverables\teaching_kg_demo_project
python -m pip install -r requirements.txt
python -m streamlit run kg\teaching_kg_app.py
```

如果本地页面可以打开，说明入口文件、依赖和基础数据基本完整。

## 4. 上传 GitHub

可以新建一个 GitHub 仓库，例如：

```text
smart-car-teaching-kg
```

然后把 `teaching_kg_demo_project` 文件夹里的内容作为仓库根目录上传。注意不要把整个 `smart_classroom` 项目上传，否则云端入口路径容易变复杂，也会带上很多不必要文件。

## 5. 在 Streamlit Cloud 创建应用

进入 Streamlit Community Cloud：

```text
https://share.streamlit.io
```

创建应用时填写：

```text
Repository: 你的 GitHub 仓库
Branch: main
Main file path: kg/teaching_kg_app.py
```

Streamlit Cloud 会自动读取 `requirements.txt` 并安装：

```text
pandas
networkx
streamlit
```

部署完成后会生成一个公开访问链接，用户打开该链接即可使用。

## 6. 云端大模型功能说明

当前应用的大模型增强功能默认调用本机 Ollama：

```text
http://localhost:11434
模型：qwen3:1.7b
```

部署到 Streamlit Cloud 后，云端服务器无法访问你个人电脑上的 `localhost`。因此：

- 图谱总览、教师端分析、学生端知识点查询、三元组检索和规则推荐可以正常使用。
- 大模型生成教学要点、学习推荐说明和自然语言问答在云端默认会降级为图谱规则结果。
- 如果需要云端也能使用大模型，需要把模型服务部署到公网服务器，或者把代码改接在线大模型 API。

## 7. 答辩展示建议

如果只是给老师或同学体验，推荐先使用云端基础版：

1. 打开云端链接。
2. 先展示 `图谱总览`，说明知识点、关系和四个章节主题。
3. 切换到 `教师端`，展示章节知识结构和三元组证据。
4. 切换到 `学生端`，展示知识点查询、学习路径和规则问答。
5. 如需展示大模型生成效果，可以在本机运行版本中演示 Ollama 增强能力。

