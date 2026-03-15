"""
Streamlit Dashboard — RAG Application Frontend.
Pages: Chat (Classic & Graph RAG), Embeddings Dashboard, Knowledge Graph, Evaluation.
"""

import streamlit as st
import requests
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import networkx as nx

API_URL = "http://localhost:8000"

# ─── Page config ───────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG Lab 6 — Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #00539C;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 4px solid #00539C;
    }
    .source-badge {
        background: #e8f4f8;
        color: #00539C;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px;
        border-radius: 8px 8px 0 0;
    }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ───────────────────────────────────────────────────────────────

def api_get(endpoint: str):
    try:
        r = requests.get(f"{API_URL}{endpoint}", timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Make sure the FastAPI server is running on port 8000.")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_post(endpoint: str, data: dict):
    try:
        r = requests.post(f"{API_URL}{endpoint}", json=data, timeout=120)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Make sure the FastAPI server is running on port 8000.")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


# ─── Sidebar ───────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## RAG Lab 6")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["💬 Chat", "📊 Embeddings", "🕸️ Knowledge Graph", "📈 Evaluation"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Stats
    stats = api_get("/stats")
    if stats:
        st.markdown("### System Info")
        st.markdown(f"**Documents:** {stats['documents']}")
        st.markdown(f"**Chunks:** {stats['chunks']}")
        st.markdown(f"**Embedding dim:** {stats['embedding_dim']}")
        st.markdown(f"**Graph nodes:** {stats['graph_nodes']}")
        st.markdown(f"**Graph edges:** {stats['graph_edges']}")
        st.markdown(f"**Communities:** {stats['communities']}")
        st.markdown("---")
        st.markdown(f"**LLM:** `{stats['llm_model']}`")
        st.markdown(f"**Embeddings:** `{stats['embed_model']}`")

    if st.button("🔄 Re-ingest Documents"):
        with st.spinner("Ingesting..."):
            result = api_post("/ingest", {})
            if result:
                st.success(f"Ingested {result.get('documents', 0)} docs, {result.get('chunks', 0)} chunks")
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: Chat
# ═══════════════════════════════════════════════════════════════════════════

if page == "💬 Chat":
    st.markdown('<div class="main-header">RAG Chat Interface</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ask questions about AI, Transformers, RAG, Fine-tuning, and Vector Databases</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col2:
        st.markdown("### Settings")
        rag_mode = st.selectbox("RAG Mode", ["Classic RAG", "Graph RAG"])
        if rag_mode == "Classic RAG":
            retrieval_method = st.selectbox("Retrieval", ["hybrid", "dense", "sparse"])
            use_reranking = st.checkbox("Use reranking", value=True)
            top_k = st.slider("Top-K retrieval", 3, 15, 5)
            rerank_top_k = st.slider("Top-K after rerank", 1, 5, 3)
        else:
            top_k_vector = st.slider("Vector chunks", 1, 5, 3)

    with col1:
        # Chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "sources" in msg:
                    with st.expander("📎 Sources"):
                        for s in msg["sources"]:
                            st.markdown(
                                f"**{s['source']}** — {s.get('section', '')} "
                                f"(score: {s['score']:.3f})"
                            )
                            st.caption(s.get("text", ""))

        question = st.chat_input("Ask a question...")

        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    if rag_mode == "Classic RAG":
                        result = api_post("/query", {
                            "question": question,
                            "method": retrieval_method,
                            "top_k": top_k,
                            "rerank_top_k": rerank_top_k,
                            "use_reranking": use_reranking,
                        })
                    else:
                        result = api_post("/query/graph", {
                            "question": question,
                            "top_k_vector": top_k_vector,
                        })

                if result and "answer" in result:
                    st.markdown(result["answer"])

                    sources = result.get("sources", [])
                    if sources:
                        with st.expander("📎 Sources"):
                            for s in sources:
                                st.markdown(
                                    f"**{s['source']}** — {s.get('section', '')} "
                                    f"(score: {s['score']:.3f})"
                                )
                                st.caption(s.get("text", ""))

                    # Graph RAG extra info
                    if "graph_entities" in result:
                        with st.expander("🕸️ Graph entities used"):
                            st.write(result["graph_entities"])
                        if result.get("graph_triples"):
                            with st.expander("🔗 Relationships found"):
                                for t in result["graph_triples"]:
                                    st.markdown(f"**{t['source']}** —[{t['relation']}]→ **{t['target']}**")

                    msg_data = {"role": "assistant", "content": result["answer"]}
                    if sources:
                        msg_data["sources"] = sources
                    st.session_state.messages.append(msg_data)
                elif result:
                    st.error(result.get("error", "Unknown error"))

    # Sample questions
    with col2:
        st.markdown("### Sample Questions")
        samples = [
            "What is self-attention in transformers?",
            "What chunking strategies are used in RAG?",
            "How does HNSW work?",
            "What is LoRA fine-tuning?",
            "When should I use RAG vs fine-tuning?",
            "How does the Transformer relate to BERT and GPT?",
        ]
        for q in samples:
            if st.button(q, key=f"sample_{q[:20]}"):
                st.session_state["_prefill"] = q
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: Embeddings Dashboard
# ═══════════════════════════════════════════════════════════════════════════

elif page == "📊 Embeddings":
    st.markdown('<div class="main-header">Embedding Space Visualization</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">PCA projection of chunk embeddings — explore semantic clusters</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["2D Projection", "3D Projection"])

    # ── 2D ──
    with tab1:
        data_2d = api_get("/embeddings")
        if data_2d and "points" in data_2d:
            df = pd.DataFrame(data_2d["points"])

            # Metrics row
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Chunks", data_2d["n_chunks"])
            c2.metric("Embedding Dim", data_2d["embedding_dim"])
            c3.metric("PC1 Variance", f"{data_2d['explained_variance'][0]:.1%}")
            c4.metric("PC2 Variance", f"{data_2d['explained_variance'][1]:.1%}")

            # Color map
            color_map = {
                "transformers.md": "#4A90D9",
                "rag_systems.md": "#E67E22",
                "vector_databases.md": "#27AE60",
                "finetuning.md": "#E74C3C",
            }

            fig = px.scatter(
                df, x="x", y="y",
                color="source",
                color_discrete_map=color_map,
                hover_data=["section", "text_preview"],
                title="Chunk Embeddings — PCA 2D Projection",
                labels={"x": "PC 1", "y": "PC 2", "source": "Document"},
                height=600,
            )
            fig.update_traces(marker=dict(size=10, line=dict(width=1, color="white")))
            fig.update_layout(
                plot_bgcolor="#fafafa",
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=12),
                ),
                font=dict(family="Inter, sans-serif"),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Source distribution
            st.markdown("### Chunk Distribution by Source")
            source_counts = df["source"].value_counts().reset_index()
            source_counts.columns = ["Source", "Count"]
            fig_bar = px.bar(
                source_counts, x="Source", y="Count",
                color="Source", color_discrete_map=color_map,
                height=300,
            )
            fig_bar.update_layout(showlegend=False, plot_bgcolor="#fafafa")
            st.plotly_chart(fig_bar, use_container_width=True)

    # ── 3D ──
    with tab2:
        data_3d = api_get("/embeddings/3d")
        if data_3d and "points" in data_3d:
            df3 = pd.DataFrame(data_3d["points"])

            c1, c2, c3 = st.columns(3)
            c1.metric("PC1 Variance", f"{data_3d['explained_variance'][0]:.1%}")
            c2.metric("PC2 Variance", f"{data_3d['explained_variance'][1]:.1%}")
            c3.metric("PC3 Variance", f"{data_3d['explained_variance'][2]:.1%}")

            color_map = {
                "transformers.md": "#4A90D9",
                "rag_systems.md": "#E67E22",
                "vector_databases.md": "#27AE60",
                "finetuning.md": "#E74C3C",
            }

            fig3d = px.scatter_3d(
                df3, x="x", y="y", z="z",
                color="source",
                color_discrete_map=color_map,
                hover_data=["section", "text_preview"],
                title="Chunk Embeddings — PCA 3D Projection",
                labels={"x": "PC 1", "y": "PC 2", "z": "PC 3"},
                height=700,
            )
            fig3d.update_traces(marker=dict(size=5, line=dict(width=0.5, color="white")))
            fig3d.update_layout(font=dict(family="Inter, sans-serif"))
            st.plotly_chart(fig3d, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: Knowledge Graph
# ═══════════════════════════════════════════════════════════════════════════

elif page == "🕸️ Knowledge Graph":
    st.markdown('<div class="main-header">Knowledge Graph Visualization</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Entity-relationship graph extracted from documents</div>', unsafe_allow_html=True)

    graph_data = api_get("/graph")
    if graph_data and "nodes" in graph_data:
        nodes = graph_data["nodes"]
        edges = graph_data["edges"]

        # Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Entities", graph_data["total_nodes"])
        c2.metric("Relationships", graph_data["total_edges"])
        c3.metric("Communities", len(graph_data.get("communities", [])))
        c4.metric("Entity Types", len(graph_data.get("type_counts", {})))

        tab1, tab2, tab3 = st.tabs(["Interactive Graph", "Community View", "Statistics"])

        # ── Interactive Graph ──
        with tab1:
            type_colors = {
                "TECHNOLOGY": "#4A90D9",
                "CONCEPT": "#E67E22",
                "METHOD": "#27AE60",
                "METRIC": "#E74C3C",
                "FRAMEWORK": "#9B59B6",
                "PERSON": "#F39C12",
                "UNKNOWN": "#95A5A6",
            }

            # Build NetworkX graph for layout
            G = nx.DiGraph()
            for n in nodes:
                G.add_node(n["id"], type=n["type"], degree=n["degree"])
            for e in edges:
                G.add_edge(e["source"], e["target"], relation=e["relation"])

            pos = nx.spring_layout(G, k=2.5, iterations=80, seed=42)

            # Edge traces
            edge_x, edge_y = [], []
            for e in edges:
                if e["source"] in pos and e["target"] in pos:
                    x0, y0 = pos[e["source"]]
                    x1, y1 = pos[e["target"]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])

            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=1.2, color="#BDC3C7"),
                hoverinfo="none",
                mode="lines",
                showlegend=False,
            )

            # Edge labels (midpoints)
            edge_label_x, edge_label_y, edge_label_text = [], [], []
            for e in edges:
                if e["source"] in pos and e["target"] in pos:
                    x0, y0 = pos[e["source"]]
                    x1, y1 = pos[e["target"]]
                    edge_label_x.append((x0 + x1) / 2)
                    edge_label_y.append((y0 + y1) / 2)
                    edge_label_text.append(e["relation"])

            edge_label_trace = go.Scatter(
                x=edge_label_x, y=edge_label_y,
                mode="text",
                text=edge_label_text,
                textfont=dict(size=8, color="#7F8C8D"),
                hoverinfo="none",
                showlegend=False,
            )

            # Node traces (one per type for legend)
            node_traces = []
            for etype, color in type_colors.items():
                type_nodes = [n for n in nodes if n["type"] == etype]
                if not type_nodes:
                    continue
                x_vals = [pos[n["id"]][0] for n in type_nodes if n["id"] in pos]
                y_vals = [pos[n["id"]][1] for n in type_nodes if n["id"] in pos]
                texts = [n["id"] for n in type_nodes if n["id"] in pos]
                sizes = [15 + 4 * n["degree"] for n in type_nodes if n["id"] in pos]
                hovers = [f"<b>{n['id']}</b><br>Type: {n['type']}<br>Degree: {n['degree']}"
                          for n in type_nodes if n["id"] in pos]

                node_traces.append(go.Scatter(
                    x=x_vals, y=y_vals,
                    mode="markers+text",
                    marker=dict(size=sizes, color=color, line=dict(width=1.5, color="white")),
                    text=texts,
                    textposition="top center",
                    textfont=dict(size=9, color="#2C3E50"),
                    hovertext=hovers,
                    hoverinfo="text",
                    name=etype,
                ))

            fig = go.Figure(
                data=[edge_trace, edge_label_trace] + node_traces,
                layout=go.Layout(
                    title=dict(text="Knowledge Graph — Entities & Relationships", font=dict(size=16, color="#2C3E50")),
                    showlegend=True,
                    legend=dict(
                        orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1, font=dict(size=11),
                    ),
                    hovermode="closest",
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    plot_bgcolor="white",
                    height=650,
                    margin=dict(l=20, r=20, t=60, b=20),
                ),
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── Community View ──
        with tab2:
            communities = graph_data.get("communities", [])
            if communities:
                # Community size chart
                comm_df = pd.DataFrame([
                    {"Community": f"Community {c['id']+1}", "Size": c["size"],
                     "Entities": ", ".join(c["entities"][:8]) + ("..." if len(c["entities"]) > 8 else "")}
                    for c in communities
                ])
                fig_comm = px.bar(
                    comm_df, x="Community", y="Size",
                    color="Size",
                    color_continuous_scale="Blues",
                    hover_data=["Entities"],
                    title="Community Sizes",
                    height=350,
                )
                fig_comm.update_layout(plot_bgcolor="#fafafa")
                st.plotly_chart(fig_comm, use_container_width=True)

                # Community details
                for c in communities:
                    with st.expander(f"Community {c['id']+1} — {c['size']} entities"):
                        st.write(c["entities"])
            else:
                st.info("No communities detected.")

        # ── Statistics ──
        with tab3:
            col1, col2 = st.columns(2)

            with col1:
                # Entity type distribution
                tc = graph_data.get("type_counts", {})
                if tc:
                    tc_df = pd.DataFrame([{"Type": k, "Count": v} for k, v in tc.items()])
                    fig_types = px.pie(
                        tc_df, names="Type", values="Count",
                        title="Entity Types Distribution",
                        color="Type",
                        color_discrete_map=type_colors,
                        height=400,
                    )
                    fig_types.update_layout(font=dict(family="Inter, sans-serif"))
                    st.plotly_chart(fig_types, use_container_width=True)

            with col2:
                # Top entities by degree
                sorted_nodes = sorted(nodes, key=lambda x: x["degree"], reverse=True)[:10]
                deg_df = pd.DataFrame(sorted_nodes)
                fig_deg = px.bar(
                    deg_df, x="id", y="degree",
                    color="type",
                    color_discrete_map=type_colors,
                    title="Top 10 Entities by Connections",
                    labels={"id": "Entity", "degree": "Degree"},
                    height=400,
                )
                fig_deg.update_layout(plot_bgcolor="#fafafa", xaxis_tickangle=-45)
                st.plotly_chart(fig_deg, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: Evaluation
# ═══════════════════════════════════════════════════════════════════════════

elif page == "📈 Evaluation":
    st.markdown('<div class="main-header">Retrieval Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Comparing dense, sparse, and hybrid retrieval on a golden set</div>', unsafe_allow_html=True)

    if st.button("Run Evaluation", type="primary"):
        with st.spinner("Evaluating retrieval methods..."):
            eval_data = api_get("/evaluate")

        if eval_data and "methods" in eval_data:
            st.success(f"Evaluated on {eval_data['golden_set_size']} queries")

            methods = eval_data["methods"]

            # Metrics cards
            cols = st.columns(len(methods))
            for col, (name, metrics) in zip(cols, methods.items()):
                with col:
                    st.markdown(f"### {name}")
                    st.metric("Hit Rate @5", f"{metrics['hit_rate']:.0%}")
                    st.metric("MRR @5", f"{metrics['mrr']:.3f}")

            st.markdown("---")

            # Bar chart comparison
            eval_df = pd.DataFrame([
                {"Method": name, "Metric": "Hit Rate @5", "Value": m["hit_rate"]}
                for name, m in methods.items()
            ] + [
                {"Method": name, "Metric": "MRR @5", "Value": m["mrr"]}
                for name, m in methods.items()
            ])

            method_colors = {
                "Dense": "#4A90D9",
                "Sparse (BM25)": "#E74C3C",
                "Hybrid (RRF)": "#27AE60",
            }

            fig = px.bar(
                eval_df, x="Method", y="Value",
                color="Method",
                color_discrete_map=method_colors,
                facet_col="Metric",
                title="Retrieval Method Comparison",
                height=400,
                barmode="group",
            )
            fig.update_layout(plot_bgcolor="#fafafa", showlegend=False)
            fig.update_yaxes(range=[0, 1.05])
            st.plotly_chart(fig, use_container_width=True)

            # Explanation
            st.markdown("""
            **Metrics explained:**
            - **Hit Rate @5**: Fraction of queries where the correct document appears in top-5 results
            - **MRR @5** (Mean Reciprocal Rank): Average of 1/rank for the first relevant result
            """)
    else:
        st.info("Click **Run Evaluation** to compare retrieval methods on the golden set.")

        st.markdown("""
        ### Golden Set Queries
        | Query | Expected Source |
        |-------|----------------|
        | What is self-attention in transformers? | transformers.md |
        | What chunking strategies exist for RAG? | rag_systems.md |
        | What is LoRA fine-tuning? | finetuning.md |
        | How does HNSW indexing work? | vector_databases.md |
        | What are the benefits of hybrid retrieval? | rag_systems.md |
        | What is DPO in LLM alignment? | finetuning.md |
        """)
