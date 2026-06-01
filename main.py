import os
import pandas as pd
import numpy as np
import yfinance as yf
import mlflow
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

load_dotenv()

# ── State ──────────────────────────────────────────────────────────────────────

class StockAgentState(TypedDict):
    ticker: str
    raw_data: Any
    patterns: List[Dict]
    similar_patterns: List[Dict]
    prediction: str
    status: str

# ── Shared resources ───────────────────────────────────────────────────────────

llm = ChatAnthropic(model="claude-sonnet-4-6")
WINDOW_SIZE = 10
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "stock-patterns")

def get_pinecone_index():
    pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    existing_index_names = [index.name for index in pinecone_client.list_indexes()]
    if INDEX_NAME not in existing_index_names:
        print(f"  Creating Pinecone index '{INDEX_NAME}' (dimension={WINDOW_SIZE})...")
        pinecone_client.create_index(
            name=INDEX_NAME,
            dimension=WINDOW_SIZE,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pinecone_client.Index(INDEX_NAME)

# ── Agent 1: DataLoader ────────────────────────────────────────────────────────

def data_loader_agent(state: StockAgentState) -> StockAgentState:
    ticker = state["ticker"]
    print(f"\n[Agent 1 -DataLoader] Downloading {ticker} (1 year)...")

    stock = yf.Ticker(ticker)
    data = stock.history(period="1y").reset_index()
    data.to_csv(f"{ticker}_data.csv", index=False)

    data["return"] = data["Close"].pct_change()
    data = data.dropna().reset_index(drop=True)

    print(f"[Agent 1 - DataLoader] {len(data)} trading days loaded -> {ticker}_data.csv")
    return {**state, "raw_data": data, "status": "data_loaded"}

# ── Agent 2: Embedder ──────────────────────────────────────────────────────────

def embedding_agent(state: StockAgentState) -> StockAgentState:
    data = state["raw_data"]
    print(f"\n[Agent 2 - Embedder] Building {WINDOW_SIZE}-day patterns -> Pinecone...")

    pinecone_index = get_pinecone_index()

    try:
        pinecone_index.delete(delete_all=True)
    except Exception:
        pass

    patterns = []
    batch = []

    for index in range(WINDOW_SIZE, len(data) - 1):
        window_returns = data["return"].iloc[index - WINDOW_SIZE : index].values
        next_return = float(data["return"].iloc[index])
        date_str = str(data["Date"].iloc[index].date())

        std_value = np.std(window_returns) + 1e-8
        embedding_vector = (window_returns / std_value).tolist()

        batch.append({
            "id": f"pattern_{index}",
            "values": embedding_vector,
            "metadata": {
                "date": date_str,
                "next_return": next_return,
                "direction": "UP" if next_return > 0 else "DOWN",
                "ticker": state["ticker"],
            },
        })
        patterns.append({
            "date": date_str,
            "window": window_returns.tolist(),
            "next_return": next_return,
        })

        if len(batch) >= 100:
            pinecone_index.upsert(vectors=batch)
            batch = []

    if batch:
        pinecone_index.upsert(vectors=batch)

    print(f"[Agent 2 -Embedder] {len(patterns)} patterns stored in Pinecone.")
    return {**state, "patterns": patterns, "status": "embedded"}

# ── Agent 3: QueryAgent ────────────────────────────────────────────────────────

def query_agent(state: StockAgentState) -> StockAgentState:
    data = state["raw_data"]
    print(f"\n[Agent 3 -QueryAgent] Finding 5 most similar historical patterns...")

    recent_returns = data["return"].iloc[-WINDOW_SIZE:].values
    std_value = np.std(recent_returns) + 1e-8
    query_embedding = (recent_returns / std_value).tolist()

    pinecone_index = get_pinecone_index()
    results = pinecone_index.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True,
    )

    similar_patterns = [
        {
            "date": match.metadata["date"],
            "next_return": match.metadata["next_return"],
            "direction": match.metadata["direction"],
            "similarity_score": match.score,
        }
        for match in results.matches
    ]

    print(f"[Agent 3 -QueryAgent] Results:")
    for pattern in similar_patterns:
        print(
            f"  ->{pattern['date']}: "
            f"next day {pattern['direction']} ({pattern['next_return']:.3%}) "
            f"| similarity: {pattern['similarity_score']:.3f}"
        )

    return {**state, "similar_patterns": similar_patterns, "status": "queried"}

# ── Agent 4: Predictor ─────────────────────────────────────────────────────────

def prediction_agent(state: StockAgentState) -> StockAgentState:
    ticker = state["ticker"]
    similar_patterns = state["similar_patterns"]
    data = state["raw_data"]

    print(f"\n[Agent 4 -Predictor] Analyzing patterns with Claude...")

    recent_returns = data["return"].iloc[-WINDOW_SIZE:].values
    recent_text = ", ".join([f"{return_value:.3%}" for return_value in recent_returns])

    up_count = sum(1 for pattern in similar_patterns if pattern["direction"] == "UP")
    down_count = len(similar_patterns) - up_count
    average_next_return = np.mean([pattern["next_return"] for pattern in similar_patterns])

    patterns_text = "\n".join([
        f"  - {pattern['date']}: next day {pattern['direction']} "
        f"({pattern['next_return']:.3%}) | similarity: {pattern['similarity_score']:.3f}"
        for pattern in similar_patterns
    ])

    prompt = f"""You are a quantitative analyst using vector similarity pattern matching.

Stock: {ticker}
Most recent {WINDOW_SIZE}-day daily returns: {recent_text}

Pinecone cosine similarity search found 5 historically similar price patterns.
What happened the day AFTER each of those patterns:
{patterns_text}

Statistics:
- UP moves after similar patterns: {up_count}/5
- DOWN moves after similar patterns: {down_count}/5
- Average next-day return in similar situations: {average_next_return:.3%}

Provide:
1. Direction: UP / DOWN / NEUTRAL
2. Expected magnitude (% range)
3. Confidence: Low / Medium / High
4. One-sentence reasoning

Be concise."""

    with mlflow.start_run():
        mlflow.log_param("ticker", ticker)
        mlflow.log_param("window_size", WINDOW_SIZE)
        mlflow.log_metric("up_count", up_count)
        mlflow.log_metric("down_count", down_count)
        mlflow.log_metric("average_next_return", average_next_return)

        response = llm.invoke([HumanMessage(content=prompt)])
        prediction_text = response.content

        mlflow.log_text(prediction_text, "prediction.txt")

    return {**state, "prediction": prediction_text, "status": "complete"}

# ── LangGraph orchestrator ─────────────────────────────────────────────────────

def build_graph():
    workflow = StateGraph(StockAgentState)

    workflow.add_node("data_loader", data_loader_agent)
    workflow.add_node("embedding", embedding_agent)
    workflow.add_node("query", query_agent)
    workflow.add_node("prediction", prediction_agent)

    workflow.set_entry_point("data_loader")
    workflow.add_edge("data_loader", "embedding")
    workflow.add_edge("embedding", "query")
    workflow.add_edge("query", "prediction")
    workflow.add_edge("prediction", END)

    return workflow.compile()

# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mlflow.set_experiment("stock-pattern-analysis")

    graph = build_graph()

    initial_state: StockAgentState = {
        "ticker": "AAPL",
        "raw_data": None,
        "patterns": [],
        "similar_patterns": [],
        "prediction": "",
        "status": "start",
    }

    print("=" * 60)
    print("  STOCK PATTERN ANALYSIS -Multi-Agent System")
    print("  LangGraph + Pinecone + Claude API + MLflow")
    print("=" * 60)

    result = graph.invoke(initial_state)

    print("\n" + "=" * 60)
    print("  PREDICTION:")
    print("=" * 60)
    print(result["prediction"])
