"""
Stock Pattern Analysis — Multi-Agent System
============================================
Architecture: 6 specialized agents orchestrated by LangGraph.
Each agent receives the shared state, does one job, and passes
the updated state to the next agent in the pipeline.

Pipeline:
  DataLoader -> Embedder -> QueryAgent -> Predictor
             -> ReversalAgent -> ChartPatternAgent

Storage:
  - Pinecone: vector database for price pattern similarity search
  - MLflow:   experiment tracking — one run per agent per ticker
"""

import json
import os

import mlflow
import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from pinecone import Pinecone, ServerlessSpec
from typing import Any, Dict, List, TypedDict

load_dotenv()

# ── State ──────────────────────────────────────────────────────────────────────
# LangGraph passes this dict between agents at every step.
# Each agent reads what it needs and writes its result back.

class StockAgentState(TypedDict):
    ticker: str
    raw_data: Any               # pandas DataFrame loaded by Agent 1
    patterns: List[Dict]        # all windows stored in Pinecone by Agent 2
    similar_patterns: List[Dict] # top-5 matches returned by Agent 3
    prediction: str             # next-day forecast from Agent 4 (LLM)
    reversal_analysis: str      # "when will it fall?" answer from Agent 5
    chart_pattern_analysis: str # double top / H&S analysis from Agent 6
    status: str                 # tracks which agent ran last

# ── Shared resources ───────────────────────────────────────────────────────────

llm = ChatAnthropic(model="claude-sonnet-4-6")

# Number of trading days in each price pattern window.
# 10 days = 2 calendar weeks — enough to capture short-term momentum shape.
WINDOW_SIZE = 10

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "stock-patterns")


def get_pinecone_index():
    """Connect to Pinecone and return the index, creating it if needed."""
    pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    existing_index_names = [index.name for index in pinecone_client.list_indexes()]
    if INDEX_NAME not in existing_index_names:
        print(f"  Creating Pinecone index '{INDEX_NAME}' (dimension={WINDOW_SIZE})...")
        # dimension must match WINDOW_SIZE — each vector is one 10-day return window
        pinecone_client.create_index(
            name=INDEX_NAME,
            dimension=WINDOW_SIZE,
            metric="cosine",            # cosine similarity ignores magnitude, compares shape
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pinecone_client.Index(INDEX_NAME)


# ── Agent 1: DataLoader ────────────────────────────────────────────────────────

def data_loader_agent(state: StockAgentState) -> StockAgentState:
    """
    Download 5 years of OHLCV data from Yahoo Finance.
    Compute daily returns (pct_change) — returns are scale-invariant,
    which makes pattern comparison valid across different price levels.
    """
    ticker = state["ticker"]
    print(f"\n[Agent 1 - DataLoader] Downloading {ticker} (5 years)...")

    stock = yf.Ticker(ticker)
    data = stock.history(period="5y").reset_index()
    data.to_csv(f"{ticker}_data.csv", index=False)

    # pct_change: converts absolute prices to relative daily moves (+2%, -1%, etc.)
    # dropna: removes the first row which has no previous day to compare against
    data["return"] = data["Close"].pct_change()
    data = data.dropna().reset_index(drop=True)

    print(f"[Agent 1 - DataLoader] {len(data)} trading days loaded -> {ticker}_data.csv")
    return {**state, "raw_data": data, "status": "data_loaded"}


# ── Agent 2: Embedder ──────────────────────────────────────────────────────────

def embedding_agent(state: StockAgentState) -> StockAgentState:
    """
    Slice the return series into overlapping 10-day windows.
    Each window becomes a vector (embedding) stored in Pinecone.

    Why normalize by std?
    Cosine similarity compares direction (shape), not magnitude.
    Normalizing removes the effect of volatility level so we match
    the SHAPE of the pattern, not the size of the moves.
    """
    data = state["raw_data"]
    print(f"\n[Agent 2 - Embedder] Building {WINDOW_SIZE}-day patterns -> Pinecone...")

    pinecone_index = get_pinecone_index()

    # Clear previous ticker's data so we don't mix patterns across tickers
    try:
        pinecone_index.delete(delete_all=True)
    except Exception:
        pass

    patterns = []
    batch = []

    for index in range(WINDOW_SIZE, len(data) - 1):
        window_returns = data["return"].iloc[index - WINDOW_SIZE : index].values
        next_return = float(data["return"].iloc[index])   # label: what happened next day
        date_str = str(data["Date"].iloc[index].date())

        # Normalize: divide by std so cosine similarity compares shape, not scale
        std_value = np.std(window_returns) + 1e-8         # 1e-8 avoids division by zero
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

        # Upsert in batches of 100 to avoid Pinecone request size limits
        if len(batch) >= 100:
            pinecone_index.upsert(vectors=batch)
            batch = []

    if batch:
        pinecone_index.upsert(vectors=batch)

    print(f"[Agent 2 - Embedder] {len(patterns)} patterns stored in Pinecone.")
    return {**state, "patterns": patterns, "status": "embedded"}


# ── Agent 3: QueryAgent ────────────────────────────────────────────────────────

def query_agent(state: StockAgentState) -> StockAgentState:
    """
    Take the most recent WINDOW_SIZE days, normalize them the same way
    as stored patterns, and run a cosine similarity search in Pinecone.
    Returns the 5 most similar historical moments.
    """
    data = state["raw_data"]
    print(f"\n[Agent 3 - QueryAgent] Finding 5 most similar historical patterns...")

    recent_returns = data["return"].iloc[-WINDOW_SIZE:].values
    std_value = np.std(recent_returns) + 1e-8
    query_embedding = (recent_returns / std_value).tolist()

    pinecone_index = get_pinecone_index()
    results = pinecone_index.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True,   # we need date, next_return, direction from metadata
    )

    similar_patterns = [
        {
            "date": match.metadata["date"],
            "next_return": match.metadata["next_return"],
            "direction": match.metadata["direction"],
            "similarity_score": match.score,   # 1.0 = identical shape, 0.0 = opposite
        }
        for match in results.matches
    ]

    print(f"[Agent 3 - QueryAgent] Results:")
    for pattern in similar_patterns:
        print(
            f"  -> {pattern['date']}: "
            f"next day {pattern['direction']} ({pattern['next_return']:.3%}) "
            f"| similarity: {pattern['similarity_score']:.3f}"
        )

    return {**state, "similar_patterns": similar_patterns, "status": "queried"}


# ── Agent 4: Predictor ─────────────────────────────────────────────────────────

def prediction_agent(state: StockAgentState) -> StockAgentState:
    """
    Pass the 5 similar patterns to Claude as context.
    The LLM acts as a quantitative analyst: it reads the historical outcomes
    and reasons about the most likely direction for tomorrow.
    MLflow logs every run so predictions are reproducible and comparable.
    """
    ticker = state["ticker"]
    similar_patterns = state["similar_patterns"]
    data = state["raw_data"]

    print(f"\n[Agent 4 - Predictor] Analyzing patterns with Claude...")

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

    with mlflow.start_run(run_name=f"{ticker}-prediction"):
        mlflow.log_param("ticker", ticker)
        mlflow.log_param("window_size", WINDOW_SIZE)
        mlflow.log_metric("up_count", up_count)
        mlflow.log_metric("down_count", down_count)
        mlflow.log_metric("average_next_return", average_next_return)

        response = llm.invoke([HumanMessage(content=prompt)])
        prediction_text = response.content

        mlflow.log_text(prediction_text, "prediction.txt")

    return {**state, "prediction": prediction_text, "status": "complete"}


# ── Agent 5: ReversalAgent ────────────────────────────────────────────────────

def reversal_agent(state: StockAgentState) -> StockAgentState:
    """
    Answer: "When will this stock start falling?"

    Method: find all historical 10-day windows where cumulative return > 3%
    (similar strong momentum), then measure how many days until price dropped
    3% from that peak. Returns distribution statistics to the LLM.
    """
    data = state["raw_data"]
    ticker = state["ticker"]
    reversal_threshold = -0.03   # define "reversal" as -3% from peak

    print(f"\n[Agent 5 - ReversalAgent] Finding historical momentum reversals...")

    # Current momentum: compound return over last 10 days
    recent_cumulative_return = (1 + data["return"].iloc[-10:]).prod() - 1

    days_until_reversal_list = []

    for index in range(10, len(data) - 20):
        window_cumulative_return = (1 + data["return"].iloc[index - 10:index]).prod() - 1

        if window_cumulative_return > 0.03:
            peak_price = data["Close"].iloc[index]
            # Scan forward up to 20 days to find when -3% drawdown occurs
            for forward_index in range(1, 21):
                if index + forward_index >= len(data):
                    break
                forward_price = data["Close"].iloc[index + forward_index]
                drawdown = (forward_price - peak_price) / peak_price
                if drawdown <= reversal_threshold:
                    days_until_reversal_list.append(forward_index)
                    break

    prompt_context = ""
    if days_until_reversal_list:
        average_days = np.mean(days_until_reversal_list)
        median_days = np.median(days_until_reversal_list)
        min_days = min(days_until_reversal_list)
        max_days = max(days_until_reversal_list)
        reversal_count = len(days_until_reversal_list)
        prompt_context = f"""
Historical strong momentum periods (>3% in 10 days) in {ticker} over 5 years:
- Found {reversal_count} similar momentum surges
- Average days until -3% reversal: {average_days:.1f} days
- Median: {median_days:.1f} days
- Fastest reversal: {min_days} days
- Slowest reversal: {max_days} days
- Current 10-day momentum: {recent_cumulative_return:.2%}
"""
    else:
        prompt_context = f"Current 10-day momentum: {recent_cumulative_return:.2%}. No similar historical momentum periods found."

    prompt = f"""You are a quantitative analyst. A user's wife asked: "When will {ticker} start falling?"

{prompt_context}

Based on this historical reversal data, answer in simple terms:
1. Typical time window before a reversal (in trading days and calendar days)
2. What would trigger the reversal (what to watch for)
3. One-sentence answer suitable for a non-trader

Keep it short and plain-language."""

    with mlflow.start_run(run_name=f"{ticker}-reversal"):
        mlflow.log_param("ticker", ticker)
        mlflow.log_param("reversal_threshold_pct", abs(reversal_threshold) * 100)
        if days_until_reversal_list:
            mlflow.log_metric("average_days_to_reversal", float(np.mean(days_until_reversal_list)))
            mlflow.log_metric("median_days_to_reversal", float(np.median(days_until_reversal_list)))
            mlflow.log_metric("min_days_to_reversal", float(min(days_until_reversal_list)))
            mlflow.log_metric("max_days_to_reversal", float(max(days_until_reversal_list)))
            mlflow.log_metric("momentum_surge_count", len(days_until_reversal_list))

        response = llm.invoke([HumanMessage(content=prompt)])
        reversal_text = response.content

        mlflow.log_text(reversal_text, "reversal_for_wife.txt")

    print(f"[Agent 5 - ReversalAgent] Done.")
    return {**state, "reversal_analysis": reversal_text, "status": "reversal_done"}


# ── Agent 6: ChartPatternAgent ─────────────────────────────────────────────────

def chart_pattern_agent(state: StockAgentState) -> StockAgentState:
    """
    Detect classic reversal patterns in the last 60 trading days:
    - Double Top: two peaks at similar price (~3% tolerance) with a trough between
    - Head and Shoulders: three peaks, middle highest, shoulders roughly equal

    For each detected pattern, compute:
    - Neckline: the support level that confirms the pattern when broken
    - Price target: neckline minus pattern height (standard projection)
    - Confirmed: whether current price has already broken the neckline
    """
    data = state["raw_data"]
    ticker = state["ticker"]
    lookback_days = 60

    print(f"\n[Agent 6 - ChartPatternAgent] Scanning for chart patterns (last {lookback_days} days)...")

    recent_data = data.tail(lookback_days).reset_index(drop=True)
    prices = recent_data["Close"].values

    # Local peak: higher than all 5 days on each side
    # Local trough: lower than all 5 days on each side
    peak_window = 5
    peaks = []
    troughs = []

    for index in range(peak_window, len(prices) - peak_window):
        left_slice = prices[index - peak_window : index]
        right_slice = prices[index + 1 : index + peak_window + 1]
        if prices[index] > max(left_slice) and prices[index] > max(right_slice):
            peaks.append({
                "index": index,
                "price": float(prices[index]),
                "date": str(recent_data["Date"].iloc[index].date())
            })
        if prices[index] < min(left_slice) and prices[index] < min(right_slice):
            troughs.append({
                "index": index,
                "price": float(prices[index]),
                "date": str(recent_data["Date"].iloc[index].date())
            })

    detected_patterns = []

    # Double Top: two peaks within 3% of each other, trough between them
    if len(peaks) >= 2:
        for first_peak_index in range(len(peaks) - 1):
            for second_peak_index in range(first_peak_index + 1, len(peaks)):
                first_peak = peaks[first_peak_index]
                second_peak = peaks[second_peak_index]
                price_difference_pct = abs(first_peak["price"] - second_peak["price"]) / first_peak["price"]

                if price_difference_pct <= 0.03:
                    middle_troughs = [
                        trough for trough in troughs
                        if first_peak["index"] < trough["index"] < second_peak["index"]
                    ]
                    if middle_troughs:
                        neckline_price = min(trough["price"] for trough in middle_troughs)
                        pattern_height = ((first_peak["price"] + second_peak["price"]) / 2) - neckline_price
                        target_price = neckline_price - pattern_height  # classic measured move
                        current_price = float(prices[-1])
                        detected_patterns.append({
                            "name": "Double Top",
                            "signal": "BEARISH",
                            "first_peak_date": first_peak["date"],
                            "first_peak_price": first_peak["price"],
                            "second_peak_date": second_peak["date"],
                            "second_peak_price": second_peak["price"],
                            "neckline": neckline_price,
                            "target_price": target_price,
                            "current_price": current_price,
                            "confirmed": current_price < neckline_price,
                        })

    # Head and Shoulders: 3 peaks, middle (head) is highest, shoulders within 4%
    if len(peaks) >= 3:
        for head_index in range(1, len(peaks) - 1):
            left_shoulder = peaks[head_index - 1]
            head = peaks[head_index]
            right_shoulder = peaks[head_index + 1]
            shoulder_difference_pct = abs(left_shoulder["price"] - right_shoulder["price"]) / left_shoulder["price"]

            if (head["price"] > left_shoulder["price"]
                    and head["price"] > right_shoulder["price"]
                    and shoulder_difference_pct <= 0.04):
                middle_troughs = [
                    trough for trough in troughs
                    if left_shoulder["index"] < trough["index"] < right_shoulder["index"]
                ]
                if len(middle_troughs) >= 2:
                    neckline_price = np.mean([trough["price"] for trough in middle_troughs])
                    pattern_height = head["price"] - neckline_price
                    target_price = neckline_price - pattern_height
                    current_price = float(prices[-1])
                    detected_patterns.append({
                        "name": "Head and Shoulders",
                        "signal": "BEARISH",
                        "left_shoulder_date": left_shoulder["date"],
                        "head_date": head["date"],
                        "right_shoulder_date": right_shoulder["date"],
                        "neckline": neckline_price,
                        "target_price": target_price,
                        "current_price": current_price,
                        "confirmed": current_price < neckline_price,
                    })

    current_price = float(prices[-1])

    if detected_patterns:
        patterns_description = "\n".join([
            f"- {pattern['name']} ({pattern['signal']}): "
            f"neckline ${pattern['neckline']:.2f}, "
            f"target ${pattern['target_price']:.2f}, "
            f"current ${pattern['current_price']:.2f}, "
            f"confirmed={'YES' if pattern['confirmed'] else 'NOT YET'}"
            for pattern in detected_patterns
        ])
        peaks_info = f"Found {len(peaks)} peaks and {len(troughs)} troughs in last {lookback_days} days."
    else:
        patterns_description = "No classic reversal patterns detected in the last 60 days."
        peaks_info = f"Found {len(peaks)} peaks and {len(troughs)} troughs — no matching patterns."

    prompt = f"""You are a technical analyst.

Stock: {ticker} | Current price: ${current_price:.2f}
Analysis window: last {lookback_days} trading days
{peaks_info}

Detected chart patterns:
{patterns_description}

Provide:
1. Pattern status (confirmed/forming/none)
2. Price target if pattern completes
3. What price level to watch (neckline / confirmation level)
4. One plain-language sentence for a non-trader

Be concise."""

    with mlflow.start_run(run_name=f"{ticker}-chart-patterns"):
        mlflow.log_param("ticker", ticker)
        mlflow.log_param("lookback_days", lookback_days)
        mlflow.log_metric("peaks_found", len(peaks))
        mlflow.log_metric("troughs_found", len(troughs))
        mlflow.log_metric("patterns_found", len(detected_patterns))
        mlflow.log_metric("current_price", current_price)

        response = llm.invoke([HumanMessage(content=prompt)])
        chart_pattern_text = response.content

        mlflow.log_text(chart_pattern_text, "chart_patterns.txt")
        if detected_patterns:
            mlflow.log_text(json.dumps(detected_patterns, indent=2), "detected_patterns.json")

    print(f"[Agent 6 - ChartPatternAgent] Found {len(detected_patterns)} pattern(s): "
          f"{[p['name'] for p in detected_patterns] or 'none'}")
    return {**state, "chart_pattern_analysis": chart_pattern_text, "status": "chart_done"}


# ── LangGraph orchestrator ─────────────────────────────────────────────────────

def build_graph():
    """
    Wire the 6 agents into a sequential pipeline using LangGraph.
    Each node is a Python function that takes state and returns updated state.
    LangGraph handles the execution loop and state passing between nodes.
    """
    workflow = StateGraph(StockAgentState)

    workflow.add_node("data_loader", data_loader_agent)
    workflow.add_node("embedding", embedding_agent)
    workflow.add_node("query", query_agent)
    workflow.add_node("prediction", prediction_agent)
    workflow.add_node("reversal", reversal_agent)
    workflow.add_node("chart_pattern", chart_pattern_agent)

    workflow.set_entry_point("data_loader")
    workflow.add_edge("data_loader", "embedding")
    workflow.add_edge("embedding", "query")
    workflow.add_edge("query", "prediction")
    workflow.add_edge("prediction", "reversal")
    workflow.add_edge("reversal", "chart_pattern")
    workflow.add_edge("chart_pattern", END)

    return workflow.compile()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mlflow.set_experiment("stock-pattern-analysis")

    graph = build_graph()

    for ticker_symbol in ["AMZN"]:
        initial_state: StockAgentState = {
            "ticker": ticker_symbol,
            "raw_data": None,
            "patterns": [],
            "similar_patterns": [],
            "prediction": "",
            "reversal_analysis": "",
            "chart_pattern_analysis": "",
            "status": "start",
        }

        print("\n" + "=" * 60)
        print(f"  ANALYZING: {ticker_symbol} (5 years)")
        print("  LangGraph + Pinecone + Claude API + MLflow")
        print("=" * 60)

        result = graph.invoke(initial_state)

        print("\n" + "=" * 60)
        print(f"  PREDICTION FOR {ticker_symbol}:")
        print("=" * 60)
        print(result["prediction"])

        print("\n" + "=" * 60)
        print(f"  WHEN WILL {ticker_symbol} FALL? (for wife)")
        print("=" * 60)
        print(result["reversal_analysis"])

        print("\n" + "=" * 60)
        print(f"  CHART PATTERNS FOR {ticker_symbol}:")
        print("=" * 60)
        print(result["chart_pattern_analysis"])
