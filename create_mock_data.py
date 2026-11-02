import os
import json
import pandas as pd

os.makedirs("data", exist_ok=True)

# 1. Unstructured News Data for Big Data & GenAI RAG
raw_news = [
    {
        "article_id": "doc_1",
        "company": "TechCorp",
        "title": "TechCorp Launches AI Cloud Suite at 15% Price Cut",
        "text": "TechCorp today announced its next-gen Enterprise AI Cloud Suite with aggressive pricing. They reduced annual enterprise licenses by 15% to capture enterprise market share from competitors.",
        "timestamp": "2026-03-01"
    },
    {
        "article_id": "doc_2",
        "company": "DataDynamics",
        "title": "DataDynamics Enterprise Q1 Expansion Strategy",
        "text": "DataDynamics expands into European markets with heavy investment in regional data centers. Analysts note potential regulatory compliance challenges with GDPR.",
        "timestamp": "2026-03-02"
    },
    {
        "article_id": "doc_3",
        "company": "TechCorp",
        "title": "TechCorp Reports Record Quarterly Cloud Revenue",
        "text": "Driven by its new pricing strategy, TechCorp reported a 28% increase in enterprise cloud adoption this quarter.",
        "timestamp": "2026-03-15"
    }
]

with open("data/raw_news.json", "w") as f:
    json.dump(raw_news, f, indent=4)

# 2. Structured Metadata for Core ML Model
metadata = pd.DataFrame({
    "sentiment_score": [0.85, -0.20, 0.90, 0.10, -0.65, 0.45, 0.75, -0.40],
    "mention_frequency": [45, 12, 88, 15, 60, 30, 52, 22],
    "competitor_activity_index": [0.9, 0.4, 0.95, 0.3, 0.8, 0.5, 0.7, 0.6],
    "high_impact_flag": [1, 0, 1, 0, 1, 0, 1, 0] # 1: Market Shifting, 0: Minor
})

metadata.to_csv("data/market_metadata.csv", index=False)
print("Mock datasets created successfully in /data folder!")