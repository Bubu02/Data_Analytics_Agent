import os
import sys
import pandas as pd
from fastapi.testclient import TestClient

import analytics_engine
from app import app, state

def test_engine():
    df = pd.DataFrame({
        "node_id": ["ND-1", "ND-2", "ND-3", "ND-4"],
        "metric_val": [0.95, 0.88, 0.42, 0.99],
        "category": ["A", "B", "A", "C"]
    })
    
    ans_shape = analytics_engine.answer_simple_query(df, "what is the dataset shape?")
    assert "4 rows" in ans_shape and "3 columns" in ans_shape, f"Failed shape test: {ans_shape}"
    
    ans_cols = analytics_engine.answer_simple_query(df, "list columns")
    assert "node_id" in ans_cols and "metric_val" in ans_cols, f"Failed cols test: {ans_cols}"

    summary = analytics_engine.route_query(df, "", "/summary")
    assert "4 rows × 3 columns" in summary, f"Failed summary test: {summary}"

    help_txt = analytics_engine.route_query(df, "", "/help")
    assert "/clean" in help_txt and "/analyze" in help_txt, f"Failed help test: {help_txt}"

    print("[PASS] Engine unit tests passed!")

def test_fastapi_endpoints():
    client = TestClient(app)

    res = client.get("/api/v1/health")
    assert res.status_code == 200, f"Health check failed: {res.json()}"

    csv_data = "timestamp,node_id,metric_val\n16:42:01,ND-77A,0.9982\n16:42:02,ND-88X,0.4012\n"
    files = {"file": ("test_sample.csv", csv_data, "text/csv")}
    upload_res = client.post("/api/v1/dataset/upload", files=files)
    assert upload_res.status_code == 200, f"Upload failed: {upload_res.json()}"
    assert upload_res.json()["rows"] == 2, f"Incorrect row count: {upload_res.json()}"

    prev_res = client.get("/api/v1/dataset/preview")
    assert prev_res.status_code == 200
    assert len(prev_res.json()["data"]) == 2

    chat_res = client.post("/api/v1/copilot/chat", json={"message": "/summary"})
    assert chat_res.status_code == 200
    assert "2 rows" in chat_res.json()["response"]

    print("[PASS] FastAPI endpoints tests passed!")

if __name__ == "__main__":
    test_engine()
    test_fastapi_endpoints()
