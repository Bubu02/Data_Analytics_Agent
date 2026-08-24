import os
import sys
from typing import Optional
from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

# Initialize FastAPI App
app = FastAPI(
    title="Stitch AI Data Workspace",
    description="Precision Analytics Engine & Multi-Agent Workspace powered by FastAPI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Mount Static Files (CSS, JS, Images)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Setup Jinja2 Templates Engine
templates = Jinja2Templates(directory=TEMPLATES_DIR)


# =====================================================================
# PAGE ROUTES (HTML VIEWS)
# =====================================================================

@app.get("/", response_class=HTMLResponse, summary="Master Workspace Hub")
async def read_root(request: Request):
    """Renders the master workspace hub with interactive dashboard viewport tabs."""
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/homepage", response_class=HTMLResponse, summary="Analytica AI Homepage (16:9)")
async def read_homepage(request: Request):
    """Renders Dashboard 1: Technical Noir Homepage Landing."""
    return templates.TemplateResponse(request=request, name="homepage.html")


@app.get("/onboarding", response_class=HTMLResponse, summary="Dataset Connection & Onboarding")
async def read_onboarding(request: Request):
    """Renders Dashboard 2: Connect Dataset & Drag-and-Drop Ingestion."""
    return templates.TemplateResponse(request=request, name="onboarding_upload.html")


@app.get("/workspace-empty", response_class=HTMLResponse, summary="Empty Workspace & AI Copilot")
async def read_workspace_empty(request: Request):
    """Renders Dashboard 3: Empty Workspace with Analytica Copilot."""
    return templates.TemplateResponse(request=request, name="workspace_empty.html")


@app.get("/workspace-populated", response_class=HTMLResponse, summary="Populated Analytics Workspace")
async def read_workspace_populated(request: Request):
    """Renders Dashboard 4: Full Analytics Dashboard with Models & KPIs."""
    return templates.TemplateResponse(request=request, name="workspace_populated.html")


# =====================================================================
# API ENDPOINTS
# =====================================================================

@app.get("/api/v1/health", summary="Engine Health Check")
async def health_check():
    """Returns precision engine health diagnostics."""
    return {
        "status": "online",
        "engine": "Stitch AI Precision Engine",
        "framework": "FastAPI",
        "version": "1.0.0"
    }


@app.get("/api/v1/dataset/metrics", summary="Get Active Dataset Metrics")
async def get_dataset_metrics():
    """Returns high-density telemetry metrics for the active dataset."""
    return {
        "total_rows": 12450,
        "total_columns": 18,
        "missing_data_pct": 0.4,
        "quality_score": 96,
        "memory_allocation_mb": 1.8,
        "health_status": "Optimal",
        "numerical_features": 8,
        "categorical_features": 10
    }


@app.post("/api/v1/dataset/upload", summary="Upload Dataset File")
async def upload_dataset(file: UploadFile = File(...)):
    """Receives and validates dataset files (CSV, JSON, Parquet)."""
    allowed_extensions = {".csv", ".json", ".parquet"}
    filename = file.filename or "dataset"
    ext = os.path.splitext(filename)[1].lower()

    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed formats: CSV, JSON, Parquet."
        )

    content = await file.read()
    file_size_kb = round(len(content) / 1024, 2)

    return {
        "status": "success",
        "filename": filename,
        "size_kb": file_size_kb,
        "message": f"Dataset '{filename}' successfully ingested into Precision Engine."
    }


@app.post("/api/v1/copilot/chat", summary="Query Analytica AI Copilot")
async def copilot_chat(query: dict):
    """Process natural language queries for automated insights."""
    user_message = query.get("message", "").strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Query message cannot be empty.")

    return {
        "query": user_message,
        "response": f"Analytica Copilot processed: '{user_message}'. Analysis shows positive revenue trajectory in North America with 96% confidence.",
        "confidence": 0.96,
        "suggested_actions": ["Filter by Region", "Export Feature Correlations", "Run Churn Prediction Model"]
    }


# =====================================================================
# SERVER RUNNER
# =====================================================================

if __name__ == "__main__":
    print("🚀 Starting Stitch AI Data Workspace FastAPI Server on http://127.0.0.1:8000 ...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
