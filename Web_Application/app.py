import os
import sys
import io
from typing import Optional
import pandas as pd
from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

import analytics_engine

# Initialize FastAPI App
app = FastAPI(
    title="Analytica AI",
    description="Precision Analytics Engine & Multi-Agent Workspace powered by FastAPI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global state for the active dataset (In a real app, use a database or cache)
class GlobalState:
    df: Optional[pd.DataFrame] = None
    filename: Optional[str] = None

state = GlobalState()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Mount Static Files (CSS, JS, Images)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Setup Jinja2 Templates Engine
templates = Jinja2Templates(directory=TEMPLATES_DIR)


# =====================================================================
# PAGE ROUTES (HTML VIEWS) — All served via the persistent layout shell.
# The shell renders the initial page content server-side (no flash), then
# subsequent navigation swaps only the #page-content div client-side.
# =====================================================================

@app.get("/", response_class=HTMLResponse, summary="Root redirect")
async def read_root(request: Request):
    """Redirects root URL to the homepage."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/homepage", status_code=302)


@app.get("/homepage", response_class=HTMLResponse, summary="Analytica AI Homepage")
async def read_homepage(request: Request):
    """Renders the shell with Homepage as the initial content (SSR first load)."""
    return templates.TemplateResponse(
        request=request, name="shell.html", context={"initial_page": "homepage"}
    )


@app.get("/onboarding", response_class=HTMLResponse, summary="Dataset Connection & Onboarding")
async def read_onboarding(request: Request):
    """Renders the shell with Onboarding as the initial content (SSR first load)."""
    return templates.TemplateResponse(
        request=request, name="shell.html", context={"initial_page": "onboarding"}
    )


@app.get("/workspace-empty", response_class=HTMLResponse, summary="Empty Workspace & AI Copilot")
async def read_workspace_empty(request: Request):
    """Renders the shell with Empty Workspace as the initial content (SSR first load)."""
    return templates.TemplateResponse(
        request=request, name="shell.html", context={"initial_page": "workspace-empty"}
    )


@app.get("/workspace-populated", response_class=HTMLResponse, summary="Populated Analytics Workspace")
async def read_workspace_populated(request: Request):
    """Renders the shell with Populated Workspace as the initial content (SSR first load)."""
    return templates.TemplateResponse(
        request=request, name="shell.html", context={"initial_page": "workspace-populated"}
    )


# =====================================================================
# PARTIAL ROUTES — Return only the inner content HTML fragment.
# Called by the client-side router when navigating between pages.
# The nav and sidebar are NOT included — only the #page-content area.
# =====================================================================

@app.get("/partial/homepage", response_class=HTMLResponse, summary="Homepage content fragment")
async def partial_homepage(request: Request):
    """Returns only the homepage content HTML (no shell) for client-side routing."""
    return templates.TemplateResponse(request=request, name="partials/homepage.html")


@app.get("/partial/onboarding", response_class=HTMLResponse, summary="Onboarding content fragment")
async def partial_onboarding(request: Request):
    """Returns only the onboarding content HTML (no shell) for client-side routing."""
    return templates.TemplateResponse(request=request, name="partials/onboarding.html")


@app.get("/partial/workspace-empty", response_class=HTMLResponse, summary="Empty workspace content fragment")
async def partial_workspace_empty(request: Request):
    """Returns only the empty workspace content HTML (no shell) for client-side routing."""
    return templates.TemplateResponse(request=request, name="partials/workspace-empty.html")


@app.get("/partial/workspace-populated", response_class=HTMLResponse, summary="Populated workspace content fragment")
async def partial_workspace_populated(request: Request):
    """Returns only the populated workspace content HTML (no shell) for client-side routing."""
    return templates.TemplateResponse(request=request, name="partials/workspace-populated.html")


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
    if state.df is None:
        return {
            "total_rows": 0,
            "total_columns": 0,
            "missing_data_pct": 0,
            "quality_score": 0,
            "memory_allocation_mb": 0,
            "health_status": "No Dataset",
            "numerical_features": 0,
            "categorical_features": 0
        }

    df = state.df
    total_cells = df.size
    missing_cells = df.isnull().sum().sum()
    missing_pct = round((missing_cells / total_cells) * 100, 2) if total_cells > 0 else 0
    
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    return {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_data_pct": missing_pct,
        "quality_score": 100 - int(missing_pct),
        "memory_allocation_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
        "health_status": "Optimal" if missing_pct < 5 else "Degraded",
        "numerical_features": len(num_cols),
        "categorical_features": len(cat_cols)
    }


@app.post("/api/v1/dataset/upload", summary="Upload Dataset File")
async def upload_dataset(file: UploadFile = File(...)):
    """Receives and validates dataset files (CSV, JSON, Parquet)."""
    allowed_extensions = {".csv", ".json", ".parquet"}
    filename = file.filename or "dataset.csv"
    ext = os.path.splitext(filename)[1].lower()

    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed: CSV, JSON, Parquet."
        )

    content = await file.read()

    try:
        if ext == ".csv":
            df = pd.read_csv(io.BytesIO(content))
        elif ext == ".json":
            df = pd.read_json(io.BytesIO(content))
        elif ext == ".parquet":
            df = pd.read_parquet(io.BytesIO(content))
        else:
            raise ValueError(f"Unsupported extension: {ext}")

        state.df = df
        state.filename = filename
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {str(e)}")

    # Build column metadata
    columns_meta = [
        {"name": col, "dtype": str(df[col].dtype)}
        for col in df.columns
    ]

    # First 10 rows, converting NaN to None for JSON safety
    preview_df = df.head(10).where(pd.notnull(df.head(10)), None)
    preview_rows = preview_df.to_dict(orient="records")

    return {
        "status": "success",
        "filename": filename,
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": columns_meta,
        "preview": preview_rows,
        "message": f"Dataset '{filename}' successfully ingested."
    }


@app.get("/api/v1/dataset/preview", summary="Get Dataset Preview")
async def get_dataset_preview():
    """Returns the first 50 rows of the active dataset."""
    if state.df is None:
        return {"data": [], "columns": []}
    
    # Convert NaNs to None for JSON serialization
    preview_df = state.df.head(50).where(pd.notnull(state.df), None)
    
    return {
        "data": preview_df.to_dict(orient="records"),
        "columns": state.df.columns.tolist()
    }


@app.post("/api/v1/copilot/chat", summary="Query Analytica AI Copilot")
async def copilot_chat(query: dict):
    """Process natural language queries for automated insights."""
    user_message = query.get("message", "").strip()
    api_key = query.get("api_key", "").strip() or os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get("GOOGLE_API_KEY", "").strip()
    model = query.get("model", "gemini-2.5-flash").strip()

    if not user_message:
        raise HTTPException(status_code=400, detail="Query message cannot be empty.")

    if state.df is None:
        return {
            "query": user_message,
            "response": "⚠️ No active dataset loaded. Please upload a dataset on the **Connect Data** page to begin analysis.",
            "status": "warning"
        }

    try:
        response = analytics_engine.route_query(
            df=state.df,
            api_key=api_key,
            prompt=user_message,
            model_name=model
        )
        return {
            "query": user_message,
            "response": response,
            "status": "success"
        }
    except Exception as e:
        return {
            "query": user_message,
            "response": f"I encountered an error: {str(e)}",
            "status": "error"
        }


# =====================================================================
# SERVER RUNNER
# =====================================================================

if __name__ == "__main__":
    print("🚀 Starting Stitch AI Data Workspace FastAPI Server on http://127.0.0.1:8000 ...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
