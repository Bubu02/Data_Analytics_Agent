import os
import re
import sys
from typing import Tuple, Optional

# Must be set BEFORE importing CrewAI
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

import pandas as pd

try:
    from crewai import Agent, Task, Crew, Process, LLM
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    Agent = Task = Crew = Process = LLM = None


# =====================================================================
# 1. GENERAL HELPERS
# =====================================================================

def get_task_output_text(task_output) -> str:
    """Safely extract text from CrewAI TaskOutput."""
    if hasattr(task_output, "raw"):
        return str(task_output.raw)
    return str(task_output)


def clean_response(text: Optional[str]) -> str:
    """Remove unnecessary leading/trailing whitespace."""
    if text is None:
        return ""
    return str(text).strip()


# =====================================================================
# 2. DATASET CONTEXT BUILDER
# =====================================================================

def build_dataset_context(df: pd.DataFrame, max_rows: int = 300) -> str:
    """
    Produce a context representation for the LLM.
    Small datasets: send complete dataframe text.
    Larger datasets: send metadata, samples, and descriptive statistics.
    """
    num_rows = len(df)
    num_cols = df.shape[1]
    columns = df.columns.tolist()
    data_types = df.dtypes.to_string()
    missing_values = df.isnull().sum().to_string()

    try:
        descriptive_stats = df.describe(include="all").to_string()
    except Exception:
        descriptive_stats = "Descriptive statistics unavailable."

    if num_rows <= max_rows:
        return (
            f"Dataset shape: {num_rows} rows x {num_cols} columns\n\n"
            f"Columns:\n{columns}\n\n"
            f"Complete dataset:\n"
            f"{df.to_string(index=False)}"
        )

    return (
        f"Dataset shape: {num_rows} rows x {num_cols} columns\n\n"
        f"Columns:\n{', '.join(map(str, columns))}\n\n"
        f"Data types:\n{data_types}\n\n"
        f"Missing values:\n{missing_values}\n\n"
        f"First 20 rows:\n"
        f"{df.head(20).to_string(index=False)}\n\n"
        f"Last 20 rows:\n"
        f"{df.tail(20).to_string(index=False)}\n\n"
        f"Descriptive statistics:\n"
        f"{descriptive_stats}"
    )


# =====================================================================
# 3. DIRECT PANDAS ANSWERS (DETERMINISTIC)
# =====================================================================

def answer_simple_query(df: pd.DataFrame, query: str) -> Optional[str]:
    """
    Answer obvious factual dataframe questions directly without LLM call.
    Returns string if direct answer found, None if LLM is required.
    """
    q = query.lower().strip()

    # Dimensions
    size_phrases = [
        "dataset size", "size of dataset", "dataset shape",
        "shape of dataset", "shape", "how big is the dataset"
    ]
    if any(phrase in q for phrase in size_phrases):
        return f"The dataset has **{df.shape[0]:,} rows** and **{df.shape[1]:,} columns**."

    if "how many rows" in q or "number of rows" in q:
        return f"The dataset has **{df.shape[0]:,} rows**."

    if "how many columns" in q or "number of columns" in q:
        return f"The dataset has **{df.shape[1]:,} columns**."

    # Columns
    if any(phrase in q for phrase in ["column names", "list columns", "what are the columns"]) or q == "columns":
        return "**Columns:**\n\n" + "\n".join(f"- `{col}`" for col in df.columns)

    # Missing values
    if any(phrase in q for phrase in ["missing values", "null values", "missing data"]):
        missing = df.isnull().sum()
        total = int(missing.sum())
        if total == 0:
            return "The dataset contains **no missing values**."
        affected = missing[missing > 0]
        details = "\n".join(f"- `{col}`: {int(cnt):,}" for col, cnt in affected.items())
        return f"The dataset contains **{total:,} missing values**.\n\n{details}"

    # Duplicates
    if "duplicate" in q:
        duplicates = int(df.duplicated().sum())
        return f"The dataset contains **{duplicates:,} duplicated rows**."

    # Data types
    if any(phrase in q for phrase in ["data types", "datatypes", "dtypes"]):
        lines = [f"- `{col}`: `{dtype}`" for col, dtype in df.dtypes.items()]
        return "**Column data types:**\n\n" + "\n".join(lines)

    # Numeric columns
    if "numeric columns" in q or "numerical columns" in q:
        numeric = df.select_dtypes(include="number").columns.tolist()
        if not numeric:
            return "The dataset contains no numeric columns."
        return "**Numeric columns:**\n\n" + "\n".join(f"- `{col}`" for col in numeric)

    # Categorical columns
    if any(phrase in q for phrase in ["categorical columns", "text columns", "object columns"]):
        categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if not categorical:
            return "The dataset contains no categorical/text columns."
        return "**Categorical columns:**\n\n" + "\n".join(f"- `{col}`" for col in categorical)

    return None


# =====================================================================
# 4. LLM INITIALIZATION
# =====================================================================

def create_llm(api_key: str, model_name: str = "gemini-2.5-flash"):
    """Create the CrewAI Gemini LLM instance."""
    if not CREWAI_AVAILABLE:
        raise RuntimeError("CrewAI package is not installed. Please install crewai via `pip install crewai` to run multi-agent queries.")

    clean_key = api_key.strip()
    if not clean_key:
        raise ValueError("API Key is required to run AI Agent queries.")

    os.environ["GEMINI_API_KEY"] = clean_key
    os.environ["GOOGLE_API_KEY"] = clean_key

    if model_name.startswith("gemini/"):
        llm_model_str = model_name
    else:
        llm_model_str = f"gemini/{model_name}"

    try:
        return LLM(model=llm_model_str, api_key=clean_key)
    except Exception as e:
        message = str(e).strip() or repr(e)
        raise RuntimeError(f"LLM initialization failed: {message}")


# =====================================================================
# 5. AGENT FACTORY
# =====================================================================

def create_agents(llm: LLM) -> Tuple[Agent, Agent, Agent]:
    """Create the three specialist agents."""
    data_cleaner = Agent(
        role="Data Cleaner & Preprocessor",
        goal="Inspect data quality, structure, missingness, duplicates, invalid values and readiness for analysis.",
        backstory="You are a meticulous senior data engineer specializing in dataset validation and preprocessing.",
        llm=llm,
        verbose=False,
    )

    data_analyst = Agent(
        role="Senior Data Analyst",
        goal="Answer quantitative questions accurately and directly using the supplied dataset.",
        backstory="You are a senior quantitative data analyst. You prioritize correct calculations and concise answers.",
        llm=llm,
        verbose=False,
    )

    business_strategist = Agent(
        role="Business Strategist",
        goal="Translate relevant quantitative findings into practical business implications and recommendations.",
        backstory="You are a senior strategy consultant who bases recommendations strictly on supplied evidence.",
        llm=llm,
        verbose=False,
    )

    return data_cleaner, data_analyst, business_strategist


# =====================================================================
# 6. SINGLE-AGENT EXECUTION
# =====================================================================

def run_single_agent(
    df: pd.DataFrame,
    api_key: str,
    query: str,
    mode: str = "analyst",
    model_name: str = "gemini-2.5-flash",
) -> str:
    """
    Run only one specialist agent (cleaner, analyst, or strategy).
    """
    cloud_llm = create_llm(api_key, model_name)
    cleaner, analyst, strategist = create_agents(cloud_llm)
    data_context = build_dataset_context(df)

    if mode == "cleaner":
        task = Task(
            description=f"Inspect the supplied dataset regarding the user's request.\n\nUser request:\n{query}\n\nDataset:\n{data_context}\n\nFocus on missing values, duplicates, invalid values, and data readiness. Be concise.",
            expected_output="A direct data-quality answer at the requested level of detail.",
            agent=cleaner,
        )
        active_agent = cleaner
    elif mode == "strategy":
        task = Task(
            description=f"Answer the user's strategic business question using evidence from the dataset.\n\nUser request:\n{query}\n\nDataset:\n{data_context}\n\nGround recommendations strictly in evidence.",
            expected_output="A concise, evidence-based strategic response.",
            agent=strategist,
        )
        active_agent = strategist
    else:
        task = Task(
            description=f"Answer the user's question using the dataset.\n\nUser question:\n{query}\n\nDataset:\n{data_context}\n\nAnswer ONLY what was asked concisely. Never invent values.",
            expected_output="A direct, concise answer addressing the question.",
            agent=analyst,
        )
        active_agent = analyst

    crew = Crew(
        agents=[active_agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )

    try:
        output = crew.kickoff()
        if hasattr(output, "tasks_output") and output.tasks_output:
            return clean_response(get_task_output_text(output.tasks_output[0]))
        return clean_response(get_task_output_text(output))
    except Exception as e:
        message = str(e).strip() or repr(e)
        raise RuntimeError(f"Crew execution failed: {message}")


# =====================================================================
# 7. FULL THREE-AGENT WORKFLOW
# =====================================================================

def run_full_analysis(
    df: pd.DataFrame,
    api_key: str,
    query: str,
    model_name: str = "gemini-2.5-flash",
) -> str:
    """Runs full 3-agent pipeline: Cleaner -> Analyst -> Strategist."""
    cloud_llm = create_llm(api_key, model_name)
    cleaner, analyst, strategist = create_agents(cloud_llm)
    data_context = build_dataset_context(df)

    clean_task = Task(
        description=f"Assess dataset readiness for request: {query}\n\nDataset:\n{data_context}",
        expected_output="A concise data-readiness assessment.",
        agent=cleaner,
    )
    analyze_task = Task(
        description=f"Perform quantitative analysis for request: {query}\n\nDataset:\n{data_context}",
        expected_output="Quantitative findings based on dataset values.",
        agent=analyst,
        context=[clean_task],
    )
    strategy_task = Task(
        description=f"Identify business implications for request: {query}\n\nUsing analysis findings, offer supported recommendations.",
        expected_output="Evidence-based strategic implications.",
        agent=strategist,
        context=[analyze_task],
    )

    crew = Crew(
        agents=[cleaner, analyst, strategist],
        tasks=[clean_task, analyze_task, strategy_task],
        process=Process.sequential,
        verbose=False,
    )

    try:
        crew_output = crew.kickoff()
        outputs = crew_output.tasks_output
        cleaner_out = get_task_output_text(outputs[0])
        analyst_out = get_task_output_text(outputs[1])
        strategist_out = get_task_output_text(outputs[2])

        return (
            "### 🧹 Data Readiness\n\n"
            f"{cleaner_out}\n\n"
            "### 📈 Quantitative Analysis\n\n"
            f"{analyst_out}\n\n"
            "### 💡 Strategic Interpretation\n\n"
            f"{strategist_out}"
        )
    except Exception as e:
        message = str(e).strip() or repr(e)
        raise RuntimeError(f"Full crew execution failed: {message}")


# =====================================================================
# 8. COMMAND PARSER & DATASET SUMMARY
# =====================================================================

COMMAND_HELP = """Available commands:

- `/clean <question>` — data quality and preprocessing assessment
- `/analyze <question>` — quantitative data analysis
- `/strategy <question>` — business strategy and recommendations
- `/full <question>` — run all three agents (Cleaner -> Analyst -> Strategist)
- `/summary` — deterministic dataset structural summary
- `/help` — display available commands
"""


def parse_command(prompt: str) -> Tuple[Optional[str], str]:
    """Parse slash commands."""
    prompt = prompt.strip()
    if not prompt.startswith("/"):
        return None, prompt
    parts = prompt.split(maxsplit=1)
    command = parts[0].lower()
    query = parts[1].strip() if len(parts) > 1 else ""
    return command, query


def dataset_summary(df: pd.DataFrame) -> str:
    """Deterministic /summary response."""
    rows, columns = df.shape
    missing = int(df.isnull().sum().sum())
    duplicates = int(df.duplicated().sum())
    numeric = df.select_dtypes(include="number").columns.tolist()
    categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()

    return f"""**Dataset Size:** {rows:,} rows × {columns:,} columns
**Missing Values:** {missing:,}
**Duplicate Rows:** {duplicates:,}
**Numeric Columns ({len(numeric)}):** {', '.join(numeric) if numeric else 'None'}
**Categorical Columns ({len(categorical)}):** {', '.join(categorical) if categorical else 'None'}
""".strip()


# =====================================================================
# 9. QUERY ROUTER (PRIMARY ENTRY POINT)
# =====================================================================

def route_query(
    df: pd.DataFrame,
    api_key: str,
    prompt: str,
    model_name: str = "gemini-2.5-flash",
) -> str:
    """Main routing logic for user prompts."""
    command, query = parse_command(prompt)

    if command is None:
        direct_answer = answer_simple_query(df, query)
        if direct_answer is not None:
            return direct_answer
        return run_single_agent(df=df, api_key=api_key, query=query, mode="analyst", model_name=model_name)

    if command == "/help":
        return COMMAND_HELP

    if command == "/summary":
        return dataset_summary(df)

    if command == "/clean":
        q = query or "Assess the overall quality and readiness of this dataset."
        return run_single_agent(df=df, api_key=api_key, query=q, mode="cleaner", model_name=model_name)

    if command == "/analyze":
        q = query or "Identify the most important quantitative patterns in this dataset."
        return run_single_agent(df=df, api_key=api_key, query=q, mode="analyst", model_name=model_name)

    if command == "/strategy":
        q = query or "Identify actionable business implications supported by this dataset."
        return run_single_agent(df=df, api_key=api_key, query=q, mode="strategy", model_name=model_name)

    if command == "/full":
        q = query or "Perform a complete multi-agent analysis of the dataset."
        return run_full_analysis(df=df, api_key=api_key, query=q, model_name=model_name)

    return f"Unknown command `{command}`. Type `/help` to see available commands."
