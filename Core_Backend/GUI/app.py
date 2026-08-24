import os

# Must be set BEFORE importing CrewAI
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

import re
import sys
from contextlib import contextmanager

import pandas as pd
import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM


# =====================================================================
# 1. PAGE CONFIGURATION
# =====================================================================

st.set_page_config(
    page_title="Agentic AI Data Analyst Chat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
    }

    .stButton > button {
        font-weight: bold;
        border-radius: 5px;
        transition: all 0.3s ease;
    }

    .stChatMessage {
        border-radius: 10px;
        margin-bottom: 10px;
    }

    .command-box {
        padding: 0.8rem 1rem;
        border-radius: 8px;
        border: 1px solid #444;
        margin-bottom: 1rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


# =====================================================================
# 2. SESSION STATE
# =====================================================================

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "current_file_name" not in st.session_state:
    st.session_state["current_file_name"] = ""

if "initial_analysis_done" not in st.session_state:
    st.session_state["initial_analysis_done"] = False


# =====================================================================
# 3. GENERAL HELPERS
# =====================================================================

def get_task_output_text(task_output):
    """Safely extract text from CrewAI TaskOutput."""

    if hasattr(task_output, "raw"):
        return task_output.raw

    return str(task_output)


def clean_response(text):
    """Remove unnecessary leading/trailing whitespace."""

    if text is None:
        return ""

    return str(text).strip()


# =====================================================================
# 4. LIVE CREWAI CONSOLE OUTPUT
# =====================================================================

ANSI_ESCAPE = re.compile(
    r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])"
)


class StreamToStreamlit:
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.buffer = ""

    def write(self, data):
        data_str = str(data)

        try:
            sys.__stdout__.write(data_str)
            sys.__stdout__.flush()
        except Exception:
            pass

        self.buffer += data_str
        clean_text = ANSI_ESCAPE.sub("", self.buffer)

        if len(clean_text) > 30000:
            clean_text = "...\n" + clean_text[-25000:]

        self.placeholder.code(clean_text)

    def flush(self):
        try:
            sys.__stdout__.flush()
        except Exception:
            pass


@contextmanager
def redirect_stdout_to_streamlit(placeholder):
    old_stdout = sys.stdout
    streamer = StreamToStreamlit(placeholder)
    sys.stdout = streamer

    try:
        yield streamer
    finally:
        sys.stdout = old_stdout


# =====================================================================
# 5. DATASET CONTEXT
# =====================================================================

def build_dataset_context(df, max_rows=300):
    """
    Produce a context representation for the LLM.

    Small datasets:
        send the complete dataframe.

    Larger datasets:
        send metadata, samples and descriptive statistics.
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
# 6. DIRECT PANDAS ANSWERS
# =====================================================================

def answer_simple_query(df, query):
    """
    Answer obvious factual dataframe questions without calling an LLM.

    Returns:
        string -> direct answer found
        None   -> send question to analyst
    """

    q = query.lower().strip()

    # -------------------------------------------------------------
    # Dataset dimensions
    # -------------------------------------------------------------

    size_phrases = [
        "dataset size",
        "size of dataset",
        "dataset shape",
        "shape of dataset",
        "shape",
        "how big is the dataset",
    ]

    if any(phrase in q for phrase in size_phrases):
        return (
            f"The dataset has **{df.shape[0]:,} rows** and "
            f"**{df.shape[1]:,} columns**."
        )

    if "how many rows" in q or "number of rows" in q:
        return f"The dataset has **{df.shape[0]:,} rows**."

    if "how many columns" in q or "number of columns" in q:
        return f"The dataset has **{df.shape[1]:,} columns**."

    # -------------------------------------------------------------
    # Column information
    # -------------------------------------------------------------

    if (
        "column names" in q
        or "list columns" in q
        or "what are the columns" in q
        or q == "columns"
    ):
        return "**Columns:**\n\n" + "\n".join(
            f"- `{column}`" for column in df.columns
        )

    # -------------------------------------------------------------
    # Missing values
    # -------------------------------------------------------------

    if (
        "missing values" in q
        or "null values" in q
        or "missing data" in q
    ):
        missing = df.isnull().sum()
        total = int(missing.sum())

        if total == 0:
            return "The dataset contains **no missing values**."

        affected = missing[missing > 0]

        details = "\n".join(
            f"- `{column}`: {int(count):,}"
            for column, count in affected.items()
        )

        return (
            f"The dataset contains **{total:,} missing values**.\n\n"
            f"{details}"
        )

    # -------------------------------------------------------------
    # Duplicate rows
    # -------------------------------------------------------------

    if "duplicate" in q:
        duplicates = int(df.duplicated().sum())

        return (
            f"The dataset contains **{duplicates:,} duplicated rows**."
        )

    # -------------------------------------------------------------
    # Data types
    # -------------------------------------------------------------

    if (
        "data types" in q
        or "datatypes" in q
        or "dtypes" in q
    ):
        lines = [
            f"- `{column}`: `{dtype}`"
            for column, dtype in df.dtypes.items()
        ]

        return "**Column data types:**\n\n" + "\n".join(lines)

    # -------------------------------------------------------------
    # Numeric columns
    # -------------------------------------------------------------

    if "numeric columns" in q or "numerical columns" in q:
        numeric = df.select_dtypes(include="number").columns.tolist()

        if not numeric:
            return "The dataset contains no numeric columns."

        return "**Numeric columns:**\n\n" + "\n".join(
            f"- `{column}`" for column in numeric
        )

    # -------------------------------------------------------------
    # Categorical columns
    # -------------------------------------------------------------

    if (
        "categorical columns" in q
        or "text columns" in q
        or "object columns" in q
    ):
        categorical = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        if not categorical:
            return "The dataset contains no categorical/text columns."

        return "**Categorical columns:**\n\n" + "\n".join(
            f"- `{column}`" for column in categorical
        )

    return None


# =====================================================================
# 7. LLM INITIALIZATION
# =====================================================================

def create_llm(api_key, model_name):
    """Create the CrewAI Gemini LLM instance."""

    clean_key = api_key.strip()

    os.environ["GEMINI_API_KEY"] = clean_key
    os.environ["GOOGLE_API_KEY"] = clean_key

    if model_name.startswith("gemini/"):
        llm_model_str = model_name
    else:
        llm_model_str = f"gemini/{model_name}"

    try:
        return LLM(
            model=llm_model_str,
            api_key=clean_key,
        )

    except Exception as e:
        message = str(e).strip() or repr(e)
        raise RuntimeError(
            f"LLM initialization failed: {message}"
        )


# =====================================================================
# 8. AGENT FACTORY
# =====================================================================

def create_agents(llm):
    """Create the three specialist agents."""

    data_cleaner = Agent(
        role="Data Cleaner & Preprocessor",
        goal=(
            "Inspect data quality, structure, missingness, duplicates, "
            "invalid values and readiness for analysis."
        ),
        backstory=(
            "You are a meticulous senior data engineer specializing "
            "in dataset validation and preprocessing."
        ),
        llm=llm,
        verbose=False,
    )

    data_analyst = Agent(
        role="Senior Data Analyst",
        goal=(
            "Answer quantitative questions accurately and directly "
            "using the supplied dataset."
        ),
        backstory=(
            "You are a senior quantitative data analyst. You prioritize "
            "correct calculations and concise answers."
        ),
        llm=llm,
        verbose=False,
    )

    business_strategist = Agent(
        role="Business Strategist",
        goal=(
            "Translate relevant quantitative findings into practical "
            "business implications and recommendations."
        ),
        backstory=(
            "You are a senior strategy consultant who bases recommendations "
            "strictly on supplied evidence."
        ),
        llm=llm,
        verbose=False,
    )

    return data_cleaner, data_analyst, business_strategist


# =====================================================================
# 9. SINGLE-AGENT EXECUTION
# =====================================================================

def run_single_agent(
    df,
    api_key,
    query,
    mode,
    model_name,
):
    """
    Run only one specialist agent.

    mode:
        cleaner
        analyst
        strategist
    """

    cloud_llm = create_llm(api_key, model_name)

    cleaner, analyst, strategist = create_agents(cloud_llm)

    data_context = build_dataset_context(df)

    if mode == "cleaner":

        task = Task(
            description=f"""
Inspect the supplied dataset with respect to the user's request.

User request:
{query}

Dataset:
{data_context}

Respond only to the user's request.

Focus on:
- missing values
- duplicates
- invalid or suspicious values
- type inconsistencies
- structural/data-readiness problems

Do not perform unrelated business analysis.
Do not add recommendations unless directly relevant.
Be concise unless the user explicitly asks for a detailed explanation.
""",
            expected_output=(
                "A direct data-quality answer at the level of detail "
                "requested by the user."
            ),
            agent=cleaner,
        )

        active_agent = cleaner

    elif mode == "strategy":

        task = Task(
            description=f"""
Answer the user's business or strategic question using only evidence
supported by the supplied dataset.

User request:
{query}

Dataset:
{data_context}

Rules:
- Answer only what the user asked.
- Do not provide a generic business essay.
- Ground recommendations in specific dataset evidence where possible.
- If the question asks for one recommendation, give one.
- If the question asks for explanation, explain.
- Otherwise be concise.
""",
            expected_output=(
                "A concise, evidence-based response to the user's "
                "strategic question."
            ),
            agent=strategist,
        )

        active_agent = strategist

    else:

        task = Task(
            description=f"""
Answer the user's question using the supplied dataset.

User question:
{query}

Dataset:
{data_context}

Strict response rules:

1. Answer ONLY what the user asked.
2. Be concise by default.
3. Do not add an introduction unless necessary.
4. Do not provide business recommendations unless requested.
5. Do not discuss data cleaning unless requested.
6. Do not summarize the entire dataset unless requested.
7. If the answer is one number, return that number with a short label.
8. If the answer requires a calculation, give the result and only the
   minimum explanation necessary.
9. If the user asks "why", "explain", "describe", "in detail", or similar,
   then provide a fuller explanation.
10. Never invent values that are not supported by the supplied dataset.
""",
            expected_output=(
                "A direct answer containing only the information "
                "necessary to answer the user's question."
            ),
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
            return clean_response(
                get_task_output_text(output.tasks_output[0])
            )

        return clean_response(get_task_output_text(output))

    except Exception as e:
        message = str(e).strip() or repr(e)

        raise RuntimeError(
            f"Crew execution failed: {message}"
        )


# =====================================================================
# 10. FULL THREE-AGENT WORKFLOW
# =====================================================================

def run_full_analysis(
    df,
    api_key,
    query,
    model_name,
):
    """
    Explicit /full mode.

    Runs:
        1. Cleaner
        2. Analyst
        3. Strategist
    """

    cloud_llm = create_llm(api_key, model_name)

    cleaner, analyst, strategist = create_agents(cloud_llm)

    data_context = build_dataset_context(df)

    clean_task = Task(
        description=f"""
Assess the dataset's readiness for the following request:

{query}

Dataset:
{data_context}

Identify only material data-quality issues that could affect the analysis.
""",
        expected_output=(
            "A concise data-readiness assessment."
        ),
        agent=cleaner,
    )

    analyze_task = Task(
        description=f"""
Perform the quantitative analysis needed to answer:

{query}

Dataset:
{data_context}

Use specific values and calculations from the dataset.
Do not add generic commentary.
""",
        expected_output=(
            "A quantitative analysis addressing the user's request."
        ),
        agent=analyst,
        context=[clean_task],
    )

    strategy_task = Task(
        description=f"""
Using the quantitative findings, identify the business implications
relevant to this request:

{query}

Give only recommendations supported by the analysis.
""",
        expected_output=(
            "Evidence-based strategic implications and recommendations."
        ),
        agent=strategist,
        context=[analyze_task],
    )

    crew = Crew(
        agents=[
            cleaner,
            analyst,
            strategist,
        ],
        tasks=[
            clean_task,
            analyze_task,
            strategy_task,
        ],
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

        raise RuntimeError(
            f"Full crew execution failed: {message}"
        )


# =====================================================================
# 11. COMMAND SYSTEM
# =====================================================================

COMMAND_HELP = """
Available commands:

- `/clean <question>` — data quality and preprocessing
- `/analyze <question>` — quantitative/data analysis
- `/strategy <question>` — business strategy and recommendations
- `/full <question>` — run all three agents
- `/summary` — general dataset summary
- `/help` — show these commands

You can also ask a normal question without a command. Normal questions
are answered concisely by default.
"""


def parse_command(prompt):
    """
    Parse slash commands.

    Returns:
        command, query
    """

    prompt = prompt.strip()

    if not prompt.startswith("/"):
        return None, prompt

    parts = prompt.split(maxsplit=1)

    command = parts[0].lower()

    query = parts[1].strip() if len(parts) > 1 else ""

    return command, query


# =====================================================================
# 12. DATASET SUMMARY
# =====================================================================

def dataset_summary(df):
    """Deterministic /summary response."""

    rows, columns = df.shape

    missing = int(df.isnull().sum().sum())
    duplicates = int(df.duplicated().sum())

    numeric = df.select_dtypes(include="number").columns.tolist()

    categorical = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    return f"""
**Dataset size:** {rows:,} rows × {columns:,} columns

**Missing values:** {missing:,}

**Duplicate rows:** {duplicates:,}

**Numeric columns:** {len(numeric)}

**Categorical/text columns:** {len(categorical)}
""".strip()


# =====================================================================
# 13. QUERY ROUTER
# =====================================================================

def route_query(
    df,
    api_key,
    prompt,
    model_name,
):
    """
    Main routing logic.

    Normal questions:
        deterministic pandas answer when possible
        otherwise analyst agent

    Slash commands:
        explicitly select specialist behavior
    """

    command, query = parse_command(prompt)

    # -------------------------------------------------------------
    # Normal conversational query
    # -------------------------------------------------------------

    if command is None:

        direct_answer = answer_simple_query(df, query)

        if direct_answer is not None:
            return direct_answer

        return run_single_agent(
            df=df,
            api_key=api_key,
            query=query,
            mode="analyst",
            model_name=model_name,
        )

    # -------------------------------------------------------------
    # Help
    # -------------------------------------------------------------

    if command == "/help":
        return COMMAND_HELP

    # -------------------------------------------------------------
    # Deterministic summary
    # -------------------------------------------------------------

    if command == "/summary":
        return dataset_summary(df)

    # -------------------------------------------------------------
    # Cleaner
    # -------------------------------------------------------------

    if command == "/clean":

        if not query:
            query = (
                "Assess the overall quality and readiness of this dataset."
            )

        return run_single_agent(
            df=df,
            api_key=api_key,
            query=query,
            mode="cleaner",
            model_name=model_name,
        )

    # -------------------------------------------------------------
    # Analyst
    # -------------------------------------------------------------

    if command == "/analyze":

        if not query:
            query = (
                "Identify the most important quantitative patterns "
                "in this dataset."
            )

        return run_single_agent(
            df=df,
            api_key=api_key,
            query=query,
            mode="analyst",
            model_name=model_name,
        )

    # -------------------------------------------------------------
    # Strategy
    # -------------------------------------------------------------

    if command == "/strategy":

        if not query:
            query = (
                "Identify the most important actionable business "
                "implications supported by this dataset."
            )

        return run_single_agent(
            df=df,
            api_key=api_key,
            query=query,
            mode="strategy",
            model_name=model_name,
        )

    # -------------------------------------------------------------
    # Full crew
    # -------------------------------------------------------------

    if command == "/full":

        if not query:
            query = (
                "Perform a complete analysis of the dataset."
            )

        return run_full_analysis(
            df=df,
            api_key=api_key,
            query=query,
            model_name=model_name,
        )

    return (
        f"Unknown command `{command}`.\n\n"
        "Type `/help` to see the available commands."
    )


# =====================================================================
# 14. SIDEBAR
# =====================================================================

st.sidebar.title("⚙️ Control Panel")
st.sidebar.markdown("---")


# API Key

gemini_api_key = st.sidebar.text_input(
    "Google Gemini API Key",
    type="password",
    help="Enter your Google AI Studio Gemini API key.",
)


# CSV uploader

uploaded_file = st.sidebar.file_uploader(
    "Upload Target CSV Dataset",
    type=["csv"],
    help="Upload the CSV dataset you want to analyze.",
)


# Model selector

selected_model = st.sidebar.selectbox(
    "Gemini Model Engine",
    options=[
        "gemini/gemini-2.5-flash",
        "gemini/gemini-2.5-pro",
    ],
    index=0,
    help="Choose the Gemini model used by CrewAI.",
)


# Clear chat

if st.sidebar.button(
    "🗑️ Clear Chat History",
    use_container_width=True,
):
    st.session_state["messages"] = []
    st.rerun()


st.sidebar.markdown("---")


# =====================================================================
# 15. LOAD DATA
# =====================================================================

df = None

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)

    except Exception as e:
        st.sidebar.error(
            f"❌ Failed to parse CSV: {str(e)}"
        )

        df = None


# =====================================================================
# 16. SIDEBAR DATASET STATISTICS
# =====================================================================

if df is not None:

    st.sidebar.subheader("📊 Dataset Statistics")

    col1, col2 = st.sidebar.columns(2)

    with col1:
        st.metric(
            "Rows",
            f"{df.shape[0]:,}",
        )

    with col2:
        st.metric(
            "Columns",
            f"{df.shape[1]:,}",
        )

    total_missing = int(
        df.isnull().sum().sum()
    )

    st.sidebar.metric(
        "Missing Values",
        f"{total_missing:,}",
    )

    st.sidebar.metric(
        "Duplicates",
        f"{df.duplicated().sum():,}",
    )

    with st.sidebar.expander(
        "🔍 Raw Data Preview",
        expanded=False,
    ):
        st.dataframe(
            df.head(10),
            use_container_width=True,
        )


# =====================================================================
# 17. MAIN INTERFACE
# =====================================================================

st.title("📊 Agentic AI Data Analyst Chat")

st.markdown(
    "##### *Cloud Multi-Agent Chat Application powered by Google Gemini & CrewAI*"
)

st.markdown("---")


# =====================================================================
# 18. PREREQUISITE MESSAGES
# =====================================================================

if not gemini_api_key:

    st.warning(
        "🔑 Enter your Google Gemini API key in the sidebar."
    )

elif df is None:

    st.info(
        "👋 Upload a CSV dataset in the sidebar to begin."
    )

else:

    # -------------------------------------------------------------
    # Environment API keys
    # -------------------------------------------------------------

    os.environ["GEMINI_API_KEY"] = gemini_api_key.strip()
    os.environ["GOOGLE_API_KEY"] = gemini_api_key.strip()


    # -------------------------------------------------------------
    # Reset conversation when file changes
    # -------------------------------------------------------------

    file_changed = (
        st.session_state.get("current_file_name")
        != uploaded_file.name
    )

    if file_changed:

        st.session_state["messages"] = []

        st.session_state["current_file_name"] = (
            uploaded_file.name
        )

        st.session_state["initial_analysis_done"] = False

        st.rerun()


    # -------------------------------------------------------------
    # Initial welcome message
    #
    # IMPORTANT:
    # No automatic 3-agent analysis anymore.
    # -------------------------------------------------------------

    if len(st.session_state["messages"]) == 0:

        welcome_message = (
            f"Loaded **{uploaded_file.name}** — "
            f"**{df.shape[0]:,} rows × {df.shape[1]:,} columns**.\n\n"
            "Ask me a question about the dataset, or type `/help` "
            "to see specialist commands."
        )

        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": welcome_message,
            }
        )


    # -------------------------------------------------------------
    # Render conversation
    # -------------------------------------------------------------

    for message in st.session_state["messages"]:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    # -------------------------------------------------------------
    # Chat input
    # -------------------------------------------------------------

    prompt = st.chat_input(
        "Ask about the dataset or type /help..."
    )

    if prompt:

        # User message

        st.session_state["messages"].append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        with st.chat_message("user"):
            st.markdown(prompt)


        # Assistant response

        with st.chat_message("assistant"):

            try:

                command, _ = parse_command(prompt)

                # Commands which do not need Gemini
                no_llm_commands = {
                    "/help",
                    "/summary",
                }

                direct_answer = None

                if command is None:
                    direct_answer = answer_simple_query(
                        df,
                        prompt,
                    )

                if (
                    command in no_llm_commands
                    or direct_answer is not None
                ):

                    agent_response = route_query(
                        df=df,
                        api_key=gemini_api_key,
                        prompt=prompt,
                        model_name=selected_model,
                    )

                else:

                    with st.status(
                        "🤖 Processing...",
                        expanded=False,
                    ) as status:

                        agent_response = route_query(
                            df=df,
                            api_key=gemini_api_key,
                            prompt=prompt,
                            model_name=selected_model,
                        )

                        status.update(
                            label="✅ Complete",
                            state="complete",
                        )

            except Exception as e:

                agent_response = (
                    "I encountered an error while processing "
                    "the request.\n\n"
                    f"**Error details:** `{str(e)}`"
                )

                st.error(agent_response)


            st.markdown(agent_response)

            st.session_state["messages"].append(
                {
                    "role": "assistant",
                    "content": agent_response,
                }
            )