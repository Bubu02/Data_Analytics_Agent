import pandas as pd
from crewai import Agent, Task, Crew, Process, LLM

# 1. Connect to local Ollama instance (Llama 3.2 1B)
local_llm = LLM(
    model="ollama/llama3.2:1b",
    base_url="http://localhost:11434"
)

# 2. Read the local CSV file
df = pd.read_csv("Data/sales_data.csv")
data_summary = df.describe(include='all').to_string()
raw_csv = df.to_string()

# 3. Define local CrewAI agents
data_cleaner = Agent(
    role="Data Cleaner & Preprocessor",
    goal="Inspect dataset structure and highlight missing values or data anomalies.",
    backstory="You are a meticulous data engineer who spots data quality issues instantly.",
    llm=local_llm,
    verbose=True
)

data_analyst = Agent(
    role="Senior Data Analyst",
    goal="Analyze metrics and extract numeric trends from data summaries.",
    backstory="You are a data analyst skilled at extracting insights from quantitative records.",
    llm=local_llm,
    verbose=True
)

business_strategist = Agent(
    role="Business Strategist",
    goal="Translate raw numerical insights into actionable strategic advice.",
    backstory="You are an executive advisor turning raw numbers into practical next steps.",
    llm=local_llm,
    verbose=True
)

# 4. Define sequential execution tasks
clean_task = Task(
    description=f"Examine this dataset summary:\n{data_summary}\nProvide a 2-bullet point assessment of data readiness.",
    expected_output="Two bullet points regarding data quality.",
    agent=data_cleaner
)

analyze_task = Task(
    description=f"Analyze the raw dataset:\n{raw_csv}\nIdentify top revenue drivers and key region performance.",
    expected_output="A bulleted summary of key numerical findings.",
    agent=data_analyst
)

strategy_task = Task(
    description="Review the analysis results and propose 2 clear strategic initiatives for next quarter.",
    expected_output="Two strategic recommendations in markdown format.",
    agent=business_strategist
)

# 5. Assemble and execute the pipeline
crew = Crew(
    agents=[data_cleaner, data_analyst, business_strategist],
    tasks=[clean_task, analyze_task, strategy_task],
    process=Process.sequential,
    verbose=True
)

if __name__ == "__main__":
    print("--- Starting Local CSV Analysis Pipeline ---")
    output = crew.kickoff()
    print("\n================ FINAL REPORT ================\n")
    print(output)