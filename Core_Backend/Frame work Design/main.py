import os
from crewai import Agent, Task, Crew, Process

# 1. Configure Environment / API Keys
# Replace with your key or use local Ollama model
os.environ["OPENAI_API_KEY"] = "your-openai-api-key-here"

# 2. Define Agents (Roles, Goals, & Backstory)
researcher = Agent(
    role="Tech Trends Researcher",
    goal="Identify top emerging AI trends for 2026",
    backstory="You are an industry analyst who tracks cutting-edge software and AI tools.",
    verbose=True
)

writer = Agent(
    role="Executive Writer",
    goal="Turn raw technical insights into simple executive summaries",
    backstory="You excel at writing clear, concise reports for non-technical stakeholders.",
    verbose=True
)

# 3. Define Tasks
research_task = Task(
    description="Find 3 major agentic AI frameworks popular in 2026.",
    expected_output="A bulleted list of 3 frameworks with brief descriptions.",
    agent=researcher
)

write_task = Task(
    description="Summarize the research findings into a brief executive report.",
    expected_output="A clean markdown summary document.",
    agent=writer
)

# 4. Create and Kick off the Crew
ai_crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential,
    verbose=True
)

if __name__ == "__main__":
    print("### Starting the Crew Execution ###")
    result = ai_crew.kickoff()
    print("\n### Final Deliverable Output ###\n")
    print(result)