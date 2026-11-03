
from langchain_groq import ChatGroq
from langchain_core.tools import tool
import requests
from langsmith import Client
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from dotenv import load_dotenv

load_dotenv()

search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str) -> str:
  """
  This function fetches the current weather data for a given city
  """
  url = f'https://api.weatherstack.com/current?access_key=f07d9636974c4120025fadf60678771b&query={city}'

  response = requests.get(url)

  return response.json()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7
)

# Step 2: Pull the ReAct prompt from LangChain Hub
client = Client()

prompt = client.pull_prompt(
    "hwchase17/react",
    dangerously_pull_public_prompt=True
)# pulls the standard ReAct agent prompt

prompt.template += """

IMPORTANT:
- Strictly follow the ReAct format.
- When using a tool, use Thought, Action, and Action Input.
- After receiving the tool observation, provide the final response using Final Answer:.
- Never use Answer: or **Answer:** as the final response label.
- Do not produce text outside the required ReAct format.
"""prompt.template += """

IMPORTANT:
- Strictly follow the ReAct format.
- When using a tool, use Thought, Action, and Action Input.
- After receiving the tool observation, provide the final response using Final Answer:.
- Never use Answer: or **Answer:** as the final response label.
- Do not produce text outside the required ReAct format.
"""

# Step 3: Create the ReAct agent manually with the pulled prompt
agent = create_react_agent(
    llm=llm,
    tools=[search_tool, get_weather_data],
    prompt=prompt
)

# Step 4: Wrap it with AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool, get_weather_data],
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True
)



# What is the release date of Dhadak 2?
# What is the current temp of gurgaon
# Identify the birthplace city of Kalpana Chawla (search) and give its current temperature.

# Step 5: Invoke
response = agent_executor.invoke({"input": "What is the current temp of gurgaon"})
print(response)

print(response['output'])