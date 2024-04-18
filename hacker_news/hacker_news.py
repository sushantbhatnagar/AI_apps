import json
import requests
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# url = "https://hacker-news.firebaseio.com/v0/topstories.json?print=pretty"

# model
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.2
)


# creating custom tools
@tool
def get_top_stories() -> list:
    """
    Get Top Stories from Hacker News API
    """
    stories_data = []
    # api_url = url
    api_url = "https://hacker-news.firebaseio.com/v0/topstories.json?print=pretty"
    response = requests.get(api_url)
    stories = response.content.decode('utf-8')
    stories_ids = json.loads(stories)[:5]
    for story_id in stories_ids:
        story_url = f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json?print=pretty"
        response = requests.get(story_url)
        if response.status_code == 200:
            story = response.json()
            title = story.get('title')
            if title:
                stories_data.append({'title': title})
        else:
            return [response.status_code]
    return stories_data


@tool
def get_job_stories() -> list:
    """
    Get JOB Stories from Hacker News API
    """
    stories_data = []
    # api_url = url
    api_url = "https://hacker-news.firebaseio.com/v0/jobstories.json?print=pretty"
    response = requests.get(api_url)
    stories = response.content.decode('utf-8')
    stories_ids = json.loads(stories)[:5]
    for story_id in stories_ids:
        story_url = f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json?print=pretty"
        response = requests.get(story_url)
        if response.status_code == 200:
            story = response.json()
            title = story.get('title')
            if title:
                stories_data.append({'title': title})
        else:
            return [response.status_code]
    return stories_data


# custom tool
tools = [get_top_stories, get_job_stories]

# prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are very powerful assistant, expert in reading news from Hacker News",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# binding tools
llm_with_tools = llm.bind_tools(tools)

agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            ),
        }
        | prompt
        | llm_with_tools
        | OpenAIToolsAgentOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
agent_executor.invoke({"input": "Get me job stories from the Hacker News?"})
