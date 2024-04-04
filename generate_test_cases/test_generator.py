import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.tools.retriever import create_retriever_tool
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Load documents
def load_pdf_docs(pdf_doc):
    pdf_reader = PdfReader(pdf_doc)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# Chunks Splitters
def get_chunks_from_pdf(pdf_content):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return text_splitter.split_text(pdf_content)


# Create Embeddings
def create_embeddings():
    return OpenAIEmbeddings()


# Create Vector Storage
def create_vector_stores(pages, embeds):
    vector = Chroma.from_texts(pages, embedding=embeds)
    db = vector.as_retriever()
    return db


# Create Retriever tool
def create_retriever_tools(vector_db):
    retriever = create_retriever_tool(
        vector_db,
        name='test_case_generator_tool',
        description='Use this tool to create test cases from given document'
    )
    return retriever


# Create Custom Tools
def create_custom_tools(retriever_tool):
    customized_tools = [retriever_tool]
    return customized_tools


# Create Prompt Template
def create_prompt_template():
    prompt_message = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="You are an expert in creating test-cases for the requirements document.You go through the "
                        "document and list out all possible test cases, based on user's ask with your expertise\n"
                        "Reply to user queries based on the document that was uploaded as requirements\n"
                        "ALWAYS Use the format of the test cases as below \n"
                        "Columns as S.No, Test Scenario Name, Step numbers Pre-requisites Expected Result\n"
                        "Write test cases in the appropriate columns and well detailed\n"
                        "Write test cases in multiple steps if needed but it should be detailed and easily understood"
            ),
            MessagesPlaceholder(
                variable_name="agent_scratchpad"
            ),
            MessagesPlaceholder(
                variable_name="chat_history"
            ),
            HumanMessagePromptTemplate.from_template(
                "{input}"
            )
        ]
    )
    return prompt_message


# bind tools with LLM
def bind_tools_with_llm(created_tools):
    llm = ChatOpenAI(
        model='gpt-3.5-turbo',
        temperature=0.8
    )

    # Bind tools to LLM
    llm_tools = llm.bind_tools(created_tools)
    return llm_tools


# Create Custom Agent
def create_agent(user_prompt, llm_tool):
    get_agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                    x["intermediate_steps"]
                ),
                "chat_history": lambda x: x["chat_history"],
            }
            | user_prompt
            | llm_tool
            | OpenAIToolsAgentOutputParser()
    )
    return get_agent


# Create Agent Executor
def create_agent_executor(custom_agent, custom_tools):
    return AgentExecutor(agent=custom_agent, tools=custom_tools, max_iterations=40,
                         max_execution_time=10, verbose=True, handle_parsing_errors=True
                         )


# Set up memory
def set_up_st_memory():
    msgs = StreamlitChatMessageHistory(key="chat_history")
    if len(msgs.messages) == 0:
        msgs.add_ai_message("Hello, I'm a Test Case Generator Bot. How can I help you?")
    return msgs


# Run Chain with History
def run_chain_with_history(agent_runnable, chat_message_history):
    history_chain = RunnableWithMessageHistory(
        agent_runnable,
        lambda session_id: chat_message_history,
        input_messages_key="input",
        history_messages_key="chat_history"
    )
    return history_chain


# Render current messages from StreamlitChatMessageHistory
def render_current_messages(chat_message_history):
    for msg in chat_message_history.messages:
        st.chat_message(msg.type).write(msg.content)


# If user inputs a new prompt, generate and draw a new response
def start_conversation(history_chain):
    if base_prompt := st.chat_input():
        st.chat_message("human").write(base_prompt)
        # New messages are saved to history automatically by Langchain during run
        config = {"configurable": {"session_id": "any"}}
        response = history_chain.invoke({"input": base_prompt}, config)
        st.chat_message("ai").write(response['output'])


st.set_page_config(page_title="Create Test Cases")
st.title("Test Cases Generator")
"""
A basic test cases generator app based on the uploaded requirement PDF document uploaded.\n
It can answer questions on Test Data Techniques like BVA, Equivalence with examples.
It can provide some test data examples, as well.\n
Find it yourself!
"""
with st.sidebar:
    st.header("Requirement Document")
    pdf = st.file_uploader("Upload Acceptance Criteria document here")
if pdf is None:
    st.info("Upload a Requirement Document in PDF format to continue")
else:
    load_dotenv()
    documents = load_pdf_docs(pdf)
    chunks = get_chunks_from_pdf(documents)
    embeddings = create_embeddings()
    vector_stores = create_vector_stores(chunks, embeddings)
    retriever_tool = create_retriever_tools(vector_stores)
    tools = create_custom_tools(retriever_tool)
    prompt = create_prompt_template()
    llm_with_tools = bind_tools_with_llm(tools)
    agent = create_agent(prompt, llm_with_tools)
    agent_executor = create_agent_executor(agent, tools)
    messages = set_up_st_memory()
    chain_with_history = run_chain_with_history(agent_executor, messages)
    render_current_messages(messages)
    start_conversation(chain_with_history)
