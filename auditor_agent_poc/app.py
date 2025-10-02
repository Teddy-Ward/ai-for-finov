# auditor_agent_poc/app.py

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.callbacks import StreamlitCallbackHandler # <<< CHANGE: Import the callback handler

# Import the custom tools we created
from tools import get_application_data, compliance_checker

# --- 1. Load Environment Variables and Page Configuration ---

# Load the API key from the .env file
load_dotenv('../.env')

# Configure the Streamlit page
st.set_page_config(page_title="Auditor Agent PoC", page_icon="🤖", layout="wide")
st.title("🤖 Compliance-Aware Auditor Agent")
st.markdown("This agent audits a mortgage application for data consistency and compliance with lender policy.")

# --- 2. Define the Agent's "Brain" (Prompt Template) ---

# This is the core instruction set for the agent.
# It tells the agent its persona, its goal, what tools it has, and how to behave.
prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are an expert mortgage underwriting assistant named "AuditBot".
    Your goal is to conduct a pre-submission audit on a mortgage application.
    You must perform a series of checks and then provide a final, structured report.

    Here is your process:
    1.  First, use the `get_application_data` tool to fetch the applicant's data using their ID.
    2.  Review the fetched data for any obvious internal inconsistencies. For example, a self-employed person should not have PAYE income details.
    3.  Next, use the `compliance_checker` tool to verify key data points against the lender's policy. You must check at least the following:
        - The applicant's credit score.
        - The Loan-to-Value (LTV) ratio.
        - The applicant's employment status and history.
    4.  You must use your tools. Do not make up answers or rely on prior knowledge.
    5.  Once you have completed all checks, provide a final answer in the structured Markdown format below. Do not add any conversational text outside of this format.

    ## Audit Report for Applicant {applicant_id}

    ### **Internal Consistency**
    - [List any inconsistencies found, or "All checks passed."]

    ### **Compliance Checks**
    - [List all policy checks performed with a PASS/FAIL status and a brief reason, citing the rule if possible.]

    ### **Final Recommendation**
    - [Provide a final status: 'Ready for Submission' or 'Action Required'.]
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Please audit the application for applicant ID: {applicant_id}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# --- 3. Set Up the Agent ---

# Initialize the Language Model
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)

# List of tools the agent can use
tools = [get_application_data, compliance_checker]

# Create the agent by binding the LLM, tools, and prompt
agent = create_openai_tools_agent(llm, tools, prompt)

# Create the Agent Executor, which is the runtime for the agent
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- 4. Build the Streamlit User Interface ---

# Load applicant IDs for the dropdown menu
try:
    applicant_df = pd.read_csv("../data/mock_applications.csv")
    applicant_ids = applicant_df['ApplicantID'].tolist()
except FileNotFoundError:
    st.error("Error: The mock_applications.csv file was not found. Please run `generate_data.py` first.")
    applicant_ids = []

# Create a container for the input controls
input_container = st.container()
with input_container:
    st.header("Select an Applicant to Audit")
    selected_id = st.selectbox("Applicant ID", options=applicant_ids)
    audit_button = st.button("Run Audit", type="primary")

# --- 5. Run the Agent and Display the Results ---

if audit_button:
    if not selected_id:
        st.warning("Please select an Applicant ID.")
    else:
        st.markdown("---")
        
        # Instantiate the callback handler
        st_callback = StreamlitCallbackHandler(st.container())
        
        # Prepare the input for the agent
        agent_input = {"applicant_id": selected_id, "chat_history": []}
        
        # Invoke the agent with the callback
        response = agent_executor.invoke(agent_input, {"callbacks": [st_callback]})
        
        # Display the final report
        st.header("Audit Complete")
        st.markdown(response['output'])