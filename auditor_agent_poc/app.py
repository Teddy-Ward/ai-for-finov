# auditor_agent_poc/app.py

import streamlit as st
import pandas as pd
import time
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.callbacks import StreamlitCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish

# Import the custom tools we created
from tools import get_application_data, compliance_checker

# --- Custom Callback Handler for Better Demo Visibility ---

class DetailedCallbackHandler(BaseCallbackHandler):
    def __init__(self, container, start_time):
        self.container = container
        self.step_counter = 0
        self.current_step = None
        self.start_time = start_time
        self.thinking_expander = None
        
    def on_agent_action(self, action: AgentAction, **kwargs):
        self.step_counter += 1
        self.current_step = f"Step {self.step_counter}"
        
        # Create or update the thinking expander
        if self.thinking_expander is None:
            self.thinking_expander = self.container.expander("🤖 Agent Thinking Process", expanded=True)
        
        with self.thinking_expander:
            st.markdown(f"### 🤔 {self.current_step}: Agent Thinking")
            
            # Show the agent's reasoning
            if hasattr(action, 'log') and action.log:
                st.markdown("**Agent's Reasoning:**")
                st.text(action.log)
            
            # Show what tool the agent is about to use
            st.markdown(f"**Action:** Using tool `{action.tool}`")
            st.markdown(f"**Tool Input:** `{action.tool_input}`")
            
            # Add a progress indicator
            progress_bar = st.progress(0)
            progress_bar.progress(min(self.step_counter * 0.2, 0.8))  # Max 80% until final step
            
            st.markdown("---")
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name", "Unknown Tool")
        if self.thinking_expander:
            with self.thinking_expander:
                st.markdown(f"### 🔧 Executing: {tool_name}")
                st.markdown(f"**Input:** {input_str}")
                
                # Show a spinner while the tool is running
                with st.spinner(f"Running {tool_name}..."):
                    time.sleep(0.5)  # Small delay for demo effect
    
    def on_tool_end(self, output, **kwargs):
        if self.thinking_expander:
            with self.thinking_expander:
                st.markdown("**Tool Output:**")
                st.code(output, language="text")
                st.markdown("---")
    
    def on_agent_finish(self, finish: AgentFinish, **kwargs):
        if self.thinking_expander:
            with self.thinking_expander:
                st.markdown("### ✅ Final Step: Generating Report")
                progress_bar = st.progress(1.0)
                st.success("Audit completed successfully!")
        
        # Calculate and store the processing time
        self.processing_time = time.time() - self.start_time

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

    IMPORTANT: Think step by step and explain your reasoning clearly. Show your work!

    Here is your detailed process:
    
    STEP 1: DATA RETRIEVAL
    - Use the `get_application_data` tool to fetch the applicant's data using their ID
    - Review the retrieved data and identify key fields: Income, CreditScore, LTV, DebtToIncomeRatio, EmploymentStatus, RejectionReason
    
    STEP 2: INTERNAL CONSISTENCY CHECK
    - Analyze the data for logical inconsistencies
    - Check if employment status matches income patterns
    - Verify that numerical values are within reasonable ranges
    - Look for any contradictory information
    
    STEP 3: COMPLIANCE VERIFICATION
    - Use the `compliance_checker` tool to verify key data points against lender policy
    - Check credit score requirements
    - Verify LTV limits
    - Validate employment status requirements
    - Check debt-to-income ratio limits
    - Ask specific policy questions as needed
    
    STEP 4: FINAL ASSESSMENT
    - Synthesize all findings
    - Determine if the application meets all requirements
    - Provide clear recommendations
    
    You must use your tools and show your reasoning. Do not make up answers or rely on prior knowledge.

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
        
        # Create containers for better organization
        progress_container = st.container()
        thinking_container = st.container()
        report_container = st.container()
        
        with progress_container:
            st.header("🔍 Audit Progress")
            st.info(f"Starting audit for Applicant ID: **{selected_id}**")
        
        with thinking_container:
            st.header("🤖 Agent Thinking Process")
            
            # Show initial status
            st.markdown("### 🚀 Initializing AuditBot...")
            st.markdown("The agent will now follow a systematic 4-step process:")
            st.markdown("1. **Data Retrieval** - Fetch applicant data")
            st.markdown("2. **Internal Consistency Check** - Look for data inconsistencies") 
            st.markdown("3. **Compliance Verification** - Check against lender policy")
            st.markdown("4. **Final Assessment** - Generate recommendations")
            st.markdown("---")
            
            # Record start time for processing time calculation
            start_time = time.time()
            
            # Instantiate our custom callback handler with start time
            detailed_callback = DetailedCallbackHandler(thinking_container, start_time)
            
            # Prepare the input for the agent
            agent_input = {"applicant_id": selected_id, "chat_history": []}
            
            # Invoke the agent with our custom callback
            response = agent_executor.invoke(agent_input, {"callbacks": [detailed_callback]})
        
        with report_container:
            st.header("📋 Final Audit Report")
            st.markdown("---")
            st.markdown(response['output'])
            
            # Add a summary section
            st.markdown("---")
            st.markdown("### 📊 Audit Summary")
            col1, col2, col3 = st.columns(3)
            
            # Get the actual processing time from the callback handler
            actual_processing_time = getattr(detailed_callback, 'processing_time', time.time() - start_time)
            
            with col1:
                st.metric("Applicant ID", selected_id)
            with col2:
                st.metric("Audit Status", "✅ Complete")
            with col3:
                st.metric("Processing Time", f"{actual_processing_time:.1f} seconds")