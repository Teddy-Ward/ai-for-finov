# auditor_agent_poc/app.py

import streamlit as st
import pandas as pd
import time
import json
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.callbacks import StreamlitCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish
from openai import OpenAI

# Import the custom tools we created
from tools import get_application_data, compliance_checker

# Initialize OpenAI client for human feedback analysis
# Load .env from parent directory (2-pocs folder)
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Create training data folder
TRAINING_DATA_FOLDER = Path(__file__).parent / 'training_data'
TRAINING_DATA_FOLDER.mkdir(exist_ok=True)

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

# Load the API key from the .env file (already loaded above)

# Configure the Streamlit page
st.set_page_config(page_title="Auditor Agent PoC", page_icon="🤖", layout="wide")

# Initialize session state
if 'audit_result' not in st.session_state:
    st.session_state.audit_result = None
if 'applicant_data' not in st.session_state:
    st.session_state.applicant_data = None
if 'selected_id' not in st.session_state:
    st.session_state.selected_id = None

st.title("🤖 Compliance-Aware Auditor Agent with Human-in-the-Loop")
st.markdown("AI audits mortgage applications, and you review/edit the results. Your feedback trains future models.")

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
    csv_path = Path(__file__).parent.parent / 'data' / 'mock_applications.csv'
    applicant_df = pd.read_csv(csv_path)
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
            
            # Store results in session state for human review
            st.session_state.audit_result = response['output']
            st.session_state.selected_id = selected_id
            
            # Fetch applicant data for context
            try:
                csv_path = Path(__file__).parent.parent / 'data' / 'mock_applications.csv'
                applicant_df = pd.read_csv(csv_path)
                applicant_row = applicant_df[applicant_df['ApplicantID'] == selected_id].iloc[0]
                st.session_state.applicant_data = applicant_row.to_dict()
            except:
                st.session_state.applicant_data = {}

# --- 6. Human-in-the-Loop Review Section ---

if st.session_state.audit_result is not None:
    st.markdown("---")
    st.markdown("---")
    st.header("👤 Human Expert Review & Feedback")
    
    st.markdown("""
    <div style='background-color: #fff3cd; padding: 15px; border-radius: 8px; border-left: 5px solid #ffc107;'>
        <strong>📝 Review the AI audit above and provide your expert feedback.</strong><br>
        Your edits and reasoning will be used to train and improve future AI models.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Create two columns for comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🤖 AI-Generated Audit")
        st.markdown("<div style='background-color: #e8f4f8; padding: 15px; border-radius: 8px;'>", unsafe_allow_html=True)
        st.markdown(st.session_state.audit_result)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("### ✏️ Your Edited Version")
        st.markdown("*Edit the report below if you disagree with the AI's assessment*")
        
        edited_report = st.text_area(
            "Edit the audit report:",
            value=st.session_state.audit_result,
            height=400,
            key="edited_audit_report",
            help="Make any necessary corrections or improvements to the AI's audit"
        )
    
    st.markdown("---")
    
    # Human assessment section
    st.markdown("### 📊 Your Assessment")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        human_verdict = st.selectbox(
            "Final Verdict",
            ["Ready for Submission", "Action Required", "Reject Application", "Request More Information"],
            key="human_verdict",
            help="Your final assessment of this application"
        )
    
    with col2:
        confidence = st.slider(
            "Confidence Level",
            0, 100, 85,
            key="confidence",
            help="How confident are you in this assessment?"
        )
    
    with col3:
        ai_agreement = st.radio(
            "Do you agree with the AI?",
            ["Fully Agree", "Partially Agree", "Disagree"],
            key="ai_agreement"
        )
    
    st.markdown("---")
    
    # Feedback and reasoning section
    st.markdown("### 💬 Your Feedback")
    
    col1, col2 = st.columns(2)
    
    with col1:
        key_changes = st.text_area(
            "What did you change? (if anything)",
            height=150,
            key="key_changes",
            placeholder="E.g., 'Changed recommendation from Ready to Action Required because...'"
        )
    
    with col2:
        additional_notes = st.text_area(
            "Additional observations or context",
            height=150,
            key="additional_notes",
            placeholder="E.g., 'The AI missed the fact that this is a remortgage, not a purchase'"
        )
    
    # Save button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("💾 Save for Training", type="primary", use_container_width=True):
            # Check if there were any changes
            has_changes = edited_report.strip() != st.session_state.audit_result.strip()
            
            # If there are changes, use AI to analyze why
            if has_changes:
                with st.spinner("🤖 AI is analyzing your changes..."):
                    # Use AI to understand the changes
                    change_analysis = analyze_human_changes(
                        st.session_state.audit_result,
                        edited_report,
                        st.session_state.applicant_data,
                        key_changes,
                        additional_notes
                    )
            else:
                change_analysis = {
                    "has_changes": False,
                    "change_summary": "No changes made - human agreed with AI assessment",
                    "reasoning_analysis": "Human expert validated AI's audit without modifications"
                }
            
            # Save the training data
            training_data = {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "applicant_id": st.session_state.selected_id,
                    "broker_id": "human_expert",  # Could be dynamic in production
                },
                "applicant_data": st.session_state.applicant_data,
                "ai_output": {
                    "original_audit": st.session_state.audit_result,
                },
                "human_review": {
                    "edited_audit": edited_report,
                    "verdict": human_verdict,
                    "confidence_level": confidence,
                    "ai_agreement": ai_agreement,
                    "key_changes_noted": key_changes,
                    "additional_notes": additional_notes
                },
                "change_analysis": change_analysis,
                "training_signals": {
                    "requires_retraining": has_changes,
                    "agreement_level": ai_agreement,
                    "error_type": change_analysis.get('error_type', 'none'),
                    "learning_priority": 'high' if has_changes and ai_agreement == 'Disagree' else 'medium' if has_changes else 'low'
                }
            }
            
            # Save to JSON file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"audit_review_{st.session_state.selected_id}_{timestamp}.json"
            filepath = TRAINING_DATA_FOLDER / filename
            
            try:
                with open(filepath, 'w') as f:
                    json.dump(training_data, f, indent=2)
                
                st.success(f"✅ Training data saved successfully!")
                st.info(f"📁 Saved to: `{filepath}`")
                
                # Show the AI's analysis of changes
                if has_changes:
                    with st.expander("🤖 AI's Analysis of Your Changes", expanded=True):
                        st.markdown("**What changed:**")
                        st.write(change_analysis.get('change_summary', 'N/A'))
                        st.markdown("**Why you might have changed it:**")
                        st.write(change_analysis.get('reasoning_analysis', 'N/A'))
                        st.markdown("**Learning points for AI:**")
                        for point in change_analysis.get('learning_points', []):
                            st.markdown(f"• {point}")
                
                # Show training data stats
                training_files = [f for f in os.listdir(TRAINING_DATA_FOLDER) if f.endswith('.json')]
                st.metric("Total Training Samples Collected", len(training_files))
                
                st.balloons()
            except Exception as e:
                st.error(f"❌ Error saving training data: {str(e)}")
    
    with col2:
        if st.button("🔄 Clear & Start New", use_container_width=True):
            st.session_state.audit_result = None
            st.session_state.applicant_data = None
            st.session_state.selected_id = None
            st.rerun()

# --- Helper Function: AI-Powered Change Analysis ---

def analyze_human_changes(original_audit, edited_audit, applicant_data, human_notes, additional_context):
    """
    Uses AI to analyze why the human expert made changes to the audit.
    This creates rich training data with explanations.
    """
    try:
        prompt = f"""You are analyzing changes made by a human mortgage underwriting expert to an AI-generated audit report.

**Original AI Audit:**
{original_audit}

**Human's Edited Version:**
{edited_audit}

**Human's Notes on Changes:**
{human_notes if human_notes else 'No specific notes provided'}

**Additional Context:**
{additional_context if additional_context else 'None provided'}

**Applicant Data Context:**
{json.dumps(applicant_data, indent=2)}

Analyze the differences and provide:

1. **change_summary**: A clear, concise summary of what changed (2-3 sentences)

2. **reasoning_analysis**: Your analysis of WHY the human likely made these changes. What did the AI miss or misunderstand? (2-3 sentences)

3. **error_type**: Categorize the AI's error (choose one):
   - 'factual_error' - AI stated incorrect facts
   - 'logic_error' - AI's reasoning was flawed
   - 'policy_misinterpretation' - AI misunderstood lending policy
   - 'tone_adjustment' - Human changed wording/tone but not substance
   - 'completeness' - AI missed important considerations
   - 'none' - No significant error, minor edits only

4. **learning_points**: Array of 2-4 specific lessons the AI should learn from this correction

5. **severity**: Rate the severity of the AI's mistake ('low', 'medium', 'high', 'critical')

Return as JSON with these exact keys."""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an AI training analyst who helps identify why humans correct AI outputs to improve future models."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        analysis = json.loads(response.choices[0].message.content)
        analysis['has_changes'] = True
        return analysis
        
    except Exception as e:
        return {
            "has_changes": True,
            "error": str(e),
            "change_summary": "Error analyzing changes",
            "reasoning_analysis": "Analysis failed",
            "error_type": "unknown",
            "learning_points": [],
            "severity": "unknown"
        }

# Sidebar with training data stats
with st.sidebar:
    st.header("📊 Training Data Collection")
    st.markdown(f"**Storage**: `{TRAINING_DATA_FOLDER}`")
    
    if os.path.exists(TRAINING_DATA_FOLDER):
        training_files = [f for f in os.listdir(TRAINING_DATA_FOLDER) if f.endswith('.json')]
        st.metric("Training Samples", len(training_files))
        
        if training_files:
            st.markdown("**Recent Reviews:**")
            for f in sorted(training_files, reverse=True)[:5]:
                st.caption(f"📄 {f}")
    
    st.divider()
    st.markdown("""
    **How it works:**
    1. AI audits the application
    2. You review and edit if needed
    3. AI asks "Why did you change this?"
    4. Your feedback → Training data
    5. Future models learn from your expertise
    """)