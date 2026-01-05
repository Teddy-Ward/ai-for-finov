import streamlit as st
import time
from datetime import datetime
from forensic_logic import (
    analyze_document_authenticity_comprehensive,
    save_training_data_comprehensive,
    save_raw_analysis_output
)
from template_library import TEMPLATES, GOLD_STANDARD_PDF, TRAINING_DATA_FOLDER
import json
import os

st.set_page_config(page_title="Finova Forensic Scanner Pro", page_icon="🔬", layout="wide")

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'human_feedback' not in st.session_state:
    st.session_state.human_feedback = {}
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

# Custom CSS
st.markdown("""
    <style>
    .module-box {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #0066cc;
        margin: 15px 0;
    }
    .critical-box {
        background-color: #ffe6e6;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #dc3545;
    }
    .success-box {
        background-color: #e6f4ea;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #ffc107;
    }
    .metric-card {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin: 10px 0;
    }
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #4facfe 0%, #00f2fe 100%);
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.title("🔬 Finova Forensic Scanner Pro")
st.markdown("### AI-Powered Document Authentication with Multi-Layer Forensic Analysis")

# Sidebar
with st.sidebar:
    st.header("🎯 Configuration")
    
    st.markdown("**Gold Standard Reference**")
    st.info(f"📄 {TEMPLATES['NLC_Payslip_Gold_Standard']['name']}")
    st.caption(f"`{GOLD_STANDARD_PDF}`")
    
    st.divider()
    
    st.markdown("**Analysis Modules**")
    st.markdown("""
    ✅ **D2D Extraction** - LLM Vision  
    ✅ **Math Validation** - Penny-Perfect  
    ✅ **Visual Forensics** - SSIM & Spatial  
    ✅ **Digital Forensics** - ELA & Metadata  
    ✅ **Risk Scoring** - Comprehensive AI  
    """)
    
    st.divider()
    
    st.markdown("**Training Data Flywheel**")
    if os.path.exists(TRAINING_DATA_FOLDER):
        training_files = [f for f in os.listdir(TRAINING_DATA_FOLDER) if f.endswith('.json')]
        raw_files = [f for f in training_files if f.startswith('raw_analysis_')]
        human_reviewed_files = [f for f in training_files if f.startswith('forensic_analysis_')]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Raw Outputs", len(raw_files), help="Automatic saves from all analyses")
        with col2:
            st.metric("Human Reviewed", len(human_reviewed_files), help="Expert-validated training data")
        
        st.metric("Total Training Samples", len(training_files))
    st.caption(f"Storage: `{TRAINING_DATA_FOLDER}`")
    
    st.divider()
    
    with st.expander("ℹ️ About This System"):
        st.markdown("""
        **Finova Forensic Scanner Pro** uses:
        - **GPT-4o Vision** for intelligent data extraction
        - **Multi-layer forensic analysis** across 5 modules
        - **Automatic JSON export** of all analysis results
        - **Human-in-the-loop** validation for enhanced training
        - **Comprehensive risk scoring** engine
        
        **Training Data Collection:**
        - 🤖 **Raw Outputs**: Every analysis automatically saved as structured JSON
        - 👤 **Human Reviewed**: Expert-validated corrections and feedback
        - 📊 **Feature Vectors**: Pre-computed ML features ready for model training
        - 🎯 **Training Labels**: Automated classification labels (authentic/tampered)
        
        Build your proprietary fraud detection model without relying on RAG!
        """)

# Main Upload Section
st.markdown("---")
uploaded_file = st.file_uploader(
    "📤 Upload Document for Comprehensive Forensic Analysis",
    type=['pdf'],
    help="Upload a payslip PDF to compare against the gold standard",
    key=f"file_uploader_{st.session_state.uploader_key}"
)

if uploaded_file:
    st.markdown("---")
    
    # Analysis Progress
    st.markdown("### 🔍 Running Comprehensive Forensic Analysis")
    
    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("📊 Module 1/5: Data Extraction (LLM Vision)...")
        progress_bar.progress(10)
        time.sleep(0.5)
        
        status_text.text("🧮 Module 2/5: Mathematical Validation...")
        progress_bar.progress(30)
        time.sleep(0.5)
        
        status_text.text("👁️ Module 3/5: Visual Forensics (SSIM, Font Analysis)...")
        progress_bar.progress(50)
        time.sleep(0.5)
        
        status_text.text("🔬 Module 4/5: Digital Forensics (ELA, Metadata X-Ray)...")
        progress_bar.progress(70)
        time.sleep(0.5)
        
        status_text.text("🎯 Module 5/5: Risk Scoring & LLM Analysis...")
        
        # Run comprehensive analysis
        results = analyze_document_authenticity_comprehensive(uploaded_file, str(GOLD_STANDARD_PDF))
        st.session_state.analysis_results = results
        
        # Automatically save raw analysis output for training
        if 'error' not in results:
            auto_save_result = save_raw_analysis_output(
                results, 
                TRAINING_DATA_FOLDER,
                uploaded_file.name
            )
            if auto_save_result.get('success'):
                st.session_state.auto_saved_file = auto_save_result.get('filename')
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis Complete!")
        time.sleep(0.5)
        status_text.empty()
        progress_bar.empty()
    
    if 'error' in results:
        st.error(f"❌ Analysis Error: {results['error']}")
    else:
        # Show auto-save notification
        if 'auto_saved_file' in st.session_state:
            st.success(f"💾 Analysis automatically saved for training: `{st.session_state.auto_saved_file}`")
        
        # ================================================================
        # EXECUTIVE DASHBOARD
        # ================================================================
        st.markdown("## 📊 Executive Dashboard")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            score = results.get('final_trust_score', 0)
            color = "🟢" if score >= 90 else "🟡" if score >= 70 else "🔴"
            st.metric(
                "Trust Score",
                f"{score:.1f}%",
                delta=f"{score-100:.1f}%",
                delta_color="inverse"
            )
        
        with col2:
            verdict = results.get('verdict', 'UNKNOWN')
            verdict_emoji = {
                "AUTHENTIC": "✅", 
                "SUSPICIOUS": "⚠️", 
                "LIKELY_TAMPERED": "🚨", 
                "TAMPERED": "❌",
                "UNKNOWN_TEMPLATE": "🔄"
            }.get(verdict, "❓")
            st.metric("Verdict", f"{verdict_emoji} {verdict}")
        
        with col3:
            risk = results.get('risk_level', 'UNKNOWN')
            risk_color = {
                "LOW": "🟢", 
                "MODERATE": "🟡", 
                "HIGH": "🟠", 
                "CRITICAL": "🔴",
                "UNKNOWN_TEMPLATE": "🔵"
            }.get(risk, "⚪")
            st.metric("Risk Level", f"{risk_color} {risk}")
        
        with col4:
            fraud_prob = results.get('llm_comprehensive_analysis', {}).get('fraud_probability', 'N/A')
            st.metric("Fraud Probability", f"{fraud_prob}%" if isinstance(fraud_prob, (int, float)) else fraud_prob)
        
        # Check if this is an unknown template
        template_detection = results.get('template_detection', {})
        if template_detection.get('is_different_template', False):
            st.warning("⚠️ **UNKNOWN TEMPLATE DETECTED**")
            st.info(f"""
**This document appears to be a completely different payslip format** (not matching your gold standard).

**Key Indicators:**
- Data Match Rate: {template_detection.get('data_match_rate', 0):.1%}
- Visual Similarity: {template_detection.get('visual_similarity', 0):.1f}%

**This is NOT fraud** - it's likely from a different employer or payroll provider.

**Recommended Actions:**
1. 🔍 Manual review required
2. ✅ If legitimate, save this as a new gold standard template
3. 📂 Build a library of gold standards for different employers/formats
            """)
            
            # Button to save as new gold standard
            if st.button("💾 Save as New Gold Standard Template", type="primary"):
                st.info("Feature coming soon: This will save the uploaded document as a new gold standard in your template library.")
        
        # Executive Summary
        llm_analysis = results.get('llm_comprehensive_analysis', {})
        if llm_analysis and 'executive_summary' in llm_analysis:
            st.markdown('<div class="module-box">', unsafe_allow_html=True)
            st.markdown("**🎯 Executive Summary**")
            st.write(llm_analysis.get('executive_summary', 'No summary available'))
            st.markdown('</div>', unsafe_allow_html=True)
        
        # ================================================================
        # MODULE 1: DATA EXTRACTION & CROSS-VALIDATION
        # ================================================================
        st.markdown("---")
        st.markdown("## 📋 Module 1: Data Extraction & Cross-Validation (D2D)")
        
        data_comparison = results.get('data_comparison', {})
        match_rate = data_comparison.get('match_rate', 0)
        discrepancies = data_comparison.get('discrepancies', [])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Data Match Rate", f"{match_rate*100:.1f}%")
        with col2:
            st.metric("Discrepancies Found", len(discrepancies))
        
        if discrepancies:
            with st.expander("🔍 View Data Discrepancies", expanded=True):
                for disc in discrepancies[:10]:  # Show first 10
                    severity_color = {"CRITICAL": "🔴", "MODERATE": "🟡", "LOW": "🟢"}.get(disc.get('severity', 'MODERATE'), "⚪")
                    st.markdown(f"{severity_color} **{disc.get('field')}**")
                    st.markdown(f"- Gold Standard: `{disc.get('gold_standard')}`")
                    st.markdown(f"- Uploaded: `{disc.get('uploaded')}`")
                    st.markdown("---")
        
        with st.expander("📊 View Extracted Data"):
            tab1, tab2 = st.tabs(["Gold Standard", "Uploaded Document"])
            with tab1:
                st.json(results.get('gold_data_extraction', {}).get('extracted_fields', {}))
            with tab2:
                st.json(results.get('uploaded_data_extraction', {}).get('extracted_fields', {}))
        
        # ================================================================
        # MODULE 2: MATHEMATICAL & LOGICAL INTEGRITY
        # ================================================================
        st.markdown("---")
        st.markdown("## 🧮 Module 2: Mathematical & Logical Integrity")
        
        math_validation = results.get('mathematical_validation', {})
        calc_errors = math_validation.get('calculation_errors', [])
        calc_verified = math_validation.get('calculations_verified', [])
        stat_flags = math_validation.get('statistical_flags', [])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Calculations Verified", len(calc_verified))
        with col2:
            st.metric("Calculation Errors", len(calc_errors))
        with col3:
            st.metric("Statistical Flags", len(stat_flags))
        
        if calc_errors:
            st.markdown('<div class="critical-box">', unsafe_allow_html=True)
            st.markdown("### ❌ Calculation Errors Detected")
            for error in calc_errors:
                st.markdown(f"**{error.get('check')}**: {error.get('result')}")
                st.markdown(f"Expected: `{error.get('expected')}` | Actual: `{error.get('actual')}` | Difference: `{error.get('difference')}`")
            st.markdown('</div>', unsafe_allow_html=True)
        elif calc_verified:
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.markdown("### ✅ All Calculations Verified")
            for verify in calc_verified:
                st.markdown(f"**{verify.get('check')}**: {verify.get('details')}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        if stat_flags:
            with st.expander("📉 Statistical Anomalies"):
                for flag in stat_flags:
                    st.warning(f"**{flag.get('flag')}** ({flag.get('suspicion_level')}): {flag.get('detail')}")
        
        with st.expander("🔍 Tax & NI Validation"):
            tax_ni = math_validation.get('tax_ni_validation', {})
            st.json(tax_ni)
        
        # ================================================================
        # MODULE 3: VISUAL FORENSIC ANALYSIS
        # ================================================================
        st.markdown("---")
        st.markdown("## 👁️ Module 3: Visual Forensic Analysis")
        
        spatial = results.get('spatial_fingerprinting', {})
        font_analysis = results.get('font_analysis', {})
        
        col1, col2 = st.columns(2)
        with col1:
            ssim_score = spatial.get('ssim_score', 0)
            st.metric("SSIM Score", f"{ssim_score:.4f}", help="Structural Similarity Index (1.0 = identical)")
            st.metric("Layout Deviations", spatial.get('total_deviations', 0))
        with col2:
            font_score = font_analysis.get('font_consistency_score', 0)
            st.metric("Font Consistency", f"{font_score*100:.1f}%")
            st.metric("Suspicious Regions", len(font_analysis.get('suspicious_regions', [])))
        
        deviation_regions = spatial.get('deviation_regions', [])
        if deviation_regions:
            with st.expander("🗺️ Spatial Deviation Regions"):
                for region in deviation_regions[:5]:
                    loc = region.get('location', {})
                    st.markdown(f"**{region.get('severity')}**: Area {region.get('area_pixels')}px at ({loc.get('x')}, {loc.get('y')})")
        
        font_regions = font_analysis.get('suspicious_regions', [])
        if font_regions:
            with st.expander("🔤 Font Weight Variations"):
                for region in font_regions[:5]:
                    st.markdown(f"**{region.get('suspicion')}** - {region.get('region')}: Edge density diff {region.get('edge_density_diff', 0):.4f}")
        
        # ================================================================
        # MODULE 4: DEEP DIGITAL FORENSICS
        # ================================================================
        st.markdown("---")
        st.markdown("## 🔬 Module 4: Deep Digital Forensics")
        
        ela = results.get('ela_analysis', {})
        metadata = results.get('metadata_analysis', {})
        linguistic = results.get('linguistic_analysis', {})
        
        tab1, tab2, tab3 = st.tabs(["Error Level Analysis", "Metadata X-Ray", "Linguistic Analysis"])
        
        with tab1:
            if ela.get('ela_performed'):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Manipulation Likelihood", f"{ela.get('manipulation_likelihood', 0)*100:.2f}%")
                with col2:
                    risk_ela = ela.get('risk_level', 'UNKNOWN')
                    risk_color = {"LOW": "🟢", "MODERATE": "🟡", "HIGH": "🔴"}.get(risk_ela, "⚪")
                    st.metric("ELA Risk", f"{risk_color} {risk_ela}")
                
                st.info(f"**Analysis**: {ela.get('analysis', 'N/A')}")
                
                if ela.get('suspicious_pixel_percentage', 0) > 10:
                    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                    st.markdown(f"⚠️ **Warning**: {ela.get('suspicious_pixel_percentage', 0):.2f}% of pixels show inconsistent compression levels")
                    st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            if metadata.get('has_suspicious_software'):
                st.markdown('<div class="critical-box">', unsafe_allow_html=True)
                st.markdown("### 🚨 SUSPICIOUS SOFTWARE DETECTED")
                for sig in metadata.get('editing_signatures', []):
                    st.markdown(f"**{sig.get('signature').upper()}** found in `{sig.get('field')}`")
                    st.markdown(f"Value: `{sig.get('value')}`")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.success("✅ No suspicious editing software signatures detected")
            
            with st.expander("📄 Full Metadata"):
                st.json(metadata.get('metadata', {}))
        
        with tab3:
            hallucinations = linguistic.get('hallucinations_detected', 0)
            if hallucinations > 0:
                st.markdown('<div class="critical-box">', unsafe_allow_html=True)
                st.markdown(f"### 🚨 {hallucinations} AI-Generated Terms Detected")
                for detection in linguistic.get('detections', []):
                    st.markdown(f"**{detection.get('severity')}**: `{detection.get('term')}`")
                    st.markdown(f"_{detection.get('description')}_")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.success("✅ No linguistic hallucinations or AI-generated forgery terms detected")
        
        # ================================================================
        # MODULE 5: LLM COMPREHENSIVE ANALYSIS
        # ================================================================
        st.markdown("---")
        st.markdown("## 🤖 Module 5: AI-Powered Comprehensive Analysis")
        
        if llm_analysis and 'critical_findings' in llm_analysis:
            st.markdown("### 🎯 Critical Findings")
            findings = llm_analysis.get('critical_findings', [])
            for i, finding in enumerate(findings, 1):
                severity = finding.get('severity', 'MODERATE')
                severity_icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MODERATE": "🟡", "LOW": "🟢"}.get(severity, "⚪")
                st.markdown(f"{severity_icon} **{i}. {finding.get('finding', 'N/A')}** ({severity})")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Mathematical Integrity")
                st.write(llm_analysis.get('mathematical_integrity', 'No analysis available'))
                
                st.markdown("### 🔍 Visual & Spatial Analysis")
                st.write(llm_analysis.get('visual_spatial_analysis', 'No analysis available'))
            
            with col2:
                st.markdown("### 🔬 Digital Manipulation")
                st.write(llm_analysis.get('digital_manipulation', 'No analysis available'))
                
                st.markdown("### 📋 Data Discrepancies")
                st.write(llm_analysis.get('data_discrepancies', 'No discrepancies analysis available'))
            
            st.markdown("---")
            st.markdown("### ✅ Recommended Actions")
            actions = llm_analysis.get('recommended_actions', [])
            for action in actions:
                st.markdown(f"• {action}")
        
        # ================================================================
        # COMPONENT SCORES
        # ================================================================
        st.markdown("---")
        st.markdown("## 📈 Component Scores Breakdown")
        
        component_scores = results.get('component_scores', {})
        if component_scores:
            cols = st.columns(5)
            for idx, (component, score) in enumerate(component_scores.items()):
                with cols[idx]:
                    st.metric(
                        component.replace('_', ' ').title(),
                        f"{score:.1f}%",
                        delta=f"{score-100:.1f}%",
                        delta_color="inverse"
                    )
        
        # ================================================================
        # HUMAN-IN-THE-LOOP VALIDATION
        # ================================================================
        st.markdown("---")
        st.markdown("## 👤 Human Expert Review & Training Data Collection")
        
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("""
        **Your expert review improves our AI models.** Review the AI's analysis above and adjust any incorrect assessments below. 
        All fields are pre-populated with AI's predictions. Your changes will be captured as training data to teach the AI what it got wrong.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Get AI's original predictions
        ai_verdict = results.get('verdict', 'UNKNOWN')
        ai_risk = results.get('risk_level', 'UNKNOWN')
        ai_trust_score = int(results.get('final_trust_score', 50))
        ai_fraud_prob = llm_analysis.get('fraud_probability', 50)
        if isinstance(ai_fraud_prob, str):
            ai_fraud_prob = 50  # Default if N/A
        
        col1, col2 = st.columns(2)
        
        with col1:
            human_verdict = st.selectbox(
                "Your Verdict",
                ["AUTHENTIC", "SUSPICIOUS", "LIKELY_TAMPERED", "TAMPERED", "INCONCLUSIVE", "UNKNOWN_TEMPLATE"],
                index=["AUTHENTIC", "SUSPICIOUS", "LIKELY_TAMPERED", "TAMPERED", "INCONCLUSIVE", "UNKNOWN_TEMPLATE"].index(ai_verdict) if ai_verdict in ["AUTHENTIC", "SUSPICIOUS", "LIKELY_TAMPERED", "TAMPERED", "INCONCLUSIVE", "UNKNOWN_TEMPLATE"] else 0,
                key="human_verdict",
                help=f"AI predicted: {ai_verdict}"
            )
            
            risk_options = ["LOW", "MODERATE", "HIGH", "CRITICAL", "UNKNOWN_TEMPLATE"]
            ai_risk_clean = ai_risk.split(' ')[0] if ' ' in ai_risk else ai_risk  # Handle "AUTHENTIC (Identical...)"
            human_risk = st.selectbox(
                "Risk Level Assessment",
                risk_options,
                index=risk_options.index(ai_risk_clean) if ai_risk_clean in risk_options else 0,
                key="human_risk",
                help=f"AI predicted: {ai_risk}"
            )
            
            trust_score_override = st.slider(
                "Trust Score Override",
                0, 100,
                ai_trust_score,
                key="trust_override",
                help=f"AI predicted: {ai_trust_score}%. Adjust if incorrect."
            )
            
            fraud_probability = st.slider(
                "Fraud Probability (%)",
                0, 100,
                int(ai_fraud_prob),
                key="fraud_prob",
                help=f"AI predicted: {ai_fraud_prob}%. Your expert estimate."
            )
        
        with col2:
            human_notes = st.text_area(
                "Detailed Observations & Corrections",
                height=150,
                key="human_notes",
                placeholder="E.g., 'Salary amount manually altered from £1,516 to £2,516 - visible kerning inconsistency in hundreds digit'"
            )
            
            key_corrections = st.text_area(
                "Corrections to AI Findings (one per line)",
                height=100,
                key="corrections",
                placeholder="E.g., 'False positive on font weight - this is standard template variation'"
            )
            
            training_notes = st.text_area(
                "Training Data Notes",
                height=80,
                key="training_notes",
                placeholder="Key patterns this case teaches (for model training)"
            )
        
        # Calculate what changed
        verdict_changed = (human_verdict != ai_verdict)
        risk_changed = (human_risk != ai_risk_clean)
        trust_changed = (trust_score_override != ai_trust_score)
        fraud_changed = (fraud_probability != int(ai_fraud_prob))
        
        # Show change indicators
        if any([verdict_changed, risk_changed, trust_changed, fraud_changed]):
            st.info(f"📝 **Changes detected:** " + ", ".join([
                "Verdict" if verdict_changed else "",
                "Risk Level" if risk_changed else "",
                "Trust Score" if trust_changed else "",
                "Fraud Probability" if fraud_changed else ""
            ]).replace(", ,", ",").strip(", "))
        
        # Save Training Data
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("💾 Save as Training Data", type="primary", use_container_width=True):
                human_feedback = {
                    "verdict": human_verdict,
                    "risk_level": human_risk,
                    "trust_score": trust_score_override,
                    "fraud_probability": fraud_probability,
                    "detailed_notes": human_notes,
                    "corrections": [c.strip() for c in key_corrections.split('\n') if c.strip()],
                    "training_notes": [n.strip() for n in training_notes.split('\n') if n.strip()],
                    "reviewer": "human_expert",
                    "document_name": uploaded_file.name,
                    "review_timestamp": datetime.now().isoformat()
                }
                
                # Add AI vs Human comparison for training
                human_feedback['ai_vs_human_comparison'] = {
                    "ai_predictions": {
                        "verdict": ai_verdict,
                        "risk_level": ai_risk,
                        "trust_score": ai_trust_score,
                        "fraud_probability": ai_fraud_prob
                    },
                    "human_corrections": {
                        "verdict": human_verdict,
                        "risk_level": human_risk,
                        "trust_score": trust_score_override,
                        "fraud_probability": fraud_probability
                    },
                    "changes_made": {
                        "verdict_changed": verdict_changed,
                        "risk_changed": risk_changed,
                        "trust_score_changed": trust_changed,
                        "fraud_probability_changed": fraud_changed,
                        "any_changes": any([verdict_changed, risk_changed, trust_changed, fraud_changed])
                    },
                    "deltas": {
                        "trust_score_delta": trust_score_override - ai_trust_score,
                        "fraud_probability_delta": fraud_probability - int(ai_fraud_prob)
                    }
                }
                
                save_result = save_training_data_comprehensive(
                    results,
                    human_feedback,
                    TRAINING_DATA_FOLDER
                )
                
                if save_result.get('success'):
                    st.success(f"✅ Training data saved successfully!")
                    st.info(f"📁 File: `{save_result['filepath']}`")
                    st.balloons()
                else:
                    st.error(f"❌ Save failed: {save_result.get('error')}")
        
        with col2:
            if st.button("🔄 Clear & Start New", use_container_width=True):
                st.session_state.analysis_results = None
                st.session_state.human_feedback = {}
                st.session_state.uploader_key += 1  # Force file uploader to reset
                st.rerun()
        
        # Final Recommendation
        st.markdown("---")
        recommendation = results.get('recommendation', 'No recommendation available')
        
        if results.get('risk_level') == 'CRITICAL':
            st.error(f"🚨 **CRITICAL ALERT**: {recommendation}")
        elif results.get('risk_level') == 'HIGH':
            st.warning(f"⚠️ **HIGH RISK**: {recommendation}")
        elif results.get('risk_level') == 'MODERATE':
            st.info(f"ℹ️ **MODERATE RISK**: {recommendation}")
        else:
            st.success(f"✅ **LOW RISK**: {recommendation}")

else:
    # Landing state
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 50px;'>
            <h2>🔬 Upload a Document to Begin</h2>
            <p>Our AI-powered forensic system will analyze:</p>
            <ul style='text-align: left; display: inline-block;'>
                <li>📊 Data extraction & cross-validation</li>
                <li>🧮 Mathematical integrity checks</li>
                <li>👁️ Visual forensics (SSIM, fonts, spacing)</li>
                <li>🔬 Digital forensics (ELA, metadata, linguistics)</li>
                <li>🎯 Comprehensive risk scoring</li>
            </ul>
            <p><strong>Every analysis becomes training data for your proprietary models.</strong></p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("Finova Forensic Scanner Pro v2.0 | Powered by GPT-4o Vision | Multi-Layer Forensic Analysis")
