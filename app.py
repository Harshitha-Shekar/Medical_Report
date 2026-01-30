import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime
from jinja2 import Environment, BaseLoader
from io import BytesIO
import base64
from PIL import Image, ImageDraw
import numpy as np

# Try to import transformers, but make it optional
import os

# Disable BioGPT on Streamlit Cloud to avoid memory issues
STREAMLIT_CLOUD = os.environ.get('STREAMLIT_RUNTIME_ENVIRONMENT') == 'cloud'

if STREAMLIT_CLOUD:
    TRANSFORMERS_AVAILABLE = False
else:
    # Try to import transformers, but make it optional
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        TRANSFORMERS_AVAILABLE = True
    except ImportError:
        TRANSFORMERS_AVAILABLE = False
        st.warning("⚠️ Transformers library not available. Using fallback text generation.")

# Disable PDF generation on Streamlit Cloud
if os.environ.get('STREAMLIT_RUNTIME_ENVIRONMENT') == 'cloud':
    PDF_AVAILABLE = False
    PDF_METHOD = None
else:
    # Try to import pdfkit and weasyprint for PDF generation (local only)
    try:
        from weasyprint import HTML
        PDF_AVAILABLE = True
        PDF_METHOD = "weasyprint"
    except ImportError:
        try:
            import pdfkit
            PDF_AVAILABLE = True
            PDF_METHOD = "pdfkit"
        except ImportError:
            PDF_AVAILABLE = False
            PDF_METHOD = None

# =========================================================
#                    GLOBAL PAGE CONFIG
# =========================================================
st.set_page_config(page_title="AI Medical Diagnostician", layout="wide")


# =========================================================
#                    PREMIUM UI CSS
# =========================================================

st.markdown("""
<style>
/* Remove extra top padding so page starts higher */
.appview-container .main .block-container {
    padding-top: 25px !important;
}

body {
    background-color:#F5F2F2;
}

.stApp {
    background-color:#F5F2F2;
}

.appview-container .main .block-container {
    background-color:#F5F2F2;
}

.section-header{
    background:#2e6f22;
    color:white;
    padding:14px 18px;
    border-radius:8px;
    font-size:20px;
    font-weight:800;
    border-left:6px solid #013220;
}

/* UNIFORM SPACING FOR ALL SECTIONS */
.uniform-spacing {
    margin-top: 20px !important;
    margin-bottom: 20px !important;
}

.block-container { 
    padding-top: 10px; 
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* ===== FIX INPUT VISIBILITY ===== */
.stTextInput input,
.stTextArea textarea,
.stSelectbox div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    color: #000000 !important;
    border: 1px solid #1f6fb2 !important;
    border-radius: 6px !important;
    font-size: 15px !important;
}

/* Placeholder text */
.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
    color: #6b6b6b !important;
    opacity: 1 !important;
}

/* Input labels */
label {
    font-weight: 700 !important;
    color: #022869 !important;
    font-size: 15px !important;
}

/* Focus (when user clicks input) */
.stTextInput input:focus,
.stTextArea textarea:focus {
    outline: none !important;
    border: 2px solid #f44336 !important;
    box-shadow: 0 0 6px rgba(244,67,54,0.4) !important;
}

</style>
""", unsafe_allow_html=True)



# ===============================
# MAIN HEADING PANEL
# ===============================

st.markdown("""
<div class="uniform-spacing" style="
background:#022869;
color:white;
padding:35px 25px;
border-radius:6px;
text-align:center;
width:100%;
margin-left:auto;
margin-right:auto;
font-family:Segoe UI;
">

<h1 style="font-weight:900; font-size:32px; margin:0 0 10px 0; color:#ffffff;">
🩺 AI Medical Diagnostician — Intelligent Patient Journey Assistant
</h1>

<h3 style="font-weight:500; margin:0 0 10px 0; color:#ffffff;">
From Patient Intake to Diagnosis — Fully Automated, Clinically Aligned.
</h3>

<p style="font-size:15px; margin:0;">
Streamline medical workflows with AI-generated lab reports, imaging
summaries, and diagnostic insights.
</p>

</div>
""", unsafe_allow_html=True)



# ===============================
# DESCRIPTION PANEL
# ===============================
st.markdown("""
<div class="uniform-spacing" style="
background:#1C4D8D;
color:white;
padding:22px;
border-radius:6px;
text-align:center;
font-size:15px;
line-height:1.6;">
This application simulates a real-world clinical pipeline.
Simply enter basic patient details, and the system will intelligently auto-fill known information,
generate lab & radiology findings, and produce an AI-driven diagnostic report.
</div>
""", unsafe_allow_html=True)



# ===============================
# PATIENT INFORMATION SECTION
# ===============================
st.markdown("""
<div class='section-header uniform-spacing'>
Patient Information
</div>

<div class="uniform-spacing" style="
background:#e3ffd4;
padding:15px;
border-radius:10px;
border:2px solid #4CAF50;
font-size:15px;
line-height:1.6;">
This section captures the essential demographic and clinical inputs required for the AI model to understand the patient context. When you type the patient's name, the system searches the knowledge base (KB) and auto-fills:
<ul style="margin:10px 0;">
<li><b>Age</b></li>
<li><b>Gender</b></li>
<li><b>Medical history</b></li>
<li><b>Presenting symptoms / clinical query</b></li>
</ul>

This reduces manual effort and ensures consistency across the entire patient journey.
</div>
            
<div class="uniform-spacing" style="
background:#1f6fb2;
color:white;
padding:18px;
border-radius:12px;
font-size:15px;
line-height:1.6;">
<b>Provide the patient's core information (Patient Name).</b><br>
If the person already exists in the system, their profile will automatically load with verified clinical details.
</div>
            
""", unsafe_allow_html=True)



# =========================================================
#                KNOWLEDGE BASE (MODIFIED TO USE CSV)
# =========================================================
class MedicalKnowledgeBase:
    def __init__(self, patients_db_path: str = "patients.csv"):
        self.patients_db_path = patients_db_path
        self.patients = self._load_patients_from_csv()
        
        # Load blood tests and scans CSV files
        try:
            for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    self.blood_tests_df = pd.read_csv("blood_tests.csv", encoding=encoding)
                    self.blood_tests_df.fillna('', inplace=True)
                    break
                except UnicodeDecodeError:
                    continue
        except FileNotFoundError:
            st.error("❌ blood_tests.csv not found!")
            self.blood_tests_df = pd.DataFrame()
            
        try:
            for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    self.scans_df = pd.read_csv("scans.csv", encoding=encoding)
                    self.scans_df.fillna('', inplace=True)
                    break
                except UnicodeDecodeError:
                    continue
        except FileNotFoundError:
            st.error("❌ scans.csv not found!")
            self.scans_df = pd.DataFrame()

    def _load_patients_from_csv(self) -> Dict[str, Any]:
        try:
            # Try multiple encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    df = pd.read_csv(self.patients_db_path, encoding=encoding)
                    df.fillna('', inplace=True)
                    if 'patient_name' in df.columns:
                        df.drop_duplicates(subset=['patient_name'], keep='first', inplace=True)
                    return df.set_index('patient_name').to_dict('index')
                except UnicodeDecodeError:
                    continue
            
            # If all encodings fail, show error
            st.error("❌ Could not decode patients.csv with any standard encoding")
            return {}
        except FileNotFoundError:
            st.error("❌ patients.csv not found!")
            return {}

    def search_patient_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        if not name:
            return None
        for p_name, data in self.patients.items():
            if str(p_name).lower().strip() == str(name).lower().strip():
                return data
        return None
    
    def get_blood_tests(self, patient_name: str) -> list:
        """Get blood test results for a patient from CSV"""
        if self.blood_tests_df.empty:
            return []
        
        tests = self.blood_tests_df[self.blood_tests_df['patient_name'].str.lower() == patient_name.lower()]
        return tests.to_dict('records')
    
    def get_scans(self, patient_name: str) -> list:
        """Get scan results for a patient from CSV"""
        if self.scans_df.empty:
            return []
        
        scans = self.scans_df[self.scans_df['patient_name'].str.lower() == patient_name.lower()]
        return scans.to_dict('records')



# =========================================================
#               TEMPLATE STRINGS
# =========================================================

class TemplateStrings:
    
    SHARED_BASE = """
    <style>
        body { 
            font-family: 'Segoe UI', Arial, sans-serif; 
            color:#222; 
            margin: 20px;
        }
        .header { 
            border-bottom: 3px solid #022869; 
            padding-bottom:15px; 
            margin-bottom:25px; 
        }
        .hospital-name { 
            font-size: 24px; 
            font-weight:bold; 
            color:#022869;
            margin-bottom: 5px;
        }
        .address { 
            font-size: 12px; 
            color:#555; 
            line-height: 1.6;
        }
        .divider {
            border-bottom: 2px solid #ddd;
            margin: 15px 0;
        }
        .verified-by {
            font-size: 11px;
            color: #666;
            margin-top: 10px;
        }
        .section-title { 
            font-size: 16px; 
            font-weight:bold; 
            color:#022869;
            background: #e8f4f8;
            padding: 10px 15px;
            margin-top:25px;
            margin-bottom: 15px;
            border-left:5px solid #022869;
        }
        .box { 
            border:1px solid #ddd; 
            padding:15px; 
            border-radius:6px; 
            background:#fafafa;
            margin-bottom: 15px;
        }
        .patient-details {
            background: #f0f8ff;
            padding: 15px;
            border-radius: 6px;
            border: 1px solid #022869;
        }
        .patient-details table {
            width: 100%;
            border-collapse: collapse;
        }
        .patient-details td {
            padding: 8px;
            font-size: 13px;
        }
        .patient-details td:first-child {
            font-weight: bold;
            width: 180px;
        }
    </style>

    <div class="header">
        <div class="hospital-name">Clariovex Diagnostics</div>
        <div class="address">
            123 Medical Center Drive, Healthcare District, City - 400001<br>
            +91-22-2345-6789 | reports@clariovexdiagnostics.com | www.clariovexdiagnostics.com
        </div>
        <div class="divider"></div>
        <div class="verified-by">
            <strong>Verified by:</strong> Dr. Rajesh Kumar, MD Pathology<br>
            <strong>24/7 Helpdesk:</strong> +91-22-2345-6789 | support@clariovexdiagnostics.com
        </div>
    </div>

    <div class="patient-details">
        <table>
            <tr>
                <td>Patient Name:</td>
                <td>{{ patient.patient_name }}</td>
            </tr>
            <tr>
                <td>Age/Sex:</td>
                <td>{{ patient.age }} / {{ patient.gender }}</td>
            </tr>
            <tr>
                <td>Patient ID:</td>
                <td>ZAI-{{ report_date | replace(' ', '-') }}-{{ patient.patient_name[:3] | upper }}{{ patient.age }}</td>
            </tr>
            <tr>
                <td>Date of Report:</td>
                <td>{{ report_date }}</td>
            </tr>
            <tr>
                <td>Referring Physician:</td>
                <td>Dr. A. Kumar, MD (Cardiology)</td>
            </tr>
        </table>
    </div>
    """

    LAB_TEMPLATE = SHARED_BASE + """
    <div class="section-title">SECTION A — LABORATORY INVESTIGATIONS</div>

    <h3 style="color: #022869; font-size: 15px; margin: 20px 0 10px 0;">1. Hematology & Blood Chemistry</h3>
    <table width="100%" border="1" cellspacing="0" cellpadding="8" style="border-collapse: collapse; font-size: 13px;">
        <tr style="background:#e8f4f8; font-weight: bold;">
            <th style="text-align: left; padding: 10px;">Test Parameter</th>
            <th style="text-align: center;">Result</th>
            <th style="text-align: center;">Reference Range</th>
            <th style="text-align: center;">Status</th>
        </tr>

        {% for t in report.test_results %}
        <tr style="{% if t.flag != 'N' %}background: #fff3e0;{% endif %}">
            <td style="padding: 8px;">{{ t.test_name }}</td>
            <td style="text-align: center;"><strong>{{ t.result }}</strong></td>
            <td style="text-align: center;">{{ t.range }}</td>
            <td style="text-align: center;">
                {% if t.flag == 'H' %}
                    <span style="color: #d32f2f; font-weight: bold;">↑ HIGH</span>
                {% elif t.flag == 'L' %}
                    <span style="color: #1976d2; font-weight: bold;">↓ LOW</span>
                {% else %}
                    <span style="color: #388e3c;">Normal</span>
                {% endif %}
            </td>
        </tr>
        {% endfor %}
    </table>

    <div class="section-title">SECTION B — CLINICAL INTERPRETATION</div>
    <div class="box" style="line-height: 1.7;">
        {{ report.interpretation_guidance }}
    </div>

    <div class="section-title">SECTION C — RECOMMENDATIONS</div>
    <div class="box">
        <ul style="line-height: 1.8; margin: 0; padding-left: 20px;">
        {% for r in report.recommendations %}
        <li style="margin-bottom: 8px;">{{ r }}</li>
        {% endfor %}
        </ul>
    </div>

    <div style="margin-top: 40px; padding-top: 20px; border-top: 2px solid #ddd;">
        <p style="font-size: 11px; color: #666; line-height: 1.6;">
            <strong>Note:</strong> This report should be interpreted in conjunction with clinical findings and other diagnostic investigations. 
            For any queries, please contact our laboratory at +91-22-2345-6789.
        </p>
    </div>
    """

    RADIOLOGY_TEMPLATE = SHARED_BASE + """
    <div class="section-title">SECTION A — EXAMINATION DETAILS</div>
    
    {% for scan in report.scans %}
    <div class="box">
        <h3 style="color: #022869; font-size: 15px; margin-top: 0;">{{ scan.scan_type }}</h3>
        <table style="width: 100%; font-size: 13px; line-height: 1.8;">
            <tr>
                <td style="width: 150px; font-weight: bold;">Findings:</td>
                <td>{{ scan.findings }}</td>
            </tr>
            <tr>
                <td style="font-weight: bold;">Interpretation:</td>
                <td>{{ scan.interpretation }}</td>
            </tr>
        </table>
    </div>
    {% endfor %}

    <div class="section-title">SECTION B — RECOMMENDATIONS</div>
    <div class="box">
        <ul style="line-height: 1.8; margin: 0; padding-left: 20px;">
        {% for r in report.recommendations %}
        <li style="margin-bottom: 8px;">{{ r }}</li>
        {% endfor %}
        </ul>
    </div>

    <div style="margin-top: 40px; padding-top: 20px; border-top: 2px solid #ddd;">
        <p style="font-size: 11px; color: #666; line-height: 1.6;">
            <strong>Reported by:</strong> Dr. S. Mehta, MD (Radiology)<br>
            <strong>Note:</strong> Clinical correlation is recommended. For any queries regarding this report, 
            please contact the Radiology Department at +91-22-2345-6789.
        </p>
    </div>
    """

    DIAGNOSIS_TEMPLATE = SHARED_BASE + """
    <div style="font-size:20px; font-weight:bold; text-align:center;
                margin-top:10px; margin-bottom:20px; color: #022869;">
        COMPREHENSIVE DIAGNOSTIC REPORT
    </div>

    <div class="section-title">SECTION A — CLINICAL SUMMARY</div>
    
    <h3 style="color: #022869; font-size: 15px; margin: 20px 0 10px 0;">1. Chief Complaint</h3>
    <div class="box" style="line-height: 1.7;">
        {{ patient.clinical_query }}
    </div>

    <h3 style="color: #022869; font-size: 15px; margin: 20px 0 10px 0;">2. History of Present Illness</h3>
    <div class="box" style="line-height: 1.7;">
        {{ diagnosis.history_description }}
    </div>

    <div class="section-title">SECTION B — RISK FACTORS</div>
    <div class="box" style="line-height: 1.7;">
        <ul style="margin: 0; padding-left: 20px; line-height: 2.0;">
            {% if patient.risk_factors %}
            {% for risk in patient.risk_factors.split(',') %}
            <li>{{ risk.strip() }}</li>
            {% endfor %}
            {% else %}
            <li>Medical History: {{ patient.medical_history }}</li>
            <li>Age: {{ patient.age }} years</li>
            <li>Gender: {{ patient.gender }}</li>
            {% endif %}
        </ul>
    </div>

    <div class="section-title">SECTION C — PHYSICAL EXAMINATION</div>
    
    {% if patient.physical_exam %}
    <div class="box" style="line-height: 1.7;">
        <strong>General</strong><br>
        {{ patient.physical_exam }}
    </div>
    {% else %}
    <div class="box" style="line-height: 1.7;">
        <strong>General</strong>
        <ul style="margin: 5px 0; padding-left: 20px;">
            <li>Patient alert and oriented</li>
            <li>Vital signs stable</li>
        </ul>
    </div>
    {% endif %}

    <div class="section-title">SECTION D — ECG INTERPRETATION (12-LEAD)</div>
    <div class="box" style="line-height: 1.7;">
        <strong>Findings</strong>
        <ul style="margin: 10px 0; padding-left: 20px;">
            {% for scan in diagnosis.ecg_findings %}
            <li>{{ scan }}</li>
            {% endfor %}
        </ul>
        <strong style="margin-top: 15px; display: block;">Conclusion:</strong> {{ diagnosis.ecg_conclusion }}
    </div>

    <div class="section-title">SECTION E — LAB INVESTIGATIONS</div>
    
    <h3 style="color: #022869; font-size: 15px; margin: 20px 0 10px 0;">1. Laboratory Results Summary</h3>
    <div class="box" style="line-height: 1.7; background: #f0f8ff;">
        {{ diagnosis.lab_interpretation }}
    </div>

    <h3 style="color: #022869; font-size: 15px; margin: 20px 0 10px 0;">2. Key Findings</h3>
    <div class="box" style="line-height: 1.7;">
        {{ diagnosis.lab_key_findings }}
    </div>

    <div class="section-title">SECTION F — IMAGING STUDIES</div>
    
    <h3 style="color: #022869; font-size: 15px; margin: 20px 0 10px 0;">Findings</h3>
    <div class="box" style="line-height: 1.7; background: #f0f8ff;">
        {{ diagnosis.imaging_interpretation }}
    </div>

    <h3 style="color: #022869; font-size: 15px; margin: 20px 0 10px 0;">Summary</h3>
    <div class="box" style="line-height: 1.7;">
        {{ diagnosis.imaging_summary }}
    </div>

    <div class="section-title">SECTION G — FINAL DIAGNOSIS & IMPRESSION</div>
    <div class="box" style="background:#fff3e0; border-left: 5px solid #ff9800; padding: 20px;">
        <strong style="font-size: 15px; color: #e65100; line-height: 1.8;">
            {{ diagnosis.impression }}
        </strong>
    </div>

    <div class="section-title">SECTION H — MANAGEMENT & RECOMMENDATIONS</div>
    <div class="box">
        <ul style="line-height: 1.8; margin: 0; padding-left: 20px;">
            {% for rec in diagnosis.recommendations %}
            <li style="margin-bottom: 8px;">{{ rec }}</li>
            {% endfor %}
        </ul>
    </div>

    <div style="margin-top: 40px; padding-top: 20px; border-top: 2px solid #ddd;">
        <table style="width: 100%; font-size: 12px;">
            <tr>
                <td style="width: 50%;">
                    <strong>Compiled by:</strong> Dr. Rajesh Kumar, MD<br>
                    <strong>Department:</strong> Internal Medicine
                </td>
                <td style="width: 50%; text-align: right;">
                    <strong>Date:</strong> {{ report_date }}<br>
                    <strong>Contact:</strong> +91-22-2345-6789
                </td>
            </tr>
        </table>
        <p style="font-size: 11px; color: #666; line-height: 1.6; margin-top: 15px;">
            <strong>Disclaimer:</strong> This comprehensive diagnostic report integrates multiple diagnostic modalities. 
            Clinical correlation and physician judgment are essential for final diagnosis and treatment planning.
        </p>
    </div>
    """



# =========================================================
#                   LLM ENGINE
# =========================================================
@st.cache_resource
def load_biogpt_model():
    # Don't load model on Streamlit Cloud
    if os.environ.get('STREAMLIT_RUNTIME_ENVIRONMENT') == 'cloud':
        st.info("💡 Running in cloud mode with optimized text generation.")
        return None, None
    
    if not TRANSFORMERS_AVAILABLE:
        return None, None
    
    try:
        with st.spinner("🔄 Loading BioGPT model... This may take a few moments on first run."):
            tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
            model = AutoModelForCausalLM.from_pretrained("microsoft/biogpt")
            st.success("✅ BioGPT model loaded successfully!")
            return model, tokenizer
    except Exception as e:
        st.error(f"❌ Error loading BioGPT model: {str(e)}")
        st.info("💡 Using fallback text generation mode.")
        return None, None


class MedicalLLMInterface:
    def __init__(self, kb):
        self.model, self.tokenizer = load_biogpt_model()
        self.kb = kb  # Store knowledge base reference

    def process(self, query, patient, report_type):
        """Process patient data and generate reports from CSV knowledge base"""
        patient_name = patient.get('patient_name', '')
        
        data = {
            "interpretation_guidance": "",
            "recommendations": [],
            "images": []
        }

        if report_type == "lab":
            # Get test results from CSV
            test_results = self.kb.get_blood_tests(patient_name)
            
            if not test_results:
                st.warning(f"No lab test data found for {patient_name}")
                return data
            
            # Filter tests based on clinical query/symptoms
            filtered_tests = self._filter_relevant_tests(test_results, query, patient)
            
            if not filtered_tests:
                # If no specific filter matches, use all tests
                filtered_tests = test_results
                
            data["test_results"] = filtered_tests
            
            # Use interpretation from CSV
            interpretation = filtered_tests[0].get('interpretation', 'Lab analysis completed') if filtered_tests else ''
            data["interpretation_guidance"] = interpretation
            
            # Generate recommendations based on findings
            recommendations = []
            high_tests = [t for t in filtered_tests if t.get('flag') == 'H']
            
            if any('troponin' in t['test_name'].lower() for t in high_tests):
                recommendations.extend([
                    "Immediate cardiology consultation required",
                    "Continuous cardiac monitoring recommended",
                    "Consider emergency coronary intervention"
                ])
            elif any('glucose' in t['test_name'].lower() or 'hba1c' in t['test_name'].lower() for t in high_tests):
                recommendations.extend([
                    "Endocrinology referral for diabetes management",
                    "Lifestyle modifications including diet and exercise counseling"
                ])
            
            if any('cholesterol' in t['test_name'].lower() or 'ldl' in t['test_name'].lower() for t in high_tests):
                recommendations.append("Consider lipid-lowering therapy if clinically indicated")
            
            if not recommendations:
                recommendations = [
                    "Follow-up testing recommended in 3 months to monitor trends",
                    "Clinical correlation with patient symptoms advised"
                ]
            
            data["recommendations"] = recommendations
            
        else:  # radiology
            # Get scan results from CSV
            scans = self.kb.get_scans(patient_name)
            
            if not scans:
                st.warning(f"No imaging data found for {patient_name}")
                return data
            
            data["scans"] = scans
            
            # Generate recommendations based on scan findings
            recommendations = []
            interpretations_text = ' '.join([s.get('interpretation', '') for s in scans]).lower()
            
            if 'stemi' in interpretations_text or 'infarction' in interpretations_text:
                recommendations.extend([
                    "Immediate cardiac catheterization and possible PCI",
                    "Dual antiplatelet therapy and appropriate cardiac medications",
                    "Cardiac rehabilitation program enrollment"
                ])
            elif 'pneumonia' in interpretations_text:
                recommendations.extend([
                    "Initiate appropriate antibiotic therapy",
                    "Follow-up chest X-ray in 4-6 weeks to confirm resolution"
                ])
            else:
                recommendations.extend([
                    "Clinical correlation with patient presentation recommended",
                    "Follow-up imaging if symptoms persist or worsen"
                ])
            
            data["recommendations"] = recommendations

        return data
    
    def _filter_relevant_tests(self, test_results, query, patient):
        """Filter lab tests based on symptoms and clinical presentation"""
        query_lower = query.lower() if query else ''
        medical_history = patient.get('medical_history', '').lower()
        combined_context = query_lower + ' ' + medical_history
        
        # Define test categories and their relevant keywords
        test_categories = {
            'cardiac': {
                'keywords': ['chest pain', 'heart', 'cardiac', 'angina', 'myocardial', 'infarction', 
                           'stemi', 'shortness of breath', 'diaphoresis', 'jaw pain', 'arm pain'],
                'tests': ['troponin', 'ck-mb', 'ck', 'ldl', 'hdl', 'cholesterol', 'triglyceride', 
                         'wbc', 'hemoglobin', 'platelet', 'creatinine', 'bun', 'k+', 'na+']
            },
            'diabetes': {
                'keywords': ['thirst', 'urination', 'glucose', 'diabetes', 'fatigue', 'weight loss',
                           'polyuria', 'polydipsia', 'hyperglycemia'],
                'tests': ['glucose', 'hba1c', 'sugar', 'cholesterol', 'ldl', 'hdl', 'triglyceride',
                         'wbc', 'hemoglobin', 'platelet', 'creatinine', 'bun', 'egfr']
            },
            'infection': {
                'keywords': ['fever', 'cough', 'infection', 'pneumonia', 'sputum', 'respiratory',
                           'breath', 'copd', 'inflammatory'],
                'tests': ['wbc', 'crp', 'hemoglobin', 'platelet', 'creatinine', 'bun']
            },
            'general': {
                'keywords': [],  # Always include these basic tests
                'tests': ['wbc', 'hemoglobin', 'platelet']
            }
        }
        
        # Determine which category matches
        relevant_test_names = set()
        matched_category = False
        
        for category, data in test_categories.items():
            # Check if any keyword matches
            if category == 'general' or any(keyword in combined_context for keyword in data['keywords']):
                relevant_test_names.update([test.lower() for test in data['tests']])
                if category != 'general':
                    matched_category = True
        
        # If no specific category matched, return all tests
        if not matched_category:
            return test_results
        
        # Filter tests based on relevant test names
        filtered = []
        for test in test_results:
            test_name_lower = test.get('test_name', '').lower()
            # Check if test name contains any of the relevant test keywords
            if any(relevant in test_name_lower for relevant in relevant_test_names):
                filtered.append(test)
        
        return filtered if filtered else test_results

    def synthesize(self, lab_data, rad_data, patient):
        """Generate final diagnostic synthesis combining all data"""
        query = patient.get('clinical_query', '')
        age = patient.get('age', 'unknown')
        gender = patient.get('gender', 'unknown')
        name = patient.get('patient_name', '')
        medical_history = patient.get('medical_history', '')
        
        lab_interp = lab_data.get('interpretation_guidance', '')
        
        # Get imaging interpretation from scans
        scans = rad_data.get('scans', [])
        
        # Generate detailed history description
        history_description = f"{name}, a {age}-year-old {gender} with {medical_history.lower()}, presented with {query.lower()}"
        
        # Extract ECG findings and conclusion
        ecg_findings = []
        ecg_conclusion = "Normal sinus rhythm"
        imaging_details = []
        
        for scan in scans:
            scan_type = scan.get('scan_type', '')
            findings = scan.get('findings', '')
            interpretation = scan.get('interpretation', '')
            
            if 'ecg' in scan_type.lower():
                # Parse ECG findings
                if findings:
                    ecg_findings = [f.strip() for f in findings.split(',')]
                ecg_conclusion = interpretation
            else:
                imaging_details.append(f"{scan_type}: {interpretation}")
        
        if not ecg_findings:
            ecg_findings = ["Normal sinus rhythm", "No ST segment changes", "No arrhythmias noted"]
        
        imaging_interpretation = '; '.join(imaging_details) if imaging_details else "Imaging studies completed"
        
        # Generate lab key findings
        test_results = lab_data.get('test_results', [])
        high_tests = [t for t in test_results if t.get('flag') == 'H']
        low_tests = [t for t in test_results if t.get('flag') == 'L']
        
        lab_key_findings = []
        if high_tests:
            lab_key_findings.append(f"Elevated: {', '.join([t['test_name'] for t in high_tests[:5]])}")
        if low_tests:
            lab_key_findings.append(f"Decreased: {', '.join([t['test_name'] for t in low_tests[:5]])}")
        if not lab_key_findings:
            lab_key_findings.append("All parameters within normal limits")
        
        lab_key_findings_text = '. '.join(lab_key_findings)
        
        # Generate imaging summary
        imaging_summary = []
        for scan in scans:
            scan_type = scan.get('scan_type', '')
            interpretation = scan.get('interpretation', '')
            if 'ecg' not in scan_type.lower():
                imaging_summary.append(f"• {scan_type}: {interpretation}")
        
        imaging_summary_text = '\n'.join(imaging_summary) if imaging_summary else "No significant abnormalities detected"
        
        # Generate diagnosis based on findings
        diagnosis_text = ""
        
        # Check for STEMI
        if any('stemi' in s.get('interpretation', '').lower() or 'infarction' in s.get('interpretation', '').lower() for s in scans):
            diagnosis_text = f"ACUTE ST-ELEVATION MYOCARDIAL INFARCTION (STEMI) — Anterior Wall\n\n"
            diagnosis_text += f"A {age}-year-old {gender} patient presenting with acute chest pain and ECG changes consistent with acute anterior wall myocardial infarction. "
            diagnosis_text += f"Elevated cardiac biomarkers confirm active myocardial necrosis. Angiography demonstrates complete LAD occlusion successfully treated with primary PCI and stent placement."
            
            recommendations = [
                "IMMEDIATE: Dual antiplatelet therapy (Aspirin 325mg + Ticagrelor 180mg loading dose)",
                "Beta-blocker therapy (Metoprolol or Carvedilol) unless contraindicated",
                "ACE inhibitor or ARB for ventricular remodeling prevention",
                "High-intensity statin therapy (Atorvastatin 80mg daily)",
                "Continuous cardiac monitoring for 24-48 hours",
                "Serial troponin monitoring until peak and trend downward",
                "Echocardiography to assess ventricular function and complications",
                "Cardiac rehabilitation program enrollment prior to discharge",
                "Aggressive risk factor modification: smoking cessation, blood pressure control, lipid management",
                "Close cardiology follow-up within 1-2 weeks post-discharge"
            ]
        
        # Check for diabetes
        elif 'glucose' in lab_interp.lower() or 'diabetes' in lab_interp.lower():
            diagnosis_text = f"TYPE 2 DIABETES MELLITUS — Inadequate Glycemic Control\n\n"
            diagnosis_text += f"A {age}-year-old {gender} patient with significantly elevated fasting glucose and HbA1c levels consistent with uncontrolled Type 2 Diabetes Mellitus. "
            diagnosis_text += f"Associated dyslipidemia present, indicating metabolic syndrome. No evidence of acute diabetic complications at this time."
            
            recommendations = [
                "Endocrinology referral for comprehensive diabetes management",
                "Initiate or adjust oral hypoglycemic therapy (Metformin as first-line unless contraindicated)",
                "Consider additional agents: SGLT2 inhibitor or GLP-1 agonist based on cardiovascular risk profile",
                "Diabetes education program enrollment",
                "Home blood glucose monitoring: fasting and 2-hour postprandial",
                "Dietary consultation with registered dietitian (carbohydrate counting, portion control)",
                "Regular exercise program: 150 minutes moderate-intensity aerobic activity per week",
                "HbA1c monitoring every 3 months until goal achieved, then every 6 months",
                "Annual comprehensive diabetic foot examination",
                "Annual diabetic retinopathy screening",
                "Annual urine microalbumin screening for diabetic nephropathy",
                "Statin therapy for cardiovascular risk reduction"
            ]
        
        # Check for pneumonia
        elif any('pneumonia' in s.get('interpretation', '').lower() for s in scans):
            diagnosis_text = f"COMMUNITY-ACQUIRED PNEUMONIA — Right Lower Lobe\n\n"
            diagnosis_text += f"A {age}-year-old {gender} patient presenting with respiratory symptoms and radiographic evidence of right lower lobe consolidation. "
            diagnosis_text += f"Elevated WBC count and inflammatory markers consistent with bacterial pneumonia. Clinical presentation supports community-acquired pneumonia requiring antibiotic therapy."
            
            recommendations = [
                "Empiric antibiotic therapy: Ceftriaxone 1g IV daily + Azithromycin 500mg PO daily",
                "Alternative regimen: Levofloxacin 750mg PO/IV daily (respiratory fluoroquinolone)",
                "Supportive care: adequate hydration (2-3L fluids daily), rest",
                "Supplemental oxygen if SpO2 <92% to maintain saturation >92%",
                "Antipyretics for fever management (Acetaminophen or Ibuprofen)",
                "Monitor for complications: pleural effusion, empyema, sepsis",
                "Follow-up chest X-ray in 4-6 weeks to confirm resolution",
                "Pulmonary function testing if symptoms persist after treatment",
                "Pneumococcal and influenza vaccination if not up to date",
                "Smoking cessation counseling if applicable",
                "Close follow-up in 2-3 days to assess treatment response"
            ]
        
        else:
            diagnosis_text = f"Clinical Assessment — Further Evaluation Required\n\n"
            diagnosis_text += f"Based on clinical presentation and diagnostic workup, this {age}-year-old {gender} patient demonstrates findings requiring medical management and close monitoring. "
            diagnosis_text += f"Comprehensive assessment including laboratory and imaging studies has been completed."
            
            recommendations = [
                "Regular follow-up appointments as scheduled",
                "Continue monitoring of relevant clinical parameters",
                "Lifestyle modifications including healthy diet and regular exercise",
                "Medication adherence counseling if applicable",
                "Patient education regarding warning signs and when to seek medical attention",
                "Repeat laboratory testing in 3 months to assess trends",
                "Consider specialty referral if symptoms persist or worsen"
            ]
        
        return {
            "history_description": history_description,
            "ecg_findings": ecg_findings,
            "ecg_conclusion": ecg_conclusion,
            "lab_interpretation": lab_interp,
            "lab_key_findings": lab_key_findings_text,
            "imaging_interpretation": imaging_interpretation,
            "imaging_summary": imaging_summary_text,
            "impression": diagnosis_text,
            "recommendations": recommendations
        }


# =========================================================
#                REPORT GENERATOR
# =========================================================
class ReportGenerator:
    def __init__(self, kb):
        self.llm = MedicalLLMInterface(kb)
        self.env = Environment(loader=BaseLoader())

    def generate(self, query, patient, rtype):
        """Generate lab or radiology report"""
        analysis = self.llm.process(query, patient, rtype)
        html = self.env.from_string(getattr(TemplateStrings,
                                            f"{rtype.upper()}_TEMPLATE")).render(
            patient=patient,
            report=analysis,
            report_date=datetime.now().strftime("%d %B %Y")
        )
        return html, analysis

    def diagnosis(self, lab, rad, patient):
        """Generate final diagnostic report"""
        diag = self.llm.synthesize(lab, rad, patient)
        html = self.env.from_string(TemplateStrings.DIAGNOSIS_TEMPLATE).render(
            patient=patient,
            diagnosis=diag,
            report_date=datetime.now().strftime("%d %B %Y")
        )
        return html, diag

    def html_to_pdf(self, html_content):
        """Convert HTML to PDF"""
        if not PDF_AVAILABLE:
            return None
        
        try:
            if PDF_METHOD == "weasyprint":
                # Using WeasyPrint - correct usage
                from weasyprint import HTML as WeasyHTML
                pdf_bytes = WeasyHTML(string=html_content).write_pdf()
                return pdf_bytes
            elif PDF_METHOD == "pdfkit":
                # Using pdfkit (requires wkhtmltopdf installed)
                pdf_bytes = pdfkit.from_string(html_content, False)
                return pdf_bytes
        except TypeError as e:
            st.error(f"PDF Generation Error: {str(e)}")
            st.info("There might be an issue with the PDF library configuration. The HTML version is still available for download.")
            return None
        except Exception as e:
            st.error(f"Error generating PDF: {str(e)}")
            st.info("PDF generation failed. Please download the HTML version instead.")
            return None


# =========================================================
#                STREAMLIT APP
# =========================================================
def main():

    kb = MedicalKnowledgeBase()
    rg = ReportGenerator(kb)

    name = st.text_input("Patient Name (Type and press Enter)")

    if "age" not in st.session_state: st.session_state.age = ""
    if "gender" not in st.session_state: st.session_state.gender = ""
    if "history" not in st.session_state: st.session_state.history = ""
    if "query" not in st.session_state: st.session_state.query = ""

    if name:
        patient = kb.search_patient_by_name(name)

        if patient:
            st.success(f"✅ Record found for **{name}** & details auto-filled")
            st.session_state.age = str(patient.get("age", ""))
            st.session_state.gender = patient.get("gender", "")
            st.session_state.history = patient.get("medical_history", "")
            st.session_state.query = patient.get("clinical_query", "")
        else:
            st.info(f"ℹ️ No existing record for **{name}**. Please enter details manually.")

    col1, col2 = st.columns(2)
    with col1:
        age = st.text_input("Age", value=st.session_state.age)
    with col2:
        gender = st.text_input("Gender", value=st.session_state.gender)

    history = st.text_area("Medical History", value=st.session_state.history)
    query = st.text_area("Clinical Query / Symptoms", value=st.session_state.query)



    # ===============================
    # LAB & RADIOLOGY REQUEST SECTION
    # ===============================
    st.markdown("""
    <div class="section-header uniform-spacing">
    Lab & Radiology Reports Request
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="uniform-spacing" style="
    background:#4169A9;
    color:white;
    padding:28px;
    border-radius:8px;
    font-size:16px;
    line-height:1.6">

    After entering patient information, this step simulates what happens in a real hospital workflow.
    The system generates:

    <ul style="margin:10px 0;">
    <li>Lab test summaries (blood tests, biomarkers, etc.)</li>
    <li>Radiology findings (X-ray/CT/MRI text-based interpretation)</li>
    </ul>

    Reports are generated from the knowledge base CSV files.<br><br>

    <b>
    Click below to automatically generate lab and imaging reports based on clinical presentation.
    These reports will be used in the next step for final diagnosis.
    </b>

    </div>
    """, unsafe_allow_html=True)

    # Information boxes for Lab and Radiology Reports
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="uniform-spacing" style="
        background:#e8f5e9;
        padding:22px;
        border-radius:8px;
        border-left:5px solid #4CAF50;
        font-size:15px;
        line-height:1.7;">
        
        <h3 style="color:#2e7d32; margin-top:0; font-size:18px;">🔬 Laboratory Report</h3>
        
        <p>A <strong>Laboratory Report</strong> is a medical document that presents the results of laboratory tests performed on patient samples (blood, urine, tissue, etc.). It provides quantitative measurements of various biomarkers and substances in the body.</p>
        
        <p style="margin-top:15px;"><strong>Purpose:</strong></p>
        <ul style="margin:5px 0 15px 0; padding-left:20px;">
        <li>Diagnose diseases and medical conditions</li>
        <li>Monitor organ function (kidney, liver, heart)</li>
        <li>Assess metabolic health (glucose, cholesterol)</li>
        <li>Detect infections or inflammation</li>
        <li>Track treatment effectiveness</li>
        </ul>
        
        <p style="margin-top:15px;"><strong>Key Components:</strong></p>
        <ul style="margin:5px 0 15px 0; padding-left:20px;">
        <li><strong>Test Results:</strong> Numerical values from blood/urine analysis</li>
        <li><strong>Reference Ranges:</strong> Normal value ranges for comparison</li>
        <li><strong>Flags:</strong> Indicators showing if results are High, Low, or Normal</li>
        <li><strong>Clinical Interpretation:</strong> Medical explanation of findings</li>
        </ul>
        
        <p style="margin-top:15px; padding:10px; background:#fff; border-radius:6px; color:#2e7d32; font-size:14px;">
        <strong>💡 In Simple Terms:</strong> Lab reports tell doctors what's happening inside your body at a chemical level through blood tests and other analyses.
        </p>
        
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="uniform-spacing" style="
        background:#e3f2fd;
        padding:22px;
        border-radius:8px;
        border-left:5px solid #2196F3;
        font-size:15px;
        line-height:1.7;">
        
        <h3 style="color:#0d47a1; margin-top:0; font-size:18px;">🩻 Radiology Report</h3>
        
        <p>A <strong>Radiology Report</strong> is a medical document that describes what a radiologist observes when examining medical images (X-rays, CT scans, MRIs, ultrasounds). It provides a detailed interpretation of the body's internal structures.</p>
        
        <p style="margin-top:15px;"><strong>Purpose:</strong></p>
        <ul style="margin:5px 0 15px 0; padding-left:20px;">
        <li>Visualize internal organs and structures</li>
        <li>Detect abnormalities (tumors, fractures, infections)</li>
        <li>Assess disease progression or healing</li>
        <li>Guide treatment decisions and surgical planning</li>
        <li>Monitor chronic conditions over time</li>
        </ul>
        
        <p style="margin-top:15px;"><strong>Key Components:</strong></p>
        <ul style="margin:5px 0 15px 0; padding-left:20px;">
        <li><strong>Imaging Technique:</strong> Type of scan performed (X-ray, CT, MRI)</li>
        <li><strong>Findings:</strong> Detailed description of what's visible</li>
        <li><strong>Impression:</strong> Radiologist's conclusion about abnormalities</li>
        <li><strong>Recommendations:</strong> Suggestions for follow-up or additional tests</li>
        </ul>
        
        <p style="margin-top:15px; padding:10px; background:#fff; border-radius:6px; color:#0d47a1; font-size:14px;">
        <strong>💡 In Simple Terms:</strong> Radiology reports explain what doctors see when they look at pictures of the inside of your body using imaging technology.
        </p>
        
        </div>
        """, unsafe_allow_html=True)


    st.markdown("""
    <style>
    div.stButton > button[kind="primary"] {
        background-color: #f44336 !important;
        color: white !important;
        font-size: 18px !important;
        font-weight: 600 !important;
        padding: 12px 24px !important;
        border-radius: 8px !important;
        border: none !important;
        width: 20% !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #d32f2f !important;
    }
    
    div.stDownloadButton > button {
        background-color: #4169A9 !important;
        color: white !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        padding: 10px 20px !important;
        border-radius: 6px !important;
        border: none !important;
        width: 20% !important;
    }
    div.stDownloadButton > button:hover {
        background-color: #345a8f !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    if st.button("Generate Lab & Radiology Reports", type="primary", use_container_width=True):
        if not name or not age or not query:
            st.warning("⚠️ Please fill in at least Patient Name, Age, and Clinical Query to generate reports.")
        else:
            patient = {
                "patient_name": name,
                "age": age,
                "gender": gender,
                "medical_history": history,
                "clinical_query": query,
                "risk_factors": kb.search_patient_by_name(name).get('risk_factors', '') if kb.search_patient_by_name(name) else ''
            }

            with st.spinner("🔬 Generating lab report from knowledge base..."):
                lab_html, lab = rg.generate(query, patient, "lab")
            
            with st.spinner("🩻 Generating radiology report from knowledge base..."):
                rad_html, rad = rg.generate(query, patient, "radiology")

            st.session_state.lab_html = lab_html
            st.session_state.rad_html = rad_html
            st.session_state.lab = lab
            st.session_state.rad = rad
            st.session_state.patient = patient
            st.session_state.show_reports = True
            
            st.success("✅ Lab and radiology reports generated successfully!")


    if st.session_state.get("show_reports"):
        
        # LAB REPORT SECTION
        st.markdown("""
        <div class="section-header uniform-spacing">
        Generated Lab Report
        </div>
        """, unsafe_allow_html=True)
        
        st.components.v1.html(st.session_state.lab_html, height=500, scrolling=True)
        
        # PDF Download for Lab Report
        if PDF_AVAILABLE:
            pdf_data = rg.html_to_pdf(st.session_state.lab_html)
            if pdf_data:
                st.download_button(
                    label="Download Lab Report",
                    data=pdf_data,
                    file_name="lab_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        else:
            st.info("💡 Install weasyprint (`pip install weasyprint`) for PDF download functionality.")

        # RADIOLOGY REPORT SECTION
        st.markdown("""
        <div class="section-header uniform-spacing">
        Generated Radiology Report
        </div>
        """, unsafe_allow_html=True)
        
        st.components.v1.html(st.session_state.rad_html, height=500, scrolling=True)
        
        # PDF Download for Radiology Report
        if PDF_AVAILABLE:
            pdf_data = rg.html_to_pdf(st.session_state.rad_html)
            if pdf_data:
                st.download_button(
                    label="Download Radiology Report",
                    data=pdf_data,
                    file_name="radiology_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        else:
            st.info("💡 Install weasyprint (`pip install weasyprint`) for PDF download functionality.")

        # DIAGNOSTIC SECTION INTRODUCTION
        st.markdown("""
        <div class="uniform-spacing" style="
        background:#e8f5e9;
        padding:25px;
        border-radius:8px;
        border-left:5px solid #4CAF50;
        font-size:15px;
        line-height:1.7;">
        
        <h3 style="color:#2e7d32; margin-top:0;">📋 Generate Final Diagnostic Report</h3>
        
        The Diagnostic Report brings together all the information entered so far—patient details, symptoms, lab results, and radiology findings—to produce a clear, clinically aligned interpretation using BioGPT.
        <br><br>
        This section acts as the final step in the patient assessment flow, where the system synthesizes insights and presents a unified diagnostic view.
        
        <br><br>
        <b>What to Expect:</b>
        <ul style="margin:10px 0;">
        <li>A concise AI-generated clinical impression</li>
        <li>Key supporting evidence drawn from earlier reports</li>
        <li>Possible risk indicators or differentials</li>
        <li>A doctor-style summary designed for quick understanding</li>
        </ul>
        
        <b>Why This Matters:</b><br>
        Instead of reviewing each report separately, this consolidated diagnostic summary helps you quickly interpret the patient's overall condition and move toward decision-making.
        <br><br>
        <b>The next step is generating the Diagnostic Report, where all your inputs are combined into a single clinical interpretation.</b>
        
        </div>
        """, unsafe_allow_html=True)



        if st.button("Generate Diagnostic Report", type="primary", use_container_width=True):
            with st.spinner("🏥 Generating comprehensive diagnostic report..."):
                diag_html, diag_data = rg.diagnosis(st.session_state.lab,
                                         st.session_state.rad,
                                         st.session_state.patient)
            st.session_state.diag_html = diag_html
            st.session_state.diag_data = diag_data
            st.session_state.show_diag = True
            st.success("✅ Diagnostic report generated successfully!")


    if st.session_state.get("show_diag"):
        st.markdown("""
        <div class="section-header uniform-spacing">
        Generated Diagnostic Report
        </div>
        """, unsafe_allow_html=True)
        
        st.components.v1.html(st.session_state.diag_html, height=700, scrolling=True)
        
        # PDF Download for Diagnostic Report
        if PDF_AVAILABLE:
            pdf_data = rg.html_to_pdf(st.session_state.diag_html)
            if pdf_data:
                st.download_button(
                    label="Download Diagnostic Report",
                    data=pdf_data,
                    file_name="diagnostic_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        else:
            st.info("💡 Install weasyprint (`pip install weasyprint`) for PDF download functionality.")

if __name__ == "__main__":
    main()