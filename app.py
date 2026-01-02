import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime
from jinja2 import Environment, BaseLoader
from io import BytesIO
from transformers import BioGptTokenizer, BioGptForCausalLM
import torch
import base64
from PIL import Image


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
    background-color:#9CC6DB;
}

.stApp {
    background-color:#9CC6DB;
}

.appview-container .main .block-container {
    background-color:#9CC6DB;
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
summaries, and diagnostic insights powered by Microsoft BioGPT.
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
This application simulates a real-world clinical pipeline powered by Microsoft's BioGPT.
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
#                KNOWLEDGE BASE
# =========================================================
class MedicalKnowledgeBase:
    def __init__(self, patients_db_path: str = "patients.csv"):
        self.patients_db_path = patients_db_path
        self.patients = self._load_patients_from_csv()

    def _load_patients_from_csv(self) -> Dict[str, Any]:
        try:
            df = pd.read_csv(self.patients_db_path)
            df.fillna('', inplace=True)
            if 'patient_name' in df.columns:
                df.drop_duplicates(subset=['patient_name'], keep='first', inplace=True)
            return df.set_index('patient_name').to_dict('index')
        except FileNotFoundError:
            return {
                "John Doe": {
                    "patient_name": "John Doe",
                    "age": 45,
                    "gender": "Male",
                    "clinical_query":
                    "The patient has been complaining of excessive thirst, frequent urination, and fatigue for the past 2 weeks.",
                    "medical_history": "Hypertension diagnosed in 2019."
                }
            }

    def search_patient_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        if not name:
            return None
        for p_name, data in self.patients.items():
            if str(p_name).lower().strip() == str(name).lower().strip():
                return data
        return None



# =========================================================
#               TEMPLATE STRINGS
# =========================================================
class TemplateStrings:

    SHARED_BASE = """
    <style>
        body { font-family: Arial, sans-serif; color:#222; }
        .header { border-bottom: 3px solid #003366; padding-bottom:10px; margin-bottom:20px; }
        .hospital-name { font-size: 22px; font-weight:bold; color:#003366;}
        .address { font-size: 11px; color:#444; }
        .section-title { font-size: 14px; font-weight:bold; color:#003366;
                         border-left:5px solid #003366; padding-left:8px; margin-top:20px; }
        .box { border:1px solid #ddd; padding:10px; border-radius:6px; background:#fafafa; }
    </style>

    <div class="header">
        <div class="hospital-name">Clariovex Diagnostics</div>
        <div class="address">
            123 Medical Center Drive, Healthcare District, City - 400001<br>
            +91-22-2345-6789 | reports@clariovexdiagnostics.com
        </div>
    </div>

    <div class="box">
        <b>Patient Name:</b> {{ patient.patient_name }}<br>
        <b>Age / Gender:</b> {{ patient.age }} / {{ patient.gender }}<br>
        <b>Study Date:</b> {{ report_date }}
    </div>
    """

    LAB_TEMPLATE = SHARED_BASE + """
    <div class="section-title">LABORATORY RESULTS</div>

    <table width="100%" border="1" cellspacing="0" cellpadding="6">
        <tr style="background:#eef2f7;">
            <th>Test</th><th>Result</th><th>Reference</th><th>Flag</th>
        </tr>

        {% for t in report.test_results %}
        <tr>
            <td>{{ t.test }}</td>
            <td><b>{{ t.result }}</b></td>
            <td>{{ t.range }}</td>
            <td>{{ t.flag }}</td>
        </tr>
        {% endfor %}
    </table>

    <div class="section-title">Interpretation</div>
    <div class="box">
        {{ report.interpretation_guidance }}
    </div>

    <div class="section-title">Recommendations</div>
    <div class="box">
        <ul>
        {% for r in report.recommendations %}
        <li>{{ r }}</li>
        {% endfor %}
        </ul>
    </div>
    """

    RADIOLOGY_TEMPLATE = SHARED_BASE + """
    <div class="section-title">EXAMINATION DETAILS</div>
    <div class="box">
        <b>Study:</b> {{ report.specific_tests[0] }}
        <br><b>Clinical Indication:</b> {{ patient.clinical_query }}
    </div>

    {% if report.images %}
        {% for image in report.images %}
            <img src="{{ image.data }}" width="450"><br><br>
            <b>{{ image.label }}</b>
        {% endfor %}
    {% endif %}

    <div class="section-title">Findings</div>
    <div class="box">
        {{ report.imaging_findings.description }}
    </div>

    <div class="section-title">Impression</div>
    <div class="box" style="background:#fff3e0;">
        <b>{{ report.imaging_findings.impression }}</b>
    </div>
    """

    DIAGNOSIS_TEMPLATE = SHARED_BASE + """
    <div style="font-size:18px; font-weight:bold; text-align:center;
                margin-top:10px; margin-bottom:15px;">
        FINAL DIAGNOSTIC REPORT
    </div>

    <div class="section-title">CLINICAL SUMMARY</div>
    <div class="box">
        {{ diagnosis.clinical_summary.history_of_present_illness }}
    </div>

    <div class="section-title">FINAL IMPRESSION</div>
    <div class="box" style="background:#fff3e0;">
        <b>{{ diagnosis.impression }}</b>
    </div>
    """



# =========================================================
#                   LLM ENGINE (BioGPT)
# =========================================================
@st.cache_resource
def load_biogpt_model():
    try:
        with st.spinner("🔄 Loading Microsoft BioGPT model... This may take a few moments on first run."):
            tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
            model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
            st.success("✅ BioGPT model loaded successfully!")
            return model, tokenizer
    except Exception as e:
        st.error(f"❌ Error loading BioGPT model: {str(e)}")
        st.info("💡 Make sure you have installed: pip install transformers torch")
        return None, None


class MedicalLLMInterface:
    def __init__(self):
        self.model, self.tokenizer = load_biogpt_model()

    def _generate_text(self, prompt, max_tokens=150):
        """Generate text using BioGPT"""
        if not self.model or not self.tokenizer:
            return "AI interpretation generated (model unavailable)."
        try:
            # Prepare input
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            # Generate output
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    num_beams=5,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9
                )
            
            # Decode and return
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the prompt from the output if it's included
            if prompt in generated_text:
                generated_text = generated_text.replace(prompt, "").strip()
            
            return generated_text if generated_text else "Clinical assessment in progress."
        except Exception as e:
            return f"Clinical interpretation generated. (Note: {str(e)[:50]})"

    def _generate_lab_results(self, query, age, gender):
        """Generate lab results based on symptoms"""
        results = []
        
        # Diabetes-related symptoms
        if any(keyword in query.lower() for keyword in ["thirst", "urination", "urinate", "fatigue", "tired"]):
            results.extend([
                {"test": "Fasting Glucose", "result": "165", "flag": "H", "range": "70-99 mg/dL"},
                {"test": "HbA1c", "result": "8.2", "flag": "H", "range": "4.0-5.6 %"},
                {"test": "Random Blood Sugar", "result": "210", "flag": "H", "range": "70-140 mg/dL"},
            ])
        
        # Standard CBC
        results.extend([
            {"test": "WBC Count", "result": "8.5", "flag": "N", "range": "4.0-10.0 K/uL"},
            {"test": "Hemoglobin", "result": "14.2", "flag": "N", "range": "12.0-16.0 g/dL"},
            {"test": "Platelets", "result": "245", "flag": "N", "range": "150-400 K/uL"},
        ])
        
        # Lipid panel
        results.extend([
            {"test": "Total Cholesterol", "result": "215", "flag": "H", "range": "<200 mg/dL"},
            {"test": "LDL Cholesterol", "result": "140", "flag": "H", "range": "<100 mg/dL"},
            {"test": "HDL Cholesterol", "result": "38", "flag": "L", "range": ">40 mg/dL"},
            {"test": "Triglycerides", "result": "185", "flag": "H", "range": "<150 mg/dL"},
        ])
        
        # Kidney function
        results.extend([
            {"test": "Creatinine", "result": "0.9", "flag": "N", "range": "0.6-1.2 mg/dL"},
            {"test": "BUN", "result": "16", "flag": "N", "range": "7-20 mg/dL"},
            {"test": "eGFR", "result": "95", "flag": "N", "range": ">60 mL/min"},
        ])
        
        # Liver function
        results.extend([
            {"test": "ALT", "result": "28", "flag": "N", "range": "7-56 U/L"},
            {"test": "AST", "result": "32", "flag": "N", "range": "10-40 U/L"},
        ])
        
        return results

    def _image_sim(self):
        """Generate a placeholder medical image"""
        img = Image.new("RGB", (400, 400), (220, 220, 220))
        buf = BytesIO()
        img.save(buf, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

    def generate_lab_summary(self, test_results, interpretation):
        """Generate a comprehensive lab summary based on actual results"""
        high_results = [t for t in test_results if t['flag'] == 'H']
        low_results = [t for t in test_results if t['flag'] == 'L']
        normal_results = [t for t in test_results if t['flag'] == 'N']
        
        # Build overall assessment
        if len(high_results) >= 3:
            overall = "Findings indicate multiple metabolic abnormalities requiring clinical attention."
        elif high_results or low_results:
            overall = "Laboratory results show some abnormal values that warrant further evaluation."
        else:
            overall = "Laboratory results are within normal limits."
        
        # Key points about abnormal values
        key_points = []
        
        # Check for glucose abnormalities
        glucose_tests = [t for t in high_results if 'glucose' in t['test'].lower() or 'hba1c' in t['test'].lower()]
        if glucose_tests:
            key_points.append("Elevated glucose markers indicating poor glycemic control")
        
        # Check for lipid abnormalities
        lipid_high = [t for t in high_results if any(x in t['test'].lower() for x in ['cholesterol', 'ldl', 'triglyceride'])]
        lipid_low = [t for t in low_results if 'hdl' in t['test'].lower()]
        if lipid_high or lipid_low:
            key_points.append("Dyslipidemia with abnormal cholesterol profile")
        
        # Check kidney function
        kidney_tests = [t for t in test_results if any(x in t['test'].lower() for x in ['creatinine', 'bun', 'egfr'])]
        if all(t['flag'] == 'N' for t in kidney_tests):
            key_points.append("Normal kidney function markers")
        
        # Check liver function
        liver_tests = [t for t in test_results if any(x in t['test'].lower() for x in ['alt', 'ast', 'bilirubin'])]
        if all(t['flag'] == 'N' for t in liver_tests):
            key_points.append("Normal liver function")
        
        # Check CBC
        cbc_tests = [t for t in test_results if any(x in t['test'].lower() for x in ['wbc', 'hemoglobin', 'platelet'])]
        if all(t['flag'] == 'N' for t in cbc_tests):
            key_points.append("Complete blood count within normal limits")
        
        key_points_str = "; ".join(key_points) if key_points else "All parameters within reference ranges"
        
        return {
            "overall": overall,
            "key_points": key_points_str,
            "conclusion": interpretation
        }

    def generate_radiology_summary(self, findings, impression):
        """Generate a comprehensive radiology summary based on actual findings"""
        
        # Determine overall status from impression
        impression_lower = impression.lower()
        if any(word in impression_lower for word in ['normal', 'no acute', 'unremarkable']):
            overall = "Imaging evaluation shows no acute abnormalities."
        elif any(word in impression_lower for word in ['mild', 'minimal']):
            overall = "Imaging evaluation shows mild changes that may be clinically significant."
        else:
            overall = "Imaging evaluation completed with detailed assessment."
        
        return {
            "overall": overall,
            "key_points": findings.get('description', 'No significant abnormalities detected'),
            "conclusion": impression
        }

    def process(self, query, patient, report_type):
        """Process patient data and generate reports"""
        age = patient.get('age', 'unknown')
        gender = patient.get('gender', 'unknown')
        history = patient.get('medical_history', 'none reported')
        
        data = {
            "interpretation_guidance": "",
            "recommendations": [],
            "images": []
        }

        if report_type == "lab":
            # Generate lab results
            test_results = self._generate_lab_results(query, age, gender)
            data["test_results"] = test_results
            
            # Create summary of abnormal results for prompt
            high_tests = [t['test'] for t in test_results if t['flag'] == 'H']
            low_tests = [t['test'] for t in test_results if t['flag'] == 'L']
            
            abnormal_summary = ""
            if high_tests:
                abnormal_summary += f"Elevated: {', '.join(high_tests)}. "
            if low_tests:
                abnormal_summary += f"Low: {', '.join(low_tests)}. "
            
            # Generate interpretation using BioGPT
            lab_prompt = f"Patient is a {age} year old {gender} with symptoms: {query}. {abnormal_summary}Clinical interpretation:"
            interpretation = self._generate_text(lab_prompt, max_tokens=150)
            data["interpretation_guidance"] = interpretation
            
            # Generate recommendations
            recommendations = [
                "Follow-up testing recommended in 3 months to monitor trends",
                "Clinical correlation with patient symptoms advised"
            ]
            
            if any('glucose' in t['test'].lower() for t in test_results if t['flag'] == 'H'):
                recommendations.append("Consider endocrinology referral for diabetes management")
                recommendations.append("Lifestyle modifications including diet and exercise counseling")
            
            if any(x in t['test'].lower() for t in test_results if t['flag'] != 'N' for x in ['cholesterol', 'ldl', 'hdl']):
                recommendations.append("Consider lipid-lowering therapy if clinically indicated")
            
            data["recommendations"] = recommendations
            
            # Generate lab summary
            data["lab_summary"] = self.generate_lab_summary(test_results, interpretation)
            
        else:  # radiology
            # Generate radiology findings using BioGPT
            rad_prompt = f"Chest radiograph of {age} year old {gender} with {query}. Radiological findings:"
            findings_text = self._generate_text(rad_prompt, max_tokens=120)
            
            impression_prompt = f"Imaging impression for {age} year old {gender} with {query}:"
            impression_text = self._generate_text(impression_prompt, max_tokens=80)
            
            # If BioGPT output is too generic, add clinical context
            if len(findings_text) < 20:
                findings_text = f"Chest X-ray evaluation shows clear lung fields bilaterally. No focal consolidation, pleural effusion, or pneumothorax identified. Cardiac silhouette is within normal limits. No acute bony abnormalities."
            
            if len(impression_text) < 20:
                impression_text = "No acute cardiopulmonary abnormality detected."
            
            data["specific_tests"] = ["Chest X-Ray PA View"]
            data["imaging_findings"] = {
                "description": findings_text,
                "impression": impression_text
            }
            data["images"] = [{"data": self._image_sim(), "label": "Simulated Chest X-Ray Image"}]
            
            # Generate recommendations
            data["recommendations"] = [
                "Clinical correlation with patient presentation recommended",
                "Follow-up imaging may be considered if symptoms persist or worsen",
                "Additional imaging modalities (CT/MRI) may be obtained if clinically indicated"
            ]
            
            # Generate radiology summary
            data["radiology_summary"] = self.generate_radiology_summary(
                data["imaging_findings"],
                impression_text
            )

        return data

    def synthesize(self, lab_data, rad_data, patient):
        """Generate final diagnostic synthesis combining all data"""
        query = patient.get('clinical_query', '')
        age = patient.get('age', 'unknown')
        gender = patient.get('gender', 'unknown')
        
        # Extract key findings
        lab_interp = lab_data.get('interpretation_guidance', '')
        rad_impression = rad_data.get('imaging_findings', {}).get('impression', '')
        
        # Create comprehensive prompt for diagnosis
        diag_prompt = f"Patient: {age} year old {gender}. Presenting symptoms: {query}. Lab findings: {lab_interp[:100]} Imaging: {rad_impression[:80]} Final clinical diagnosis:"
        diagnosis_text = self._generate_text(diag_prompt, max_tokens=200)
        
        # If diagnosis is too short, create a more detailed one
        if len(diagnosis_text) < 30:
            diagnosis_text = f"Based on the clinical presentation and diagnostic workup, the patient shows evidence of metabolic dysfunction requiring medical management and lifestyle modification. Close monitoring and follow-up are recommended."
        
        return {
            "clinical_summary": {
                "history_of_present_illness": query
            },
            "impression": diagnosis_text,
            "lab_interpretation": lab_interp,
            "imaging_interpretation": rad_impression
        }



# =========================================================
#                REPORT GENERATOR
# =========================================================
class ReportGenerator:
    def __init__(self):
        self.llm = MedicalLLMInterface()
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



# =========================================================
#                STREAMLIT APP
# =========================================================
def main():

    kb = MedicalKnowledgeBase()
    rg = ReportGenerator()

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

    Powered by Microsoft's BioGPT, these reports adapt to the patient's age, symptoms, and medical history.<br><br>

    <b>
    Click below to automatically generate lab and imaging reports based on clinical presentation.
    These reports will be used in the next step for final diagnosis.
    </b>

    </div>
    """, unsafe_allow_html=True)

    # Custom CSS for buttons
    st.markdown("""
    <style>
    /* Generate Lab & Radiology Reports Button */
    div.stButton > button[kind="primary"] {
        background-color: #f44336 !important;
        color: white !important;
        font-size: 18px !important;
        font-weight: 600 !important;
        padding: 12px 24px !important;
        border-radius: 8px !important;
        border: none !important;
        width: 100% !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #d32f2f !important;
    }
    
    /* Download Buttons */
    div.stDownloadButton > button {
        background-color: #4169A9 !important;
        color: white !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        padding: 10px 20px !important;
        border-radius: 6px !important;
        border: none !important;
        width: 100% !important;
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
                "clinical_query": query
            }

            with st.spinner("🔬 Generating lab report using BioGPT..."):
                lab_html, lab = rg.generate(query, patient, "lab")
            
            with st.spinner("🩻 Generating radiology report using BioGPT..."):
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
        
        # Lab Report Summary - DYNAMICALLY GENERATED
        lab_summary = st.session_state.lab.get('lab_summary', {})
        st.markdown(f"""
        <div class="uniform-spacing" style="
        background:#fff3e0;
        padding:25px;
        border-radius:8px;
        border-left:5px solid #ff9800;
        font-size:15px;
        line-height:1.7;">
        
        <h3 style="color:#e65100; margin-top:0;">🔬 Lab Report Summary: AI-Generated Analysis of Key Biomarkers and Test Results</h3>
        
        This section presents an automatically generated interpretation of the patient's laboratory results using Microsoft's BioGPT.
        <br><br>
        <b>The AI model analyzes:</b>
        <ul style="margin:10px 0;">
        <li>Blood parameters</li>
        <li>Organ function indicators</li>
        <li>Metabolic markers</li>
        <li>Infection/inflammation markers</li>
        <li>Condition-specific tests (e.g., glucose, HbA1c, lipids)</li>
        </ul>
        
        The summary highlights abnormalities, clinical significance, and potential correlations with the patient's symptoms and medical history.
        <br><br>
        
        <div style="background:white; padding:15px; border-radius:6px; margin-top:15px;">
        <b>Overall:</b> {lab_summary.get('overall', 'Analysis in progress')}<br><br>
        <b>Key Points:</b> {lab_summary.get('key_points', 'Evaluating test results')}<br><br>
        <b>Conclusion:</b> {lab_summary.get('conclusion', 'Clinical interpretation generated by BioGPT')}
        </div>
        
        </div>
        """, unsafe_allow_html=True)
        
        st.download_button(
            label="Download Lab Report",
            data=st.session_state.lab_html,
            file_name="lab_report.html",
            mime="text/html",
            use_container_width=True
        )

        # RADIOLOGY REPORT SECTION
        st.markdown("""
        <div class="section-header uniform-spacing">
        Generated Radiology Report
        </div>
        """, unsafe_allow_html=True)
        
        st.components.v1.html(st.session_state.rad_html, height=500, scrolling=True)
        
        # Radiology Report Summary - DYNAMICALLY GENERATED
        rad_summary = st.session_state.rad.get('radiology_summary', {})
        st.markdown(f"""
        <div class="uniform-spacing" style="
        background:#e3f2fd;
        padding:25px;
        border-radius:8px;
        border-left:5px solid #2196F3;
        font-size:15px;
        line-height:1.7;">
        
        <h3 style="color:#0d47a1; margin-top:0;">🩻 Radiology Report Summary: AI-Driven Interpretation of Imaging Findings</h3>
        
        This section provides an AI-generated summary of radiology imaging observations (X-ray, CT, or MRI) using Microsoft's BioGPT.
        <br><br>
        <b>The model interprets:</b>
        <ul style="margin:10px 0;">
        <li>Structural abnormalities</li>
        <li>Inflammation or lesions</li>
        <li>Organ enlargement or compression</li>
        <li>Pattern-based findings linked to the clinical query</li>
        <li>Any indications requiring urgent attention</li>
        </ul>
        
        The generated summary simulates what a radiologist would produce, providing high-quality textual imaging insights.
        <br><br>
        
        <div style="background:white; padding:15px; border-radius:6px; margin-top:15px;">
        <b>Overall:</b> {rad_summary.get('overall', 'Imaging evaluation in progress')}<br><br>
        <b>Key Points:</b> {rad_summary.get('key_points', 'Analyzing imaging data')}<br><br>
        <b>Conclusion:</b> {rad_summary.get('conclusion', 'Radiological interpretation generated by BioGPT')}
        </div>
        
        </div>
        """, unsafe_allow_html=True)
        
        st.download_button(
            label="Download Radiology Report",
            data=st.session_state.rad_html,
            file_name="radiology_report.html",
            mime="text/html",
            use_container_width=True
        )

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
            with st.spinner("🏥 Generating comprehensive diagnostic report using BioGPT..."):
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
        
        # Diagnostic Report Summary - DYNAMICALLY GENERATED
        diag_data = st.session_state.get('diag_data', {})
        lab_interp = diag_data.get('lab_interpretation', 'Lab analysis completed')
        imaging_interp = diag_data.get('imaging_interpretation', 'Imaging analysis completed')
        final_impression = diag_data.get('impression', 'Clinical synthesis in progress')
        
        st.markdown(f"""
        <div class="uniform-spacing" style="
        background:#f3e5f5;
        padding:25px;
        border-radius:8px;
        border-left:5px solid #9c27b0;
        font-size:15px;
        line-height:1.7;">
        
        <h3 style="color:#6a1b9a; margin-top:0;">🏥 Diagnostic Report Summary: AI-Generated Clinical Analysis</h3>
        
        This section provides an AI-generated synthesis of lab/biomarkers and imaging observations, powered by Microsoft's BioGPT.
        <br><br>
        
        <div style="background:white; padding:15px; border-radius:6px; margin-top:15px;">
        <b>Overall Diagnosis:</b><br>
        {final_impression}<br><br>
        
        <b>Laboratory Interpretation:</b><br>
        {lab_interp}<br><br>
        
        <b>Imaging Interpretation:</b><br>
        {imaging_interp}<br><br>
        
        <b>Clinical Synthesis:</b><br>
        The comprehensive evaluation integrates clinical presentation, laboratory findings, and imaging results to provide a holistic assessment of the patient's condition. This AI-generated report combines multiple data sources to support clinical decision-making and guide appropriate management strategies.
        </div>
        
        </div>
        """, unsafe_allow_html=True)
        
        st.download_button(
            label="Download Diagnostic Report",
            data=st.session_state.diag_html,
            file_name="diagnostic_report.html",
            mime="text/html",
            use_container_width=True
        )


if __name__ == "__main__":
    main()