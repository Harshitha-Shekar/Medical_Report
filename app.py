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
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    st.warning("⚠️ Transformers library not available. Using fallback text generation.")

# Try to import pdfkit and weasyprint for PDF generation
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
        st.warning("⚠️ PDF generation libraries not available. Install weasyprint or pdfkit for PDF downloads.")


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
                <td>Dr. A. Kumar, MD</td>
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
            <td style="padding: 8px;">{{ t.test }}</td>
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
    <div class="box">
        <table style="width: 100%; font-size: 13px; line-height: 1.8;">
            <tr>
                <td style="width: 150px; font-weight: bold;">Study Performed:</td>
                <td>{{ report.specific_tests[0] }}</td>
            </tr>
            <tr>
                <td style="font-weight: bold;">Clinical Indication:</td>
                <td>{{ patient.clinical_query }}</td>
            </tr>
            <tr>
                <td style="font-weight: bold;">Technique:</td>
                <td>Standard radiographic protocol</td>
            </tr>
        </table>
    </div>

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
    
    <h3 style="color: #022869; font-size: 15px; margin: 20px 0 10px 0;">1. History of Present Illness</h3>
    <div class="box" style="line-height: 1.7;">
        {{ diagnosis.clinical_summary.history_of_present_illness }}
    </div>

    <h3 style="color: #022869; font-size: 15px; margin: 20px 0 10px 0;">2. Risk Factors</h3>
    <div class="box" style="line-height: 1.7;">
        <ul style="margin: 0; padding-left: 20px;">
            <li>Medical History: {{ patient.medical_history }}</li>
            <li>Age: {{ patient.age }} years</li>
            <li>Gender: {{ patient.gender }}</li>
        </ul>
    </div>

    <div class="section-title">SECTION B — DIAGNOSTIC FINDINGS SUMMARY</div>
    
    <h3 style="color: #022869; font-size: 15px; margin: 20px 0 10px 0;">Laboratory Interpretation</h3>
    <div class="box" style="line-height: 1.7; background: #f0f8ff;">
        {{ diagnosis.lab_interpretation }}
    </div>

    <h3 style="color: #022869; font-size: 15px; margin: 20px 0 10px 0;">Imaging Interpretation</h3>
    <div class="box" style="line-height: 1.7; background: #f0f8ff;">
        {{ diagnosis.imaging_interpretation }}
    </div>

    <div class="section-title">SECTION C — FINAL IMPRESSION</div>
    <div class="box" style="background:#fff3e0; border-left: 5px solid #ff9800; padding: 20px;">
        <strong style="font-size: 15px; color: #e65100; line-height: 1.8;">
            {{ diagnosis.impression }}
        </strong>
    </div>

    <div class="section-title">SECTION D — CLINICAL RECOMMENDATIONS</div>
    <div class="box">
        <ul style="line-height: 1.8; margin: 0; padding-left: 20px;">
            <li>Immediate medical consultation recommended</li>
            <li>Follow-up investigations as clinically indicated</li>
            <li>Lifestyle modifications and risk factor management</li>
            <li>Regular monitoring of relevant parameters</li>
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
    def __init__(self):
        self.model, self.tokenizer = load_biogpt_model()

    def _generate_text_fallback(self, prompt, context):
        """Fallback text generation when BioGPT is not available"""
        query_lower = context.get('query', '').lower()
        age = context.get('age', 'unknown')
        gender = context.get('gender', 'unknown')
        
        # Generate context-aware clinical text
        if 'lab' in prompt.lower():
            if any(kw in query_lower for kw in ['thirst', 'urination', 'fatigue', 'diabetes']):
                return f"Laboratory findings for this {age}-year-old {gender} patient reveal elevated fasting glucose (165 mg/dL) and HbA1c (8.2%), consistent with uncontrolled diabetes mellitus. The lipid panel shows dyslipidemia with elevated total cholesterol, LDL, and triglycerides, alongside reduced HDL levels. Complete blood count and renal function tests remain within normal limits. These findings suggest metabolic syndrome requiring comprehensive diabetes management and cardiovascular risk modification."
            else:
                return f"Laboratory analysis shows some metabolic abnormalities requiring clinical correlation. Baseline hematology and organ function parameters are within acceptable limits."
        
        elif 'imaging' in prompt.lower() or 'radiological' in prompt.lower():
            if any(kw in query_lower for kw in ['chest', 'cough', 'breath', 'lung']):
                return "No acute cardiopulmonary abnormality detected on chest radiography. Lung fields are clear bilaterally without focal consolidation, pleural effusion, or pneumothorax. Cardiac silhouette is within normal limits."
            else:
                return "Imaging study demonstrates no acute abnormality. Clinical correlation recommended."
        
        elif 'diagnosis' in prompt.lower() or 'impression' in prompt.lower():
            if any(kw in query_lower for kw in ['thirst', 'urination', 'fatigue', 'diabetes']):
                return f"Based on clinical presentation and diagnostic workup, this {age}-year-old {gender} patient demonstrates features consistent with Type 2 Diabetes Mellitus with associated dyslipidemia. The constellation of elevated glucose markers (fasting glucose 165 mg/dL, HbA1c 8.2%) alongside metabolic syndrome components indicates need for comprehensive diabetes management including pharmacotherapy, lifestyle modification, and cardiovascular risk factor control. Regular monitoring and endocrinology follow-up recommended."
            else:
                return f"Clinical assessment suggests metabolic dysfunction requiring medical management and lifestyle modifications. Close monitoring and appropriate follow-up are recommended for this {age}-year-old {gender} patient."
        
        return "Clinical interpretation generated based on patient presentation and diagnostic findings."

    def _generate_text(self, prompt, max_tokens=150, context=None):
        """Generate text using BioGPT or fallback"""
        if not self.model or not self.tokenizer:
            return self._generate_text_fallback(prompt, context or {})
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
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
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if prompt in generated_text:
                generated_text = generated_text.replace(prompt, "").strip()
            
            return generated_text if generated_text else self._generate_text_fallback(prompt, context or {})
        except Exception as e:
            return self._generate_text_fallback(prompt, context or {})

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

    def _generate_xray_image(self):
        """Generate a realistic X-ray-like medical image placeholder"""
        width, height = 500, 500
        
        # Create gradient background
        img_array = np.zeros((height, width), dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                distance = np.sqrt((i - height/2)**2 + (j - width/2)**2)
                max_distance = np.sqrt((height/2)**2 + (width/2)**2)
                value = int(180 - (distance / max_distance) * 80)
                img_array[i, j] = value
        
        img = Image.fromarray(img_array, mode='L')
        draw = ImageDraw.Draw(img)
        
        # Add anatomical shapes
        draw.ellipse([80, 120, 220, 360], fill=200, outline=150, width=2)
        draw.ellipse([280, 120, 420, 360], fill=200, outline=150, width=2)
        draw.ellipse([190, 260, 310, 400], fill=130, outline=100, width=2)
        draw.rectangle([235, 50, 265, 450], fill=120)
        draw.arc([100, 80, 200, 140], start=180, end=0, fill=140, width=3)
        draw.arc([300, 80, 400, 140], start=180, end=0, fill=140, width=3)
        
        try:
            draw.text((20, 20), "CHEST X-RAY - PA VIEW", fill=240)
            draw.text((20, height-30), "Simulated Radiograph", fill=240)
            draw.text((width-150, height-30), "R", fill=240)
        except:
            pass
        
        buf = BytesIO()
        img.save(buf, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

    def generate_detailed_lab_summary(self, test_results, interpretation, patient):
        """Generate comprehensive lab summary"""
        high_results = [t for t in test_results if t['flag'] == 'H']
        low_results = [t for t in test_results if t['flag'] == 'L']
        normal_results = [t for t in test_results if t['flag'] == 'N']
        
        summary_parts = []
        
        summary_parts.append("<h4 style='color:#e65100; margin-top:0;'>Overall Assessment:</h4>")
        if len(high_results) >= 3:
            summary_parts.append(f"<p>The laboratory panel reveals <strong>{len(high_results)} elevated parameters</strong> and <strong>{len(low_results)} decreased parameters</strong>, indicating significant metabolic abnormalities requiring immediate clinical attention. Out of {len(test_results)} total tests performed, {len(normal_results)} parameters remain within normal limits.</p>")
        elif high_results or low_results:
            summary_parts.append(f"<p>The laboratory evaluation shows <strong>{len(high_results) + len(low_results)} abnormal values</strong> out of {len(test_results)} tests performed, warranting further clinical correlation. {len(normal_results)} parameters are within reference ranges.</p>")
        else:
            summary_parts.append(f"<p>All {len(test_results)} laboratory parameters tested are within normal reference ranges.</p>")
        
        summary_parts.append("<h4 style='color:#e65100; margin-top:20px;'>Detailed Analysis by System:</h4>")
        
        # 1. Glucose Metabolism
        glucose_tests = [t for t in test_results if any(x in t['test'].lower() for x in ['glucose', 'hba1c', 'sugar'])]
        if glucose_tests:
            summary_parts.append("<p><strong>🔹 Glucose Metabolism:</strong></p><ul>")
            for test in glucose_tests:
                status_word = "elevated" if test['flag'] == 'H' else "decreased" if test['flag'] == 'L' else "normal"
                summary_parts.append(f"<li><strong>{test['test']}:</strong> {test['result']} ({status_word}, reference: {test['range']})")
                
                if 'glucose' in test['test'].lower() and test['flag'] == 'H':
                    summary_parts.append(" - This elevation suggests impaired glucose regulation, consistent with prediabetes or diabetes mellitus.")
                elif 'hba1c' in test['test'].lower() and test['flag'] == 'H':
                    summary_parts.append(" - HbA1c reflects average blood glucose over the past 2-3 months. This elevated level indicates poor long-term glycemic control and increased risk of diabetic complications.")
                
                summary_parts.append("</li>")
            summary_parts.append("</ul>")
        
        # 2. Lipid Profile
        lipid_tests = [t for t in test_results if any(x in t['test'].lower() for x in ['cholesterol', 'ldl', 'hdl', 'triglyceride'])]
        if lipid_tests:
            summary_parts.append("<p><strong>🔹 Lipid Profile:</strong></p><ul>")
            for test in lipid_tests:
                status_word = "elevated" if test['flag'] == 'H' else "decreased" if test['flag'] == 'L' else "within normal limits"
                summary_parts.append(f"<li><strong>{test['test']}:</strong> {test['result']} ({status_word}, reference: {test['range']})")
                
                if 'total cholesterol' in test['test'].lower() and test['flag'] == 'H':
                    summary_parts.append(" - Elevated total cholesterol increases cardiovascular disease risk.")
                elif 'ldl' in test['test'].lower() and test['flag'] == 'H':
                    summary_parts.append(" - High LDL ('bad cholesterol') contributes to atherosclerotic plaque formation and coronary artery disease.")
                elif 'hdl' in test['test'].lower() and test['flag'] == 'L':
                    summary_parts.append(" - Low HDL ('good cholesterol') reduces protective cardiovascular effects and increases cardiac risk.")
                elif 'triglyceride' in test['test'].lower() and test['flag'] == 'H':
                    summary_parts.append(" - Elevated triglycerides may indicate metabolic syndrome and increase pancreatitis risk.")
                
                summary_parts.append("</li>")
            summary_parts.append("</ul>")
        
        # 3. Complete Blood Count
        cbc_tests = [t for t in test_results if any(x in t['test'].lower() for x in ['wbc', 'hemoglobin', 'platelet'])]
        if cbc_tests:
            summary_parts.append("<p><strong>🔹 Complete Blood Count (CBC):</strong></p><ul>")
            for test in cbc_tests:
                status_word = "elevated" if test['flag'] == 'H' else "decreased" if test['flag'] == 'L' else "within normal limits"
                summary_parts.append(f"<li><strong>{test['test']}:</strong> {test['result']} ({status_word}, reference: {test['range']})")
                
                if 'wbc' in test['test'].lower():
                    if test['flag'] == 'H':
                        summary_parts.append(" - Elevated white blood cells may indicate infection, inflammation, or stress response.")
                    elif test['flag'] == 'N':
                        summary_parts.append(" - Normal white blood cell count suggests no active infection or immune compromise.")
                elif 'hemoglobin' in test['test'].lower():
                    if test['flag'] == 'L':
                        summary_parts.append(" - Low hemoglobin indicates anemia, which may cause fatigue and reduced oxygen delivery.")
                    elif test['flag'] == 'N':
                        summary_parts.append(" - Normal hemoglobin suggests adequate red blood cell production and oxygen-carrying capacity.")
                elif 'platelet' in test['test'].lower():
                    if test['flag'] == 'N':
                        summary_parts.append(" - Normal platelet count indicates adequate clotting function.")
                
                summary_parts.append("</li>")
            summary_parts.append("</ul>")
        
        # 4. Kidney Function
        kidney_tests = [t for t in test_results if any(x in t['test'].lower() for x in ['creatinine', 'bun', 'egfr'])]
        if kidney_tests:
            summary_parts.append("<p><strong>🔹 Kidney Function Tests:</strong></p><ul>")
            for test in kidney_tests:
                status_word = "elevated" if test['flag'] == 'H' else "decreased" if test['flag'] == 'L' else "within normal limits"
                summary_parts.append(f"<li><strong>{test['test']}:</strong> {test['result']} ({status_word}, reference: {test['range']})")
                
                if 'creatinine' in test['test'].lower() and test['flag'] == 'N':
                    summary_parts.append(" - Normal creatinine suggests preserved kidney function.")
                elif 'egfr' in test['test'].lower() and test['flag'] == 'N':
                    summary_parts.append(" - Normal eGFR indicates adequate glomerular filtration and kidney health.")
                
                summary_parts.append("</li>")
            summary_parts.append("</ul>")
        
        # 5. Liver Function
        liver_tests = [t for t in test_results if any(x in t['test'].lower() for x in ['alt', 'ast', 'bilirubin', 'alp'])]
        if liver_tests:
            summary_parts.append("<p><strong>🔹 Liver Function Tests:</strong></p><ul>")
            for test in liver_tests:
                status_word = "elevated" if test['flag'] == 'H' else "decreased" if test['flag'] == 'L' else "within normal limits"
                summary_parts.append(f"<li><strong>{test['test']}:</strong> {test['result']} ({status_word}, reference: {test['range']})")
                
                if test['flag'] == 'N':
                    summary_parts.append(" - Normal levels indicate healthy hepatic function.")
                
                summary_parts.append("</li>")
            summary_parts.append("</ul>")
        
        # Clinical Correlation
        summary_parts.append("<h4 style='color:#e65100; margin-top:20px;'>Clinical Correlation:</h4>")
        summary_parts.append(f"<p>{interpretation}</p>")
        
        # Recommendations based on findings
        summary_parts.append("<h4 style='color:#e65100; margin-top:20px;'>Clinical Significance:</h4>")
        summary_parts.append("<p>")
        
        if any('glucose' in t['test'].lower() for t in high_results):
            summary_parts.append("The elevated glucose markers indicate impaired glucose homeostasis, requiring endocrinology evaluation, lifestyle modifications, and possible pharmacotherapy to prevent diabetic complications. ")
        
        if any(x in t['test'].lower() for t in high_results for x in ['cholesterol', 'ldl']) or any('hdl' in t['test'].lower() for t in low_results):
            summary_parts.append("The dyslipidemia pattern significantly increases cardiovascular risk and may warrant statin therapy, dietary intervention, and regular cardiovascular monitoring. ")
        
        if all(t['flag'] == 'N' for t in kidney_tests):
            summary_parts.append("Preserved kidney function is reassuring and suggests no acute renal compromise. ")
        
        summary_parts.append("</p>")
        
        return "".join(summary_parts)

    def generate_detailed_radiology_summary(self, findings, impression, recommendations):
        """Generate a comprehensive, detailed radiology summary"""
        
        summary_parts = []
        
        # Overall Assessment
        summary_parts.append("<h4 style='color:#0d47a1; margin-top:0;'>Overall Imaging Assessment:</h4>")
        
        impression_lower = impression.lower()
        if any(word in impression_lower for word in ['normal', 'no acute', 'unremarkable', 'clear']):
            summary_parts.append("<p>The radiological evaluation demonstrates <strong>no acute abnormalities</strong> on the imaging study performed. All visualized structures appear within normal anatomical limits with no evidence of acute pathology.</p>")
        elif any(word in impression_lower for word in ['mild', 'minimal', 'slight']):
            summary_parts.append("<p>The imaging study reveals <strong>mild changes</strong> that may have clinical significance depending on the patient's symptoms and medical history. These findings warrant correlation with clinical presentation.</p>")
        elif any(word in impression_lower for word in ['moderate', 'significant']):
            summary_parts.append("<p>The radiological examination shows <strong>moderate findings</strong> that require clinical attention and possible further diagnostic workup or intervention.</p>")
        else:
            summary_parts.append("<p>Imaging evaluation completed with detailed assessment of visualized structures. Findings are described below.</p>")
        
        # Detailed Findings Breakdown
        summary_parts.append("<h4 style='color:#0d47a1; margin-top:20px;'>Detailed Imaging Findings:</h4>")
        summary_parts.append(f"<p>{findings.get('description', 'No significant abnormalities detected on imaging.')}</p>")
        
        # Anatomical Structures Assessed
        summary_parts.append("<h4 style='color:#0d47a1; margin-top:20px;'>Anatomical Structures Evaluated:</h4>")
        summary_parts.append("<ul>")
        summary_parts.append("<li><strong>Lung Fields:</strong> Assessed for consolidation, infiltrates, masses, nodules, and pleural abnormalities</li>")
        summary_parts.append("<li><strong>Cardiac Silhouette:</strong> Evaluated for size, contour, and any abnormal enlargement</li>")
        summary_parts.append("<li><strong>Mediastinum:</strong> Examined for widening, masses, or lymphadenopathy</li>")
        summary_parts.append("<li><strong>Pleural Spaces:</strong> Checked for effusions, thickening, or pneumothorax</li>")
        summary_parts.append("<li><strong>Bony Structures:</strong> Reviewed for fractures, lesions, or degenerative changes</li>")
        summary_parts.append("<li><strong>Soft Tissues:</strong> Assessed for any abnormal masses or swelling</li>")
        summary_parts.append("</ul>")
        
        # Radiological Impression
        summary_parts.append("<h4 style='color:#0d47a1; margin-top:20px;'>Radiological Impression:</h4>")
        summary_parts.append(f"<p style='background:#fff3e0; padding:15px; border-radius:6px; border-left:4px solid #ff9800;'><strong>{impression}</strong></p>")
        
        # Clinical Significance
        summary_parts.append("<h4 style='color:#0d47a1; margin-top:20px;'>Clinical Significance & Recommendations:</h4>")
        summary_parts.append("<ul>")
        for rec in recommendations:
            summary_parts.append(f"<li>{rec}</li>")
        summary_parts.append("</ul>")
        
        # Technical Quality
        summary_parts.append("<h4 style='color:#0d47a1; margin-top:20px;'>Technical Quality:</h4>")
        summary_parts.append("<p>The imaging study was performed using standard radiographic protocol with adequate penetration and positioning. Image quality is sufficient for diagnostic interpretation. No technical limitations significantly impair the assessment.</p>")
        
        # Comparison Statement
        summary_parts.append("<h4 style='color:#0d47a1; margin-top:20px;'>Comparison:</h4>")
        summary_parts.append("<p>No prior imaging studies available for comparison. If symptoms persist or worsen, follow-up imaging may be valuable to assess for interval changes.</p>")
        
        return "".join(summary_parts)

    def _determine_imaging_modality(self, query):
        """Determine the appropriate imaging modality based on clinical query"""
        query_lower = query.lower()
        
        # Brain/Head related
        if any(word in query_lower for word in ['head', 'brain', 'headache', 'stroke', 'neuro', 'seizure', 'dizziness']):
            return "mri", "MRI Brain"
        
        # Spine related
        if any(word in query_lower for word in ['spine', 'back pain', 'neck pain', 'spinal', 'vertebra']):
            return "mri", "MRI Spine"
        
        # Abdomen/Pelvis
        if any(word in query_lower for word in ['abdomen', 'abdominal', 'pelvis', 'pelvic', 'kidney', 'liver', 'pancreas']):
            return "ct", "CT Abdomen and Pelvis"
        
        # Chest/Respiratory (default to X-ray for chest complaints)
        if any(word in query_lower for word in ['chest', 'cough', 'breath', 'lung', 'respiratory', 'pneumonia']):
            return "xray", "Chest X-Ray (PA and Lateral Views)"
        
        # Default to chest X-ray for general symptoms
        return "xray", "Chest X-Ray (PA and Lateral Views)"

    def _generate_medical_image(self, modality, query, age, gender):
        """Generate appropriate medical image based on modality"""
        if modality == "xray":
            return self._generate_xray_image()
        elif modality == "ct":
            return self._generate_ct_image()
        elif modality == "mri":
            return self._generate_mri_image(query)
        else:
            return self._generate_xray_image()

    def _generate_xray_image(self):
        """Generate a realistic X-ray-like medical image placeholder"""
        from PIL import ImageDraw
        import numpy as np
        
        # Create a grayscale image that looks more like an X-ray
        width, height = 500, 500
        
        # Create gradient background (darker at edges, lighter in center)
        img_array = np.zeros((height, width), dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                # Create radial gradient
                distance = np.sqrt((i - height/2)**2 + (j - width/2)**2)
                max_distance = np.sqrt((height/2)**2 + (width/2)**2)
                value = int(180 - (distance / max_distance) * 80)
                img_array[i, j] = value
        
        img = Image.fromarray(img_array, mode='L')
        draw = ImageDraw.Draw(img)
        
        # Add some anatomical-looking shapes (simplified chest anatomy)
        # Left lung area
        draw.ellipse([80, 120, 220, 360], fill=200, outline=150, width=2)
        # Right lung area
        draw.ellipse([280, 120, 420, 360], fill=200, outline=150, width=2)
        # Heart shadow
        draw.ellipse([190, 260, 310, 400], fill=130, outline=100, width=2)
        # Spine representation
        draw.rectangle([235, 50, 265, 450], fill=120)
        # Clavicles
        draw.arc([100, 80, 200, 140], start=180, end=0, fill=140, width=3)
        draw.arc([300, 80, 400, 140], start=180, end=0, fill=140, width=3)
        
        # Add text overlay
        try:
            draw.text((20, 20), "CHEST X-RAY - PA VIEW", fill=240)
            draw.text((20, height-30), "Simulated Radiograph for Demonstration", fill=240)
            draw.text((width-150, height-30), "R", fill=240)  # Right marker
        except:
            pass  # In case font issues
        
        # Convert to base64
        buf = BytesIO()
        img.save(buf, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

    def _generate_ct_image(self):
        """Generate a CT scan-like image"""
        from PIL import ImageDraw
        import numpy as np
        
        width, height = 500, 500
        
        # Create base image with CT-like appearance
        img_array = np.ones((height, width), dtype=np.uint8) * 40
        
        # Add circular field of view
        for i in range(height):
            for j in range(width):
                distance = np.sqrt((i - height/2)**2 + (j - width/2)**2)
                if distance < 220:
                    img_array[i, j] = int(100 + 50 * np.random.random())
        
        img = Image.fromarray(img_array, mode='L')
        draw = ImageDraw.Draw(img)
        
        # Add anatomical structures
        draw.ellipse([150, 150, 350, 350], fill=120, outline=100, width=2)
        draw.ellipse([200, 200, 300, 300], fill=80, outline=70, width=2)
        
        try:
            draw.text((20, 20), "CT SCAN - AXIAL VIEW", fill=240)
            draw.text((20, height-30), "Simulated CT for Demonstration", fill=240)
        except:
            pass
        
        buf = BytesIO()
        img.save(buf, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

    def _generate_mri_image(self, query):
        """Generate an MRI-like image"""
        from PIL import ImageDraw
        import numpy as np
        
        width, height = 500, 500
        
        # Create base image with MRI-like appearance
        img_array = np.ones((height, width), dtype=np.uint8) * 30
        
        # Add brain-like structures if it's a brain MRI
        if any(word in query.lower() for word in ['head', 'brain', 'neuro']):
            for i in range(height):
                for j in range(width):
                    distance = np.sqrt((i - height/2)**2 + (j - width/2)**2)
                    if distance < 200:
                        img_array[i, j] = int(150 + 40 * np.random.random())
        else:
            # Spine-like structures
            for i in range(height):
                for j in range(width):
                    if 200 < j < 300:
                        img_array[i, j] = int(120 + 30 * np.random.random())
        
        img = Image.fromarray(img_array, mode='L')
        draw = ImageDraw.Draw(img)
        
        try:
            if any(word in query.lower() for word in ['head', 'brain', 'neuro']):
                draw.text((20, 20), "MRI BRAIN - AXIAL T2", fill=240)
            else:
                draw.text((20, 20), "MRI SPINE - SAGITTAL T2", fill=240)
            draw.text((20, height-30), "Simulated MRI for Demonstration", fill=240)
        except:
            pass
        
        buf = BytesIO()
        img.save(buf, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

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
            test_results = self._generate_lab_results(query, age, gender)
            data["test_results"] = test_results
            
            high_tests = [t['test'] for t in test_results if t['flag'] == 'H']
            low_tests = [t['test'] for t in test_results if t['flag'] == 'L']
            
            abnormal_summary = ""
            if high_tests:
                abnormal_summary += f"Elevated: {', '.join(high_tests)}. "
            if low_tests:
                abnormal_summary += f"Low: {', '.join(low_tests)}. "
            
            lab_prompt = f"Patient is a {age} year old {gender} with symptoms: {query}. {abnormal_summary}Clinical interpretation:"
            interpretation = self._generate_text(lab_prompt, max_tokens=150)
            data["interpretation_guidance"] = interpretation
            
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
            
            # Generate detailed lab summary
            data["detailed_summary"] = self.generate_detailed_lab_summary(test_results, interpretation, patient)
            
        else:  # radiology
            # Determine appropriate imaging modality
            modality, test_name = self._determine_imaging_modality(query)
            
            rad_prompt = f"{test_name} of {age} year old {gender} with {query}. Radiological findings:"
            findings_text = self._generate_text(rad_prompt, max_tokens=120)
            
            impression_prompt = f"Imaging impression for {age} year old {gender} with {query}:"
            impression_text = self._generate_text(impression_prompt, max_tokens=80)
            
            # Generate detailed findings based on modality
            if modality == "xray":
                if len(findings_text) < 20:
                    findings_text = "Chest X-ray evaluation shows clear lung fields bilaterally. No focal consolidation, pleural effusion, or pneumothorax identified. Cardiac silhouette is within normal limits. Bony structures are intact without acute fractures. No acute cardiopulmonary abnormality detected."
            elif modality == "ct":
                if len(findings_text) < 20:
                    findings_text = "CT examination demonstrates normal lung parenchyma without focal consolidation or mass lesion. Mediastinal structures are unremarkable. No pleural or pericardial effusion. Osseous structures show no acute abnormality. Airways are patent."
            elif modality == "mri":
                if len(findings_text) < 20:
                    if any(word in query.lower() for word in ['head', 'brain', 'neuro', 'headache']):
                        findings_text = "MRI brain demonstrates normal gray-white matter differentiation. Ventricular system is normal in size and configuration. No abnormal signal intensity or mass lesion. No evidence of acute infarction or hemorrhage. Midline structures are normal."
                    else:
                        findings_text = "MRI examination shows normal vertebral body alignment and height. Intervertebral disc spaces are preserved. Spinal cord demonstrates normal signal intensity without evidence of compression. Neural foramina are patent. Paraspinal soft tissues are unremarkable."
            
            if len(impression_text) < 20:
                impression_text = "No acute abnormality detected on imaging."
            
            data["specific_tests"] = [test_name]
            data["imaging_findings"] = {
                "description": findings_text,
                "impression": impression_text
            }
            
            # Generate appropriate medical image based on symptoms and modality
            image_data = self._generate_medical_image(modality, query, age, gender)
            data["images"] = [{"data": image_data, "label": f"Simulated {test_name}"}]
            
            # Generate recommendations based on modality
            recommendations = [
                "Clinical correlation with patient presentation recommended",
            ]
            
            if modality == "xray":
                recommendations.extend([
                    "Follow-up chest X-ray may be considered in 6-8 weeks if symptoms persist",
                    "CT chest may be obtained for further characterization if clinically indicated"
                ])
            elif modality == "ct":
                recommendations.extend([
                    "PET-CT may be considered if malignancy is suspected",
                    "Follow-up CT imaging in 3-6 months to assess stability or interval changes"
                ])
            elif modality == "mri":
                recommendations.extend([
                    "MRI with contrast enhancement may provide additional information if needed",
                    "Clinical neurology consultation recommended for correlation"
                ])
            
            data["recommendations"] = recommendations
            
            # Generate detailed radiology summary
            data["detailed_summary"] = self.generate_detailed_radiology_summary(
                data["imaging_findings"],
                impression_text,
                data["recommendations"]
            )

        return data

    def synthesize(self, lab_data, rad_data, patient):
        """Generate final diagnostic synthesis combining all data"""
        query = patient.get('clinical_query', '')
        age = patient.get('age', 'unknown')
        gender = patient.get('gender', 'unknown')
        
        lab_interp = lab_data.get('interpretation_guidance', '')
        rad_impression = rad_data.get('imaging_findings', {}).get('impression', '')
        
        diag_prompt = f"Patient: {age} year old {gender}. Presenting symptoms: {query}. Lab findings: {lab_interp[:100]} Imaging: {rad_impression[:80]} Final clinical diagnosis:"
        diagnosis_text = self._generate_text(diag_prompt, max_tokens=200)
        
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
        
        # DYNAMICALLY GENERATED LAB SUMMARY
        st.markdown(f"""
        <div class="uniform-spacing" style="
        background:#fff3e0;
        padding:25px;
        border-radius:8px;
        border-left:5px solid #ff9800;
        font-size:15px;
        line-height:1.7;">
        
        <h3 style="color:#e65100; margin-top:0;">🔬 Lab Report Summary: AI-Generated Analysis of Key Biomarkers and Test Results</h3>
        
        <p>This section presents an automatically generated interpretation of the patient's laboratory results using Microsoft's BioGPT, analyzing blood parameters, organ function indicators, metabolic markers, and condition-specific tests.</p>
        
        <div style="background:white; padding:20px; border-radius:6px; margin-top:15px;">
        {st.session_state.lab.get('detailed_summary', 'Generating detailed analysis...')}
        </div>
        
        </div>
        """, unsafe_allow_html=True)
        
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
        
        # DYNAMICALLY GENERATED RADIOLOGY SUMMARY
        st.markdown(f"""
        <div class="uniform-spacing" style="
        background:#e3f2fd;
        padding:25px;
        border-radius:8px;
        border-left:5px solid #2196F3;
        font-size:15px;
        line-height:1.7;">
        
        <h3 style="color:#0d47a1; margin-top:0;">🩻 Radiology Report Summary: AI-Driven Interpretation of Imaging Findings</h3>
        
        <p>This section provides an AI-generated summary of radiology imaging observations using Microsoft's BioGPT, interpreting structural abnormalities, inflammation, organ changes, and pattern-based findings.</p>
        
        <div style="background:white; padding:20px; border-radius:6px; margin-top:15px;">
        {st.session_state.rad.get('detailed_summary', 'Generating detailed analysis...')}
        </div>
        
        </div>
        """, unsafe_allow_html=True)
        
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
        
        # Diagnostic Report Summary
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
        
        <h3 style="color:#6a1b9a; margin-top:0;">🏥 Diagnostic Report Summary: Comprehensive AI-Generated Clinical Synthesis</h3>
        
        <p>This section provides an AI-generated synthesis of all diagnostic findings, powered by Microsoft's BioGPT, integrating laboratory biomarkers, imaging observations, clinical presentation, and medical history into a cohesive assessment.</p>
        
        <div style="background:white; padding:20px; border-radius:6px; margin-top:15px;">
        
        <h4 style="color:#6a1b9a;">Overall Clinical Diagnosis:</h4>
        <p style="background:#fff3e0; padding:15px; border-radius:6px; border-left:4px solid #ff9800;">
        <strong>{final_impression}</strong>
        </p>
        
        <h4 style="color:#6a1b9a; margin-top:20px;">Laboratory Interpretation:</h4>
        <p>{lab_interp}</p>
        
        <h4 style="color:#6a1b9a; margin-top:20px;">Imaging Interpretation:</h4>
        <p>{imaging_interp}</p>
        
        <h4 style="color:#6a1b9a; margin-top:20px;">Integrated Clinical Synthesis:</h4>
        <p>The comprehensive evaluation integrates clinical presentation with objective laboratory and imaging data to provide a holistic assessment of the patient's condition. The AI analysis considers:</p>
        <ul>
        <li><strong>Presenting Symptoms:</strong> Patient complaints and clinical manifestations</li>
        <li><strong>Laboratory Biomarkers:</strong> Quantitative assessment of metabolic, hematologic, and organ function parameters</li>
        <li><strong>Imaging Findings:</strong> Structural and anatomical evaluation through radiological studies</li>
        <li><strong>Medical History:</strong> Pre-existing conditions and risk factors</li>
        <li><strong>Age & Gender Factors:</strong> Demographic considerations in disease prevalence and presentation</li>
        </ul>
        
        <p>This multi-modal approach ensures a thorough diagnostic assessment that supports evidence-based clinical decision-making and guides appropriate management strategies. The AI-generated synthesis serves as a clinical decision support tool, providing physicians with a comprehensive view of the patient's condition while maintaining the necessity of clinical judgment and patient-specific considerations.</p>
        
        </div>
        
        </div>
        """, unsafe_allow_html=True)
        
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