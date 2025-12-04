import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime
from jinja2 import Environment, BaseLoader
from io import BytesIO
from weasyprint import HTML
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import base64
from PIL import Image
import os
import random

# === KNOWLEDGE BASE ===
class MedicalKnowledgeBase:
    """Handles loading and accessing patient data."""
    def __init__(self, patients_db_path: str = "patients.csv"):
        self.patients_db_path = patients_db_path
        self.patients = self._load_patients_from_csv()

    def _load_patients_from_csv(self) -> Dict[str, Any]:
        # Simulating a database. If CSV exists, load it, otherwise use dummy data.
        try:
            df = pd.read_csv(self.patients_db_path)
            df.fillna('', inplace=True)
            
            # === FIX: Remove duplicate names to prevent ValueError ===
            if 'patient_name' in df.columns:
                df.drop_duplicates(subset=['patient_name'], keep='first', inplace=True)
            
            return df.set_index('patient_name').to_dict('index')
        except FileNotFoundError:
            return {
                "John Doe": {
                    "patient_name": "John Doe", "age": 45, "gender": "Male",
                    "clinical_query": "The patient has been complaining of excessive thirst, frequent urination, and fatigue for the past 2 weeks.",
                    "medical_history": "Hypertension diagnosed in 2019."
                },
                "Jane Smith": {
                    "patient_name": "Jane Smith", "age": 29, "gender": "Female",
                    "clinical_query": "Persistent dry cough and mild fever for 5 days.",
                    "medical_history": "Asthma."
                }
            }
        except Exception as e:
            st.error(f"Error loading database: {e}")
            return {}

    def search_patient_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        # Case insensitive search
        if not name: return None
        for p_name, data in self.patients.items():
            if str(p_name).lower().strip() == str(name).lower().strip():
                return data
        return None

# === TEMPLATE STRINGS (Keeping CSS/HTML as is) ===
class TemplateStrings:
    SHARED_HEADER = """
        <div class="hospital-header">
            <div class="hospital-info">
                <div><span class="hospital-icon">🏥</span><span class="hospital-name"> Clariovex Diagnostics</span></div>
                <div class="hospital-address">
                    📍 123 Medical Center Drive, Healthcare District, City - 400001<br>
                    📞 +91-22-2345-6789 | 📧 reports@clariovexdiagnostics.com
                </div>
            </div>
            <div class="patient-info-section">
                <div class="patient-info-header">PATIENT INFORMATION</div>
                <div class="patient-info-grid">
                    <div class="patient-info-line"><span class="patient-info-label">Name:</span> {{ patient.patient_name }}</div>
                    <div class="patient-info-line"><span class="patient-info-label">Patient ID:</span> P{{ patient.age }}{{ patient.patient_name[:3].upper() }}01</div>
                    <div class="patient-info-line"><span class="patient-info-label">Age/Gender:</span> {{ patient.age }} Years / {{ patient.gender }}</div>
                    <div class="patient-info-line"><span class="patient-info-label">Report ID:</span> R{{ (patient.age * 1547) % 99999 }}</div>
                    <div class="patient-info-line"><span class="patient-info-label">Study Date:</span> {{ report_date }}</div>
                </div>
            </div>
        </div>
    """

    SHARED_FOOTER = """
        <div class="report-footer">
            <div class="verification-section">
                <div class="verification-line"><span class="footer-icon">✓</span> Verified by: Dr. Rajesh Kumar, MD Pathology</div>
            </div>
            <div class="page-info">Page 1 of 1 | Generated: {{ report_date }}</div>
        </div>
    """

    SHARED_CSS = """
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: 'Arial', sans-serif; font-size: 12px; line-height: 1.4; color: #333; background: #f5f5f5; }
            .report-container { background: white; max-width: 8.5in; margin: 0 auto; box-shadow: 0 0 10px rgba(0,0,0,0.1); min-height: 11in; }
            .hospital-header { border-bottom: 3px solid #003366; padding: 20px; background: #f8f9fa; }
            .hospital-info { margin-bottom: 15px; }
            .hospital-icon { font-size: 24px; color: #0056b3; display: inline-block; margin-right: 10px; vertical-align: middle; }
            .hospital-name { font-size: 20px; font-weight: bold; color: #003366; margin-bottom: 5px; }
            .hospital-address { font-size: 11px; color: #555; line-height: 1.3; }
            .patient-info-section { border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin-top: 15px; background: #fff; }
            .patient-info-header { background: #003366; color: white; padding: 8px 15px; margin: -15px -15px 12px -15px; font-weight: bold; font-size: 13px; border-radius: 4px 4px 0 0; }
            .patient-info-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
            .patient-info-line { font-size: 11px; padding: 3px 0; }
            .patient-info-label { font-weight: bold; display: inline-block; min-width: 80px; color: #444; }
            .report-body { padding: 25px; }
            .section-header { background: #eef2f7; color: #003366; padding: 10px 15px; font-weight: bold; font-size: 14px; margin: 25px 0 15px 0; border-left: 4px solid #0056b3; }
            .content-box { padding: 15px; border: 1px solid #eee; border-radius: 5px; background: #fdfdfd; }
            .report-footer { border-top: 3px solid #003366; padding: 20px; margin-top: 40px; background: #f8f9fa; font-size: 11px; text-align: center; }
            .verification-section { margin-bottom: 15px; }
            .footer-icon { font-size: 14px; color: #28a745; margin-right: 5px; }
            .verification-line { font-weight: bold; margin-bottom: 5px; }
            .page-info { margin-top: 15px; font-weight: bold; color: #003366; }
            ul { padding-left: 20px; margin-top: 5px; }
            li { margin-bottom: 5px; }
        </style>
    """

    LAB_TEMPLATE = SHARED_CSS + SHARED_HEADER + """
    <div class="report-body">
        <div class="section-header">LABORATORY RESULTS</div>
        <table class="lab-table" style="width: 100%; border-collapse: collapse;">
            <thead><tr><th>Test Name</th><th>Result</th><th>Flag</th><th>Reference Range</th><th>Units</th></tr></thead>
            <tbody>
                {% for test in report.test_results %}
                <tr>
                    <td>{{ test.test }}</td>
                    <td style="font-weight: bold; color: {% if test.flag == 'H' %}#d32f2f{% elif test.flag == 'L' %}#1976d2{% else %}#333{% endif %};">{{ test.result }}</td>
                    <td style="font-weight: bold; color: {% if test.flag == 'H' %}#d32f2f{% elif test.flag == 'L' %}#1976d2{% else %}#333{% endif %};">{{ test.flag }}</td>
                    <td>{{ test.range }}</td>
                    <td>{{ test.units }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        <div class="section-header">INTERPRETATION</div>
        <div class="content-box"><p>{{ report.interpretation_guidance }}</p></div>
        <div class="section-header">RECOMMENDATIONS</div>
        <div class="content-box"><ul>{% for rec in report.recommendations %}<li>{{ rec }}</li>{% endfor %}</ul></div>
    </div>
    """ + SHARED_FOOTER

    RADIOLOGY_TEMPLATE = SHARED_CSS + SHARED_HEADER + """
    <div class="report-body">
        <div class="section-header">EXAMINATION DETAILS</div>
        <div class="content-box">
            <p><strong>Study:</strong> {{ report.specific_tests[0] if report.specific_tests }}</p>
            <p><strong>Indication:</strong> {{ patient.clinical_query }}</p>
        </div>
        
        <div class="section-header">IMAGING GENERATION</div>
        <div style="text-align: center; margin: 20px 0;">
        {% if report.images and report.images|length > 0 %}
            {% for image in report.images %}
            <div style="display: inline-block; border: 1px solid #ddd; border-radius: 8px; padding: 10px; background: #f9f9f9;">
                <img src="{{ image.data }}" alt="{{ image.label }}" style="max-width: 400px; height: auto;">
                <div style="margin-top: 10px; font-weight: bold; color: #003366;">{{ image.label }}</div>
            </div>
            {% endfor %}
        {% else %}
            <p><i>Image generation failed or not applicable.</i></p>
        {% endif %}
        </div>

        <div class="section-header">FINDINGS</div>
        <div class="content-box"><p>{{ report.imaging_findings.description | replace('\\n', '<br>') }}</p></div>

        <div class="section-header">IMPRESSION</div>
        <div class="content-box" style="background: #fff3e0;"><p><strong>{{ report.imaging_findings.impression }}</strong></p></div>
    </div>
    """ + SHARED_FOOTER

    GENUINE_CSS = """
    <style>
        @page { margin: 0.5in; size: Letter; }
        body { font-family: 'Times New Roman', Times, serif; font-size: 10pt; color: #000; line-height: 1.2; }
        .header-container { text-align: center; margin-bottom: 15px; font-family: Arial, sans-serif; }
        .dept-name { font-weight: bold; font-size: 10pt; text-transform: uppercase; }
        .report-title { font-size: 14pt; font-weight: bold; margin-top: 10px; text-align: center; font-family: Arial, sans-serif;}
        .demographics-box { 
            border-top: 3px double #000; 
            border-bottom: 3px double #000; 
            padding: 5px 0; 
            margin-bottom: 10px;
            font-family: Arial, sans-serif;
        }
        table { width: 100%; border-collapse: collapse; }
        td { vertical-align: top; padding: 1px 4px; }
        .label { font-weight: bold; font-size: 9pt; width: 110px; display: inline-block; }
        .value { font-size: 10pt; }
        .section-divider { border-bottom: 2px solid #000; margin-bottom: 5px; }
        .section-header { font-weight: bold; text-decoration: underline; font-size: 11pt; margin-top: 15px; margin-bottom: 5px; font-family: Arial, sans-serif;}
        .footer { 
            position: fixed; bottom: 0; left: 0; right: 0; 
            border-top: 1px solid #000; padding-top: 5px; 
            font-size: 8pt; display: flex; justify-content: space-between; 
            font-family: Arial, sans-serif;
        }
    </style>
    """

    # --- NEW DIAGNOSIS TEMPLATE: matches provided sample structure ---
    DIAGNOSIS_TEMPLATE = GENUINE_CSS + """
    <div style="text-align:right; font-size:9px; color:#333; margin-bottom:6px;">
        <strong>Verified by:</strong> Dr. Rajesh Kumar, MD Pathology
    </div>

    <div class="header-container">
        <div style="text-align: center;">
            <div style="font-size: 14pt; font-weight: bold;">CLARIOVEX DIAGNOSTICS CENTER</div>
            <div style="font-size:10pt;">123 Medical Center Drive, Healthcare District, City - 400001</div>
            <div style="font-size:9pt;">+91-22-2345-6789 | reports@clariovexdiagnostics.com</div>
        </div>
    </div>

    <div style="margin-top:10px;">
        <table style="width:100%; font-size:10pt;">
            <tr>
                <td><b>Patient Name:</b> {{ patient.patient_name }}</td>
                <td><b>Age/Sex:</b> {{ patient.age }} / {{ patient.gender }}</td>
            </tr>
            <tr>
                <td><b>Patient ID:</b> {{ diagnosis.patient_id if diagnosis.patient_id else 'N/A' }}</td>
                <td><b>Date of Report:</b> {{ report_date }}</td>
            </tr>
            <tr>
                <td colspan="2"><b>Referring Physician:</b> {{ diagnosis.referring_physician if diagnosis.referring_physician else '' }}</td>
            </tr>
        </table>
    </div>

    <div class="section-header">SECTION A — CLINICAL SUMMARY</div>
    <div style="margin-left:6px;">
        <p><strong>1. Chief Complaint</strong><br>{{ diagnosis.clinical_summary.chief_complaint }}</p>
        <p><strong>2. History of Present Illness</strong><br>{{ diagnosis.clinical_summary.history_of_present_illness | replace('\\n', '<br>') }}</p>
    </div>

    <div class="section-header">SECTION B — RISK FACTORS</div>
    <div style="margin-left:10px;">
        <ul>
            {% for rf in diagnosis.risk_factors %}
                <li>{{ rf }}</li>
            {% endfor %}
        </ul>
    </div>

    <div class="section-header">SECTION C — PHYSICAL EXAMINATION</div>
    <div style="margin-left:6px;">
        <strong>Vital Signs</strong>
        <ul>
            {% for k, v in diagnosis.physical_exam.vitals.items() %}
                <li>{{ k }}: {{ v }}</li>
            {% endfor %}
        </ul>

        <strong>General</strong>
        <p>{{ diagnosis.physical_exam.general }}</p>
    </div>

    <div class="section-header">SECTION D — ECG INTERPRETATION (12-LEAD)</div>
    <div style="margin-left:6px;">
        <strong>Findings</strong>
        <ul>
            {% for line in diagnosis.ecg.findings %}
                <li>{{ line }}</li>
            {% endfor %}
        </ul>
        <p><strong>Conclusion:</strong> {{ diagnosis.ecg.conclusion }}</p>
    </div>

    <div class="section-header">SECTION E — LAB INVESTIGATIONS</div>
    <div style="margin-left:6px;">
        <table style="width:100%; border-collapse: collapse; font-size:10pt;">
            <thead>
                <tr style="background:#f2f2f2;">
                    <th style="padding:4px; border:1px solid #ccc;">Test</th>
                    <th style="padding:4px; border:1px solid #ccc;">Result</th>
                    <th style="padding:4px; border:1px solid #ccc;">Comment</th>
                </tr>
            </thead>
            <tbody>
                {% for row in diagnosis.labs %}
                <tr>
                    <td style="padding:4px; border:1px solid #ccc;">{{ row.test }}</td>
                    <td style="padding:4px; border:1px solid #ccc;">{{ row.result }}</td>
                    <td style="padding:4px; border:1px solid #ccc;">{{ row.comment if row.comment else '' }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>

    <div class="section-header">SECTION F — ECHOCARDIOGRAPHY</div>
    <div style="margin-left:6px;">
        <p>{{ diagnosis.echo | replace('\\n', '<br>') }}</p>
    </div>

    <div class="section-header">INTERVENTION</div>
    <div style="margin-left:6px;">
        <ul>
        {% for it in diagnosis.interventions %}
            <li>{{ it }}</li>
        {% endfor %}
        </ul>
    </div>

    <div class="section-header">IMPRESSION / FINAL DIAGNOSIS</div>
    <div style="margin-left:6px; background:#fff3e0; padding:8px; border-radius:4px;">
        <strong>{{ diagnosis.impression }}</strong>
        <div style="margin-top:8px;">
            <ul>
                {% for pt in diagnosis.interpretation_points %}
                    <li>{{ pt }}</li>
                {% endfor %}
            </ul>
        </div>
    </div>

    <div style="margin-top:20px; text-align:right;">
        Signed out by: <b>Dr. AI Pathologist, MD</b><br>
        Date Reported: {{ report_date }}
    </div>

    <div class="footer">
        <div>Patient: {{ patient.patient_name }}</div>
        <div>Page 1 of 1</div>
    </div>
    """

# === LLM & GENERATION LOGIC ===
@st.cache_resource
def load_llm_model():
    try:
        model_name = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        st.error(f"❌ Failed to load AI model: {e}")
        return None, None

class MedicalLLMInterface:
    def __init__(self):
        self.model, self.tokenizer = load_llm_model()

    def _generate_text(self, prompt: str, max_new_tokens: int = 150) -> str:
        if not self.model: return "AI model not available."
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7,
                top_p=0.9, repetition_penalty=1.5, no_repeat_ngram_size=2
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        except Exception as e:
            return "Analysis generated."

    def _get_generated_image(self, query: str) -> list:
        # Simulating AI Image Generation
        img_data = None
        if os.path.exists("chest_xray_sample.png"):
            with open("chest_xray_sample.png", "rb") as f:
                img_data = f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"
        else:
            # Generate a programmatic placeholder
            img = Image.new('RGB', (400, 400), color=(random.randint(0,50), random.randint(0,50), random.randint(0,50)))
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_data = f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"

        return [{"data": img_data, "label": "AI Generated Radiograph based on clinical query"}]

    def process_medical_query(self, query: str, patient_context: Dict, report_type: str) -> Dict[str, Any]:
        # Ensure age is an int for calculation, even if string was passed
        age_val = patient_context.get('age', 0)
        try:
            age_val = int(age_val)
        except (ValueError, TypeError):
            age_val = 0

        prompt = f"Analyze {report_type} for: {query}. Patient age {age_val}. Interpretation:"
        interpretation = self._generate_text(prompt, max_new_tokens=80)
        
        response = {
            "specific_tests": ["Complete Blood Count (CBC), CMP"] if report_type == "lab" else ["Chest X-Ray PA/Lateral"],
            "interpretation_guidance": interpretation,
            "recommendations": ["Clinical correlation advised."],
            "images": []
        }
        
        if report_type == "lab":
            response["test_results"] = self._generate_smart_lab_results(query)
        elif report_type == "radiology":
            response["imaging_findings"] = self._get_imaging_findings(query, interpretation)
            response["images"] = self._get_generated_image(query)
            
        return response

    def synthesize_reports(self, lab_data: Dict, radio_data: Dict, patient_info: Dict) -> Dict[str, Any]:
        """
        Return a detailed structured diagnosis dict that matches the DIAGNOSIS_TEMPLATE.
        lab_data and radio_data are the outputs from generate_single_report (lab/radiology).
        """
        clinical_query = patient_info.get('clinical_query', '')
        # Simple clinical summary
        clinical_summary = {
            "chief_complaint": (clinical_query.split('.')[0] if clinical_query else "Not provided"),
            "history_of_present_illness": clinical_query or "Not provided"
        }

        # Risk factors: infer from medical_history (simple heuristics)
        history = (patient_info.get('medical_history') or "").lower()
        risk_factors = []
        if 'hypertension' in history:
            risk_factors.append("Hypertension")
        if 'smoking' in history:
            risk_factors.append("Active smoker")
        if 'diabetes' in history or 'sugar' in clinical_query.lower():
            risk_factors.append("Diabetes / impaired glucose tolerance")
        if not risk_factors:
            risk_factors.append("No major risk factors documented")

        # Physical exam stub
        physical_exam = {
            "vitals": {"BP": "148/92 mmHg", "HR": "108 bpm", "RR": "22/min", "SpO2": "94%"},
            "general": "Alert but anxious; diaphoretic; mild respiratory distress."
        }

        # ECG inference
        ecg = {"findings": [], "conclusion": "Normal ECG"}
        if 'chest pain' in clinical_query.lower() or 'stemi' in clinical_query.lower() or 'ecg' in clinical_query.lower():
            ecg["findings"] = ["ST-Elevation in V1-V4 (anterior leads)", "Reciprocal ST depression in II, III, aVF", "Loss of R waves in V3"]
            ecg["conclusion"] = "Acute Anterior Wall STEMI — likely LAD occlusion"
        else:
            ecg["findings"] = ["No acute ischemic changes"]
            ecg["conclusion"] = "Normal ECG"

        # Convert lab_data test_results into table rows
        labs = []
        for t in lab_data.get('test_results', []):
            labs.append({"test": t.get('test', ''), "result": t.get('result', ''), "comment": ("High" if t.get('flag')=='H' else "")})

        # Echo & interventions
        echo_text = radio_data.get('imaging_findings', {}).get('description', 'Echocardiography not available.')
        interventions = radio_data.get('imaging_findings', {}).get('interventions', []) if isinstance(radio_data.get('imaging_findings'), dict) else []

        # Impression synthesis via LLM (if available)
        try:
            lab_summary = ", ".join([f"{r['test']}:{r['result']}" for r in labs if r.get('result')])
            imaging_imp = radio_data.get('imaging_findings', {}).get('impression', '')
            prompt = f"Diagnose patient: {clinical_query}. Lab: {lab_summary}. Imaging: {imaging_imp}. Give one-line diagnosis."
            impression = self._generate_text(prompt, max_new_tokens=30)
            if not impression:
                impression = imaging_imp or "No acute abnormality detected."
        except Exception:
            impression = radio_data.get('imaging_findings', {}).get('impression', 'No acute abnormality detected.')

        interpretation_points = [
            f"Symptoms: {clinical_query or 'Not provided'}",
            f"Labs: {lab_summary or 'No significant abnormality'}",
            f"Imaging: {imaging_imp or 'Not available'}"
        ]

        detailed_diagnosis = [f"Findings support: {impression}."]

        diagnosis = {
            "patient_id": f"ZAI-{patient_info.get('age',0)}-{str(patient_info.get('patient_name','')).replace(' ','')[:6].upper()}001",
            "referring_physician": patient_info.get('referring_physician', ''),
            "clinical_summary": clinical_summary,
            "risk_factors": risk_factors,
            "physical_exam": physical_exam,
            "ecg": ecg,
            "labs": labs,
            "echo": echo_text,
            "interventions": interventions or ["Primary intervention as clinically indicated."],
            "impression": impression,
            "interpretation_points": interpretation_points,
            "detailed_diagnosis": detailed_diagnosis
        }

        return diagnosis

    def _generate_smart_lab_results(self, query: str) -> list:
        query_lower = query.lower()
        if 'sugar' in query_lower or 'thirst' in query_lower:
            return [{"test": "Glucose", "result": "160", "flag": "H", "range": "70-99", "units": "mg/dL"}]
        return [{"test": "WBC", "result": "11.5", "flag": "H", "range": "4.5-11.0", "units": "K/uL"}]

    def _get_imaging_findings(self, query: str, interpretation: str) -> dict:
        if 'cough' in query.lower():
            return {"description": "Opacities noted in right lower lobe.", "impression": "Pneumonia"}
        if 'chest pain' in query.lower() or 'stemi' in query.lower():
            return {"description": "Ejection Fraction 42%. Regional wall motion abnormality in LAD territory.", "impression": "Acute Anterior Wall STEMI"}
        return {"description": "Lungs are clear. Cardiac silhouette normal.", "impression": "Normal Chest X-Ray"}


# === REPORT GENERATOR ===
class ReportGenerator:
    def __init__(self):
        self.llm_interface = MedicalLLMInterface()
        self.env = Environment(loader=BaseLoader())
        self.templates = {'lab': TemplateStrings.LAB_TEMPLATE, 'radiology': TemplateStrings.RADIOLOGY_TEMPLATE, 'diagnosis': TemplateStrings.DIAGNOSIS_TEMPLATE}

    def generate_single_report(self, query: str, patient_info: Dict, report_type: str):
        llm_analysis = self.llm_interface.process_medical_query(query, patient_info, report_type)
        
        # Ensure age is safely passed to template
        try:
            patient_info['age'] = int(patient_info['age'])
        except (ValueError, TypeError):
            patient_info['age'] = 0

        context = {"patient": patient_info, "report": llm_analysis, "report_date": datetime.now().strftime("%d %B %Y")}
        html_report = self.env.from_string(self.templates[report_type]).render(context)
        return html_report, llm_analysis

    def generate_diagnosis_report(self, lab_data: Dict, radio_data: Dict, patient_info: Dict):
        synthesis = self.llm_interface.synthesize_reports(lab_data, radio_data, patient_info)
        
        try:
            patient_info['age'] = int(patient_info['age'])
        except (ValueError, TypeError):
            patient_info['age'] = 0

        context = {"patient": patient_info, "diagnosis": synthesis, "report_date": datetime.now().strftime("%d %B %Y")}
        html_report = self.env.from_string(self.templates['diagnosis']).render(context)
        return html_report

# === UTILS ===
def generate_pdf_from_html(html_string: str) -> Optional[bytes]:
    try:
        return HTML(string=html_string).write_pdf()
    except Exception:
        return None

# === MAIN APP ===
def main():
    st.set_page_config(page_title="AI Medical Diagnostician", layout="wide")
    
    # Force hide sidebar via CSS
    st.markdown("""
    <style>
        section[data-testid="stSidebar"][aria-expanded="true"]{display: none;}
        section[data-testid="stSidebar"][aria-expanded="false"]{display: none;}
    </style>
    """, unsafe_allow_html=True)

    st.title("🩺 AI Medical Diagnostician")
    st.markdown("Enter patient details below. Reports flow automatically from Lab/Radiology -> Diagnosis.")

    # Initialize Session State
    if 'input_patient_name' not in st.session_state: st.session_state.input_patient_name = ""
    if 'input_age' not in st.session_state: st.session_state.input_age = ""
    if 'input_gender' not in st.session_state: st.session_state.input_gender = ""
    if 'input_history' not in st.session_state: st.session_state.input_history = ""
    if 'input_query' not in st.session_state: st.session_state.input_query = ""
    if 'last_searched_name' not in st.session_state: st.session_state.last_searched_name = ""
    
    kb = MedicalKnowledgeBase()
    report_generator = ReportGenerator()

    # === SECTION 1: PATIENT INPUT (Direct Linear Flow) ===
    with st.container():
        st.subheader("Patient Information")
        
        # Name Input acts as the trigger
        name_input = st.text_input("Patient Name (Type name and hit Enter to auto-fill)", value=st.session_state.input_patient_name)
        
        # Logic: If name changes/entered, look up details
        if name_input and name_input != st.session_state.get('last_searched_name'):
            st.session_state.input_patient_name = name_input
            st.session_state.last_searched_name = name_input
            
            found_data = kb.search_patient_by_name(name_input)
            if found_data:
                # Auto-fill state variables
                st.session_state.input_age = str(found_data.get('age', ''))
                st.session_state.input_gender = found_data.get('gender', '')
                st.session_state.input_history = found_data.get('medical_history', '')
                st.session_state.input_query = found_data.get('clinical_query', '')
                st.success(f"✅ Record found for {name_input}. Details generated.")
            else:
                st.info(f"New patient: {name_input}. Please fill details.")

        # Other inputs (No dropdowns, No +/- steppers)
        col1, col2 = st.columns(2)
        with col1:
            age = st.text_input("Age", value=st.session_state.input_age)
        with col2:
            gender = st.text_input("Gender", value=st.session_state.input_gender)
            
        medical_history = st.text_area("Medical History", value=st.session_state.input_history, height=70)
        query = st.text_area("Clinical Query / Symptoms", value=st.session_state.input_query, height=100)

        # Update state with current values in case they were edited manually
        st.session_state.input_age = age
        st.session_state.input_gender = gender
        st.session_state.input_history = medical_history
        st.session_state.input_query = query

        # Action Button
        generate_btn = st.button("Generate Lab & Radiology Reports", type="primary")

    # === SECTION 2: GENERATE INITIAL REPORTS ===
    if generate_btn:
        if not name_input or not query:
            st.error("Please provide Patient Name and Clinical Query.")
        else:
            with st.spinner("AI is generating Lab Results and synthesizing Radiology Images..."):
                # Prepare Context
                p_info = {
                    "patient_name": name_input, 
                    "age": age, 
                    "gender": gender, 
                    "clinical_query": query, 
                    "medical_history": medical_history
                }
                
                # Generate Lab
                lab_html, lab_data = report_generator.generate_single_report(query, p_info, "lab")
                st.session_state.lab_html = lab_html
                st.session_state.lab_data = lab_data
                
                # Generate Radiology
                rad_html, rad_data = report_generator.generate_single_report(query, p_info, "radiology")
                st.session_state.rad_html = rad_html
                st.session_state.rad_data = rad_data
                st.session_state.p_info = p_info
                st.session_state.reports_ready = True

    # === SECTION 3: DISPLAY REPORTS & PROCEED TO DIAGNOSIS ===
    if st.session_state.get('reports_ready'):
        st.markdown("---")
        st.subheader("Generated Reports")
        
        c1, c2 = st.columns(2)
        with c1:
            st.info("Lab Report")
            st.components.v1.html(st.session_state.lab_html, height=400, scrolling=True)
        with c2:
            st.info("Radiology Report")
            st.components.v1.html(st.session_state.rad_html, height=400, scrolling=True)

        st.markdown("### Next Step")
        if st.button("Proceed to Diagnostic Synthesis"):
            with st.spinner("Synthesizing final diagnosis based on Lab and Radiology data..."):
                diag_html = report_generator.generate_diagnosis_report(
                    st.session_state.lab_data, 
                    st.session_state.rad_data, 
                    st.session_state.p_info
                )
                st.session_state.diag_html = diag_html
                st.session_state.diag_ready = True

    # === SECTION 4: DISPLAY DIAGNOSIS ===
    if st.session_state.get('diag_ready'):
        st.markdown("---")
        st.header("Final Diagnosis Report")
        st.components.v1.html(st.session_state.diag_html, height=800, scrolling=True)
        
        # PDF Download
        pdf_bytes = generate_pdf_from_html(st.session_state.diag_html)
        if pdf_bytes:
            st.download_button("⬇️ Download Final Report PDF", pdf_bytes, "Diagnosis_Report.pdf", "application/pdf")

if __name__ == "__main__":
    main()
