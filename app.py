import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime
from jinja2 import Environment, BaseLoader
from io import BytesIO
from weasyprint import HTML
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import base64
from PIL import Image
import os
import re

# === KNOWLEDGE BASE ===
class MedicalKnowledgeBase:
    """Handles loading and accessing patient data from a CSV file."""
    def __init__(self, patients_db_path: str = "patients.csv"):
        self.patients_db_path = patients_db_path
        self.patients = self._load_patients_from_csv()

    def _load_patients_from_csv(self) -> Dict[str, Any]:
        try:
            df = pd.read_csv(self.patients_db_path)
            # Fill any missing values with an empty string to prevent errors
            df.fillna('', inplace=True)
            # Convert numeric types after filling NaNs
            for col in ['age', 'weight', 'height']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

            df.drop_duplicates(subset=['patient_name'], keep='first', inplace=True)
            
            # Convert the DataFrame to a dictionary keyed by patient_name
            patients_dict = df.set_index('patient_name').to_dict('index')
            return patients_dict
        except FileNotFoundError:
            st.error(f"Error: The patient database '{self.patients_db_path}' was not found.")
            # Return a default dictionary to allow the app to run
            return {
                "John Doe": {
                    "patient_name": "John Doe", "age": 45, "gender": "Male", "weight": 70.0, "height": 175.0,
                    "blood_group": "A+", "phone": "+1-555-0123", "emergency_contact": "Jane Doe - +1-555-0456",
                    "insurance_id": "INS123456789", "allergies": "None known", "medications": "None",
                    "medical_history": "No significant past medical history", "family_history": "No significant family history",
                    "clinical_query": "The patient has been complaining of excessive thirst, frequent urination, and fatigue for the past 2 weeks."
                }
            }

    def search_patient_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        return self.patients.get(name)

# === TEMPLATE STRINGS ===
class TemplateStrings:
    """Contains all the HTML and CSS for the report templates."""
    
    # Using a shared header and footer structure to keep reports consistent
    SHARED_HEADER = """
        <div class="hospital-header">
            <div class="hospital-info">
                <div><span class="hospital-icon">🏥</span><span class="hospital-name">Apollo Diagnostics</span></div>
                <div class="hospital-address">
                    📍 123 Medical Center Drive, Healthcare District, City - 400001<br>
                    📞 +91-22-2345-6789 | 📧 reports@apollodiagnostics.com | 🌐 www.apollodiagnostics.com
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
                    <div class="patient-info-line"><span class="patient-info-label">Ref. Doctor:</span> Dr. Sarah Mitchell, MD</div>
                </div>
            </div>
        </div>
    """

    SHARED_FOOTER = """
        <div class="report-footer">
            <div class="verification-section">
                <div class="verification-line"><span class="footer-icon">✓</span> Verified by: Dr. Rajesh Kumar, MD Pathology</div>
                <div style="font-size: 10px; color: #666;">Digital Signature Applied</div>
            </div>
            <div class="disclaimer">This report is for physician reference only. Not valid for medico-legal purposes without original signature.</div>
            <div class="contact-info">📞 24/7 Helpdesk: +91-22-2345-6789 | 📧 support@apollodiagnostics.com</div>
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
            .disclaimer { font-style: italic; color: #666; margin: 10px 0; }
            .contact-info { margin: 10px 0; }
            .page-info { margin-top: 15px; font-weight: bold; color: #003366; }
            ul { padding-left: 20px; margin-top: 5px; }
            li { margin-bottom: 5px; }
        </style>
    """

    # --- CORRECTED DEFINITION ---
    # No 'f' prefix, just a regular multi-line string.
    # Jinja will handle the {{...}} and {%...%} parts later.
    DIAGNOSIS_TEMPLATE = SHARED_CSS + SHARED_HEADER + """
<div class="report-body">
    <div class="section-header">CLINICAL SUMMARY</div>
    <div class="content-box"><p>{{ diagnosis.clinical_summary }}</p></div>

    <div class="section-header">KEY DIAGNOSTIC FINDINGS</div>
    <div class="content-box">
        <h4>🔬 Key Laboratory Findings</h4>
        <ul>{% for finding in diagnosis.key_lab_findings %}<li>{{ finding }}</li>{% endfor %}</ul>
    </div>
    <div class="content-box">
        <h4>📷 Key Radiology Findings</h4>
        <ul>{% for finding in diagnosis.key_radio_findings %}<li>{{ finding }}</li>{% endfor %}</ul>
    </div>

    <div class="section-header">DIFFERENTIAL DIAGNOSIS</div>
    <div class="content-box"><ul>{% for item in diagnosis.differential_diagnosis %}<li>{{ item }}</li>{% endfor %}</ul></div>

    <div class="section-header">DIAGNOSTIC RATIONALE & IMPRESSION</div>
    <div class="content-box">
        <p>{{ diagnosis.rationale }}</p>
        <hr style="margin: 15px 0; border: 0; border-top: 1px solid #ddd;">
        <p><strong>IMPRESSION: {{ diagnosis.impression }}</strong></p>
    </div>

    <div class="section-header">RECOMMENDED ACTION PLAN</div>
    <div class="content-box">
        <h4>Diagnostic Plan:</h4><ul>{% for item in diagnosis.plan_diagnostic %}<li>{{ item }}</li>{% endfor %}</ul>
        <h4 style="margin-top: 15px;">Therapeutic Plan:</h4><ul>{% for item in diagnosis.plan_therapeutic %}<li>{{ item }}</li>{% endfor %}</ul>
        <h4 style="margin-top: 15px;">Consultations & Follow-up:</h4><ul>{% for item in diagnosis.plan_consultation %}<li>{{ item }}</li>{% endfor %}</ul>
    </div>
</div>
""" + SHARED_FOOTER

    # --- CORRECTED DEFINITION ---
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

    # --- CORRECTED DEFINITION ---
    RADIOLOGY_TEMPLATE = SHARED_CSS + SHARED_HEADER + """
<div class="report-body">
    <div class="section-header">EXAMINATION DETAILS</div>
    <div class="content-box">
        <p><strong>Study:</strong> {{ report.specific_tests[0] if report.specific_tests }}</p>
        <p><strong>Indication:</strong> {{ patient.clinical_query }}</p>
    </div>
    
    <div class="section-header">IMAGING STUDIES</div>
    <div style="text-align: center; margin: 20px 0;">
    {% if report.images and report.images|length > 0 %}
        {% for image in report.images %}
        <div style="display: inline-block; border: 1px solid #ddd; border-radius: 8px; padding: 10px; background: #f9f9f9;">
            <img src="{{ image.data }}" alt="{{ image.label }}" style="max-width: 400px; height: auto;">
            <div style="margin-top: 10px; font-weight: bold; color: #003366;">{{ image.label }}</div>
        </div>
        {% endfor %}
    {% else %}
        <p><i>No images attached to this report.</i></p>
    {% endif %}
    </div>

    <div class="section-header">FINDINGS</div>
    <div class="content-box"><p>{{ report.imaging_findings.description | replace('\\n', '<br>') }}</p></div>

    <div class="section-header">IMPRESSION</div>
    <div class="content-box" style="background: #fff3e0;"><p><strong>{{ report.imaging_findings.impression }}</strong></p></div>
    
    <div class="section-header">RECOMMENDATIONS</div>
    <div class="content-box"><ul>{% for rec in report.recommendations %}<li>{{ rec }}</li>{% endfor %}</ul></div>
</div>
""" + SHARED_FOOTER


# === LLM INTEGRATION ===
@st.cache_resource
def load_llm_model():
    """Loads the T5-base model and tokenizer. Cached for performance."""
    try:
        # Upgraded to the 'base' model for better quality results.
        model_name = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        generator = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=150  # Increased token limit for more detailed responses
        )
        st.success(f"✅ Loaded AI Model ({model_name}).")
        return generator
    except Exception as e:
        st.error(f"❌ Failed to load AI model: {e}")
        return None

class MedicalLLMInterface:
    """Interface for interacting with the LLM to process queries and generate content."""
    def __init__(self):
        self.generator = load_llm_model()

    def _generate_text(self, prompt: str, max_tokens: int = 150) -> str:
        """Generates text from a prompt using the loaded LLM."""
        if not self.generator: return "AI model not available."
        try:
            outputs = self.generator(prompt, num_return_sequences=1, num_beams=3, no_repeat_ngram_size=2, max_new_tokens=max_tokens)
            generated_text = outputs[0]['generated_text'].strip() if outputs else "AI could not generate a response."
            # Clean up common artifacts
            return re.sub(r'^(-|\*|\d+\.)\s*', '', generated_text)
        except Exception as e:
            return f"An error occurred during AI generation: {e}"

    def process_medical_query(self, query: str, patient_context: Dict, report_type: str) -> Dict[str, Any]:
        """Generates content for a single lab or radiology report."""
        prompt = f"For a {report_type} report, analyze the following case: A {patient_context.get('age')} year old {patient_context.get('gender')} with symptoms: {query.lower()}. Provide a concise interpretation."
        interpretation = self._generate_text(prompt)
        
        response = {
            "specific_tests": self._get_relevant_tests(report_type, query),
            "interpretation_guidance": interpretation,
            "recommendations": self._get_recommendations(report_type, query),
            "images": patient_context.get("uploaded_images", [])
        }
        if report_type == "lab":
            response["test_results"] = self._generate_smart_lab_results(query)
        elif report_type == "radiology":
            response["imaging_findings"] = self._get_imaging_findings(query, interpretation)
            if not response["images"]:
                if os.path.exists("chest x-ray image 1.png") and (img_data := load_local_image("chest x-ray image 1.png")):
                    response["images"].append({"data": img_data, "label": "Default Chest X-Ray (AP View)"})
        return response

    def synthesize_reports(self, lab_data: Dict, radio_data: Dict, patient_info: Dict) -> Dict[str, Any]:
        """Synthesizes lab and radiology data into a comprehensive diagnostic report."""
        lab_summary = ", ".join([f"{t['test']}: {t['result']} {t['units']} (Flag: {t['flag']})" for t in lab_data.get('test_results', [])])
        radio_impression = radio_data.get('imaging_findings', {}).get('impression', 'Not available')

        base_prompt = f"""Synthesize a diagnostic report for the following case:
- Patient: A {patient_info['age']}-year-old {patient_info['gender']}.
- Clinical Query: {patient_info['clinical_query']}.
- Past Medical History: {patient_info.get('medical_history', 'N/A')}.
- Key Lab Results: {lab_summary}.
- Key Radiology Impression: {radio_impression}.

Based on ALL the information above, """

        return {
            "clinical_summary": self._generate_text(base_prompt + "provide a concise, one-paragraph clinical summary integrating the patient's symptoms and findings."),
            "key_lab_findings": [f"{t['test']}: {t['result']} {t['units']} (Ref: {t['range']})" for t in lab_data.get('test_results', []) if t.get('flag') != 'N'],
            "key_radio_findings": [radio_impression],
            "differential_diagnosis": self._generate_text(base_prompt + "list the top three differential diagnoses.", max_tokens=96).split('\n'),
            "rationale": self._generate_text(base_prompt + "write a detailed diagnostic rationale. Explain how the lab and radiology findings support the most likely diagnosis over the other differentials."),
            "impression": self._generate_text(base_prompt + "state the final, primary diagnostic impression in one definitive sentence.", max_tokens=64),
            "plan_diagnostic": self._generate_text(base_prompt + "suggest two necessary follow-up diagnostic tests as a list.", max_tokens=96).split('\n'),
            "plan_therapeutic": self._generate_text(base_prompt + "suggest two initial therapeutic interventions as a list.", max_tokens=96).split('\n'),
            "plan_consultation": self._generate_text(base_prompt + "suggest one or two specialist consultations required for this case.", max_tokens=64).split('\n')
        }

    def _generate_smart_lab_results(self, query: str) -> list:
        query_lower = query.lower()
        if any(s in query_lower for s in ['thirst', 'urination', 'glucose', 'diabetes']):
            return [{"test": "Fasting Glucose", "result": "145", "flag": "H", "range": "70-99", "units": "mg/dL"}, {"test": "HbA1c", "result": "7.2", "flag": "H", "range": "<5.7", "units": "%"}]
        elif any(s in query_lower for s in ['chest pain', 'heart', 'cardiac']):
            return [{"test": "High-Sensitivity Troponin I", "result": "0.09", "flag": "H", "range": "<0.04", "units": "ng/mL"}, {"test": "BNP", "result": "150", "flag": "H", "range": "<100", "units": "pg/mL"}]
        return [{"test": "Complete Blood Count", "result": "Normal", "flag": "N", "range": "WNL", "units": "N/A"}]

    def _get_relevant_tests(self, report_type: str, query: str) -> list:
        if report_type == "lab": return ["Comprehensive Metabolic Panel, Complete Blood Count"]
        if report_type == "radiology": return ["Chest X-Ray, 2 Views"]
        return ["Clinical Assessment"]

    def _get_recommendations(self, report_type: str, query: str) -> list:
        base = ["Clinical correlation is advised.", "Follow-up with referring physician to discuss results."]
        if "diabetes" in query.lower(): base.append("Consider endocrinology consultation for diabetes management.")
        if "cardiac" in query.lower(): base.append("Cardiology consultation is recommended due to abnormal cardiac markers.")
        return base

    def _get_imaging_findings(self, query: str, interpretation: str) -> dict:
        query_lower = query.lower()
        description = "The lungs are clear. No evidence of focal consolidation, pleural effusion, or pneumothorax.\nThe cardiomediastinal silhouette is within normal limits for age."
        impression = "No acute cardiopulmonary process identified."
        if any(s in query_lower for s in ['chest pain', 'heart', 'cardiac']):
             impression = "Mild cardiomegaly noted. No acute pulmonary edema. Clinical correlation with cardiac enzymes is recommended."
        elif 'cough' in query_lower or 'infection' in query_lower:
             description = "There are patchy opacities in the right lower lobe, suggestive of early pneumonia. The remainder of the lungs are clear."
             impression = "Findings consistent with right lower lobe pneumonia."
        
        return {"description": description, "impression": f"{impression} AI-assisted interpretation suggests: {interpretation}"}


# === REPORT GENERATOR ===
class ReportGenerator:
    """Handles the creation of HTML and PDF reports."""
    def __init__(self):
        self.llm_interface = MedicalLLMInterface()
        self.templates = {
            'lab': TemplateStrings.LAB_TEMPLATE,
            'radiology': TemplateStrings.RADIOLOGY_TEMPLATE,
            'diagnosis': TemplateStrings.DIAGNOSIS_TEMPLATE
        }
        self.env = Environment(loader=BaseLoader())

    def generate_single_report(self, query: str, patient_info: Dict, report_type: str) -> tuple[str, str, Optional[Dict]]:
        llm_analysis = self.llm_interface.process_medical_query(query, patient_info, report_type)
        context = {"patient": patient_info, "report": llm_analysis, "report_date": datetime.now().strftime("%B %d, %Y")}
        html_report = self.env.from_string(self.templates[report_type]).render(context)
        return f"{report_type.capitalize()} Report", html_report, llm_analysis

    def generate_diagnosis_report(self, lab_data: Dict, radio_data: Dict, patient_info: Dict) -> tuple[str, str, Dict]:
        synthesis = self.llm_interface.synthesize_reports(lab_data, radio_data, patient_info)
        context = {"patient": patient_info, "diagnosis": synthesis, "report_date": datetime.now().strftime("%B %d, %Y")}
        html_report = self.env.from_string(self.templates['diagnosis']).render(context)
        return "Diagnostic Synthesis Report", html_report, synthesis


# === UTILS & CACHING ===
def generate_pdf_from_html(html_string: str) -> Optional[bytes]:
    """Generates a PDF file from an HTML string using WeasyPrint."""
    try:
        return HTML(string=html_string).write_pdf()
    except Exception as e:
        st.error(f"Error generating PDF: {e}")
        st.warning("PDF generation requires system libraries. If this fails on deployment, ensure `packages.txt` is configured correctly.")
        return None

def handle_image_upload(uploaded_file):
    if uploaded_file is None: return None
    try:
        image = Image.open(uploaded_file)
        image.thumbnail((800, 800), Image.Resampling.LANCZOS)
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"
    except Exception: return None

def load_local_image(image_path: str) -> Optional[str]:
    if not os.path.exists(image_path): return None
    try:
        with open(image_path, 'rb') as f:
            return f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"
    except Exception: return None

@st.cache_data(show_spinner="Generating Report...")
def generate_cached_report(_report_generator, query, patient_info, report_type):
    return _report_generator.generate_single_report(query, patient_info, report_type)

@st.cache_data(show_spinner="Synthesizing Diagnosis...")
def generate_cached_diagnosis(_report_generator, lab_data, radio_data, patient_info):
    return _report_generator.generate_diagnosis_report(lab_data, radio_data, patient_info)

# === UI RENDERING FUNCTIONS ===
def render_generation_page(kb: MedicalKnowledgeBase, report_generator: ReportGenerator):
    st.header("Step 1: Generate Individual Reports")

    patient_list = [""] + list(kb.patients.keys())
    selected_patient_name = st.selectbox("Select Existing Patient (or leave blank for new)", options=patient_list)

    patient_data = kb.search_patient_by_name(selected_patient_name) if selected_patient_name else {}

    with st.form(key="patient_form"):
        st.subheader("Patient Information")
        col1, col2 = st.columns(2)
        with col1:
            patient_name = st.text_input("Patient Name", value=patient_data.get("patient_name", ""))
            age = st.number_input("Age", 0, 120, int(patient_data.get("age", 0)))
        with col2:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=["Male", "Female", "Other"].index(patient_data.get("gender", "Male")))
            medical_history = st.text_area("Past Medical History", value=patient_data.get("medical_history", ""))
        
        query = st.text_area("Clinical Query / Patient Symptoms", value=patient_data.get("clinical_query", ""), height=100)
        
        uploaded_files = st.file_uploader("Upload Imaging (Optional, for Radiology Report)", type=["png", "jpg"], accept_multiple_files=True)
        
        submitted = st.form_submit_button("Generate Reports")

    if submitted:
        if not patient_name or not query:
            st.warning("Please fill in Patient Name and Clinical Query.")
        else:
            patient_info = {"patient_name": patient_name, "age": age, "gender": gender, "clinical_query": query, "medical_history": medical_history}
            
            with st.spinner("Generating Lab Report..."):
                _, html, analysis = generate_cached_report(report_generator, query, patient_info, "lab")
                st.session_state.generated_lab_report = analysis
                st.session_state.patient_info_for_report = patient_info
                st.session_state.lab_report_html = html
                st.session_state.lab_report_pdf = generate_pdf_from_html(html)
            
            with st.spinner("Generating Radiology Report..."):
                patient_info_radio = patient_info.copy()
                patient_info_radio["uploaded_images"] = [handle_image_upload(f) for f in uploaded_files if f]
                _, html, analysis = generate_cached_report(report_generator, query, patient_info_radio, "radiology")
                st.session_state.generated_radiology_report = analysis
                st.session_state.radiology_report_html = html
                st.session_state.radiology_report_pdf = generate_pdf_from_html(html)

            st.success("✅ Lab and Radiology reports generated!")
           # st.balloons()
    
    st.markdown("---")
    st.subheader("Generated Report Previews")
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.markdown("##### Lab Report")
        if 'lab_report_html' in st.session_state:
            st.download_button("⬇️ Download PDF", st.session_state.lab_report_pdf, f"Lab_{st.session_state.patient_info_for_report['patient_name']}.pdf", "application/pdf")
            st.components.v1.html(st.session_state.lab_report_html, height=400, scrolling=True)
    with dl_col2:
        st.markdown("##### Radiology Report")
        if 'radiology_report_html' in st.session_state:
            st.download_button("⬇️ Download PDF", st.session_state.radiology_report_pdf, f"Radiology_{st.session_state.patient_info_for_report['patient_name']}.pdf", "application/pdf")
            st.components.v1.html(st.session_state.radiology_report_html, height=400, scrolling=True)


def render_diagnosis_page(report_generator: ReportGenerator):
    st.header("Step 2: Generate Combined Diagnostic Synthesis")
    st.markdown("This tool synthesizes the generated reports into a single, detailed diagnostic conclusion.")

    if 'generated_lab_report' not in st.session_state or 'generated_radiology_report' not in st.session_state:
        st.warning("⚠️ Please generate both Lab and Radiology reports in 'Step 1' before proceeding.")
        return

    st.success(f"✅ Reports for **{st.session_state.patient_info_for_report['patient_name']}** are ready for synthesis.")

    if st.button("🧠 **Generate Detailed Diagnostic Synthesis**", type="primary", use_container_width=True):
        lab_data = st.session_state.generated_lab_report
        radio_data = st.session_state.generated_radiology_report
        patient_info = st.session_state.patient_info_for_report

        _, html, synthesis_data = generate_cached_diagnosis(report_generator, lab_data, radio_data, patient_info)
        st.session_state.diagnosis_html = html
        st.session_state.diagnosis_pdf = generate_pdf_from_html(html)
        st.success("🎉 Combined Diagnosis Report Generated!")

    if 'diagnosis_html' in st.session_state:
        st.subheader("Consolidated Diagnostic Report")
        patient_name = st.session_state.patient_info_for_report['patient_name'].replace(' ', '_')
        if st.session_state.get('diagnosis_pdf'):
            st.download_button("⬇️ **Download Full Diagnosis Report (PDF)**", st.session_state.diagnosis_pdf, f"Diagnosis_{patient_name}.pdf", "application/pdf", use_container_width=True)
        st.components.v1.html(st.session_state.diagnosis_html, height=800, scrolling=True)

# === MAIN APP ===
def main():
    st.set_page_config(page_title="AI Medical Report Generator", layout="wide", initial_sidebar_state="expanded")
    
    st.sidebar.title("🩺 AI Medical Diagnostician")
    app_mode = st.sidebar.radio("Navigation", ("Step 1: Generate Reports", "Step 2: Combined Diagnosis"))
    st.sidebar.markdown("---")
    
    if st.sidebar.button("Clear All Data", use_container_width=True):
        st.cache_data.clear()
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.sidebar.info("This is a demo tool. All generated content is from an AI and is for illustrative purposes only. **Not for clinical use.**")

    kb = MedicalKnowledgeBase()
    report_generator = ReportGenerator()

    if app_mode == "Step 1: Generate Reports":
        render_generation_page(kb, report_generator)
    elif app_mode == "Step 2: Combined Diagnosis":
        render_diagnosis_page(report_generator)

if __name__ == "__main__":
    main()