import os
import streamlit as st
import pandas as pd
import numpy as np
import re
import google.generativeai as genai
from datetime import datetime
from PyPDF2 import PdfReader
import docx2txt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from fpdf import FPDF
import plotly.express as px

# Load environment variables
load_dotenv('.env')

# Initialize Gemini
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-pro')
except Exception as e:
    st.error(f"Failed to initialize Gemini API: {str(e)}")
    st.stop()

# Session state initialization
steps = ["Job Analysis", "CV Processing", "Shortlisting", "Scheduling"]
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'cvs' not in st.session_state:
    st.session_state.cvs = []
if 'results' not in st.session_state:
    st.session_state.results = pd.DataFrame()
if 'filtered' not in st.session_state:
    st.session_state.filtered = pd.DataFrame()

# Sidebar navigation
with st.sidebar:
    st.markdown("## Process Steps👋")
    for i, step in enumerate(steps):
        class_name = "step-indicator active-step" if i <= st.session_state.current_step else "step-indicator"
        st.markdown(f'<div class="{class_name}">{step}</div>', unsafe_allow_html=True)

st.markdown("""
<style>
    .step-indicator {
        padding: 10px 20px;
        margin: 8px 0;
        border-radius: 5px;
        background: #f8f9fa;
        color: #6c757d;
        transition: all 0.3s ease;
    }
    .active-step {
        background: #007bff;
        color: white !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- Utility Functions -----------------

def extract_text(file):
    if file.name.endswith(".pdf"):
        pdf = PdfReader(file)
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
        return text
    elif file.name.endswith(".docx"):
        return docx2txt.process(file)
    else:
        raise ValueError("Unsupported file format")

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip().lower()

def extract_keywords(text, top_n=10):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform([text])
    scores = zip(tfidf.get_feature_names_out(), tfidf_matrix.toarray()[0])
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return [word for word, score in sorted_scores[:top_n]]

def extract_skills(text):
    skills = ["python", "sql", "excel", "communication", "machine learning", "data analysis", "project management"]
    found = [skill for skill in skills if skill in text.lower()]
    return list(set(found))

def calculate_similarity(job_desc, cvs):
    job_vector = clean_text(job_desc)
    cvs_clean = [clean_text(cv) for cv in cvs]
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([job_vector] + cvs_clean)
    similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    return similarities, vectors

def analyze_with_gemini(prompt):
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

def generate_pdf_report(results_df, job_desc):
    filename = "candidate_report.pdf"
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="SkillMatch Candidate Report", ln=True, align='C')
    pdf.ln(10)
    pdf.multi_cell(0, 10, f"Job Description Summary: {job_desc}")
    pdf.ln(10)
    pdf.cell(200, 10, txt="Candidate Scores", ln=True, align='L')
    
    for idx, row in results_df.iterrows():
        pdf.cell(200, 10, txt=f"{row['File Name']}: {round(row['Match Score']*100, 2)}%", ln=True)
    
    pdf.output(filename)
    return filename

# ----------------- Main Streamlit App -----------------

st.title("🤖SkillMatch: Recruitment Platform")

# Step 1: Job Description Analysis
with st.expander("Step 1: Job Description Analysis", expanded=True):
    job_desc_file = st.file_uploader("Upload Job Description", type=["pdf", "docx"])
    if job_desc_file:
        try:
            job_desc = extract_text(job_desc_file)
            if not job_desc.strip():
                raise ValueError("Empty document or failed text extraction")

            st.session_state.job_desc = job_desc
            st.success("✅ Analysis Complete!")

            st.subheader("🔍Job Requirements")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**🧿Keywords**")
                keywords = extract_keywords(job_desc)
                st.write(" ".join([f'`{kw}`' for kw in keywords]))

            with col2:
                st.markdown("**🎯Required Skills**")
                skills = extract_skills(job_desc)
                st.write("\n".join([f"- {s.title()}" for s in skills]) or "No specific skills detected")

            st.session_state.current_step = 1

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Step 2: CV Upload & Processing
if 'job_desc' in st.session_state:
    with st.expander("Step 2: CV Processing", expanded=st.session_state.current_step >= 1):
        cv_files = st.file_uploader("Upload Candidate CVs", type=["pdf", "docx"], accept_multiple_files=True)
        if cv_files and not st.session_state.processed:
            with st.spinner("⌛Analyzing CVs..."):
                try:
                    st.session_state.cvs = [extract_text(cv) for cv in cv_files]
                    file_names = [cv.name for cv in cv_files]
                    similarity_scores, _ = calculate_similarity(st.session_state.job_desc, st.session_state.cvs)
                    st.session_state.results = pd.DataFrame({
                        "Candidate": [f"CV {i+1}" for i in range(len(file_names))],
                        "Match Score": similarity_scores,
                        "File Name": file_names
                    }).sort_values("Match Score", ascending=False)
                    st.session_state.processed = True
                    st.session_state.current_step = 2
                except Exception as e:
                    st.error(f"CV Processing Error: {str(e)}")

# Step 3: Shortlisting
if not st.session_state.results.empty:
    with st.expander("Step 3: Candidate Shortlisting", expanded=st.session_state.current_step >= 2):
        st.subheader("📊 AI-Powered Ranking")
        fig_data = st.session_state.results.copy()
        fig_data["Score Group"] = pd.cut(fig_data["Match Score"], bins=[0, 0.3, 0.6, 1], labels=["Low", "Medium", "High"])
        st.bar_chart(fig_data["Score Group"].value_counts())

        threshold = st.slider("Match Threshold (%)", 0, 100, 50) / 100
        st.session_state.filtered = st.session_state.results[st.session_state.results["Match Score"] >= threshold]

        if not st.session_state.filtered.empty:
            st.session_state.current_step = max(st.session_state.current_step, 3)
            st.dataframe(
                st.session_state.filtered[["Candidate", "Match Score", "File Name"]].style.format({"Match Score": "{:.2%}"}),
                height=300
            )

            selected = st.selectbox("View Candidate Details", st.session_state.filtered["File Name"])
            if selected:
                idx = st.session_state.results[st.session_state.results["File Name"] == selected].index[0]
                candidate_text = st.session_state.cvs[idx]

                col1, col2 = st.columns([1, 2])
                with col1:
                    st.subheader("🌟Key Qualifications")
                    st.write(candidate_text[:300] + "...")  # Replace with actual parsing if needed

                


# Step 4: Report Generation
if st.session_state.current_step >= 3:
    if st.button("⚙️ Generate PDF Report"):
        try:
            report_path = generate_pdf_report(st.session_state.results, st.session_state.job_desc[:300])
            with open(report_path, "rb") as f:
                st.download_button("Download Report", data=f, file_name=report_path, mime="application/pdf")
            os.remove(report_path)
        except:
            csv = st.session_state.results.to_csv(index=False)
            st.download_button("Download CSV", data=csv, file_name="candidate_report.csv", mime="text/csv")

# Step 5: Interview Scheduling
if st.session_state.current_step >= 3:
    with st.expander("Step 4: Interview Scheduling", expanded=True):
        st.subheader("📅 Schedule Interviews")
        selected_candidates = st.multiselect("Select candidates to schedule interviews:", st.session_state.filtered["Candidate"])
        date = st.date_input("Interview Date")
        time = st.time_input("Interview Time")
        if st.button("Schedule Interviews"):
            if selected_candidates:
                schedule = pd.DataFrame({
                    "Candidate": selected_candidates,
                    "Date": [date.strftime("%Y-%m-%d")] * len(selected_candidates),
                    "Time": [time.strftime("%H:%M")] * len(selected_candidates)
                })
                st.success("✅ Interviews scheduled!")
                st.dataframe(schedule)
            else:
                st.warning("Please select at least one candidate.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("⚓**About SkillMatch**\n\nSkillMatch automates the recruitment process from job analysis to interview scheduling.")
st.sidebar.markdown("**SkillMatch**\n👨‍💻 Created by **Team Tech_Burner**© 2025")
