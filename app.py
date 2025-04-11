import os
import requests
import streamlit as st
import pandas as pd
import numpy as np
import re
import google.generativeai as genai
import smtplib
from datetime import datetime
from PyPDF2 import PdfReader
import docx2txt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from fpdf import FPDF
import plotly.express as px
import plotly.graph_objects as go

# Load environment variables
load_dotenv('.env')

# Initialize Gemini
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-pro')
except Exception as e:
    st.error(f"Failed to initialize Gemini API: {str(e)}")
    st.stop()

# DeepSeek API configuration
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}"
}

def generate_with_deepseek(prompt):
    """Generate content using DeepSeek API for email-related tasks"""
    try:
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 500
        }
        response = requests.post(DEEPSEEK_API_URL, json=payload, headers=DEEPSEEK_HEADERS)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"

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
if 'candidate_details' not in st.session_state:
    st.session_state.candidate_details = {}

# ----------------- Custom Styling -----------------
st.set_page_config(page_title="SkillMatch", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    /* Specific to job description upload section */
    div[data-testid="stExpander"]:has(> div[aria-label="📋 1. Job Description Analysis"]) .upload-section {
        background: #ffffff !important;
        border: 2px dashed #1e88e5 !important;
        border-radius: 15px;
        padding: 3rem;
        margin: 2rem 0;
        text-align: center;
    }

    div[data-testid="stExpander"]:has(> div[aria-label="📋 1. Job Description Analysis"]) .upload-section h3 {
        color: #1a237e !important;
        font-size: 1.4rem;
        margin-bottom: 0.5rem;
    }

    div[data-testid="stExpander"]:has(> div[aria-label="📋 1. Job Description Analysis"]) .file-upload-text {
        color: #2d3436 !important;
        font-size: 1rem;
        opacity: 0.9;
    }

    /* Hide other upload sections */
    div[data-testid="stExpander"]:not(:has(> div[aria-label="📋 1. Job Description Analysis"])) .upload-section {
        display: none !important;
    }
    
    .candidate-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
    }
    
    .candidate-table th, .candidate-table td {
        border: 1px solid #e0e0e0;
        padding: 8px 12px;
        text-align: left;
    }
    
    .candidate-table th {
        background-color: #1e88e5;
        color: white;
    }
    
    .candidate-table tr:nth-child(even) {
        background-color: #f5f5f5;
    }
    
    .match-score {
        font-weight: bold;
        color: #1e88e5;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- Navigation Progress Bar -----------------
def show_progress():
    progress = st.session_state.current_step/(len(steps)-1)
    progress_html = f"""
    <div style="margin: 40px 0 60px 0;">
        <div style="height: 20px; background: #e3f2fd; border-radius: 10px; overflow: hidden;">
            <div style="width: {progress*100}%; height: 100%; background: #1e88e5; transition: width 0.5s ease;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 20px; padding: 0 15px;">
            {''.join([f'<div style="text-align: center; color: {"#1e88e5" if i <= st.session_state.current_step else "#90a4ae"}; font-weight: {"600" if i == st.session_state.current_step else "500"}">{step}</div>' for i, step in enumerate(steps)])}
        </div>
    </div>
    """
    st.markdown(progress_html, unsafe_allow_html=True)

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
    skills = ["python", "sql", "excel", "communication", "machine learning", 
             "data analysis", "project management", "team leadership", 
             "cloud computing", "statistical modeling", "java", "javascript",
             "html", "css", "react", "angular", "node.js", "django", "flask",
             "aws", "azure", "docker", "kubernetes", "git", "agile"]
    found = [skill for skill in skills if skill in text.lower()]
    return list(set(found))

def extract_contact_info(text):
    # Extract email
    email = re.findall(r'[\w\.-]+@[\w\.-]+', text)
    email = email[0] if email else "Not found"
    
    # Extract phone number
    phone = re.findall(r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})', text)
    phone = phone[0] if phone else "Not found"
    
    # Extract name (simple heuristic - first line that's not empty)
    name = "Not found"
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if line and not any(word in line.lower() for word in ['curriculum vitae', 'resume', 'cv', 'phone', 'email']):
            name = line
            break
    
    return {
        'name': name,
        'email': email,
        'phone': phone
    }

def extract_qualifications(text):
    qualifications = []
    education_keywords = [
        'education', 'qualification', 'degree', 'bachelor', 'master', 
        'phd', 'diploma', 'academic', 'university', 'college', 
        'school', 'coursework', 'certification'
    ]
    
    lines = text.split('\n')
    
    # Method 1: Section-based detection
    for i, line in enumerate(lines):
        if any(keyword in line.lower() for keyword in education_keywords):
            qualifications.extend(line.strip() for line in lines[i:i+6] if line.strip())
            break
    
    # Method 2: Degree pattern matching
    if not qualifications:
        degree_patterns = [
            r'\b(bachelor\'s? degree\b|\bbs\b|\bb\.?s?c\b|\bb\.?tech\b|\bb\.?e\b)',
            r'\b(master\'s? degree\b|\bms\b|\bm\.?s?c\b|\bm\.?tech\b|\bmba\b)',
            r'\b(ph\.?d\b|doctorate\b)',
            r'\bdiploma\b',
            r'\bassociate\'s? degree\b',
            r'\bcertificate\b'
        ]
        for line in lines:
            if any(re.search(pattern, line.lower()) for pattern in degree_patterns):
                qualifications.append(line.strip())
    
    # Method 3: Gemini API fallback
    if not qualifications:
        prompt = f"Extract educational qualifications from this resume. Return only degrees and institutions:\n{text[:3000]}"
        response = analyze_with_gemini(prompt)
        if "Error:" not in response:
            qualifications.extend(response.split('\n'))
    
    return qualifications[:3] if qualifications else ["Education information not found"]

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
    pdf.cell(200, 10, txt="MillMatch Candidate Report", ln=True, align='C')
    pdf.ln(10)
    pdf.multi_cell(0, 10, f"Job Description Summary: {job_desc}")
    pdf.ln(10)
    pdf.cell(200, 10, txt="Candidate Scores", ln=True, align='L')
    
    for idx, row in results_df.iterrows():
        pdf.cell(200, 10, txt=f"{row['File Name']}: {round(row['Match Score']*100, 2)}%", ln=True)
    
    pdf.output(filename)
    return filename

# ----------------- Main App -----------------
st.title("🤖 SkillMatch: Intelligent Recruitment Platform")
show_progress()

# Step 1: Job Description Analysis
with st.expander("📋 1. Job Description Analysis", expanded=st.session_state.current_step == 0):
    st.markdown("""
    <div class="upload-section">
        <h3>Drag and drop job description file here</h3>
        <p class="file-upload-text">Limit 200MB per file • PDF, DOCX</p>
    </div>
    """, unsafe_allow_html=True)
    
    job_desc_file = st.file_uploader("Upload Job Description", type=["pdf", "docx"], key="job_upload", label_visibility="collapsed")
    
    if job_desc_file:
        with st.spinner("🔍 Analyzing Job Requirements..."):
            try:
                job_desc = extract_text(job_desc_file)
                st.session_state.job_desc = job_desc
                
                cols = st.columns([1,2])
                with cols[0]:
                    st.subheader("📌 Key Insights")
                    keywords = extract_keywords(job_desc)
                    st.markdown(f"**Top Keywords:** {' '.join([f'`{kw}`' for kw in keywords[:5]])}")
                    
                    skills = extract_skills(job_desc)
                    st.markdown("**🔧 Required Skills:**")
                    st.write("\n".join([f"- {s.title()}" for s in skills]) or "No specific skills detected")
                
                with cols[1]:
                    st.subheader("📊 Requirements Breakdown")
                    fig = px.pie(values=[len(keywords), len(skills)], 
                                names=['Keywords', 'Skills'],
                                color_discrete_sequence=['#1e88e5', '#1565c0'])
                    st.plotly_chart(fig, use_container_width=True)
                
                st.session_state.current_step = 1
                st.success("✅ Job Analysis Complete!")

            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

# Step 2: CV Processing
if 'job_desc' in st.session_state:
    with st.expander("📄 2. CV Processing", expanded=st.session_state.current_step == 1):
        st.markdown("""
        <div class="upload-section">
            <h3>Drag and drop candidate CVs here</h3>
            <p class="file-upload-text">Limit 200MB per file • PDF, DOCX</p>
        </div>
        """, unsafe_allow_html=True)
        
        cv_files = st.file_uploader("Upload Candidate CVs", type=["pdf", "docx"], 
                                  accept_multiple_files=True, key="cv_upload", 
                                  label_visibility="collapsed")
        
        if cv_files and not st.session_state.processed:
            with st.spinner("⏳ Processing CVs..."):
                try:
                    st.session_state.cvs = [extract_text(cv) for cv in cv_files]
                    file_names = [cv.name for cv in cv_files]
                    similarity_scores, _ = calculate_similarity(st.session_state.job_desc, st.session_state.cvs)
                    
                    # Extract candidate details
                    for i, cv_text in enumerate(st.session_state.cvs):
                        contact_info = extract_contact_info(cv_text)
                        qualifications = extract_qualifications(cv_text)
                        skills = extract_skills(cv_text)
                        
                        st.session_state.candidate_details[file_names[i]] = {
                            'name': contact_info['name'],
                            'email': contact_info['email'],
                            'phone': contact_info['phone'],
                            'qualifications': qualifications,
                            'skills': skills
                        }
                    
                    st.session_state.results = pd.DataFrame({
                        "Candidate": [f"CV {i+1}" for i in range(len(file_names))],
                        "Match Score": similarity_scores,
                        "File Name": file_names,
                        "Name": [st.session_state.candidate_details[fn]['name'] for fn in file_names],
                        "Email": [st.session_state.candidate_details[fn]['email'] for fn in file_names],
                        "Top Skills": [', '.join(st.session_state.candidate_details[fn]['skills'][:3]) for fn in file_names],
                        "Education": [st.session_state.candidate_details[fn]['qualifications'][0] if st.session_state.candidate_details[fn]['qualifications'] else "Not specified" for fn in file_names]
                    }).sort_values("Match Score", ascending=False)
                    
                    # Visualization
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total CVs", len(cv_files), help="Number of submitted applications")
                    col2.metric("Average Match", f"{np.mean(similarity_scores)*100:.1f}%", 
                              delta=f"{np.mean(similarity_scores)*100 - 50:.1f}% vs baseline")
                    col3.metric("Top Score", f"{np.max(similarity_scores)*100:.1f}%", 
                              help="Highest matching candidate score")
                    
                    st.session_state.processed = True
                    st.session_state.current_step = 2
                    st.success(f"✅ Processed {len(cv_files)} CVs!")

                except Exception as e:
                    st.error(f"CV Processing Error: {str(e)}")

# Step 3: Shortlisting
if not st.session_state.results.empty:
    with st.expander("🎯 3. Candidate Shortlisting", expanded=st.session_state.current_step == 2):
        st.subheader("📈 Match Distribution")
        threshold = st.slider("Set Match Threshold (%)", 0, 100, 50, key="threshold") / 100
        st.session_state.filtered = st.session_state.results[st.session_state.results["Match Score"] >= threshold]
        
        # Visualization
        fig = px.histogram(st.session_state.results, x="Match Score", nbins=20,
                         title="Candidate Match Score Distribution",
                         color_discrete_sequence=['#1e88e5'])
        fig.add_vline(x=threshold, line_dash="dash", line_color="#d32f2f",
                    annotation_text=f"Threshold: {threshold*100:.0f}%")
        st.plotly_chart(fig, use_container_width=True)
        
        if not st.session_state.filtered.empty:
            st.subheader("🏆 Top Candidates")
            
            # Add custom styling for candidate cards
            st.markdown("""
            <style>
                .candidate-card {
                    border: 2px solid #1e88e5;
                    border-radius: 10px;
                    padding: 15px;
                    margin: 10px 0;
                    background: black;
                }
                .candidate-score {
                    font-size: 1.4em;
                    color: #1e88e5;
                    font-weight: bold;
                    text-align: right;
                }
                .candidate-details {
                    margin-left: 15px;
                }
            </style>
            """, unsafe_allow_html=True)

            # Display candidates in cards
            for _, row in st.session_state.filtered.iterrows():
                candidate = st.session_state.candidate_details[row['File Name']]
                with st.container():
                    st.markdown(f"""
                    <div class="candidate-card">
                        <div class="candidate-details">
                            <h4>{row['Name']}</h4>
                            <p>📧 {row['Email']}</p>
                            <p>🛠️ <strong>Skills:</strong> {', '.join(candidate['skills'][:3])}</p>
                            <p>🎓 <strong>Education:</strong> {candidate['qualifications'][0][:50]}{'...' if len(candidate['qualifications'][0]) > 50 else ''}</p>
                            <div class='candidate-score' style='margin-top: 10px; color: #1e88e5; font-size: 1.2em;'>
                                Match Score: {row['Match Score']*100:.1f}%
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Candidate Details
            selected = st.selectbox("Select Candidate for Detailed View", st.session_state.filtered["File Name"])
            if selected:
                candidate_data = st.session_state.candidate_details[selected]
                
                tab1, tab2 = st.tabs(["📄 Profile Summary", "📊 Skills Analysis"])
                with tab1:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### Contact Information")
                        st.write(f"**⚙️Name:** {candidate_data['name']}")
                        st.write(f"**📧Email:** {candidate_data['email']}")
                        st.write(f"**📞Phone:** {candidate_data['phone']}")
                        
    
                    
                    with col2:
                        st.markdown("### Skills")
                        skills_cols = st.columns(3)
                        for i, skill in enumerate(candidate_data['skills']):
                            with skills_cols[i % 3]:
                                st.markdown(f"✅ {skill.title()}")
                
                with tab2:
                    st.subheader("Skills Radar")
                    job_skills = extract_skills(st.session_state.job_desc)
                    candidate_skills = candidate_data['skills']
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(
                        r=[1]*len(job_skills),
                        theta=job_skills,
                        fill='toself',
                        name='Job Requirements',
                        line_color='#1e88e5'
                    ))
                    fig.add_trace(go.Scatterpolar(
                        r=[1 if skill in candidate_skills else 0 for skill in job_skills],
                        theta=job_skills,
                        name='Candidate Skills',
                        line_color='#d32f2f'
                    ))
                    fig.update_layout(polar=dict(radialaxis=dict(visible=True)),
                                    showlegend=True)
                    st.plotly_chart(fig, use_container_width=True)
                
                st.session_state.current_step = 3

# Step 4: Scheduling
if st.session_state.current_step >= 3:
    with st.expander("📅 4. Interview Scheduling", expanded=st.session_state.current_step == 3):
        st.subheader("🗓 Schedule Interviews")
        
        if 'selected_candidates' not in st.session_state:
            st.session_state.selected_candidates = []
        
        st.session_state.selected_candidates = st.multiselect(
            "Select candidates to interview:", 
            st.session_state.filtered["File Name"],
            format_func=lambda x: f"{st.session_state.candidate_details[x]['name']} ({x})"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            date = st.date_input("Interview Date")
        with col2:
            time = st.time_input("Interview Time")
        
        interview_details = st.text_area("Interview Details", 
                                       value="Please join the interview at the scheduled time. Bring any relevant documents.")
        
        # Move email template generation outside the send button
        if st.session_state.selected_candidates:
            position_name = " ".join(job_desc_file.name.split('.')[0].split('_')).title()
            
            # Generate email template only once
            if 'email_body' not in st.session_state:
                with st.spinner("🤖 Generating email template..."):
                    prompt = f"""Create a professional interview invitation email template with placeholders for:
                    - Candidate name
                    - Position name
                    - Interview date
                    - Interview time
                    - Meeting details
                    - Company name
                    Use a friendly but professional tone."""
                    
                    ai_template = generate_with_deepseek(prompt)
                    default_template = """Dear {name},
We're pleased to invite you for the {position} position interview.
🗓 Date: {date}
⏰ Time: {time}
📍 Details: {details}
Please confirm your attendance by replying to this email.
Best regards,
{company} HR Team"""
                    
                    st.session_state.email_body = st.text_area("Email Content", 
                        value=ai_template if "Error:" not in ai_template else default_template,
                        height=250
                    )
                    
                    st.session_state.email_subject = st.text_input("Email Subject", 
                        value=f"Invitation: {position_name} Position"
                    )
        
        # Fixed indentation for the send button block
        if st.button("📨 Send Invitations", type="primary"):
            if st.session_state.selected_candidates:
                successful_emails = 0
                for candidate_file in st.session_state.selected_candidates:
                    candidate = st.session_state.candidate_details[candidate_file]
                    if candidate['email'] == "Not found":
                        st.warning(f"⚠️ No email found for {candidate['name']}")
                        continue
                    
                    try:
                        # Format email body
                        formatted_body = st.session_state.email_body.format(
                            name=candidate['name'],
                            position=position_name,
                            date=date.strftime("%B %d, %Y"),
                            time=time.strftime("%I:%M %p"),
                            details=interview_details,
                            company="Your Company"
                        )
                        
                        # Create secure connection
                        with smtplib.SMTP(os.getenv("SMTP_SERVER"), int(os.getenv("SMTP_PORT"))) as server:
                            server.ehlo()
                            server.starttls()
                            server.login(
                                os.getenv("EMAIL_ADDRESS"), 
                                os.getenv("EMAIL_PASSWORD")
                            )
                            server.sendmail(
                                os.getenv("EMAIL_ADDRESS"),
                                candidate['email'],
                                f"Subject: {st.session_state.email_subject}\n\n{formatted_body}".encode('utf-8')
                            )
                        
                        successful_emails += 1
                        st.success(f"✉️ Sent to {candidate['name']} ({candidate['email']})")
                        
                    except smtplib.SMTPAuthenticationError:
                        st.error("Authentication failed. Check your email credentials in .env file")
                    except Exception as e:
                        st.error(f"Failed to send to {candidate['name']}: {str(e)}")
                
                st.balloons()
                st.success(f"🎉 Successfully sent {successful_emails}/{len(st.session_state.selected_candidates)} emails!")
            else:
                st.warning("Please select candidates first!")


# Report Generation
if st.session_state.current_step >= 3:
    st.sidebar.markdown("### Report Generation")
    report_type = st.sidebar.radio("Select Report Type", ["CSV", "PDF"])
    
    if st.sidebar.button("📥 Download Report"):
        if report_type == "CSV":
            csv = st.session_state.results.to_csv(index=False)
            st.sidebar.download_button(
                label="Download CSV",
                data=csv,
                file_name="candidate_report.csv",
                mime="text/csv"
            )
        else:
            pdf_file = generate_pdf_report(st.session_state.results, st.session_state.job_desc[:1000])
            with open(pdf_file, "rb") as f:
                st.sidebar.download_button(
                    label="Download PDF",
                    data=f,
                    file_name="candidate_report.pdf",
                    mime="application/pdf"
                )

# Footer
st.sidebar.markdown("**🌟 Key Features**")
st.sidebar.markdown("- AI-Powered Matching\n- Interactive Visualizations\n- Automated Scheduling\n- PDF/CSV Reports")
st.sidebar.markdown("⚓**About SkillMatch**\n\nSkillMatch automates the recruitment process from job analysis to interview scheduling.")
st.sidebar.markdown("**SkillMatch**👨‍💻 Created by Tech_Burner ©2025 All rights reserved")
