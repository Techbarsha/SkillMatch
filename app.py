import os
import streamlit as st
import pandas as pd
import numpy as np
import re
import google.generativeai as genai
from datetime import datetime
from PyPDF2 import PdfReader
import docx2txt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from fpdf import FPDF
import plotly.express as px
from datetime import timedelta

# Load environment variables
load_dotenv('.env')

# Initialize Gemini with error handling
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-pro')
except Exception as e:
    st.error(f"Failed to initialize Gemini API: {str(e)}")
    st.stop()

# Define process steps
steps = ["Job Analysis", "CV Processing", "Shortlisting", "Scheduling"]

# Initialize session state
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0

# Sidebar Navigation
with st.sidebar:
    st.markdown("## Process Steps👋")
    for i, step in enumerate(steps):
        if i <= st.session_state.current_step:
            st.markdown(f'<div class="step-indicator active-step">{step}</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="step-indicator">{step}</div>', 
                       unsafe_allow_html=True)
    
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
    .skill-gap {color: #dc3545; font-weight: bold;}
    .match-score {color: #28a745; font-weight: bold;}
    .candidate-card {padding: 15px; border: 1px solid #dee2e6; border-radius: 8px; margin: 10px 0;}
    table {width: 100%; border-collapse: collapse; margin: 10px 0;}
    th, td {border: 1px solid #dee2e6; padding: 8px; text-align: left;}
    th {background-color: #f8f9fa;}
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'cvs' not in st.session_state:
    st.session_state.cvs = []
if 'results' not in st.session_state:
    st.session_state.results = pd.DataFrame()

def extract_text(file):
    try:
        if file.type == "application/pdf":
            reader = PdfReader(file)
            text = " ".join([page.extract_text() for page in reader.pages])
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = docx2txt.process(file)
        else:
            raise ValueError("Unsupported file format")
        return clean_text(text)
    except Exception as e:
        st.error(f"Error extracting text: {str(e)}")
        return ""

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower()

def extract_skills(text):
    skill_keywords = [
        'python', 'java', 'c++', 'javascript', 'sql', 'html', 'css',
        'react', 'node.js', 'aws', 'docker', 'git', 'linux', 'debugging',
        'machine learning', 'data analysis', 'cloud computing', 'agile', 'scrum',
        'rest api', 'mongodb', 'postgresql', 'tensorflow', 'pytorch', 'keras'
    ]
    return list(set([skill for skill in skill_keywords if skill in text.lower()]))

def analyze_with_gemini(prompt):
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Error: {str(e)}"

def calculate_similarity(job_desc, cvs):
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        vectors = vectorizer.fit_transform([job_desc] + cvs)
        similarity = cosine_similarity(vectors[0:1], vectors[1:])[0]
        return similarity, vectorizer.get_feature_names_out()
    except Exception as e:
        st.error(f"Similarity calculation error: {str(e)}")
        return np.zeros(len(cvs)), []

def send_email(recipient, subject, body):
    if recipient == 'N/A':
        return False
    try:
        # Create message container
        msg = MIMEMultipart()
        msg['From'] = os.getenv("SMTP_USER")
        msg['To'] = recipient
        msg['Subject'] = subject
        
        # Attach body to email
        msg.attach(MIMEText(body, 'plain'))
        
        # Create SMTP connection
        with smtplib.SMTP(os.getenv("SMTP_SERVER"), int(os.getenv("SMTP_PORT"))) as server:
            server.starttls()
            server.login(os.getenv("SMTP_USER"), os.getenv("SMTP_PASSWORD"))
            server.send_message(msg)
        
        return True
    except Exception as e:
        st.error(f"Email sending failed: {str(e)}")
        return False

def extract_email(text):
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    match = re.search(email_pattern, text)
    return match.group(0) if match else 'N/A'

def plot_score_distribution(results):
    """Visualize the distribution of match scores"""
    try:
        # Convert scores to percentages for better readability
        results = results.copy()
        results['Match Score'] = results['Match Score'] * 100
        
        fig = px.histogram(
            results,
            x="Match Score",
            nbins=20,
            title="Distribution of Candidate Match Scores",
            labels={"Match Score": "Match Score (%)"},
            color_discrete_sequence=['#007bff']
        )
        fig.update_layout(
            xaxis_title="Match Score (%)",
            yaxis_title="Number of Candidates",
            bargap=0.1,
            xaxis=dict(tickformat=".0f")
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating score distribution plot: {str(e)}")

# Main Application
st.title("SkillMatch: Recruitment Platform")

# Step 1: Job Analysis
with st.expander("Step 1: Job Description Analysis", expanded=True):
    job_desc_file = st.file_uploader("Upload Job Description", type=["pdf", "docx"])
    
    if job_desc_file:
        try:
            job_desc = extract_text(job_desc_file)
            if not job_desc.strip():
                raise ValueError("Empty document or failed text extraction")
            
            st.session_state.job_desc = job_desc
            st.success("✅ Analysis Complete!")
            
            st.subheader("🔍 Job Requirements")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🧿 Key Skills**")
                skills = extract_skills(job_desc)
                if skills:
                    for skill in skills:
                        st.write(f"- {skill.title()}")
                else:
                    st.write("No specific skills detected")
                
            with col2:
                st.markdown("**📈 Experience Level**")
                exp_years = analyze_with_gemini(f"Determine required experience years from: {job_desc[:2000]}")
                st.write(exp_years.split('\n')[0])
            
            st.session_state.current_step = 1
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Step 2: CV Processing
if 'job_desc' in st.session_state:
    with st.expander("Step 2: CV Processing", expanded=st.session_state.current_step >=1):
        cv_files = st.file_uploader("Upload Candidate CVs", 
                                  type=["pdf", "docx"], 
                                  accept_multiple_files=True)
        
        if cv_files and not st.session_state.processed:
            with st.spinner("⌛ Analyzing CVs..."):
                try:
                    st.session_state.cvs = []
                    file_names = []
                    emails = []
                    skills_list = []
                    
                    for cv in cv_files:
                        text = extract_text(cv)
                        st.session_state.cvs.append(text)
                        file_names.append(cv.name)
                        emails.append(extract_email(text))
                        skills_list.append(", ".join(extract_skills(text)))
                    
                    similarity_scores, _ = calculate_similarity(st.session_state.job_desc, st.session_state.cvs)
                    
                    st.session_state.results = pd.DataFrame({
                        "Candidate": [f"CV {i+1}" for i in range(len(file_names))],
                        "Match Score": similarity_scores,
                        "File Name": file_names,
                        "Email": emails,
                        "Skills": skills_list
                    }).sort_values("Match Score", ascending=False)
                    
                    st.session_state.processed = True
                    st.session_state.current_step = 2
                except Exception as e:
                    st.error(f"CV Processing Error: {str(e)}")

# Step 3: Shortlisting
if 'results' in st.session_state:
    with st.expander("Step 3: Candidate Shortlisting", expanded=st.session_state.current_step >=2):
        st.subheader("AI-Powered Ranking📊")
        
        filtered = pd.DataFrame()
        try:
            if not st.session_state.results.empty:
                plot_score_distribution(st.session_state.results)
                
                threshold = st.slider("Match Threshold (%)", 0, 100, 60)
                filtered = st.session_state.results[st.session_state.results["Match Score"] >= (threshold/100)]
        except Exception as e:
            st.error(f"Data processing error: {str(e)}")

        if not filtered.empty:
            st.session_state.current_step = max(st.session_state.current_step, 3)
            
            st.dataframe(filtered[["Candidate", "Match Score", "File Name", "Email"]]
                        .style.format({"Match Score": "{:.2%}"}),
                        height=300)
            
            selected = st.selectbox("View Candidate Details", filtered["File Name"])
            if selected:
                try:
                    idx = st.session_state.results[st.session_state.results["File Name"] == selected].index[0]
                    candidate_text = st.session_state.cvs[idx]
                    
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.subheader("🌟 Candidate Profile")
                        qualifications_data = {
                            'Field': ['Name', 'Email', 'Skills', 'Experience'],
                            'Value': [
                                selected,
                                st.session_state.results.loc[idx, 'Email'],
                                st.session_state.results.loc[idx, 'Skills'],
                                analyze_with_gemini(f"Summarize experience from: {candidate_text[:3000]}")
                            ]
                        }
                        st.table(pd.DataFrame(qualifications_data))
                        
                    with col2:
                        st.subheader("🎯 Interview Preparation")
                        tab1, tab2 = st.tabs(["Questions", "Scoring Guide"])
                        
                        with tab1:
                            questions = analyze_with_gemini(f"Generate 5 technical and 3 behavioral questions for {selected}")
                            st.write(questions)
                        
                        with tab2:
                            st.write("""
                            **Scoring Rubric**
                            - Technical Skills (0-5)
                            - Cultural Fit (0-3)
                            - Communication (0-2)
                            """)
                            st.download_button("Download Scorecard", 
                                             data=pd.DataFrame(columns=['Criteria', 'Score']).to_csv(index=False),
                                             file_name="scorecard.csv")
                except Exception as e:
                    st.error(f"Error loading candidate details: {str(e)}")
        else:
            st.warning("No candidates match the current criteria")

# Step 4: Interview Scheduling
if st.session_state.current_step >= 3:
    with st.expander("Step 4: Interview Scheduling", expanded=True):
        st.subheader("📅 Schedule Interviews")
        
        try:
            if not filtered.empty:
                selected_candidates = st.multiselect(
                    "Select candidates to schedule:",
                    filtered["Candidate"],
                    format_func=lambda x: filtered[filtered["Candidate"] == x]["File Name"].values[0]
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input("First Interview Date")
                with col2:
                    start_time = st.time_input("First Interview Time")

                if st.button("📩 Schedule & Notify"):
                    if selected_candidates and start_date and start_time:
                        schedule_data = []
                        success_count = 0
                        current_time = datetime.combine(start_date, start_time)
                        
                        for i, candidate in enumerate(selected_candidates):
                            candidate_info = filtered[filtered["Candidate"] == candidate].iloc[0]
                            interview_time = current_time + timedelta(minutes=30*i)
                            
                            # Send email
                            email_body = f"""Dear Candidate,

Your application has been shortlisted! Please join your interview at:
Date: {interview_time.strftime('%d %B %Y')}
Time: {interview_time.strftime('%H:%M')} 
Duration: 45 minutes

Meeting Link: {os.getenv("MEETING_URL", "https://meet.example.com/your-room")}

Please confirm your availability by replying to this email.

Best regards,
{os.getenv("HR_NAME", "HR Team")}"""
                            
                            email_sent = send_email(
                                candidate_info['Email'],
                                f"Interview Invitation: {candidate_info['File Name']}",
                                email_body
                            )
                            
                            schedule_data.append({
                                "Candidate": candidate_info['File Name'],
                                "Email": candidate_info['Email'],
                                "Date": interview_time.date(),
                                "Time": interview_time.time(),
                                "Email Sent": "✅" if email_sent else "❌"
                            })
                            
                            if email_sent:
                                success_count += 1

                        st.success(f"Scheduled {len(selected_candidates)} interviews | Emails sent: {success_count}")
                        st.dataframe(pd.DataFrame(schedule_data))
                        
                        # Export schedule
                        csv = pd.DataFrame(schedule_data).to_csv(index=False)
                        st.download_button(
                            label="Download Schedule",
                            data=csv,
                            file_name="interview_schedule.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("Please select candidates and specify date/time")
            else:
                st.warning("No candidates available for scheduling")
        except Exception as e:
            st.error(f"Scheduling error: {str(e)}")

st.sidebar.markdown("---")
st.sidebar.markdown("⚓**About SkillMatch**  \nAI-powered recruitment automation system")
st.sidebar.markdown("**v2.1** | 🚀 Enhanced Features:\n- Smart Scheduling\n- Email Automation\n- AI Assessments")