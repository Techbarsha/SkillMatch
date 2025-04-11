# 🤖 SkillMatch - AI-Powered Recruitment Assistant

SkillMatch is an intelligent recruitment platform built with multi-agent architecture that automates the entire hiring workflow—from job description analysis to resume matching, shortlisting, and interview scheduling. Designed during **Hack the Future: A Gen AI Sprint**, it leverages NLP, machine learning, and LLM APIs to reduce human bias and streamline the recruitment process.

![Project Screenshot](https://github.com/Techbarsha/SkillMatch/blob/main/job_descriptions/Screenshot%202025-04-11%20184149.png)

---

## 📌 Problem Statement

Recruiters often deal with:
- ⌛ High volume of resumes needing manual review  
- 🤷‍♀️ Subjectivity and potential bias in candidate screening  
- 🤯 Difficulty identifying top applicants efficiently  
- 📅 Manual and error-prone interview scheduling  

---

## ✅ Our Solution

SkillMatch automates recruitment using coordinated AI agents:

1. **JD Analyzer Agent**: Summarizes job descriptions via Gemini API.  
2. **CV Parser & Matcher Agent**: Parses resumes (PDF/DOCX) and scores them using TF-IDF and cosine similarity.  
3. **Shortlisting Module**: Filters applicants with score ≥ 80%.  
4. **Interview Scheduler Agent**: Auto-generates personalized interview emails.  
5. **Database Layer**: Uses SQLite for persistent resume, JD, and shortlist storage.

---

## 🚀 Features

- 📂 Upload and parse multiple CVs and JDs  
- 🧠 Gemini-powered JD summarization and email generation  
- 📊 Match scoring using TF-IDF & NLP  
- 📈 Visuals with Plotly: match distribution, shortlisted candidate views  
- 📑 Download PDF reports  
- 📬 Automated interview email drafts  
- 🧾 SQLite persistence for all data  
- 🖥️ Clean, interactive Streamlit interface

---

## 🛠 Technology Stack

### Frontend
- Streamlit  
- Plotly  
- FPDF2

### Backend & AI
- Python  
- scikit-learn (TF-IDF + cosine similarity)  
- spaCy (NLP parsing)  
- Gemini API (Job summarization, email drafting)  
- PyPDF2, docx2txt (File parsing)  
- SQLite (Data storage)

---

## 📁 Project Structure

```
skillmatch/
├── job_descriptions/       # Folder containing uploaded job descriptions
├── .env                    # Environment variables (e.g., API keys)
├── .gitignore              # Git ignored files (like .env, __pycache__, etc.)
├── LICENSE                 # MIT License file
├── README.md               # Project documentation
├── SECURITY.md             # Security policy
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python package dependencies
├── resumes.zip             # Compressed file containing resumes

```

---

## 🧑‍💻 Installation Guide

### ⚡ Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/your-username/skillmatch.git
cd skillmatch

# 2. Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the Streamlit app
streamlit run app.py
```

> ✅ Requires **Python 3.8+**

---

## 📘 How to Use

1. **Upload Job Description**  
   - Accepted formats: `.txt`, `.docx`, `.pdf`  
   - Summarized using Gemini API  

2. **Upload Resumes**  
   - Bulk upload of `.pdf` or `.docx` files  

3. **Matching Interface**  
   - View candidate match scores and keyword overlaps  
   - Scores computed using TF-IDF + cosine similarity  

4. **Shortlisting Panel**  
   - Automatically select candidates above threshold (80%)  

5. **Email Generator**  
   - Auto-generate personalized email invitations for shortlisted candidates  

6. **Reports**  
   - Download PDF summary of results  

---

## 🔌 Optional API Endpoints (CLI/Backend)

```http
POST /summarize-jd
POST /match
POST /schedule
```

Use these to access backend functionality programmatically.

---

## ⚙ How It Works

- **Job Description Summarization**: Gemini API + prompt engineering  
- **Resume Parsing**: PyPDF2 & docx2txt + spaCy entity recognition  
- **Scoring**: TF-IDF vectorization + cosine similarity  
- **Shortlisting**: Scores above a fixed threshold (default 80%)  
- **Scheduling**: Scheduler Agent drafts interview invitations  

---

## 🔮 Future Improvements

- 🔐 Admin login & role-based access  
- 💬 Candidate feedback loop  
- 🧠 LLM-based semantic JD-CV matching  
- 🌐 Integration with ATS & job boards  
- 📧 One-click email dispatch (SMTP support)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙋 Contact

📧 Feel free to reach out for feedback, collaboration, or hiring us 😄  
- 🌐 [Live App](https://skillmatch-tbeyh4okgjeeuq8uc5zced.streamlit.app)  
- 🎥 [Demo Video](https://youtu.be/g3hcY44_xL8)

---

> Built with ❤️ by **Team Tech_Burner**  
> Barsha Saha (Team Lead) • Rohan Ghosh
