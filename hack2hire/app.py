import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from PyPDF2 import PdfReader
import time
import os
from dotenv import load_dotenv
import json
import random
import re

load_dotenv()
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    temperature=0.4
)

# Helpers
def parse_resume(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = "".join(page.extract_text() or "" for page in reader.pages)
    prompt = PromptTemplate.from_template(
        """Extract key information from this resume as strict JSON only. 
        Infer 'target_role' from title, summary, objective or most prominent role (e.g. 'Data Analyst', 'Data Scientist', 'Python Developer', 'Fresher').
        Do NOT assume 'Software Engineer' unless clearly stated.
        Output ONLY valid JSON:
        {"skills": ["skill1", "skill2"], "experience_years": integer or 0, "projects": ["project1"], "target_role": "string"}
        Resume text: {text}"""
    )
    response = llm.invoke(prompt.format(text=text))
    raw = response.content.strip()
    try:
        data = json.loads(raw)
        return {
            "skills": data.get("skills", []),
            "experience_years": data.get("experience_years", 0),
            "projects": data.get("projects", []),
            "target_role": data.get("target_role", "Unknown")
        }
    except:
        return {"skills": [], "experience_years": 0, "projects": [], "target_role": "Unknown"}

def parse_jd(jd_text):
    prompt = PromptTemplate.from_template(
        "Extract from job description as strict JSON only. Output ONLY valid JSON: "
        "{\"required_skills\": [\"skill1\", \"skill2\"], \"role\": \"string\"}. JD: {text}"
    )
    response = llm.invoke(prompt.format(text=jd_text))
    raw = response.content.strip()
    try:
        data = json.loads(raw)
        return {
            "required_skills": data.get("required_skills", []),
            "role": data.get("role", "General Tech Role")
        }
    except:
        return {"required_skills": [], "role": "General Tech Role"}

def generate_question(resume_data, jd_data, difficulty, qtype):
    levels = ["very easy (beginner level)", "easy", "medium"]
    prompt = PromptTemplate.from_template(
        """Generate one simple {level} {qtype} interview question for a beginner in role: {resume_role}.
        Focus on resume skills: {skills}.
        Lightly consider JD role {jd_role} and skills {req_skills}.
        Keep it simple — conceptual, basic explanation, behavioral, or project-related.
        Output only the question text."""
    )
    return llm.invoke(prompt.format(
        level=levels[difficulty],
        qtype=qtype,
        resume_role=resume_data.get("target_role", "Unknown"),
        jd_role=jd_data.get("role", "Unknown"),
        skills=", ".join(resume_data.get("skills", [])),
        req_skills=", ".join(jd_data.get("required_skills", []))
    )).content.strip()

def evaluate_answer(question, answer, jd_data, elapsed, max_time=120):
    overtime = max(0, elapsed - max_time) / 10
    prompt = PromptTemplate.from_template(
        """Score this answer 0–10 for beginner level (average of accuracy, clarity, depth, relevance, time efficiency — deduct for overtime).
        Role: {role} | Skills: {skills}
        Question: {question}
        Answer: {answer}
        Return ONLY valid JSON: {"avg_score": number 0 to 10, "feedback": "one short sentence"}"""
    )
    resp = llm.invoke(prompt.format(
        question=question,
        answer=answer[:1200],
        role=jd_data.get("role", "Unknown"),
        skills=", ".join(jd_data.get("required_skills", []))
    )).content.strip()

    cleaned = re.sub(r'^```json\s*|\s*```$', '', resp.strip(), flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(r'^\s*json\s*', '', cleaned, flags=re.IGNORECASE).strip()

    try:
        data = json.loads(cleaned)
        score = float(data.get("avg_score", 5.0)) - overtime
        return max(0, min(10, score)), data.get("feedback", "No feedback")
    except:
        simple_score = 4.0 + min(3.0, len(answer) / 100) - overtime
        return max(0, min(10, simple_score)), "Evaluation issue — partial score"

# App
st.title("AI Mock Interview Platform")

if "stage" not in st.session_state:
    st.session_state.update({
        "stage": "input",
        "scores": [],
        "feedbacks": [],
        "questions": [],
        "difficulty": 0,
        "question_index": 0
    })

if st.session_state.stage == "input":
    resume_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
    jd_text = st.text_area("Job Description (optional)")
    
    if st.button("Start Interview") and resume_file:
        with st.spinner("Analyzing..."):
            resume_data = parse_resume(resume_file)
            jd_data = parse_jd(jd_text) if jd_text else {"required_skills": [], "role": "General Tech Role"}
            st.session_state.resume_data = resume_data
            st.session_state.jd_data = jd_data
        
        st.session_state.stage = "interview"
        st.rerun()

elif st.session_state.stage == "interview":
    if st.session_state.question_index < 7:
        qtypes = ["technical", "conceptual", "behavioral", "project-based"]
        qtype = random.choice(qtypes)
        question = generate_question(
            st.session_state.resume_data,
            st.session_state.jd_data,
            st.session_state.difficulty,
            qtype
        )
        st.session_state.questions.append(question)

        st.subheader(f"Question {st.session_state.question_index + 1}")
        st.write(question)

        if st.button("Hear Question"):
            escaped = question.replace("`", "\\`").replace('"', '\\"').replace("\n", " ")
            tts_script = f"""
            <script>
            const synth = window.speechSynthesis;
            const utterance = new SpeechSynthesisUtterance(`{escaped}`);
            utterance.rate = 0.95;
            utterance.pitch = 1.0;
            utterance.volume = 1.0;
            synth.cancel();
            synth.speak(utterance);
            </script>
            """
            st.markdown(tts_script, unsafe_allow_html=True)

        st.progress((st.session_state.question_index + 1) / 7.0)

        start = time.time()
        text_answer = st.text_area("Your Answer:", height=140, key=f"text_{st.session_state.question_index}")

        if st.button("Submit"):
            elapsed = time.time() - start
            score, fb = evaluate_answer(question, text_answer, st.session_state.jd_data, elapsed)
            st.session_state.scores.append(score)
            st.session_state.feedbacks.append(fb)

            if score >= 7:
                st.success(f"Score: {score:.1f}/10 | {fb}")
            elif score >= 4:
                st.info(f"Score: {score:.1f}/10 | {fb}")
            else:
                st.warning(f"Score: {score:.1f}/10 | {fb}")

            if score > 7:
                st.session_state.difficulty = min(2, st.session_state.difficulty + 1)
            elif score < 4:
                st.session_state.difficulty = max(0, st.session_state.difficulty - 1)

            if len(st.session_state.scores) >= 3 and sum(st.session_state.scores)/len(st.session_state.scores) < 3.0:
                st.session_state.stage = "end"
                st.rerun()

            st.session_state.question_index += 1
            st.rerun()

    else:
        st.session_state.stage = "end"
        st.rerun()

elif st.session_state.stage == "end":
    if st.session_state.scores:
        avg = sum(st.session_state.scores) / len(st.session_state.scores) * 10
        category = "Strong" if avg > 75 else "Average" if avg > 45 else "Needs Improvement"
        st.header(f"Final Readiness Score: {avg:.1f}/100 ({category})")

        st.subheader("Performance Breakdown")
        for i, (s, f) in enumerate(zip(st.session_state.scores, st.session_state.feedbacks)):
            st.write(f"Question {i+1}: {s:.1f}/10 → {f}")

        summary_prompt = PromptTemplate.from_template(
            "Act as a friendly mentor. Summarize strengths, weaknesses, and give 3 simple, actionable tips for improvement based on these feedbacks: {fb}"
        )
        summary = llm.invoke(summary_prompt.format(fb="; ".join(st.session_state.feedbacks))).content
        st.subheader("Feedback & Recommendations")
        st.write(summary)

        readiness = "Ready to apply" if avg > 65 else "Good start — keep practicing"
        st.success(f"Hiring Readiness: {readiness}")
    else:
        st.warning("Interview ended early due to low performance.")

    if st.button("Start New Interview"):
        st.session_state.clear()
        st.rerun()
