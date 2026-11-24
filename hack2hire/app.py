import copy
import json
import os
import random
import time

import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

import core

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

st.set_page_config(page_title="AI Mock Interview", page_icon="🎤", layout="wide")

DEFAULTS = dict(stage="setup", profile=None, jd=None, resume_text="", history=[], current=None,
                idx=0, level=0, cfg={}, last=None, summary=None, qtypes=[], followup_ctx=None)


def init_state():
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(k, copy.deepcopy(v))


init_state()
S = st.session_state


# ---------------- API key / client ----------------
def get_api_key():
    try:
        key = st.secrets["GROQ_API_KEY"]
    except Exception:
        key = None
    return key or os.getenv("GROQ_API_KEY")


@st.cache_resource
def client_for(key):
    return core.get_client(key)


api_key = get_api_key()
if not api_key:
    st.title("🎤 AI Mock Interview")
    st.error("GROQ_API_KEY not found.")
    st.markdown(
        "This app runs 100% on **free** services. Get a free key at "
        "[console.groq.com/keys](https://console.groq.com/keys) (no card needed), then either put it in a "
        "`.env` file as `GROQ_API_KEY=gsk_...` or in `.streamlit/secrets.toml`, and restart.")
    st.stop()
client = client_for(api_key)


# ---------------- helpers ----------------
def reset_interview(keep_profile):
    keep = {k: S[k] for k in ("profile", "jd", "resume_text")} if keep_profile else {}
    for k in list(S.keys()):
        if k.startswith(("ans_", "stt_")):
            del S[k]
    for k, v in DEFAULTS.items():
        S[k] = copy.deepcopy(v)
    S.update(keep)
    S.stage = "review" if keep_profile and S.profile else "setup"
    st.rerun()


def timer_html(question, remaining):
    # question text is LLM output: escape every HTML-significant char so it can only ever be a JS string
    q = json.dumps(question).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return f"""
<div style="font-family:sans-serif;display:flex;gap:14px;align-items:center">
  <button id="spk" style="padding:6px 14px;border-radius:8px;border:1px solid #555;background:#262730;
          color:#fff;cursor:pointer">🔊 Hear question</button>
  <span id="t" style="font-size:18px;font-weight:600"></span>
</div>
<script>
const Q = {q}; let rem = {int(remaining)};
const el = document.getElementById('t');
function fmt(s){{const a=Math.abs(s);return (s<0?'+':'')+Math.floor(a/60)+':'+String(a%60).padStart(2,'0');}}
function tick(){{el.textContent=(rem>=0?'⏱ ':'⚠️ Overtime ')+fmt(rem);
  el.style.color=rem<0?'#e74c3c':(rem<20?'#f5a623':'#2ecc71');rem--;}}
tick(); setInterval(tick,1000);
document.getElementById('spk').onclick=()=>{{const s=window.speechSynthesis;s.cancel();
  const u=new SpeechSynthesisUtterance(Q);u.rate=0.95;s.speak(u);}};
</script>"""


def render_html(html, height):
    if hasattr(st, "iframe"):  # newer Streamlit (components.v1.html is deprecated there)
        st.iframe(html, height=height)
    else:
        components.html(html, height=height)


def score_color(x):
    return "#2ecc71" if x >= 7 else "#f5a623" if x >= 4 else "#e74c3c"


# ---------------- sidebar ----------------
locked = S.stage in ("interview", "feedback")
st.sidebar.header("⚙️ Interview settings")
mode = st.sidebar.selectbox("Interview type", list(core.QTYPES), disabled=locked, key="w_mode")
experience = st.sidebar.selectbox("Your level", list(core.EXPERIENCE), disabled=locked, key="w_exp")
total_q = st.sidebar.slider("Number of questions", 3, 10, 6, disabled=locked, key="w_total")
max_time = st.sidebar.slider("Time per question (sec)", 60, 300, 120, step=30, disabled=locked, key="w_time")
use_followups = st.sidebar.toggle("Adaptive follow-up questions", True, disabled=locked, key="w_follow")
use_voice = st.sidebar.toggle("Voice answers (free Whisper)", True, disabled=locked, key="w_voice")
st.sidebar.caption("💸 100% free stack: Groq free tier + browser text-to-speech. No card needed. "
                   "If you hit the free rate limit the app falls back to a smaller free model.")

if S.stage in ("interview", "feedback", "report") and S.history:
    st.sidebar.divider()
    st.sidebar.metric("Running average", f"{core.mean(h['final'] for h in S.history):.1f}/10")

st.title("🎤 AI Mock Interview Platform")

# =====================================================================
# STAGE 1: setup
# =====================================================================
if S.stage == "setup":
    st.write("Upload your resume, optionally paste a job description, and practice a realistic interview "
             "that adapts to your answers.")
    resume_file = st.file_uploader("Resume (PDF)", type="pdf")
    jd_text = st.text_area("Job description (optional)", height=160)

    if st.button("Analyze resume →", type="primary"):
        if not resume_file:
            st.warning("Please upload your resume PDF first.")
        else:
            try:
                text = core.extract_pdf_text(resume_file)
            except Exception as e:
                st.error(f"Couldn't read that PDF ({e}). Try another file.")
                st.stop()
            if len(text) < 50:
                st.error("No text found in this PDF. It may be a scanned image; please upload a text-based PDF.")
                st.stop()
            try:
                with st.spinner("Analyzing resume and job description…"):
                    S.profile = core.parse_resume(client, text)
                    S.jd = core.parse_jd(client, jd_text) if jd_text.strip() else None
            except RuntimeError as e:
                st.error(str(e))
                st.stop()
            S.resume_text = text
            S.stage = "review"
            st.rerun()

# =====================================================================
# STAGE 2: review detected profile
# =====================================================================
elif S.stage == "review":
    st.subheader("Check what the AI understood")
    p = S.profile
    c1, c2 = st.columns(2)
    p["target_role"] = c1.text_input("Target role", p["target_role"])
    skills_txt = c2.text_area("Skills (comma separated)", ", ".join(p["skills"]), height=110)
    p["skills"] = [s.strip() for s in skills_txt.split(",") if s.strip()]
    if p["projects"]:
        with st.expander("Detected projects"):
            for pr in p["projects"]:
                st.write(f"- {pr}")

    if S.jd and S.jd["required_skills"]:
        gap = core.skill_gap(p["skills"], S.jd["required_skills"])
        st.markdown(f"**Skill match with _{S.jd['role']}_: {gap['pct']}%**")
        g1, g2 = st.columns(2)
        g1.success("✅ Matched: " + (", ".join(gap["matched"]) or "none"))
        g2.warning("⚠️ Missing: " + (", ".join(gap["missing"]) or "none"))

    b1, b2 = st.columns([1, 5])
    if b1.button("Start interview", type="primary"):
        S.cfg = dict(mode=mode, experience=experience, total=total_q, max_time=max_time,
                     followups=use_followups, voice=use_voice)
        S.level = core.EXPERIENCE[experience]
        types = list(core.QTYPES[mode])
        random.shuffle(types)
        S.qtypes = types
        S.stage = "interview"
        st.rerun()
    if b2.button("← Upload a different resume"):
        reset_interview(keep_profile=False)

# =====================================================================
# STAGE 3: answering a question
# =====================================================================
elif S.stage == "interview":
    cfg = S.cfg
    if S.current is None:  # generate ONCE per question (not on every rerun)
        try:
            with st.spinner("Preparing your next question…"):
                fu = S.followup_ctx
                qtype = "follow-up" if fu else S.qtypes[S.idx % len(S.qtypes)]
                text = core.generate_question(client, S.profile, S.jd, S.level, qtype, cfg["experience"],
                                              [h["question"] for h in S.history], followup=fu)
        except RuntimeError as e:
            st.error(str(e))
            if st.button("Retry"):
                st.rerun()
            st.stop()
        S.followup_ctx = None
        S.current = dict(text=text, qtype=qtype, level=core.LEVELS[S.level], followup=qtype == "follow-up",
                         started=time.time())
    cur = S.current

    st.progress(S.idx / cfg["total"], text=f"Question {S.idx + 1} of {cfg['total']}")
    st.caption(f"🏷️ {cur['qtype'].title()}  •  🎚️ {cur['level']}")
    st.markdown(f"### {cur['text']}")
    remaining = cfg["max_time"] - (time.time() - cur["started"])
    render_html(timer_html(cur["text"], remaining), 48)

    ans_key = f"ans_{S.idx}"
    col_text, col_voice = st.columns([3, 2])

    with col_voice:  # handled first so a transcript can be written into the text box before it renders
        if cfg["voice"]:
            st.write("🎤 Or speak your answer")
            audio = st.audio_input("Record", key=f"aud_{S.idx}")
            if audio is not None:
                data = audio.getvalue()
                if S.get(f"stt_{S.idx}") != hash(data):
                    try:
                        with st.spinner("Transcribing…"):
                            spoken = core.transcribe(client, data)
                        S[f"stt_{S.idx}"] = hash(data)
                        prev = S.get(ans_key, "").strip()
                        S[ans_key] = (prev + " " + spoken).strip()
                        st.rerun()
                    except Exception as e:
                        S[f"stt_{S.idx}"] = hash(data)
                        st.warning(f"Couldn't transcribe the recording ({e}). Please type your answer.")

    with col_text:
        st.text_area("Your answer", height=190, key=ans_key,
                     placeholder="Type, or record your voice and edit the transcript here…")

    b1, b2 = st.columns([1, 1])
    if b1.button("Submit answer ✅", type="primary", key="submit_btn"):
        answer = S.get(ans_key, "")
        elapsed = time.time() - cur["started"]
        with st.spinner("Evaluating your answer…"):
            res = core.evaluate_answer(client, cur["text"], answer, S.profile["target_role"],
                                       S.profile["skills"], elapsed, cfg["max_time"], cfg["experience"])
        entry = dict(question=cur["text"], qtype=cur["qtype"], level=cur["level"], answer=answer.strip(),
                     elapsed=round(elapsed), followup=cur["followup"], **res)
        S.history.append(entry)
        S.last = entry
        S.level = core.next_level(S.level, res["final"])
        if cfg["followups"] and not cur["followup"] and 6 <= res["final"] < 9.5 and S.idx + 1 < cfg["total"]:
            S.followup_ctx = {"question": cur["text"], "answer": answer}
        S.stage = "feedback"
        st.rerun()
    if b2.button("End interview now", key="end_btn"):
        S.stage = "report"
        st.rerun()

# =====================================================================
# STAGE 4: feedback on the last answer
# =====================================================================
elif S.stage == "feedback":
    e, cfg = S.last, S.cfg
    st.progress((S.idx + 1) / cfg["total"], text=f"Question {S.idx + 1} of {cfg['total']} done")
    st.markdown(f"**{e['question']}**")
    st.markdown(f"<h2 style='color:{score_color(e['final'])};margin:0'>{e['final']:.1f} / 10</h2>",
                unsafe_allow_html=True)
    cols = st.columns(len(core.DIMENSIONS) + 1)
    for c, d in zip(cols, core.DIMENSIONS):
        c.metric(d.title(), f"{e['scores'][d]:.1f}")
    over = e["elapsed"] - cfg["max_time"]
    cols[-1].metric("Time", f"{e['elapsed']}s", delta=f"-{e['penalty']:.1f} pts overtime" if over > 0 else "on time",
                    delta_color="inverse" if over > 0 else "off")
    if e.get("fallback"):
        st.warning(e["feedback"])
    else:
        st.info(e["feedback"])
    if e["strength"]:
        st.success("👍 " + e["strength"])
    if e["improve"]:
        st.warning("🔧 " + e["improve"])
    if e["ideal"]:
        with st.expander("💡 See a model answer"):
            st.write(e["ideal"])
    if S.followup_ctx:
        st.caption("🔍 Nice answer. The interviewer wants to dig deeper, so the next question is a follow-up.")

    last = S.idx + 1 >= cfg["total"]
    if st.button("Finish & see report 📊" if last else "Next question →", type="primary", key="next_btn"):
        if last:
            S.stage = "report"
        else:
            S.idx += 1
            S.current = None
            S.stage = "interview"
        st.rerun()

# =====================================================================
# STAGE 5: final report
# =====================================================================
elif S.stage == "report":
    if not S.history:
        st.warning("No answers were submitted.")
        if st.button("Start over"):
            reset_interview(keep_profile=True)
        st.stop()

    if S.summary is None:
        with st.spinner("Writing your mentor feedback…"):
            S.summary = core.build_summary(client, S.history, S.profile)
    summary = S.summary
    score = core.overall_score(S.history)
    label = core.category_label(score)

    st.header(f"Final Readiness Score: {score}/100 ({label})")
    if summary["verdict"]:
        st.write(summary["verdict"])
    st.success("🚀 Ready to apply!" if score > 65 else "📈 Good start. Keep practicing!")

    ch1, ch2 = st.columns(2)
    dims = core.dimension_stats(S.history)
    if dims:
        labels = [d.title() for d in dims]
        fig = go.Figure(go.Scatterpolar(r=list(dims.values()) + [list(dims.values())[0]],
                                        theta=labels + [labels[0]], fill="toself", name="Skills"))
        fig.update_layout(template="plotly_dark", height=340, title="Answer quality",
                          polar=dict(radialaxis=dict(range=[0, 10])), showlegend=False,
                          margin=dict(l=40, r=40, t=60, b=30))
        ch1.plotly_chart(fig, width="stretch")
    bar = go.Figure(go.Bar(x=[f"Q{i + 1}" for i in range(len(S.history))],
                           y=[h["final"] for h in S.history],
                           marker_color=[score_color(h["final"]) for h in S.history]))
    bar.update_layout(template="plotly_dark", height=340, title="Score per question",
                      yaxis=dict(range=[0, 10]), margin=dict(l=20, r=20, t=60, b=30))
    ch2.plotly_chart(bar, width="stretch")

    cats = core.category_stats(S.history)
    if len(cats) > 1:
        st.markdown("**By question type:** " + "  |  ".join(f"{k.title()}: **{v:.1f}**" for k, v in cats.items()))

    r1, r2, r3 = st.columns(3)
    with r1:
        st.subheader("💪 Strengths")
        for x in summary["strengths"] or ["-"]:
            st.write(f"- {x}")
    with r2:
        st.subheader("🔧 To improve")
        for x in summary["weaknesses"] or ["-"]:
            st.write(f"- {x}")
    with r3:
        st.subheader("🎯 Action tips")
        for x in summary["tips"] or ["-"]:
            st.write(f"- {x}")

    gap = None
    if S.jd and S.jd["required_skills"]:
        gap = core.skill_gap(S.profile["skills"], S.jd["required_skills"])
        st.subheader(f"🧩 Skill match for {S.jd['role']}: {gap['pct']}%")
        if gap["missing"]:
            st.write("Skills to learn next: " + ", ".join(gap["missing"]))

    st.subheader("📝 Question by question")
    for i, h in enumerate(S.history, 1):
        with st.expander(f"Q{i} • {h['final']:.1f}/10 • {h['question'][:90]}"):
            st.write(f"**Your answer:** {h['answer'] or '_(none)_'}")
            st.write(f"**Feedback:** {h['feedback']}")
            if h["ideal"]:
                st.write(f"**Model answer:** {h['ideal']}")

    st.download_button("⬇️ Download report (Markdown)",
                       core.report_markdown(S.profile, S.jd, S.history, summary, gap),
                       file_name="mock_interview_report.md", mime="text/markdown")
    a1, a2, _ = st.columns([1, 1, 4])
    if a1.button("🔁 Practice again", type="primary"):
        reset_interview(keep_profile=True)
    if a2.button("📄 New resume"):
        reset_interview(keep_profile=False)
