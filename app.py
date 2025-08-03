
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from utils.skill_extractor import extract_skills_from_text
from PyPDF2 import PdfReader


# ========== Load Data ==========
jobs_df = pd.read_csv("data/jobs.csv")
courses_df = pd.read_csv("data/resources.csv")

# ========== Load AI Model and Encoder ==========
model = joblib.load("models/job_predictor.pkl")
role_encoder = joblib.load("models/role_encoder.pkl")

# ========== Streamlit Setup ==========
st.set_page_config(page_title="Skill Gap Analyzer", layout="wide")

st.title("🎓 Skill Gap Analyzer for Engineering Graduates")

tab1, tab2, tab3 = st.tabs(["Manual Skill Matcher", "AI Job Role Predictor", " Resume Skill Extractor"])

# ========== Tab 1 ==========
with tab1:
    st.header("Skill Gap Analysis (Manual)")
    user_skills = st.text_input("Enter your skills (comma-separated):", placeholder="e.g., python, sql, excel")
    selected_role = st.selectbox("Select a Job Role to Compare:", jobs_df["role"].unique())

    if st.button("Check Skill Match"):
        if user_skills.strip() == "":
            st.warning("Please enter skills.")
        else:
            user_skills_set = set(s.strip().lower() for s in user_skills.split(","))
            role_row = jobs_df[jobs_df["role"] == selected_role]

            if not role_row.empty:
                required_skills = set(skill.strip().lower() for skill in role_row.iloc[0]["skills"].split(","))
                matched_skills = user_skills_set.intersection(required_skills)
                missing_skills = required_skills.difference(user_skills_set)
                match_percent = (len(matched_skills) / len(required_skills)) * 100

                st.subheader("✅ Match Score")
                st.progress(match_percent / 100)
                st.write(f"🎯 Your Skill Match: **{match_percent:.2f}%**")

                # Pie Chart
                pie_fig = go.Figure(data=[go.Pie(
                    labels=["Matched Skills", "Missing Skills"],
                    values=[len(matched_skills), len(missing_skills)],
                    hole=0.5
                )])
                st.plotly_chart(pie_fig, use_container_width=True)

                st.subheader("❌ Missing Skills")
                for skill in missing_skills:
                    st.markdown(f"- {skill}")

                st.subheader("📘 Recommended Courses")
                for skill in missing_skills:
                    course_row = courses_df[courses_df["skill"].str.lower() == skill]
                    if not course_row.empty:
                        st.markdown(f"- **{skill.title()}**: [{course_row.iloc[0]['course']}]({course_row.iloc[0]['url']})")
                    else:
                        st.markdown(f"- No course found for: **{skill.title()}**")

# ========== Tab 2 ==========
with tab2:
    st.header(" AI-Based Job Role Recommender")
    ai_input = st.text_input("Enter your skills (comma-separated):", placeholder="e.g., html, css, javascript")

    if st.button("🔍 Show Role Suggestions"):
        if ai_input.strip() == "":
            st.warning("Please enter your skills.")
        else:
            user_skills_set = set(s.strip().lower() for s in ai_input.split(","))
            all_results = []

            for index, row in jobs_df.iterrows():
                role = row["role"].title()
                required_skills = set(s.strip().lower() for s in row["skills"].split(","))
                matched = user_skills_set.intersection(required_skills)
                missing = required_skills.difference(user_skills_set)
                score = len(matched) / len(required_skills) * 100

                if score > 0:
                    all_results.append({
                        "role": role,
                        "match": score,
                        "matched": matched,
                        "missing": missing
                    })

            if not all_results:
                st.warning("No suitable roles found.")
            else:
                all_results = sorted(all_results, key=lambda x: x["match"], reverse=True)
                
                # Bar chart for Top 5
                top5 = all_results[:5]
                bar_fig = go.Figure(data=go.Bar(
                    x=[r["role"] for r in top5],
                    y=[r["match"] for r in top5],
                    marker_color='teal'
                ))
                bar_fig.update_layout(title="Top Role Matches", yaxis_title="Match %")
                st.plotly_chart(bar_fig, use_container_width=True)

                for result in top5:
                    st.markdown(f"### 🎯 {result['role']}")
                    st.markdown(f"**✅ Match:** {result['match']:.1f}%")
                    st.markdown(f"**🧠 Matched Skills:** {', '.join(result['matched'])}")
                    st.markdown("**❌ Missing Skills & Courses:**")
                    for skill in result["missing"]:
                        course_row = courses_df[courses_df["skill"].str.lower() == skill]
                        if not course_row.empty:
                            st.markdown(f"- **{skill.title()}**: [{course_row.iloc[0]['course']}]({course_row.iloc[0]['url']})")
                        else:
                            st.markdown(f"- **{skill.title()}**: No course found")
                    st.markdown("---")


# ========== Tab 3 ==========
with tab3:
    st.header(" Upload Resume to Analyze Skills")
    uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])

    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            reader = PdfReader(uploaded_file)
            resume_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        else:
            resume_text = uploaded_file.read().decode("utf-8")

        all_skills = set()
        for skills in jobs_df["skills"]:
            all_skills.update([s.strip().lower() for s in skills.split(",")])

        extracted_skills = extract_skills_from_text(resume_text, list(all_skills))

        if extracted_skills:
            st.success("✅ Extracted Skills:")
            st.markdown(", ".join(extracted_skills))

            match_scores = []
            for index, row in jobs_df.iterrows():
                role = row["role"].title()
                required_skills = set(s.strip().lower() for s in row["skills"].split(","))
                matched = set(extracted_skills).intersection(required_skills)
                missing = required_skills.difference(extracted_skills)
                score = len(matched) / len(required_skills) * 100
                if score > 0:
                    match_scores.append({
                        "role": role,
                        "match": score,
                        "matched": matched,
                        "missing": missing
                    })

            match_scores = sorted(match_scores, key=lambda x: x["match"], reverse=True)

            # Display Skill Score
            top_score = match_scores[0]['match']
            st.subheader("📊 Job Readiness Score")
            st.metric("Skill Score", f"{top_score:.1f}%")

            for result in match_scores[:5]:
                st.markdown(f"### 🎯 {result['role']}")
                st.markdown(f"**✅ Match:** {result['match']:.1f}%")
                st.markdown(f"**🧠 Matched Skills:** {', '.join(result['matched'])}")
                st.markdown(f"**❌ Missing Skills & Courses:**")
                for skill in result["missing"]:
                    course_row = courses_df[courses_df["skill"].str.lower() == skill]
                    if not course_row.empty:
                        st.markdown(f"- **{skill.title()}**: [{course_row.iloc[0]['course']}]({course_row.iloc[0]['url']})")
                    else:
                        st.markdown(f"- **{skill.title()}**: No course found")

                # Tips + Feedback
                if result['match'] == 100:
                    st.success("🏅 Perfect Match! You’re fully job-ready!")
                    st.balloons()
                    if "ai" in result['role'].lower():
                        st.markdown("🎖 **Badge:** AI Ready")
                    elif "frontend" in result['role'].lower():
                        st.markdown("🎖 **Badge:** Frontend Pro")
                elif result['match'] >= 70:
                    st.info("💡 You're almost ready. Just polish a few missing skills.")
                else:
                    st.warning("🛠 Try adding more technical and relevant keywords to your resume.")

                st.markdown("---")
        else:
            st.warning("No skills found in your resume.")
