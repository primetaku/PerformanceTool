import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from fpdf import FPDF
import io
import tempfile
import os
import unicodedata
import re
from collections import Counter

st.set_page_config(page_title="ML Performance Planning Tool", layout="wide")

def extract_keywords_from_dev_areas(dev_area_list):
    keywords = []
    for area in dev_area_list:
        if not isinstance(area, str):
            continue
        # Split by number+dot, number+paren, semicolon, comma, or newline
        items = re.split(r'\d+\.\s*|\d+\)\s*|;|,|\n', area)
        for item in items:
            cleaned = item.strip().lower()
            # Remove very short words or too generic ones (like "the", "and")
            if cleaned and len(cleaned) > 2 and cleaned not in ['and', 'for', 'with', 'the', 'act']:
                keywords.append(cleaned)
    return Counter(keywords)

def clean_unicode(text):
    if pd.isnull(text):
        return ""
    text = str(text)
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    return text


def clean_dev_area(area):
    if pd.isnull(area):
        return ""
    area_str = str(area).strip()
    if area_str.lower() in ["nan", "none", ""]:
        return ""
    return area_str


def map_question_to_category(question):
    q = str(question).lower()
    if 'leader' in q:
        return 'Leadership'
    elif 'engage' in q or 'engagement' in q:
        return 'Engagement'
    elif 'team' in q:
        return 'Teamwork'
    elif 'communication' in q or 'communicat' in q:
        return 'Communication'
    elif 'innovation' in q or 'innovate' in q:
        return 'Innovation'
    elif 'motivation' in q or 'motivate' in q:
        return 'Motivation'
    elif 'customer' in q or 'client' in q:
        return 'Customer Focus'
    elif 'learning' in q or 'development' in q or 'grow' in q:
        return 'Growth/Development'
    elif 'planning' in q or 'goal' in q:
        return 'Planning'
    else:
        return 'Other'


@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    employee_columns = [col for col in df.columns if not col.startswith('Unnamed') and col != 'Timestamp']

    # Find the row labeled 'Development Areas' or similar (case insensitive)
    dev_row_idx = df['Timestamp'].str.lower().str.strip().eq("development areas")
    if dev_row_idx.any():
        dev_areas = df.loc[dev_row_idx, employee_columns].iloc[0].tolist()
    else:
        dev_areas = ["" for _ in employee_columns]

    # Find employee names (e.g., from the 1st data row after headers)
    employee_names = df.loc[0, employee_columns].tolist()

    # Get the actual questions
    # (skip headers and dev areas row)
    # Find all question rows as those not header and not 'Development Areas'
    is_question_row = (~df['Timestamp'].str.lower().str.strip().eq("development areas")) & (df.index > 2)
    questions = df['Timestamp'][is_question_row].reset_index(drop=True)

    # Get answers (per employee) for the question rows
    question_indices = df.index[is_question_row]
    employee_data = []
    for col, name, dev_area in zip(employee_columns, employee_names, dev_areas):
        responses = df.loc[question_indices, col].reset_index(drop=True).tolist()
        employee_dict = {
            "name": name,
            "responses": responses,
            "development_area": dev_area
        }
        employee_data.append(employee_dict)
    employee_df = pd.DataFrame({
        "name": [e["name"] for e in employee_data],
        "development_area": [e["development_area"] for e in employee_data],
        **{f"Q{i+1}": [e["responses"][i] for e in employee_data] for i in range(len(questions))}
    })
    return employee_df, questions, [map_question_to_category(q) for q in questions]


def score_extract(ans):
    if pd.isnull(ans): return None
    if str(ans).strip().lower() in ["nan", "none", ""]:
        return None
    for s in ['6', '5', '4', '3', '2', '1']:
        if str(ans).strip().startswith(s): return int(s)
    return None


def get_recommendation(dev_area):
    dev_area_clean = clean_dev_area(dev_area)
    if dev_area_clean == "":
        return "No development area provided."
    if 'leader' in dev_area_clean.lower():
        return "Consider targeted leadership training and mentorship opportunities."
    if 'loyal' in dev_area_clean.lower():
        return "Support with recognition programs and clear progression paths."
    return f"Suggested focus: {dev_area_clean}"


def plotly_radar(category_scores, title="Competency Radar"):
    categories = list(category_scores.keys())
    values = list(category_scores.values())
    values += values[:1]
    categories += categories[:1]
    fig = go.Figure(
        data=[
            go.Scatterpolar(r=values, theta=categories, fill='toself', name='Scores')
        ]
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 6])),
        showlegend=False,
        title=title,
        autosize=True,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig


def plotly_hist(scores, title="Score Distribution"):
    scores = [s for s in scores if s is not None]
    fig = px.histogram(scores, nbins=6, range_x=[0.5, 6.5], labels={'value': 'Score'}, title=title)
    fig.update_xaxes(dtick=1)
    fig.update_layout(bargap=0.2, autosize=True)
    return fig


def plotly_pie(labels, sizes, title="Development Areas"):
    fig = px.pie(values=sizes, names=labels, title=title, hole=0.3)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig


def plotly_line(scores, questions, title="Survey Scores (Answered Only)"):
    xs, ys, qtext = [], [], []
    for i, s in enumerate(scores):
        if s is not None:
            xs.append(i + 1)
            ys.append(s)
            qtext.append(questions[i])
    fig = px.line(x=xs, y=ys, markers=True, title=title, labels={'x': 'Question #', 'y': 'Score'})
    fig.update_traces(hovertemplate='Q%{x}: %{y}')
    fig.update_xaxes(dtick=1)
    return fig


def fig_to_bytes(fig, dpi=200):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def get_category_scores(responses, categories):
    cat_to_scores = {}
    for ans, cat in zip(responses, categories):
        s = score_extract(ans)
        if s is not None:
            cat_to_scores.setdefault(cat, []).append(s)
    return {cat: np.mean(scores) if scores else 0 for cat, scores in cat_to_scores.items()}


def export_pdf_emp(emp_row, questions, question_categories):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, clean_unicode(f"Performance Report: {emp_row['name']}"), ln=1)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Development Area:", ln=1)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 8, clean_unicode(clean_dev_area(emp_row['development_area'])))
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Recommendation:", ln=1)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 8, clean_unicode(get_recommendation(emp_row['development_area'])))
    # Charts at the top
    scores = [score_extract(ans) for ans in emp_row.drop(['name', 'development_area'])]
    if any(s is not None for s in scores):
        fig, ax = plt.subplots()
        xs, ys = [], []
        for i, s in enumerate(scores):
            if s is not None:
                xs.append(i + 1)
                ys.append(s)
        ax.plot(xs, ys, marker='o', linestyle='-', label="Scores")
        ax.set_xlabel("Question #")
        ax.set_ylabel("Score")
        ax.set_title("Survey Scores (Answered Only)")
        img = fig_to_bytes(fig, dpi=200)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            tmpfile.write(img)
            tmpfile.flush()
            pdf.image(tmpfile.name, w=160)
        os.unlink(tmpfile.name)
    responses = list(emp_row.drop(['name', 'development_area']))
    category_scores = get_category_scores(responses, question_categories)
    if category_scores:
        cats = list(category_scores.keys())
        vals = list(category_scores.values())
        vals += vals[:1]
        angles = np.linspace(0, 2 * np.pi, len(cats) + 1, endpoint=True)
        fig, ax = plt.subplots(subplot_kw=dict(polar=True))
        ax.plot(angles, vals, linewidth=2, linestyle='solid', marker='o')
        ax.fill(angles, vals, alpha=0.25)
        ax.set_thetagrids(np.degrees(angles[:-1]), cats)
        ax.set_title("Competency Radar")
        img = fig_to_bytes(fig, dpi=200)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            tmpfile.write(img)
            tmpfile.flush()
            pdf.image(tmpfile.name, w=120)
        os.unlink(tmpfile.name)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Survey Answers (Answered Only):", ln=1)
    pdf.set_font("Arial", '', 12)
    for i, ans in enumerate(emp_row.drop(['name', 'development_area'])):
        if pd.isnull(ans) or ans == "" or str(ans).strip().lower() in ["nan", "none"]:
            continue
        q_txt = str(questions[i])
        pdf.set_font("Arial", 'B', 12)
        pdf.multi_cell(0, 8, clean_unicode(q_txt), 0)
        pdf.set_font("Arial", '', 12)
        pdf.multi_cell(0, 8, clean_unicode(f"Answer: {ans}"), 0)
    return pdf.output(dest='S').encode('latin-1')


def export_pdf_exec(employee_df, questions, question_categories):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Executive Performance Report", ln=1)
    pdf.set_font("Arial", '', 12)
    # Pie chart of dev areas
    dev_table = employee_df[['name', 'development_area']].copy()
    dev_table['development_area'] = dev_table['development_area'].apply(clean_dev_area)
    dev_table = dev_table[dev_table['development_area'] != ""]
    dev_counts = dev_table['development_area'].value_counts()
    if len(dev_counts) > 0:
        fig, ax = plt.subplots()
        ax.pie(dev_counts.values, labels=dev_counts.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        ax.set_title("Development Areas")
        img = fig_to_bytes(fig, dpi=200)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            tmpfile.write(img)
            tmpfile.flush()
            pdf.image(tmpfile.name, w=120)
        os.unlink(tmpfile.name)
    pdf.cell(0, 10, "Employees and Development Areas:", ln=1)
    for i, row in dev_table.iterrows():
        pdf.cell(0, 8, clean_unicode(f"{row['name']}: {row['development_area']}"), ln=1)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Average Competency Radar", ln=1)
    all_category_scores = {}
    for idx, row in employee_df.iterrows():
        responses = list(row.drop(['name', 'development_area']))
        cat_scores = get_category_scores(responses, question_categories)
        for cat, score in cat_scores.items():
            if not np.isnan(score):
                all_category_scores.setdefault(cat, []).append(score)
    avg_category_scores = {cat: np.mean(scores) for cat, scores in all_category_scores.items() if len(scores) > 0}
    if avg_category_scores:
        cats = list(avg_category_scores.keys())
        vals = list(avg_category_scores.values())
        vals += vals[:1]
        angles = np.linspace(0, 2 * np.pi, len(cats) + 1, endpoint=True)
        fig, ax = plt.subplots(subplot_kw=dict(polar=True))
        ax.plot(angles, vals, linewidth=2, linestyle='solid', marker='o')
        ax.fill(angles, vals, alpha=0.25)
        ax.set_thetagrids(np.degrees(angles[:-1]), cats)
        ax.set_title("Average Competency Radar")
        img = fig_to_bytes(fig, dpi=200)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            tmpfile.write(img)
            tmpfile.flush()
            pdf.image(tmpfile.name, w=120)
        os.unlink(tmpfile.name)
    leader_qs = [i for i, q in enumerate(questions[:-2]) if 'leader' in str(q).lower()]
    engagement_qs = [i for i, q in enumerate(questions[:-2]) if
                     'engage' in str(q).lower() or 'engagement' in str(q).lower()]
    for q_idxs, title in [(leader_qs, "Leader Qualities"), (engagement_qs, "Employee Engagement")]:
        if q_idxs:
            all_scores = []
            for q in q_idxs:
                q_col = f"Q{q + 1}"
                vals = employee_df[q_col].map(score_extract).dropna().tolist()
                all_scores += vals
            if all_scores:
                fig, ax = plt.subplots()
                ax.hist(all_scores, bins=range(1, 8), align='left', rwidth=0.7)
                ax.set_xlabel("Score")
                ax.set_ylabel("Frequency")
                ax.set_title(f"{title} Scores")
                img = fig_to_bytes(fig, dpi=200)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                    tmpfile.write(img)
                    tmpfile.flush()
                    pdf.image(tmpfile.name, w=120)
                os.unlink(tmpfile.name)
    return pdf.output(dest='S').encode('latin-1')


st.title("ML Performance Planning Tool")

uploaded_file = st.sidebar.file_uploader("Upload Employee Data CSV", type=["csv"])
if uploaded_file:
    employee_df, questions, question_categories = load_data(uploaded_file)
    view = st.sidebar.selectbox("Report Type", ["Executive Report", "Individual Report"])

    if view == "Executive Report":
        st.header("Executive Summary")
        dev_table = employee_df[['name', 'development_area']].copy()
        dev_table['development_area'] = dev_table['development_area'].apply(clean_dev_area)
        dev_table = dev_table[dev_table['development_area'] != ""]
        st.subheader("Development Areas Distribution (Grouped by Theme)")

        dev_keywords = extract_keywords_from_dev_areas(dev_table['development_area'].tolist())
        if dev_keywords:
            most_common = dev_keywords.most_common(10)  # Show top 10 themes
            labels, counts = zip(*most_common)
            fig = px.pie(values=counts, names=labels, title="Top Development Area Themes", hole=0.3)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No development area themes found.")

        st.subheader("Employees & Development Areas")
        if not dev_table.empty:
            st.dataframe(dev_table.rename(columns={"name": "Employee", "development_area": "Development Area"}))
        else:
            st.info("No development areas found.")

        st.subheader("Average Competency Radar (Interactive)")
        all_category_scores = {}
        for idx, row in employee_df.iterrows():
            responses = list(row.drop(['name', 'development_area']))
            cat_scores = get_category_scores(responses, question_categories)
            for cat, score in cat_scores.items():
                if not np.isnan(score):
                    all_category_scores.setdefault(cat, []).append(score)
        avg_category_scores = {cat: np.mean(scores) for cat, scores in all_category_scores.items() if len(scores) > 0}
        if avg_category_scores:
            fig = plotly_radar(avg_category_scores, "Average Competency Radar")
            st.plotly_chart(fig, use_container_width=True)
        # Leader/engagement charts
        leader_qs = [i for i, q in enumerate(questions[:-2]) if 'leader' in str(q).lower()]
        engagement_qs = [i for i, q in enumerate(questions[:-2]) if
                         'engage' in str(q).lower() or 'engagement' in str(q).lower()]
        if leader_qs:
            st.subheader("Leader Qualities Score Distribution")
            scores = []
            for q in leader_qs:
                q_col = f"Q{q + 1}"
                vals = employee_df[q_col].map(score_extract).dropna().tolist()
                scores += vals
            if scores:
                fig = plotly_hist(scores, "Leader Qualities")
                st.plotly_chart(fig, use_container_width=True)
        if engagement_qs:
            st.subheader("Engagement Score Distribution")
            scores = []
            for q in engagement_qs:
                q_col = f"Q{q + 1}"
                vals = employee_df[q_col].map(score_extract).dropna().tolist()
                scores += vals
            if scores:
                fig = plotly_hist(scores, "Employee Engagement")
                st.plotly_chart(fig, use_container_width=True)
        # PDF export
        pdf_bytes = export_pdf_exec(employee_df, questions, question_categories)
        st.download_button(
            label="Download Executive Report as PDF",
            data=pdf_bytes,
            file_name="executive_report.pdf",
            mime="application/pdf"
        )

    elif view == "Individual Report":
        emp = st.sidebar.selectbox("Select Employee", employee_df['name'])
        emp_row = employee_df[employee_df['name'] == emp].iloc[0]
        st.header(f"Report for {emp}")

        # --- Development Area at the top ---
        dev_area = clean_dev_area(emp_row['development_area'])
        st.subheader("Development Areas")
        if dev_area == "":
            st.write("*No development areas provided*")
        else:
            st.write(dev_area)

        # --- Charts at the top ---
        responses = list(emp_row.drop(['name', 'development_area']))
        scores = [score_extract(ans) for ans in responses]
        category_scores = get_category_scores(responses, question_categories)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Competency Radar Chart (Interactive)")
            if category_scores:
                fig = plotly_radar(category_scores, "Competency Radar")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough data for a radar chart.")

        with col2:
            st.subheader("Score Chart (Interactive)")
            if any(s is not None for s in scores):
                fig = plotly_line(scores, questions)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No scores available.")

        st.subheader("Development Recommendation")
        st.write(get_recommendation(dev_area))

        st.subheader("Survey Answers (Answered Only)")
        for i, ans in enumerate(responses):
            if pd.isnull(ans) or ans == "" or str(ans).strip().lower() in ["nan", "none"]:
                continue
            st.markdown(f"**{questions[i]}**: {ans}")

        pdf_bytes = export_pdf_emp(emp_row, questions, question_categories)
        st.download_button(
            label="Download This Employee's Report as PDF",
            data=pdf_bytes,
            file_name=f"{emp.replace(' ', '_').lower()}_report.pdf",
            mime="application/pdf"
        )
else:
    st.info("Please upload your Employee Data CSV to get started.")
