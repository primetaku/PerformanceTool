# ML Performance Planning Tool

This app generates executive and individual performance reports (with interactive diagrams) from your organization’s employee survey data. It provides powerful visual insights for leaders and employees, including:

- Interactive radar/spider and line charts for competencies and scores.
- Executive report with grouped development area themes (pie chart).
- Per-employee development area, Q&A, and recommendations.
- High-resolution PDF export for both individual and executive reports.

---

## Features

- **Interactive Dashboard:** Web app built on Streamlit.
- **Automatic Grouping:** Groups and visualizes common development area themes across employees.
- **Personalized Reports:** One-click per-employee or all-employee reporting.
- **Professional Export:** Download as crisp, ready-to-share PDF.
- **Flexible Input:** Works with CSV files exported from Google Forms, Excel, or similar.

---

1. Installation

**Requirements:**
- Python 3.12+
- [pip](https://pip.pypa.io/en/stable/installation/)

**Install dependencies:**

```bash
pip install streamlit pandas numpy matplotlib plotly fpdf

---

2. Data Preparation

- Use a CSV file where each column after "Timestamp" is an employee.
- Each employee's name should be in the first data row.
- Include a row labeled "Development Areas" (case-insensitive) in the "Timestamp" column.
- Each employee's development area(s) go in their respective column in that row (e.g., "Communication, Leadership").
- Survey questions/answers should follow after the "Development Areas" row.

CSV Example:

| Timestamp          | Alice       | Bob         | Carol    |
|--------------------|-------------|-------------|----------|
|                    | Alice Smith | Bob Brown   | Carol C. |
| ...                | ...         | ...         | ...      |
| Development Areas  | Leadership  | Teamwork    | Feedback |
| Q1: ...            | 6           | 5           | 6        |
| Q2: ...            | 5           | 6           | 6        |
| ...                | ...         | ...         | ...      |

---

3. Running the App

1. Save the Python script (tool.py).
2. Open a terminal in the script's folder.
3. Run:

streamlit run tool.py

4. In the app sidebar:
   - Upload the CSV file (Employee data.csv)
   - Select "Executive Report" or "Individual Report".
   - Download PDF reports if needed.

---

4. Notes

- For best results, use simple, consistent words/phrases for development areas (e.g., "Leadership", "Teamwork").
- Pie chart in the executive report groups the most common themes across all employees' development areas.

---

Contact:
Takudzwa Zinyengere  
takudzwazinyengere@email.com
