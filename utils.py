import os
import json
from pyexpat import features
import pandas as pd
import joblib
import sys
import ast
from werkzeug.security import generate_password_hash, check_password_hash
from io import StringIO

# ==========================================================
# PATHS
# ==========================================================
DATA_DIR = "data"
MODEL_DIR = "model"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

MASTER_CSV = os.path.join(DATA_DIR, "master_students.csv")
RECYCLE_CSV = os.path.join(DATA_DIR, "recycle_bin.csv")
ADMINS_JSON = os.path.join(DATA_DIR, "admins.json")

# ==========================================================
# LOAD ML MODEL
# ==========================================================
def resource_path(relative_path):
    """Make paths work inside PyInstaller EXE."""
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# Use EXE-safe paths
# ==========================================================
# LOAD NEW ML MODEL
# ==========================================================

MODEL_PATH = resource_path("model/best_model.pkl")
ENCODER_PATH = resource_path("model/label_encoder.pkl")

MODEL = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
ENCODER = joblib.load(ENCODER_PATH) if os.path.exists(ENCODER_PATH) else None


# ==========================================================
# ADMIN MANAGEMENT
# ==========================================================
def read_admins():
    if not os.path.exists(ADMINS_JSON):
        return {}
    with open(ADMINS_JSON, "r") as f:
        return json.load(f)


def save_admins(admins):
    with open(ADMINS_JSON, "w") as f:
        json.dump(admins, f, indent=2)


def create_admin(username, password):
    admins = read_admins()
    if username in admins:
        raise ValueError("Admin already exists")

    admins[username] = generate_password_hash(password)
    save_admins(admins)


def verify_admin(username, password):
    admins = read_admins()
    return username in admins and check_password_hash(admins[username], password)


# ==========================================================
# CSV DATA HANDLING
# ==========================================================
def read_master_df():
    if not os.path.exists(MASTER_CSV):
        return pd.DataFrame()
    return pd.read_csv(MASTER_CSV)


def save_master_df(df):
    df.to_csv(MASTER_CSV, index=False)


def read_recycle_df():
    if not os.path.exists(RECYCLE_CSV):
        return pd.DataFrame()
    return pd.read_csv(RECYCLE_CSV)


def save_recycle_df(df):
    df.to_csv(RECYCLE_CSV, index=False)



# ==========================================================
# SKILL INDEX CALCULATION
# ==========================================================
def compute_skill_index_for_row(row):

    try:

        # -----------------------------
        # Academic Score
        # -----------------------------
        cgpa = float(row.get("CGPA", 0))

        projects = float(row.get("Projects_Done", 0))
        papers = float(row.get("Research_Papers_Published", 0))
        hackathons = float(row.get("Hackathons_Participated", 0))

        academic_score = (
            cgpa * 10 +
            projects * 4 +
            papers * 8 +
            hackathons * 5
        )

        # -----------------------------
        # Certificate Score
        # -----------------------------
        levels = str(row.get("Certificate_Course_Level", "")).split(",")
        platforms = str(row.get("Certificate_Platform", "")).split(",")
        domains = str(row.get("Certificate_Domain", "")).split(",")
        years = str(row.get("Certificate_Completion_Year", "")).split(",")

        level_weight = {
            "Beginner": 5,
            "Intermediate": 10,
            "Advanced": 15,
            "Professional": 20
        }

        platform_weight = {
            "Coursera": 1.4,
            "Google": 1.4,
            "Microsoft": 1.4,
            "Cisco": 1.5,
            "NPTEL": 1.5,
            "Udemy": 1.2
        }

        domain_weight = {
            "AI": 1.5,
            "Data Science": 1.4,
            "Cybersecurity": 1.4,
            "Cloud Computing": 1.3,
            "Programming": 1.2,
            "Networking": 1.2
        }

        cert_score = 0

        for i in range(len(levels)):

            level = levels[i].strip()
            platform = platforms[i].strip() if i < len(platforms) else ""
            domain = domains[i].strip() if i < len(domains) else ""

            base = level_weight.get(level, 5)
            p_weight = platform_weight.get(platform, 1.0)
            d_weight = domain_weight.get(domain, 1.0)

            cert_score += base * p_weight * d_weight

        # -----------------------------
        # Final Skill Index
        # -----------------------------
        features = [[
            academic_score,
            cert_score,
            float(row.get("Number_of_Certifications", 0)),
            float(row.get("Hackathons_Participated", 0))
        ]]

        feature_df = pd.DataFrame(features, columns=[
            "academic_score",
            "certificate_score",
            "certificate_count",
            "hackathons"
        ])
        feature_df = pd.DataFrame(features, columns=[
            "academic_score",
            "certificate_score",
            "certificate_count",
            "hackathons"
        ])

        
        if MODEL is not None:
            proba = MODEL.predict_proba(feature_df)[0]

            score = (
                proba[1] * 40 +
                proba[2] * 70 +
                proba[3] * 90
            )

            return round(score, 2)

        return round(min((0.5 * academic_score) + (0.5 * cert_score), 100), 2)

    except Exception as e:
        print("Skill index error:", e)
        return 0

    
def parse_list_field(val):
    """
    Safely parse a sem subjects/IA1/IA2/model list field.
    Supports:
      - Python lists: ['A','B']
      - JSON lists: ["A","B"]
      - CSV lists: A,B,C
      - Pipe lists: A|B|C
      - Broken formats: ['A', "B"] etc.
    """
    if pd.isna(val) or val == "":
        return []

    if isinstance(val, (list, tuple)):
        return list(val)

    s = str(val).strip()

    # Try Python literal list → ['A','B']
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed]
    except Exception:
        pass

    # Try JSON list → ["A","B"]
    try:
        parsed = json.loads(s.replace("'", '"'))
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed]
    except Exception:
        pass

    # Fallback parsing
    if "," in s:
        return [x.strip() for x in s.split(",") if x.strip()]

    if "|" in s:
        return [x.strip() for x in s.split("|") if x.strip()]

    # Single item fallback
    return [s]
def compute_analysis_from_df(df, selected_sem=None):
    """
    Compute numeric summary, language distribution, subject-level raw & normalized scores.
    All returned values are JSON-safe (int, float, str).
    """
    out = {}
    source_semester_map = {}

    # ----- NUMERIC SUMMARY -----
    numeric_cols = [
        c for c in df.columns
        if c.lower() in ("cgpa", "weekly_aptitude_avg", "avg_communication_score")
        or df[c].dtype.kind in "biuf"
    ]

    summary = {}
    for col in numeric_cols:
        try:
            ser = pd.to_numeric(df[col], errors="coerce").dropna()
            summary[col] = {
                "count": int(ser.count()),
                "mean": float(round(ser.mean(), 2)) if not ser.empty else None,
                "min": float(round(ser.min(), 2)) if not ser.empty else None,
                "max": float(round(ser.max(), 2)) if not ser.empty else None
            }
        except:
            summary[col] = {"count": 0, "mean": None, "min": None, "max": None}

    out["summary_numeric"] = summary

    # ----- LANGUAGE DISTRIBUTION -----
    if "Foreign_Language_Known" in df.columns:
        langs = df["Foreign_Language_Known"].fillna("").astype(str)
        dist = {str(k): int(v) for k, v in langs.value_counts().to_dict().items()}

        lang_students = {}
        for lang in dist:
            if lang.strip() == "":
                continue
            sub = df[langs == lang]
            lang_students[lang] = [
            {
                k: ("" if pd.isna(v) else v)
                for k, v in r.to_dict().items()
            }
            for _, r in sub.iterrows()
            ]

        out["foreign_language_dist"] = dist
        out["foreign_language_students"] = lang_students
    else:
        out["foreign_language_dist"] = {}
        out["foreign_language_students"] = {}

    # ----- SUBJECT-PERFORMANCE EXTRACTION -----
    subjects_map = {}

    for sem in range(1, 9):
        subj_col = f"Sem{sem}_Subjects"
        ia1_col = f"Sem{sem}_IA1"
        ia2_col = f"Sem{sem}_IA2"
        model_col = f"Sem{sem}_Model"

        if subj_col not in df.columns:
            continue

        for _, row in df.iterrows():
            subj_list = parse_list_field(row.get(subj_col, ""))
            ia1_list = parse_list_field(row.get(ia1_col, ""))
            ia2_list = parse_list_field(row.get(ia2_col, ""))
            model_list = parse_list_field(row.get(model_col, ""))

            for i, subj in enumerate(subj_list):
                subj_name = str(subj).strip()
                if subj_name == "":
                    continue
                source_semester_map[subj_name] = sem

                def safe(lst):
                    try:
                        return float(lst[i])
                    except:
                        return 0.0

                raw_total = safe(ia1_list) + safe(ia2_list) + safe(model_list)

                subjects_map.setdefault(subj_name, []).append({
                    "student_id": str(row.get("Student_ID", "")),
                    "student_name": str(row.get("Student_Name", "")),
                    "raw_score": float(raw_total),
                    "cgpa": float(row.get("CGPA", 0))
                })

    # ----- SUBJECT SUMMARY -----
    subject_summary = {}

    for subject, recs in subjects_map.items():
        raw_scores = [r["raw_score"] for r in recs]
        max_raw = max(raw_scores) if raw_scores else 1.0
        if max_raw <= 0:
            max_raw = 1.0

        for r in recs:
            r["normalized_score"] = float(round((r["raw_score"] / max_raw) * 100, 2))

        scores = [r["normalized_score"] for r in recs]
        avg = float(round(sum(scores) / len(scores), 2)) if scores else 0.0

        def clean(lst):
            return [
                {
                    "student_id": r["student_id"],
                    "student_name": r["student_name"],
                    "raw_score": float(r["raw_score"]),
                    "normalized_score": float(r["normalized_score"]),
                    "cgpa": float(r["cgpa"])
                }
                for r in lst
            ]

        top5 = clean(sorted(recs, key=lambda x: x["normalized_score"], reverse=True)[:5])
        least5 = clean(sorted(recs, key=lambda x: x["normalized_score"])[:5])

        mn, mx = min(scores), max(scores)

        # Buckets
        if mn == mx:
            buckets = [{
                "label": f"{mn}",
                "min": float(mn),
                "max": float(mx),
                "count": len(scores)
            }]
        else:
            step = (mx - mn) / 5
            buckets = []
            for b in range(5):
                low = mn + b * step
                high = mn + (b + 1) * step if b < 4 else mx
                count = sum(low <= s <= high for s in scores)
                buckets.append({
                    "label": f"{round(low,1)} - {round(high,1)}",
                    "min": float(low),
                    "max": float(high),
                    "count": int(count)
                })

        subject_summary[subject] = {
            "avg": avg,
            "top": top5,
            "least": least5,
            "buckets": buckets,
            "subject_max_raw": float(max_raw)
        }

    out["subject_summary"] = subject_summary
    out["source_semester_map"] = source_semester_map
    return out

def classify_skill_level(score):

    try:
        score = float(score)
    except:
        return "Unknown"

    if score < 40:
        return "Beginner"
    elif score < 60:
        return "Intermediate"
    elif score < 80:
        return "Advanced"
    else:
        return "Expert"

# ==========================================================
# MERGE CSV INTO MASTER
# ==========================================================
def merge_upload_df(upload_df, admin_user="system"):
    master = read_master_df()

    upload_df["Register_Number"] = upload_df["Register_Number"].astype(str)
    if not master.empty:
        master["Register_Number"] = master["Register_Number"].astype(str)

    if master.empty:
        # first upload → compute skill index for all rows
        upload_df["skill_index"] = upload_df.apply(compute_skill_index_for_row, axis=1)
        save_master_df(upload_df)
        return {"updated": len(upload_df)}

    # update or insert rows
    for _, row in upload_df.iterrows():
        reg = row["Register_Number"]
        exists = master[master["Register_Number"] == reg]

        if not exists.empty:
            idx = exists.index[0]
            for col in upload_df.columns:
                if pd.notna(row[col]) and row[col] != "":
                    master.at[idx, col] = row[col]
        else:
            master = pd.concat([master, pd.DataFrame([row])], ignore_index=True)

    # recompute skill index for all
    master["skill_index"] = master.apply(compute_skill_index_for_row, axis=1)
    master["skill_level"] = master["skill_index"].apply(classify_skill_level)
    save_master_df(master)
    return {"updated": len(upload_df)}


# ==========================================================
# ELIGIBILITY CHECK LOGIC
# ==========================================================

# ==========================================================
# UPDATED ELIGIBILITY LOGIC (IA1/IA2 out of 20, Model out of 100)
# ==========================================================

import ast

# DOMAIN → REQUIRED SUBJECTS
DOMAIN_SUBJECTS = {
    "Machine Learning / AI": [
        "Machine Learning", "Deep Learning", "Neural Networks",
        "Artificial Intelligence", "Statistics", "Probability",
        "Linear Algebra", "Python"
    ],
    "Web Development / Full Stack": [
        "Web Technology", "Internet Programming", "HTML", "CSS",
        "JavaScript", "React", "NodeJS", "DBMS"
    ],
    "Data Science / Analytics": [
        "Statistics", "Data Mining", "Data Science",
        "Python", "R", "Big Data", "Probability"
    ],
    "Cyber Security": [
        "Computer Networks", "Network Security",
        "Cryptography", "Ethical Hacking", "Operating Systems"
    ],
    "Cloud Computing / DevOps": [
        "Cloud Computing", "Distributed Systems", "Operating Systems",
        "Virtualization", "Computer Networks"
    ],
    "Software Development": [
        "Data Structures", "Algorithms", "OOPS",
        "Java", "C++", "Software Engineering"
    ],
    "IoT / Embedded Systems": [
        "IoT", "Embedded Systems", "Microcontrollers",
        "Wireless Sensor Networks"
    ],
    "Robotics / Automation": [
        "Robotics", "Control Systems", "AI", "Image Processing"
    ]
}

# Convert raw marks into percentage
def calc_percentage(ia1, ia2, model):
    ia1_pct = (ia1 / 20) * 100
    ia2_pct = (ia2 / 20) * 100
    model_pct = (model / 100) * 100
    return ia1_pct, ia2_pct, model_pct


# A subject is valid if ANY 2/3 percentages ≥ 70
def subject_pass(ia1, ia2, model):
    ia1_pct, ia2_pct, model_pct = calc_percentage(ia1, ia2, model)
    passed = sum([
        ia1_pct >= 70,
        ia2_pct >= 70,
        model_pct >= 70
    ])
    return passed >= 2   # ANY 2 must be ≥ 70%


# MAIN eligibility function
def is_student_eligible(student, company):
    domain = company["domain_required"]
    min_cgpa = float(company["min_cgpa"])

    # 1) CGPA check
    if float(student["CGPA"]) < min_cgpa:
        return False

    required = DOMAIN_SUBJECTS.get(domain, [])
    if not required:
        return False

    eligible_subjects = 0

    # 2) Check subjects + marks across Sem1–Sem6
    for i in range(1, 7):
        sub_col = f"Sem{i}_Subjects"
        ia1_col = f"Sem{i}_IA1"
        ia2_col = f"Sem{i}_IA2"
        model_col = f"Sem{i}_Model"

        if sub_col not in student:
            continue

        try:
            subjects = ast.literal_eval(student[sub_col])
            ia1_list = ast.literal_eval(student[ia1_col])
            ia2_list = ast.literal_eval(student[ia2_col])
            model_list = ast.literal_eval(student[model_col])
        except:
            continue

        # Check each subject in the semester
        for idx, sub in enumerate(subjects):
            if sub in required:
                ia1 = ia1_list[idx] if idx < len(ia1_list) else 0
                ia2 = ia2_list[idx] if idx < len(ia2_list) else 0
                model = model_list[idx] if idx < len(model_list) else 0

                if subject_pass(ia1, ia2, model):
                    eligible_subjects += 1

    # 3) Apply 30% subject requirement
    subject_ratio = eligible_subjects / len(required)
    return subject_ratio >= 0.30   # UPDATED threshold


# Wrapper: return eligible students list
def get_eligible_students_for_company(company_row):
    master = read_master_df()
    eligible = []

    for _, student in master.iterrows():
        if is_student_eligible(student, company_row):
            eligible.append(student.to_dict())

    return eligible
