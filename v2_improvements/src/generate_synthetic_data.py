"""Generate a reproducible, explicitly synthetic training data set.

The generator derives field names from the read-only master CSV, then gives its
numeric academic/portfolio fields correlated distributions. Targets are noisy,
weighted combinations of these fields rather than independent random labels.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parents[0] / "data" / "master_students.csv"
OUTPUT = ROOT / "data" / "synthetic_students.csv"

def generate_synthetic_data(rows: int = 900, seed: int = 42) -> pd.DataFrame:
    if not 800 <= rows <= 1000:
        raise ValueError("rows must be between 800 and 1000")
    rng = np.random.default_rng(seed)
    schema = pd.read_csv(SOURCE, nrows=1).columns.tolist()  # read only
    data = pd.DataFrame(index=range(rows), columns=schema)
    ability = rng.normal(0, 1, rows)
    cgpa = np.clip(7.35 + .82 * ability + rng.normal(0, .38, rows), 5, 10)
    projects = np.clip(np.rint(3.8 + 1.65 * ability + rng.normal(0, 1.8, rows)), 0, 12).astype(int)
    hackathons = np.clip(np.rint(3.1 + 1.35 * ability + rng.normal(0, 2.0, rows)), 0, 15).astype(int)
    research = np.clip(np.rint(.7 + .75 * ability + rng.normal(0, .9, rows)), 0, 7).astype(int)
    certs = np.clip(np.rint(9 + 3.2 * ability + rng.normal(0, 3.2, rows)), 0, 30).astype(int)
    leetcode = np.clip(np.rint(350 + 210 * ability + 28 * projects + rng.normal(0, 115, rows)), 0, 1800).astype(int)
    aptitude = np.clip(72 + 8.5 * ability + rng.normal(0, 6, rows), 40, 100)
    communication = np.clip(74 + 6.7 * ability + rng.normal(0, 7, rows), 40, 100)
    codechef = np.clip(np.rint(1450 + 290 * ability + 31 * hackathons + rng.normal(0, 170, rows)), 800, 3000).astype(int)
    certificate_score = np.clip(57 + 1.2 * certs + 4.5 * ability + rng.normal(0, 7, rows), 30, 100)
    academic = np.clip(45 + 5.25 * cgpa + .18 * aptitude + rng.normal(0, 4, rows), 40, 100)
    latent = (18 * (cgpa - 5) + 1.55 * projects + 1.1 * hackathons + 2.1 * research + .7 * certs + .014 * leetcode + .009 * (codechef - 800) + .11 * aptitude + .09 * communication)
    final_index = np.clip(20 + latent * .74 + rng.normal(0, 5.2, rows), 0, 100)
    # The regressor target has independent measurement noise; it is not a direct feature copy.
    skill_index = np.clip(final_index + rng.normal(0, 4.5, rows), 0, 100)
    skill_level = pd.cut(skill_index, [-1, 49, 68, 83, 101], labels=["Beginner", "Intermediate", "Advanced", "Expert"]).astype(str)
    scheme_probability = 1 / (1 + np.exp(-((final_index - 63) / 8 + .13 * projects + .07 * certs - 1.1)))
    scheme_eligible = rng.binomial(1, np.clip(scheme_probability, .03, .97))
    data.update(pd.DataFrame({
        "Batch": "2022-2026", "Department": "CSE (AI & ML)", "Student_ID": [f"S{i:04d}" for i in range(1, rows + 1)],
        "Register_Number": [f"V2CSE{i:04d}" for i in range(1, rows + 1)], "Student_Name": [f"Synthetic Student {i}" for i in range(1, rows + 1)],
        "Current_Sem": rng.integers(3, 9, rows), "Foreign_Language_Known": rng.choice(["German", "French", "Japanese", "Spanish"], rows),
        "CGPA": cgpa.round(2), "Projects_Done": projects, "Research_Papers_Published": research, "Review_Papers_Published": np.maximum(0, research - 1),
        "Patents_Granted": np.clip(rng.poisson(.25 + np.maximum(ability, 0) * .35), 0, 3), "Hackathons_Participated": hackathons,
        "Hackathons_Won_Top100": np.minimum(hackathons, rng.binomial(np.maximum(hackathons, 1), .16)), "Weekly_Aptitude_Avg": aptitude.round(1),
        "Avg_Communication_Score": communication.round(1), "Number_of_Certifications": certs, "LeetCode_Problems_Solved": leetcode,
        "CodeChef_Rating": codechef, "certificate_score": certificate_score.round(1), "academic_score": academic.round(1),
        "final_skill_index": final_index.round(2), "skill_index": skill_index.round(2), "skill_level": skill_level,
        "scheme_eligible": scheme_eligible, "Gender": rng.choice(["Female", "Male", "Non-binary"], rows), "Password": "SYNTHETIC",
        "Certificate_Course_Level": rng.choice(["Beginner", "Intermediate", "Advanced"], rows), "Certificate_Platform": rng.choice(["Coursera", "edX", "Udemy"], rows),
        "Certificate_Domain": rng.choice(["AI", "Programming", "Cloud"], rows), "Certificate_Completion_Year": rng.integers(2023, 2027, rows),
    }))
    # These are V2 derived fields, intentionally added beyond the source schema.
    # DataFrame.update ignores non-source columns, so assign them explicitly.
    data["certificate_score"] = certificate_score.round(1)
    data["academic_score"] = academic.round(1)
    data["final_skill_index"] = final_index.round(2)
    data["scheme_eligible"] = scheme_eligible
    return data

def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame = generate_synthetic_data()
    frame.to_csv(OUTPUT, index=False)
    print(f"Generated {len(frame)} synthetic rows at {OUTPUT}")

if __name__ == "__main__": main()
