# utils/skill_extractor.py

import re
import pandas as pd

def extract_skills_from_text(resume_text, skill_list):
    resume_text = resume_text.lower()
    extracted_skills = []

    for skill in skill_list:
        if re.search(r'\b' + re.escape(skill.lower()) + r'\b', resume_text):
            extracted_skills.append(skill.lower())

    return list(set(extracted_skills))
