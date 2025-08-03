from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_skill_gap(user_skills, job_skills):
    user_text = " ".join(user_skills).lower()
    job_text = " ".join(job_skills).lower()

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([user_text, job_text])
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]

    matched = list(set(user_skills).intersection(set(job_skills)))
    missing = list(set(job_skills) - set(user_skills))

    return {
        "matched": matched,
        "missing": missing,
        "score": round(similarity * 100, 2)
    }
