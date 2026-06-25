# ============================================
#   Tech Stack Recommender — Pro Version
#   DecodeLabs Industrial Training | Batch 2026
#   Project 3 — AI Recommendation Logic
#   By: Drashtee
#   Algorithm: Content-Based Filtering
#             TF-IDF Vectorization + Cosine Similarity
# ============================================

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import time

# ============================================
# COLORS FOR TERMINAL
# ============================================
class C:
    PINK   = '\033[95m'
    BLUE   = '\033[94m'
    CYAN   = '\033[96m'
    GREEN  = '\033[92m'
    YELLOW = '\033[93m'
    RED    = '\033[91m'
    BOLD   = '\033[1m'
    DIM    = '\033[2m'
    RESET  = '\033[0m'

def p(text, color=C.RESET):
    print(f"{color}{text}{C.RESET}")

# ============================================
# KNOWLEDGE BASE — Extended Dataset
# ============================================

FALLBACK_DATA = {
    "job_title": [
        "Data Scientist", "Web Developer", "DevOps Engineer",
        "AI/ML Engineer", "Backend Developer", "Data Analyst",
        "Cybersecurity Engineer", "Cloud Architect", "Frontend Developer",
        "Full Stack Developer", "Mobile Developer", "Database Administrator",
        "NLP Engineer", "Blockchain Developer", "Game Developer",
        "Computer Vision Engineer", "Robotics Engineer", "AR/VR Developer",
        "Site Reliability Engineer", "Business Intelligence Developer"
    ],
    "skills": [
        "python machine learning data analysis statistics pandas numpy sql deep learning visualization scikit-learn",
        "html css javascript react nodejs frontend backend web design bootstrap jquery",
        "aws docker kubernetes cloud linux git ci cd automation deployment infrastructure terraform",
        "python tensorflow pytorch deep learning neural network nlp computer vision model training keras",
        "python java nodejs sql apis rest databases django flask spring microservices postgresql",
        "excel sql python data visualization power bi tableau statistics reporting dashboards analytics",
        "networking security ethical hacking linux firewalls encryption penetration testing vulnerability",
        "aws azure google cloud infrastructure deployment devops automation scalability serverless lambda",
        "html css javascript react vuejs typescript ui ux design responsive figma animation sass",
        "html css javascript react nodejs python django sql mongodb rest apis fullstack express",
        "flutter react native android ios kotlin swift java mobile app development cross-platform",
        "sql mysql postgresql mongodb database management backup optimization queries indexing redis",
        "python nlp text processing bert transformers huggingface sentiment analysis chatbot spacy",
        "solidity ethereum blockchain web3 smart contracts cryptocurrency decentralized dapp",
        "unity unreal engine c++ c# game design 3d animation physics rendering opengl",
        "python opencv deep learning image processing cnn object detection yolo segmentation",
        "python c++ ros embedded systems sensors actuators control systems automation hardware",
        "unity c# augmented reality virtual reality 3d spatial computing mixed reality",
        "linux monitoring automation python go reliability scalability incident management sre",
        "sql power bi tableau data warehousing etl reporting analytics business intelligence"
    ]
}

# Related skills to learn for each job role
RELATED_SKILLS = {
    "Data Scientist":              ["Scikit-learn", "TensorFlow", "Tableau", "Apache Spark"],
    "Web Developer":               ["TypeScript", "GraphQL", "Docker", "AWS"],
    "DevOps Engineer":             ["Terraform", "Ansible", "Prometheus", "GitLab CI"],
    "AI/ML Engineer":              ["MLOps", "Hugging Face", "FastAPI", "Ray"],
    "Backend Developer":           ["Redis", "GraphQL", "Kafka", "gRPC"],
    "Data Analyst":                ["Python", "Power BI", "Looker", "dbt"],
    "Cybersecurity Engineer":      ["Metasploit", "Wireshark", "Splunk", "Zero Trust"],
    "Cloud Architect":             ["Terraform", "Kubernetes", "FinOps", "Multi-Cloud"],
    "Frontend Developer":          ["Next.js", "Storybook", "Cypress", "Web3.js"],
    "Full Stack Developer":        ["Docker", "CI/CD", "Redis", "GraphQL"],
    "Mobile Developer":            ["Firebase", "App Store Optimization", "CI/CD", "RevenueCat"],
    "Database Administrator":      ["Redis", "Cassandra", "Query Optimization", "Replication"],
    "NLP Engineer":                ["LangChain", "Vector Databases", "RAG", "LLM Fine-tuning"],
    "Blockchain Developer":        ["IPFS", "Layer 2", "DeFi Protocols", "Hardhat"],
    "Game Developer":              ["Shader Programming", "Multiplayer Networking", "VR/AR", "PhysX"],
    "Computer Vision Engineer":    ["CUDA", "TensorRT", "Point Cloud", "Depth Estimation"],
    "Robotics Engineer":           ["SLAM", "Computer Vision", "Path Planning", "ROS2"],
    "AR/VR Developer":             ["WebXR", "Shader Graph", "Spatial Audio", "Hand Tracking"],
    "Site Reliability Engineer":   ["Chaos Engineering", "eBPF", "OpenTelemetry", "SLO/SLI"],
    "Business Intelligence Developer": ["dbt", "Snowflake", "Looker", "Apache Airflow"]
}

# Avg salary ranges (LPA India)
SALARY_DATA = {
    "Data Scientist":              "8–25 LPA",
    "Web Developer":               "4–18 LPA",
    "DevOps Engineer":             "8–28 LPA",
    "AI/ML Engineer":              "10–35 LPA",
    "Backend Developer":           "6–22 LPA",
    "Data Analyst":                "4–15 LPA",
    "Cybersecurity Engineer":      "7–25 LPA",
    "Cloud Architect":             "15–45 LPA",
    "Frontend Developer":          "4–18 LPA",
    "Full Stack Developer":        "6–25 LPA",
    "Mobile Developer":            "5–20 LPA",
    "Database Administrator":      "5–18 LPA",
    "NLP Engineer":                "10–30 LPA",
    "Blockchain Developer":        "8–30 LPA",
    "Game Developer":              "4–20 LPA",
    "Computer Vision Engineer":    "10–30 LPA",
    "Robotics Engineer":           "8–25 LPA",
    "AR/VR Developer":             "6–22 LPA",
    "Site Reliability Engineer":   "12–40 LPA",
    "Business Intelligence Developer": "5–18 LPA"
}


# ============================================
# STEP 1 — INGESTION
# ============================================

def load_dataset(filepath="raw_skills.csv"):
    try:
        df = pd.read_csv(filepath)
        p(f"  ✅ Dataset loaded: {len(df)} job roles found.", C.GREEN)
        return df
    except FileNotFoundError:
        p("  ⚠️  raw_skills.csv not found — using built-in dataset.", C.YELLOW)
        return pd.DataFrame(FALLBACK_DATA)


# ============================================
# STEP 2 — VECTORIZATION (TF-IDF)
# ============================================

def build_tfidf_matrix(df):
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(df['skills'])
    return vectorizer, tfidf_matrix


# ============================================
# STEP 3 — SCORING (Cosine Similarity)
# ============================================

def score_and_rank(user_skills, vectorizer, tfidf_matrix, df):
    user_profile = " ".join(user_skills).lower()
    user_vector  = vectorizer.transform([user_profile])
    scores       = cosine_similarity(user_vector, tfidf_matrix)[0]

    df = df.copy()
    df['score']   = scores
    df['percent'] = (scores * 100).round(2)
    return df.sort_values('score', ascending=False).reset_index(drop=True)


# ============================================
# STEP 4 — FILTERING + DISPLAY (Top-N)
# ============================================

def render_bar(score, width=25):
    filled = int((score / 100) * width)
    empty  = width - filled
    if score >= 60:   color = C.GREEN
    elif score >= 35: color = C.YELLOW
    else:             color = C.CYAN
    bar = "█" * filled + "░" * empty
    return f"{color}[{bar}]{C.RESET}"


def display_results(top_df, user_skills, df_len):
    medals = ["🥇", "🥈", "🥉"]
    line   = C.DIM + "─" * 58 + C.RESET

    print(f"\n{line}")
    p(f"  🎯  TOP {len(top_df)} JOB ROLES MATCHING YOUR PROFILE", C.BOLD + C.PINK)
    print(line)
    p(f"  Your Skills  : {C.CYAN}{', '.join(s.title() for s in user_skills)}{C.RESET}")
    p(f"  Dataset      : {df_len} job roles analyzed", C.DIM)
    p(f"  Algorithm    : TF-IDF + Cosine Similarity", C.DIM)
    print(line)

    for i, (_, row) in enumerate(top_df.iterrows()):
        medal   = medals[i] if i < 3 else "⭐"
        title   = row['job_title']
        pct     = row['percent']
        bar     = render_bar(pct)
        salary  = SALARY_DATA.get(title, "N/A")
        related = RELATED_SKILLS.get(title, [])

        print()
        p(f"  {medal}  {C.BOLD}{title}{C.RESET}", C.RESET)
        print(f"      {bar}  ", end="")

        if pct >= 60:   p(f"{pct}% match 🔥", C.GREEN)
        elif pct >= 35: p(f"{pct}% match ✨", C.YELLOW)
        else:           p(f"{pct}% match", C.CYAN)

        p(f"      💰 Avg Salary  : {salary}", C.DIM)
        if related:
            p(f"      📚 Also learn  : {', '.join(related[:3])}", C.DIM)

    print(f"\n{line}\n")


# ============================================
# SAVE RESULTS
# ============================================

def save_results(top_df, user_skills, filename="recommendation_output.txt"):
    with open(filename, 'w') as f:
        f.write("=" * 55 + "\n")
        f.write("  Tech Stack Recommender — DecodeLabs 2026\n")
        f.write("  By: Drashtee | Project 3\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Your Skills: {', '.join(user_skills)}\n\n")
        f.write("TOP 3 RECOMMENDATIONS:\n\n")
        for i, (_, row) in enumerate(top_df.iterrows(), 1):
            title  = row['job_title']
            pct    = row['percent']
            salary = SALARY_DATA.get(title, "N/A")
            f.write(f"  Rank {i}: {title}\n")
            f.write(f"  Match : {pct}%\n")
            f.write(f"  Salary: {salary}\n\n")
    p(f"  💾 Results saved to {filename}", C.GREEN)


# ============================================
# COLD START HANDLER
# ============================================

def handle_cold_start(df):
    p("\n  ⚠️  No skills detected — showing top trending roles!\n", C.YELLOW)
    return df.head(3)


# ============================================
# BANNER
# ============================================

def print_banner():
    os.system('cls' if os.name == 'nt' else 'clear')
    print()
    p("  ╔══════════════════════════════════════════════════╗", C.PINK)
    p("  ║      🤖  Tech Stack Recommender                  ║", C.PINK)
    p("  ║      DecodeLabs Industrial Training 2026         ║", C.PINK)
    p("  ║      Project 3 — AI Recommendation Logic         ║", C.PINK)
    p("  ║      By: Drashtee                                ║", C.PINK)
    p("  ╚══════════════════════════════════════════════════╝", C.PINK)
    print()


# ============================================
# MAIN
# ============================================

def main():
    print_banner()

    p("  📂 Loading dataset...", C.CYAN)
    df = load_dataset("raw_skills.csv")

    p("\n  🔧 Building TF-IDF matrix...", C.CYAN)
    vectorizer, tfidf_matrix = build_tfidf_matrix(df)
    p("  ✅ Model ready!\n", C.GREEN)

    while True:
        print()
        p("  💡 Enter your skills for personalized job recommendations", C.BOLD)
        p("     (Minimum 3 skills | Press Enter to skip optional ones)\n", C.DIM)

        s1 = input(f"  {C.CYAN}Skill 1{C.RESET} : ").strip()
        s2 = input(f"  {C.CYAN}Skill 2{C.RESET} : ").strip()
        s3 = input(f"  {C.CYAN}Skill 3{C.RESET} : ").strip()
        s4 = input(f"  {C.DIM}Skill 4 (optional){C.RESET} : ").strip()
        s5 = input(f"  {C.DIM}Skill 5 (optional){C.RESET} : ").strip()

        user_skills = [s for s in [s1, s2, s3, s4, s5] if s]

        if not user_skills:
            top_df = handle_cold_start(df)
        else:
            p("\n  🔍 Analyzing your profile...", C.CYAN)
            time.sleep(0.6)
            ranked = score_and_rank(user_skills, vectorizer, tfidf_matrix, df)
            top_df = ranked.head(3)

        display_results(top_df, user_skills if user_skills else ["Trending"], len(df))

        # Save option
        save = input(f"  {C.DIM}Save results to file? (yes/no){C.RESET} : ").strip().lower()
        if save in ['yes', 'y']:
            save_results(top_df, user_skills)

        # Try again
        again = input(f"\n  {C.DIM}Try with different skills? (yes/no){C.RESET} : ").strip().lower()
        if again not in ['yes', 'y']:
            print()
            p("  Goodbye! Keep building amazing things 🚀", C.PINK)
            print()
            break


if __name__ == "__main__":
    main()