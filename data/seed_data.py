import json
import random
import os

def generate_seed_data():
    random.seed(42)

    roles_skills = {
        "Data Scientist": ["python", "r", "sql", "machine learning", "deep learning", "tensorflow", "pytorch", "pandas", "numpy", "statistics", "data visualization", "scikit-learn"],
        "Software Engineer": ["java", "python", "c++", "algorithms", "data structures", "git", "agile", "system design", "microservices", "rest apis"],
        "Web Developer": ["html", "css", "javascript", "react", "node.js", "typescript", "rest apis", "responsive design", "git", "webpack"],
        "DevOps Engineer": ["docker", "kubernetes", "aws", "terraform", "jenkins", "ci/cd", "linux", "ansible", "monitoring", "bash"],
        "Data Analyst": ["sql", "excel", "python", "tableau", "power bi", "statistics", "data visualization", "data cleaning", "reporting"],
        "ML Engineer": ["python", "tensorflow", "pytorch", "mlops", "docker", "kubernetes", "feature engineering", "model deployment", "aws sagemaker"],
        "Backend Developer": ["java", "python", "node.js", "sql", "postgresql", "mongodb", "rest apis", "microservices", "redis", "rabbitmq"],
        "Frontend Developer": ["javascript", "react", "vue", "angular", "html", "css", "typescript", "webpack", "sass", "responsive design"],
        "Mobile Developer": ["kotlin", "swift", "react native", "flutter", "java", "rest apis", "firebase", "ui/ux", "android", "ios"],
        "QA Engineer": ["selenium", "python", "java", "test automation", "jira", "agile", "api testing", "performance testing", "cypress"],
        "Project Manager": ["agile", "scrum", "jira", "stakeholder management", "risk management", "budgeting", "communication", "leadership", "ms project"],
        "Business Analyst": ["sql", "excel", "requirements gathering", "data analysis", "jira", "stakeholder management", "uml", "power bi", "communication"],
        "UX Designer": ["figma", "sketch", "adobe xd", "user research", "wireframing", "prototyping", "usability testing", "html", "css"],
        "Cloud Architect": ["aws", "azure", "gcp", "terraform", "docker", "kubernetes", "networking", "security", "serverless", "microservices"],
        "Cybersecurity Analyst": ["network security", "penetration testing", "siem", "firewalls", "incident response", "python", "linux", "compliance", "vulnerability assessment"]
    }

    first_names = [
        "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda", 
        "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica", 
        "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Lisa", "Daniel", "Nancy", 
        "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra", "Donald", "Ashley", 
        "Steven", "Kimberly", "Paul", "Emily", "Andrew", "Donna", "Joshua", "Michelle", 
        "Kenneth", "Carol", "Kevin", "Amanda", "Brian", "Melissa", "George", "Deborah",
        "Edward", "Stephanie", "Ronald", "Rebecca", "Timothy", "Sharon", "Jason", "Laura"
    ]

    last_names = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", 
        "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", 
        "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson", 
        "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker", 
        "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores", 
        "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell", 
        "Carter", "Roberts", "Gomez", "Phillips", "Evans", "Turner", "Diaz", "Parker", "Cruz"
    ]

    all_skills = [skill for skills in roles_skills.values() for skill in skills]

    data = []
    
    # Generate ~13 resumes per role
    for role, skills_pool in roles_skills.items():
        for _ in range(14):
            first = random.choice(first_names)
            last = random.choice(last_names)
            name = f"{first} {last}"
            
            experience = random.randint(1, 15)
            
            # Select skills
            num_skills = random.randint(6, min(10, len(skills_pool)))
            selected_skills = random.sample(skills_pool, num_skills)
            
            # Cross-pollination
            if random.random() < 0.2:
                extra_skill = random.choice(all_skills)
                if extra_skill not in selected_skills:
                    selected_skills.append(extra_skill)
            
            skills_str = ", ".join(selected_skills)
            
            # Education
            education = random.choices(
                ["PhD", "Masters", "Bachelors", "Associate"], 
                weights=[10, 30, 50, 10]
            )[0]
            
            record = {
                "Name": name,
                "YearsExperience": experience,
                "Skills": skills_str,
                "Education": education,
                "JobRole": role
            }
            data.append(record)

    # Shuffle dataset
    random.shuffle(data)

    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "resume_dataset.json")
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)
        
    print(f"Generated {len(data)} resume records.")
    print(f"Saved dataset to {output_path}")

if __name__ == "__main__":
    generate_seed_data()
