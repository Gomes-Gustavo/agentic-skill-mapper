import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Note: The first time this is run, it will download the model.
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_top_skills_by_category(dataframe: pd.DataFrame, category: str, top_n: int = 30) -> list:
    """
    Filters a dataframe by job category, aggregates all skills, normalizes them
    semantically using embeddings and clustering, and returns the top N.
    """
    df_filtered = dataframe[dataframe['Category'].str.contains(category, case=False, na=False)]
    if df_filtered.empty:
        return []

    tech_skills = [skill for sublist in df_filtered['extracted_technical_skills'] for skill in sublist if skill]
    soft_skills = [skill for sublist in df_filtered['extracted_soft_skills'] for skill in sublist if skill]
    all_skills_raw = tech_skills + soft_skills

    unique_skills = sorted(list(set(skill.strip().lower() for skill in all_skills_raw)))
    if not unique_skills:
        return []

    # Generate embeddings and cluster
    skill_embeddings = model.encode(unique_skills, convert_to_tensor=True)
    clusters = util.community_detection(skill_embeddings, min_community_size=2, threshold=0.85)

    # Auto-generate normalization map
    skill_mapping = {}
    for cluster in clusters:
        cluster_skills = [unique_skills[skill_id] for skill_id in cluster]
        canonical_name = min(cluster_skills, key=len)
        for skill in cluster_skills:
            skill_mapping[skill] = canonical_name
            
    # Apply normalization and calculate final frequencies
    normalized_skills = [skill.strip().lower() for skill in all_skills_raw]
    final_skills = [skill_mapping.get(skill, skill) for skill in normalized_skills]
    
    top_skills = pd.Series(final_skills).value_counts().head(top_n)
    return top_skills.index.tolist()