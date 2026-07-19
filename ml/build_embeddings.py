import os
import sys
import json

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.embeddings import EmbeddingEngine
from app.utils.helpers import setup_logger

log = setup_logger(__name__)

def build_index():
    log.info("Starting FAISS index building process...")
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(root_dir, "data", "raw", "resume_dataset.json")
    
    with open(data_path, "r") as f:
        data = json.load(f)
        
    log.info(f"Loaded {len(data)} records for indexing.")
    
    embedding_engine = EmbeddingEngine()
    
    texts = []
    metadata_list = []
    
    for i, record in enumerate(data):
        # Construct composite text
        text = f"{record['Name']} | {record['JobRole']} | Skills: {record['Skills']} | Education: {record['Education']} | Experience: {record['YearsExperience']} years"
        texts.append(text)
        
        # Build metadata
        meta = {
            "name": record["Name"],
            "job_role": record["JobRole"],
            "skills": record["Skills"],
            "education": record["Education"],
            "experience": record["YearsExperience"],
            "original_index": i
        }
        metadata_list.append(meta)
        
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(data)} records...")
            
    log.info("Building and saving index via EmbeddingEngine...")
    embedding_engine.build_index(texts, metadata_list)
    embedding_engine.save_index()
    print("FAISS index building complete.")

if __name__ == "__main__":
    build_index()
