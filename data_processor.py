import csv
import json
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path

class DataProcessor:
    def __init__(self):
        pass
    
    def load_abstracts_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        file_path = Path(file_path)
        abstracts = []
        processed_base_accessions = set()
        
        if file_path.suffix == '.json':
            # handle JSON input with ID and CONTEXT keys
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # support both list of objects and objects with documents array
            documents = data if isinstance(data, list) else data.get('documents', [])
            
            for doc in documents:
                doc_id = doc.get('ID', doc.get('StudyId', ''))
                context = doc.get('CONTEXT', doc.get('Description', ''))
                
                # extract base ID for deduplication
                base_id = doc_id.split('.')[0] if '.' in doc_id else doc_id
                
                if base_id in processed_base_accessions:
                    continue
                    
                abstracts.append({
                    'accession': doc_id,
                    'abstract': context
                })
                processed_base_accessions.add(base_id)
        
        elif file_path.suffix == '.csv':
            # handle CSV input (backward compatibility)
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    from config import Config
                    accession = row.get(Config.CSV_ACCESSION_COLUMN, '')
                    # extract base accession for deduplication
                    base_accession = accession.split('.')[0] if '.' in accession else accession
                    
                    if base_accession in processed_base_accessions:
                        continue
                        
                    abstracts.append({
                        'accession': accession,
                        'abstract': row.get(Config.CSV_DESCRIPTION_COLUMN, '')
                    })
                    processed_base_accessions.add(base_accession)
        
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}. Use .json or .csv")
        
        return abstracts
    
    def load_dataset(self, file_path: str) -> Dict[str, Any]:
        file_path = Path(file_path)
        
        if file_path.suffix == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
            return {"questions": df.to_dict('records')}
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def save_dataset(self, data: Dict[str, Any], output_path: str) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)