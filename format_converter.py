"""Format converter for different evaluation dataset formats"""

import json
from typing import Dict, Any, List, Optional
from pathlib import Path
import pandas as pd


class FormatConverter:
    """Convert between different evaluation dataset formats"""
    
    def __init__(self):
        self.supported_formats = [
            "ragas_standard",
            "simple_qa",
            "evaluation_ready"
        ]
    
    def convert_to_ragas_format(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Convert any dataset to RAGAS evaluation format"""
        
        if not isinstance(dataset, dict) or "questions" not in dataset:
            raise ValueError("Dataset must contain 'questions' key")
        
        ragas_questions = []
        
        for question in dataset["questions"]:
            # Handle different input formats
            if isinstance(question, dict):
                ragas_item = {
                    "user_input": question.get("user_input", question.get("question", "")),
                    "retrieved_contexts": question.get("retrieved_contexts", question.get("context", [])),
                    "reference": question.get("reference", question.get("expected_answer", "")),
                    "response": question.get("response", "")  # Will be filled during evaluation
                }
                
                # Ensure retrieved_contexts is a list
                if isinstance(ragas_item["retrieved_contexts"], str):
                    ragas_item["retrieved_contexts"] = [ragas_item["retrieved_contexts"]]
                
                ragas_questions.append(ragas_item)
        
        return {
            "dataset_info": dataset.get("dataset_info", dataset.get("testset_info", {})),
            "questions": ragas_questions
        }
    
    def convert_simple_qa_to_evaluation_ready(self, qa_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Convert simple question-answer pairs to evaluation-ready format"""
        
        evaluation_questions = []
        
        for i, qa_item in enumerate(qa_data):
            if isinstance(qa_item, dict):
                evaluation_question = {
                    "question_id": f"Q{i+1:03d}",
                    "user_input": qa_item.get("question", ""),
                    "retrieved_contexts": [qa_item.get("context", "")],
                    "reference": qa_item.get("answer", qa_item.get("expected_answer", "")),
                    "question_type": "simple",
                    "difficulty": "medium",
                    "evaluation_criteria": ["accuracy", "completeness"],
                    "answerable": True,
                    "metadata": {
                        "generation_method": "simple_qa_conversion",
                        "original_index": i
                    }
                }
                evaluation_questions.append(evaluation_question)
        
        return {
            "dataset_info": {
                "total_questions": len(evaluation_questions),
                "original_format": "simple_qa",
                "conversion_method": "simple_to_evaluation_ready"
            },
            "questions": evaluation_questions
        }
    
    def export_to_csv(self, dataset: Dict[str, Any], output_path: str) -> None:
        """Export dataset to CSV format for easy viewing"""
        
        if "questions" not in dataset:
            raise ValueError("Dataset must contain 'questions' key")
        
        rows = []
        for question in dataset["questions"]:
            row = {
                "question_id": question.get("question_id", ""),
                "question": question.get("user_input", ""),
                "context": " | ".join(question.get("retrieved_contexts", [])),
                "expected_answer": question.get("reference", ""),
                "question_type": question.get("question_type", ""),
                "difficulty": question.get("difficulty", ""),
                "answerable": question.get("answerable", True),
                "evaluation_criteria": ", ".join(question.get("evaluation_criteria", []))
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"Dataset exported to CSV: {output_path}")
    
    def _count_question_types(self, questions: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count question types in dataset"""
        type_counts = {}
        for question in questions:
            q_type = question.get("question_type", "unknown")
            type_counts[q_type] = type_counts.get(q_type, 0) + 1
        return type_counts
    
    def _count_difficulty_levels(self, questions: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count difficulty levels in dataset"""
        difficulty_counts = {}
        for question in questions:
            difficulty = question.get("difficulty", "unknown")
            difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
        return difficulty_counts
    
    def convert_format(self, dataset: Dict[str, Any], target_format: str) -> Dict[str, Any]:
        """Generic format conversion method"""
        
        if target_format not in self.supported_formats:
            raise ValueError(f"Unsupported format: {target_format}")
        
        if target_format == "ragas_standard":
            return self.convert_to_ragas_format(dataset)
        elif target_format == "simple_qa":
            return self.convert_simple_qa_to_evaluation_ready(dataset.get("questions", []))
        elif target_format == "evaluation_ready":
            return dataset  # Assume already in evaluation ready format
        else:
            return dataset
    
    def validate_dataset(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Validate dataset format and return validation report"""
        
        validation_report = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "statistics": {}
        }
        
        # Check basic structure
        if not isinstance(dataset, dict):
            validation_report["valid"] = False
            validation_report["errors"].append("Dataset must be a dictionary")
            return validation_report
        
        if "questions" not in dataset:
            validation_report["valid"] = False
            validation_report["errors"].append("Dataset must contain 'questions' key")
            return validation_report
        
        # Check questions format
        questions = dataset["questions"]
        if not isinstance(questions, list):
            validation_report["valid"] = False
            validation_report["errors"].append("Questions must be a list")
            return validation_report
        
        # Validate individual questions
        required_fields = ["user_input", "retrieved_contexts", "reference"]
        empty_questions = 0
        missing_fields = []
        
        for i, question in enumerate(questions):
            if not isinstance(question, dict):
                validation_report["errors"].append(f"Question {i+1} must be a dictionary")
                continue
            
            # Check required fields
            for field in required_fields:
                if field not in question:
                    missing_fields.append(f"Question {i+1} missing field: {field}")
                elif not question[field]:
                    empty_questions += 1
        
        if missing_fields:
            validation_report["valid"] = False
            validation_report["errors"].extend(missing_fields)
        
        if empty_questions > 0:
            validation_report["warnings"].append(f"{empty_questions} questions have empty required fields")
        
        # Calculate statistics
        validation_report["statistics"] = {
            "total_questions": len(questions),
            "question_types": self._count_question_types(questions),
            "difficulty_levels": self._count_difficulty_levels(questions),
            "answerable_questions": len([q for q in questions if q.get("answerable", True)]),
            "unanswerable_questions": len([q for q in questions if not q.get("answerable", True)])
        }
        
        return validation_report


# Example usage
def main():
    converter = FormatConverter()
    
    # Example: Convert to RAGAS format
    # ragas_data = converter.convert_to_ragas_format(dataset)
    
    # Example: Export to CSV
    # converter.export_to_csv(dataset, "evaluation_dataset.csv")
    
    print("Format converter ready for use!")


if __name__ == "__main__":
    main()