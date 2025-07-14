import json
from typing import Dict, Any, List
from pathlib import Path
from datetime import datetime
from config import Config
from ragas_evaluator import RAGASEvaluator

class Evaluator:
    def __init__(self):
        self.results_dir = Path(Config.RESULTS_DIR)
        self.results_dir.mkdir(exist_ok=True)
        self.ragas_evaluator = RAGASEvaluator()
    
    def evaluate_api_results(self, api_results: Dict[str, Any], output_prefix: str = "evaluation", include_ragas: bool = True) -> Dict[str, Any]:
        results = api_results['results']
        test_info = api_results['test_info']
        
        evaluation = {
            "evaluation_info": {
                "total_questions": test_info['total_questions'],
                "successful_requests": test_info['successful_requests'],
                "evaluation_date": datetime.now().isoformat()
            },
            "performance_metrics": {
                "success_rate": test_info['success_rate'],
                "average_response_time": test_info['average_response_time'],
                "error_rate": test_info['failed_requests'] / test_info['total_questions']
            },
            "question_analysis": self._analyze_questions(results),
            "detailed_results": results
        }
        
        # add RAGAS evaluation if requested and available
        if include_ragas and self.ragas_evaluator.is_available():
            print("Running RAGAS evaluation...")
            ragas_results = self.ragas_evaluator.evaluate_with_ragas(api_results, output_prefix)
            if "error" not in ragas_results:
                evaluation["ragas_evaluation"] = ragas_results["ragas_scores"]
                evaluation["ragas_info"] = ragas_results["evaluation_info"]
            else:
                print(f"RAGAS evaluation failed: {ragas_results['error']}")
        elif include_ragas:
            print("RAGAS evaluation skipped - not available")
        
        output_file = self.results_dir / f"{output_prefix}_evaluation.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation, f, indent=2, ensure_ascii=False)
        
        print(f"Evaluation saved to: {output_file}")
        return evaluation
    
    def _analyze_questions(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        successful_results = [r for r in results if r['api_status'] == 'success']
        
        if not successful_results:
            return {"error": "No successful API responses to analyze"}
        
        response_times = [r['response_time'] for r in successful_results if r['response_time']]
        answer_lengths = [len(r['actual_answer']) for r in successful_results]
        
        analysis = {
            "successful_questions": len(successful_results),
            "response_time_stats": {
                "min": min(response_times) if response_times else 0,
                "max": max(response_times) if response_times else 0,
                "average": sum(response_times) / len(response_times) if response_times else 0
            },
            "answer_length_stats": {
                "min": min(answer_lengths),
                "max": max(answer_lengths), 
                "average": sum(answer_lengths) / len(answer_lengths)
            }
        }
        
        return analysis
    
    def generate_evaluation_report(self, evaluation: Dict[str, Any]) -> str:
        report = []
        report.append("API Evaluation Report")
        report.append("=" * 40)
        report.append("")
        
        eval_info = evaluation['evaluation_info']
        metrics = evaluation['performance_metrics']
        analysis = evaluation['question_analysis']
        
        report.append(f"Total Questions: {eval_info['total_questions']}")
        report.append(f"Successful Requests: {eval_info['successful_requests']}")
        report.append(f"Success Rate: {metrics['success_rate']:.2%}")
        report.append(f"Average Response Time: {metrics['average_response_time']:.2f}s")
        report.append(f"Error Rate: {metrics['error_rate']:.2%}")
        report.append("")
        
        if 'error' not in analysis:
            report.append("Response Analysis:")
            report.append(f"  Response Time - Min: {analysis['response_time_stats']['min']:.2f}s, Max: {analysis['response_time_stats']['max']:.2f}s")
            report.append(f"  Answer Length - Min: {analysis['answer_length_stats']['min']}, Max: {analysis['answer_length_stats']['max']}, Avg: {analysis['answer_length_stats']['average']:.0f}")
        
        # add RAGAS scores if available
        if "ragas_evaluation" in evaluation:
            report.append("")
            report.append("RAGAS Quality Scores:")
            report.append("-" * 20)
            ragas_scores = evaluation["ragas_evaluation"]
            for metric, score in ragas_scores.items():
                report.append(f"  {metric}: {score:.3f}")
            
            avg_score = sum(ragas_scores.values()) / len(ragas_scores)
            report.append(f"  Overall Quality: {avg_score:.3f}")
        
        return "\n".join(report)
    
    def compare_datasets(self, results_files: List[str], output_prefix: str = "comparison") -> Dict[str, Any]:
        comparison_data = {}
        
        for i, file_path in enumerate(results_files):
            dataset_name = f"dataset_{i+1}"
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if 'test_info' in data:
                    comparison_data[dataset_name] = {
                        "source_file": file_path,
                        "success_rate": data['test_info']['success_rate'],
                        "average_response_time": data['test_info']['average_response_time'],
                        "total_questions": data['test_info']['total_questions']
                    }
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        comparison = {
            "comparison_info": {
                "datasets_compared": len(comparison_data),
                "comparison_date": datetime.now().isoformat()
            },
            "dataset_performance": comparison_data,
            "summary": self._create_comparison_summary(comparison_data)
        }
        
        output_file = self.results_dir / f"{output_prefix}_comparison.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        
        print(f"Comparison saved to: {output_file}")
        return comparison
    
    def _create_comparison_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not data:
            return {}
        
        success_rates = [d['success_rate'] for d in data.values()]
        response_times = [d['average_response_time'] for d in data.values()]
        
        best_success_rate = max(data.items(), key=lambda x: x[1]['success_rate'])
        best_response_time = min(data.items(), key=lambda x: x[1]['average_response_time'])
        
        return {
            "best_success_rate": {
                "dataset": best_success_rate[0],
                "rate": best_success_rate[1]['success_rate']
            },
            "best_response_time": {
                "dataset": best_response_time[0], 
                "time": best_response_time[1]['average_response_time']
            },
            "overall_stats": {
                "average_success_rate": sum(success_rates) / len(success_rates),
                "average_response_time": sum(response_times) / len(response_times)
            }
        }