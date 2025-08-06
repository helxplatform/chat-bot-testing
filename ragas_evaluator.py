import os
from typing import Dict, Any, List
from pathlib import Path
from datetime import datetime

try:
    from ragas import evaluate
    from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness, AnswerRelevancy
    from ragas.llms import LangchainLLMWrapper
    from ragas import EvaluationDataset
    from langchain_openai import ChatOpenAI
    from langchain_ollama import OllamaLLM
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

from config import Config

class RAGASEvaluator:
    def __init__(self):
        if not RAGAS_AVAILABLE:
            self.evaluator_llm = None
            return
            
        self.results_dir = Path(Config.RESULTS_DIR)
        self.results_dir.mkdir(exist_ok=True)
        
        # get evaluation LLM config
        eval_config = Config.get_ragas_evaluation_config()
        
        # initialize evaluator LLM 
        try:
            if eval_config["provider"] == "openai":
                if eval_config["api_key"]:
                    os.environ["OPENAI_API_KEY"] = eval_config["api_key"]
                self.evaluator_llm = LangchainLLMWrapper(
                    ChatOpenAI(model=eval_config["model"], temperature=eval_config["temperature"])
                )
            elif eval_config["provider"] == "ollama":
                # Set a dummy OpenAI key to prevent RAGAS from failing
                if "OPENAI_API_KEY" not in os.environ:
                    os.environ["OPENAI_API_KEY"] = "dummy-key-for-ollama"
                self.evaluator_llm = LangchainLLMWrapper(
                    OllamaLLM(
                        model=eval_config["model"],
                        base_url=eval_config["base_url"],
                        temperature=eval_config["temperature"]
                    )
                )
            else:
                raise ValueError(f"Unsupported provider: {eval_config['provider']}")
        except Exception as e:
            print(f"Warning: Could not initialize evaluation LLM: {e}")
            self.evaluator_llm = None
    
    def is_available(self) -> bool:
        return RAGAS_AVAILABLE and self.evaluator_llm is not None
    
    def convert_api_results_to_ragas_format(self, api_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert API test results to RAGAS evaluation format"""
        ragas_data = []
        
        for result in api_results['results']:
            if result['api_status'] == 'success':
                # extract context from question data if available
                context = result.get('retrieved_contexts', [])
                
                ragas_item = {
                    "user_input": result['question'],
                    "reference_contexts": context,
                    "reference": result['expected_answer'],
                    "response": result['actual_answer']
                }
                ragas_data.append(ragas_item)
        
        return ragas_data
    
    def evaluate_with_ragas(self, api_results: Dict[str, Any], output_prefix: str = "ragas_evaluation") -> Dict[str, Any]:
        """Evaluate API results using RAGAS metrics"""
        if not self.is_available():
            return {"error": "RAGAS evaluation not available"}
        
        # convert to RAGAS format
        ragas_data = self.convert_api_results_to_ragas_format(api_results)
        
        if not ragas_data:
            return {"error": "No successful API responses to evaluate"}
        
        # Limit evaluation to first 10 samples for performance
        if len(ragas_data) > 10:
            print(f"Limiting RAGAS evaluation to first 10 samples (out of {len(ragas_data)}) for performance...")
            ragas_data = ragas_data[:10]
        else:
            print(f"Evaluating {len(ragas_data)} responses with RAGAS...")
        
        try:
            # create evaluation dataset
            evaluation_dataset = EvaluationDataset.from_list(ragas_data)
            
            # define metrics
            metrics = []
            if "context_recall" in Config.RAGAS_METRICS:
                metrics.append(LLMContextRecall(llm=self.evaluator_llm))
            if "faithfulness" in Config.RAGAS_METRICS:
                metrics.append(Faithfulness(llm=self.evaluator_llm))
            if "factual_correctness" in Config.RAGAS_METRICS:
                metrics.append(FactualCorrectness(llm=self.evaluator_llm))
            if "answer_relevancy" in Config.RAGAS_METRICS:
                metrics.append(AnswerRelevancy(llm=self.evaluator_llm))
            
            # run evaluation
            result = evaluate(
                dataset=evaluation_dataset,
                metrics=metrics
            )
            
            # convert to dictionary
            ragas_scores = {}
            for metric_score in result.scores:
                ragas_scores.update(metric_score)
            
            # create comprehensive evaluation
            evaluation = {
                "evaluation_info": {
                    "total_responses_evaluated": len(ragas_data),
                    "evaluation_date": datetime.now().isoformat(),
                    "metrics_used": [m.name for m in metrics]
                },
                "ragas_scores": ragas_scores,
                "api_performance": api_results['test_info'],
                "detailed_results": ragas_data
            }
            
            # save results
            output_file = self.results_dir / f"{output_prefix}_ragas.json"
            import json
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(evaluation, f, indent=2, ensure_ascii=False)
            
            print(f"RAGAS evaluation saved to: {output_file}")
            print("RAGAS Scores:")
            for metric, score in ragas_scores.items():
                print(f"  {metric}: {score:.3f}")
            
            return evaluation
            
        except Exception as e:
            print(f"Error during RAGAS evaluation: {e}")
            return {"error": f"RAGAS evaluation failed: {str(e)}"}
    
    def generate_ragas_report(self, evaluation: Dict[str, Any]) -> str:
        """Generate human-readable RAGAS evaluation report"""
        if "error" in evaluation:
            return f"RAGAS Evaluation Error: {evaluation['error']}"
        
        report = []
        report.append("RAGAS Evaluation Report")
        report.append("=" * 40)
        report.append("")
        
        eval_info = evaluation['evaluation_info']
        ragas_scores = evaluation['ragas_scores']
        api_performance = evaluation['api_performance']
        
        report.append(f"Evaluation Date: {eval_info['evaluation_date']}")
        report.append(f"Responses Evaluated: {eval_info['total_responses_evaluated']}")
        report.append(f"API Success Rate: {api_performance['success_rate']:.2%}")
        report.append(f"Average Response Time: {api_performance['average_response_time']:.2f}s")
        report.append("")
        
        report.append("RAGAS Quality Scores:")
        report.append("-" * 20)
        for metric, score in ragas_scores.items():
            report.append(f"{metric}: {score:.3f}")
        
        # overall assessment
        avg_score = sum(ragas_scores.values()) / len(ragas_scores)
        report.append("")
        report.append(f"Overall Quality Score: {avg_score:.3f}")
        
        if avg_score >= 0.8:
            assessment = "Excellent"
        elif avg_score >= 0.7:
            assessment = "Good"
        elif avg_score >= 0.6:
            assessment = "Fair"
        else:
            assessment = "Needs Improvement"
        
        report.append(f"Assessment: {assessment}")
        
        return "\n".join(report)