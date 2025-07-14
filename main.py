#!/usr/bin/env python3

import argparse
from config import Config
from qa_generator import QAGenerator
from api_tester import APITester
from evaluator import Evaluator
from data_processor import DataProcessor

def generate_questions(args):
    generator = QAGenerator()
    generator.generate_from_file(args.input_file, args.output, args.num_questions)

def test_api(args):
    processor = DataProcessor()
    dataset = processor.load_dataset(args.dataset)
    
    tester = APITester()
    results = tester.test_dataset(dataset, args.output_prefix)
    
    if args.report:
        report = tester.generate_report(results, f"{args.output_prefix}_report.txt")
        print("\n" + report)

def evaluate_results(args):
    evaluator = Evaluator()
    
    processor = DataProcessor()
    api_results = processor.load_dataset(args.results_file)
    
    include_ragas = args.with_ragas
    evaluation = evaluator.evaluate_api_results(api_results, args.output_prefix, include_ragas)
    report = evaluator.generate_evaluation_report(evaluation)
    
    print("\n" + report)

def compare_results(args):
    evaluator = Evaluator()
    comparison = evaluator.compare_datasets(args.results_files, args.output_prefix)
    
    print(f"Compared {len(args.results_files)} datasets")
    print(f"Best success rate: {comparison['summary']['best_success_rate']['dataset']} ({comparison['summary']['best_success_rate']['rate']:.2%})")

def main():
    parser = argparse.ArgumentParser(description="Biomedical API Testing Framework")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # generate questions 
    gen_parser = subparsers.add_parser('generate', help='Generate questions from CSV or JSON file')
    gen_parser.add_argument('input_file', help='Input file with abstracts (CSV or JSON)')
    gen_parser.add_argument('-o', '--output', default='generated_questions.json', help='Output JSON file')
    gen_parser.add_argument('-n', '--num_questions', type=int, default=Config.DEFAULT_NUM_QUESTIONS, help='Number of questions to generate')
    
    # test API 
    test_parser = subparsers.add_parser('test', help='Test API with questions')
    test_parser.add_argument('dataset', help='Questions dataset JSON file')
    test_parser.add_argument('-o', '--output_prefix', default='api_test', help='Output file prefix')
    test_parser.add_argument('-r', '--report', action='store_true', help='Generate text report')
    
    # evaluate results 
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate API test results')
    eval_parser.add_argument('results_file', help='API test results JSON file')
    eval_parser.add_argument('-o', '--output_prefix', default='evaluation', help='Output file prefix')
    eval_parser.add_argument('--with-ragas', action='store_true', help='Enable RAGAS evaluation (may be slow)')
    
    # compare results 
    comp_parser = subparsers.add_parser('compare', help='Compare multiple test results')
    comp_parser.add_argument('results_files', nargs='+', help='Multiple results JSON files')
    comp_parser.add_argument('-o', '--output_prefix', default='comparison', help='Output file prefix')
    
    args = parser.parse_args()
    
    if args.command == 'generate':
        generate_questions(args)
    elif args.command == 'test':
        test_api(args)
    elif args.command == 'evaluate':
        evaluate_results(args)
    elif args.command == 'compare':
        compare_results(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()