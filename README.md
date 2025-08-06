# Biomedical RAG API Testing Framework

Testing framework for biomedical RAG applications via FastAPI endpoints. Generate questions from study abstracts and evaluate API performance.

## Features

- Generate test questions from biomedical abstracts CSV
- Test DugBot/BDCBot APIs via HTTP endpoints  
- Performance evaluation and reporting
- RAGAS evaluation for answer quality (context recall, faithfulness, etc.)
- Question types: factual, analytical, comparative, unanswerable

## Setup

```bash
pip install -r requirements.txt
```
### Configuration 

To Configure the application,  please copy [.env-template](.env-template) to a new `.env` file and modify the appropriate variables. 
When the program starts it will load in .env file automatically.

### Process Flow

1. **Load Abstracts** → CSV with study abstracts 
2. **Generate Questions** → 4 types using configurable LLM (factual, analytical, comparative, unanswerable)  
3. **Test API** → Send questions to DUGBot endpoint, track performance
4. **Store Results** → Raw API responses with timing and status
5. **Compute Basic Metrics** → Success rate, response time, error analysis
6. **Run RAGAS** → Answer quality evaluation using OpenAI or Ollama (faithfulness, context recall, etc.)
7. **Generate Report** → Combined performance + quality assessment

## Usage

### Generate Questions
```bash
# From JSON file 
python main.py generate documents.json -o questions.json -n 40

# From CSV file 
python main.py generate abstracts.csv -o questions.json -n 40
```

### Test API
```bash
python main.py test questions.json -o test_results -r
```

### Evaluate Results
```bash
python main.py evaluate test_results_results.json -o evaluation
python main.py evaluate test_results_results.json --with-ragas  # enable RAGAS evaluation
```

### Compare Multiple Tests
```bash
python main.py compare test1_results.json test2_results.json -o comparison
```

## File Structure

```
testing_framework/
├── main.py              # CLI interface
├── config.py            # Configuration 
├── qa_generator.py      # Question generation from abstracts
├── api_tester.py        # API testing (DugBot and BDCBot)
├── evaluator.py         # Results evaluation
├── data_processor.py    # Data loading/saving
├── format_converter.py  # Dataset format conversion
├── requirements.txt     # Dependencies
└── results/             # Generated results
```

## Input Format

### JSON Format 
```json
[
  {
    "ID": "study_001",
    "CONTEXT": "This study examines the ..."
  },
  {
    "ID": "study_002", 
    "CONTEXT": "The C4R study ..."
  }
]
```


## Output

- Question datasets in JSON format
- API test results with response times and success rates
- Evaluation reports with performance metrics
- Comparison analysis across multiple tests (this is just an idea to track the performance for a given period)


Question generation uses configurable LLM (default: Ollama/Llama) for specific biomedical question types.

### JSON Input Examples

**Simple document list:**
```json
[
  {"ID": "doc1", "CONTEXT": "The C4R studies are ..."},
  {"ID": "doc2", "CONTEXT": "The Covid studies..."}
]
```

**With metadata wrapper:**
```json
{
  "documents": [
    {"ID": "doc1", "CONTEXT": "The C4R studies are..."},
    {"ID": "doc2", "CONTEXT": "The Covid studies..."}
  ]
}
```

## LLM Configuration

### Question Generation
- **Default**: Ollama with Llama 3.1 OR "gemma3:12b"
- **Configurable**: Any Ollama-compatible model that are limited to GPU resources on Sterling (Cluster)
- **Purpose**: Generate test questions from abstracts

### RAGAS Evaluation
- **Option 1**: OpenAI GPT-4 
- **Option 2**: Ollama with Llama 3.1/Gemma3:12b 
- **Purpose**: Evaluate answer quality with RAGAS metrics

### Configuration Examples

**OpenAI for RAGAS :**
```python
RAGAS_EVALUATION_LLM_PROVIDER = "openai"
RAGAS_EVALUATION_LLM_API_KEY = "your-openai-key"
```

**Ollama for RAGAS :**
```python
RAGAS_EVALUATION_LLM_PROVIDER = "ollama"
RAGAS_EVALUATION_LLM_MODEL = "llama3.1:latest" OR "gemma3:12b"
```