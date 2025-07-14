class Config:
    # Question generation LLM settings
    QUESTION_GENERATION_LLM_URL = "http://localhost:11434"
    QUESTION_GENERATION_LLM_MODEL = "gemma3:12b"
    QUESTION_GENERATION_TEMPERATURE = 0
    
    # API settings for RAG tool (DubBot/BdcBot)
    API_BASE_URL = "http://localhost:8000"  # change to your API endpoint
    API_ENDPOINT = "/qv-app/invoke"         # change to your endpoint format
    API_TIMEOUT = 30                        # seconds
    API_DELAY = 1                          # seconds between requests
    
    # Output directories
    RESULTS_DIR = "results"
    DATASETS_DIR = "datasets"
    
    # Question generation settings
    DEFAULT_NUM_QUESTIONS = 400
    QUESTION_TYPES = {
        "factual": 100,
        "analytical": 100,
        "comparative": 100,
        "unanswerable": 100
    }
    
    # CSV settings
    CSV_ACCESSION_COLUMN = "Accession"
    CSV_DESCRIPTION_COLUMN = "Description"
    
    # RAGAS evaluation LLM settings (LLM will act as evaluator)
    RAGAS_EVALUATION_LLM_PROVIDER = "ollama"  # "openai" or "ollama"
    RAGAS_EVALUATION_LLM_API_KEY = None       # for OpenAI
    RAGAS_EVALUATION_LLM_URL = "http://localhost:11434"  # for Ollama
    RAGAS_EVALUATION_LLM_MODEL = "gemma3:12b"      # for Ollama
    RAGAS_EVALUATION_TEMPERATURE = 0.1
    RAGAS_METRICS = [
        "context_recall",
        "faithfulness", 
        "factual_correctness",
        "answer_relevancy"
    ]
    
    @classmethod
    def update_api_config(cls, base_url=None, endpoint=None, timeout=None, delay=None):
        """Update API configuration at runtime"""
        if base_url:
            cls.API_BASE_URL = base_url
        if endpoint:
            cls.API_ENDPOINT = endpoint
        if timeout:
            cls.API_TIMEOUT = timeout
        if delay:
            cls.API_DELAY = delay
    
    @classmethod
    def get_api_config(cls):
        """Get current API configuration"""
        return {
            "base_url": cls.API_BASE_URL,
            "query_endpoint": cls.API_ENDPOINT,
            "timeout": cls.API_TIMEOUT,
            "delay": cls.API_DELAY
        }
    
    @classmethod
    def get_question_generation_config(cls):
        """Get question generation LLM configuration"""
        return {
            "model": cls.QUESTION_GENERATION_LLM_MODEL,
            "base_url": cls.QUESTION_GENERATION_LLM_URL,
            "temperature": cls.QUESTION_GENERATION_TEMPERATURE
        }
    
    @classmethod
    def get_ragas_evaluation_config(cls):
        """Get RAGAS evaluation LLM configuration"""
        return {
            "provider": cls.RAGAS_EVALUATION_LLM_PROVIDER,
            "api_key": cls.RAGAS_EVALUATION_LLM_API_KEY,
            "base_url": cls.RAGAS_EVALUATION_LLM_URL,
            "model": cls.RAGAS_EVALUATION_LLM_MODEL,
            "temperature": cls.RAGAS_EVALUATION_TEMPERATURE
        }