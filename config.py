from dotenv import load_dotenv
import os
import json
# Load environment variables from a .env file if it exists.
# This line looks for a file named '.env' in the current directory or parent directories.
load_dotenv()

class Config:
    """
    Configuration class for the application.

    Reads settings from environment variables or a .env file.
    Provides default values for all settings.
    """

    # Helper function to read boolean values from environment variables
    @staticmethod
    def _get_bool(key, default_value):
        """Converts string environment variables to booleans."""
        return os.getenv(key, str(default_value)).lower() in ('true', '1', 't')

    # --- Question generation LLM settings ---
    QUESTION_GENERATION_LLM_URL = os.getenv("QUESTION_GENERATION_LLM_URL", "http://localhost:11434")
    QUESTION_GENERATION_LLM_MODEL = os.getenv("QUESTION_GENERATION_LLM_MODEL", "gemma3:12b")
    QUESTION_GENERATION_TEMPERATURE = int(os.getenv("QUESTION_GENERATION_TEMPERATURE", 0))

    # --- API settings for RAG tool (DugBot/BdcBot) ---
    API_BASE_URL = os.getenv("API_BASE_URL", "https://search-dev.biodatacatalyst.renci.org")
    API_ENDPOINT = os.getenv("API_ENDPOINT", "/agent/invoke_test")
    API_TIMEOUT = int(os.getenv("API_TIMEOUT", 120))
    API_DELAY = int(os.getenv("API_DELAY", 1))

    # --- Output directories ---
    RESULTS_DIR = os.getenv("RESULTS_DIR", "results")
    DATASETS_DIR = os.getenv("DATASETS_DIR", "datasets")

    # --- Question generation settings ---
    DEFAULT_NUM_QUESTIONS = int(os.getenv("DEFAULT_NUM_QUESTIONS", 400))

    # For complex types like dictionaries, it's best to store them as a JSON string
    # in the environment variable. We provide a default dictionary and then load
    # the value from the environment if it exists.
    _default_question_types = {
        "factual": 100,
        "analytical": 100,
        "comparative": 100,
        "unanswerable": 100
    }
    QUESTION_TYPES = json.loads(
        os.getenv("QUESTION_TYPES", json.dumps(_default_question_types))
    )

    # --- CSV settings ---
    CSV_ACCESSION_COLUMN = os.getenv("CSV_ACCESSION_COLUMN", "Accession")
    CSV_DESCRIPTION_COLUMN = os.getenv("CSV_DESCRIPTION_COLUMN", "Description")

    # --- RAGAS evaluation LLM settings ---
    RAGAS_EVALUATION_LLM_PROVIDER = os.getenv("RAGAS_EVALUATION_LLM_PROVIDER", "openai")
    RAGAS_EVALUATION_LLM_API_KEY = os.getenv("RAGAS_EVALUATION_LLM_API_KEY", "EMPTY")
    RAGAS_EVALUATION_LLM_URL = os.getenv("RAGAS_EVALUATION_LLM_URL", "http://localhost:9091/v1")
    RAGAS_EVALUATION_LLM_MODEL = os.getenv("RAGAS_EVALUATION_LLM_MODEL", "google/gemma-3-12b-it")
    RAGAS_EVALUATION_TEMPERATURE = int(os.getenv("RAGAS_EVALUATION_TEMPERATURE", 0))

    # For lists, we can store them as a comma-separated string.
    _default_ragas_metrics = [
        # "context_recall",
        # "faithfulness",
        "factual_correctness",
        # "answer_relevancy"
    ]
    # We get the environment variable, fall back to the default joined list, then split by comma.
    _raw_metrics = os.getenv("RAGAS_METRICS", ",".join(_default_ragas_metrics))
    RAGAS_METRICS = [metric.strip() for metric in _raw_metrics.split(',') if metric.strip()]

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
