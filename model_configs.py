# ============================================================================
# DEFAULT CONFIGURATION (used by library if no config provided)
# ============================================================================

DEFAULT_CONFIG = {
    'DATA_PATH': "IMDB Dataset.csv",
    'OUTPUT_DIR': "./sentiment_deep_models_output",
    'CLASSICAL_MODELS_DIR': "./sentiment_student1_output",
    'MAX_FEATURES': 20000,  # Vocabulary size
    'MAX_LEN': 200,         # Maximum sequence length
    'EMBEDDING_DIM': 128,   # Embedding dimension
    'BATCH_SIZE': 64,
    'EPOCHS': 10,
    'VALIDATION_SPLIT': 0.2,
    'RANDOM_STATE': 42
}

# ============================================================================
# CONFIGURATION PRESETS
# ============================================================================

# Quick testing configuration (fast training)
QUICK_TEST_CONFIG = {
    'DATA_PATH': "IMDB Dataset.csv",
    'OUTPUT_DIR': "./sentiment_deep_models_output",
    'CLASSICAL_MODELS_DIR': "./sentiment_student1_output",
    'MAX_FEATURES': 5000,   # Small vocabulary for speed
    'MAX_LEN': 50,          # Short sequences
    'EMBEDDING_DIM': 32,    # Small embeddings
    'BATCH_SIZE': 128,
    'EPOCHS': 5,            # Very few epochs
    'VALIDATION_SPLIT': 0.2,
    'RANDOM_STATE': 42,
    'USE_GLOVE': False,     # Disable GloVe for quick testing
    'GLOVE_PATH': None
}

# Standard configuration (balanced performance/speed)
STANDARD_CONFIG = {
    'DATA_PATH': "IMDB Dataset.csv",
    'OUTPUT_DIR': "./sentiment_deep_models_output", 
    'CLASSICAL_MODELS_DIR': "./sentiment_student1_output",
    'MAX_FEATURES': 20000,
    'MAX_LEN': 200,
    'EMBEDDING_DIM': 128,
    'BATCH_SIZE': 64,
    'EPOCHS': 10,
    'VALIDATION_SPLIT': 0.2,
    'RANDOM_STATE': 42,
    'USE_GLOVE': False,     
    'GLOVE_PATH': None
}

# High performance configuration (longer training, better results)
HIGH_PERFORMANCE_CONFIG = {
    'DATA_PATH': "IMDB Dataset.csv",
    'OUTPUT_DIR': "./sentiment_deep_models_output",
    'CLASSICAL_MODELS_DIR': "./sentiment_student1_output", 
    'MAX_FEATURES': 50000,  # Large vocabulary
    'MAX_LEN': 400,         # Long sequences
    'EMBEDDING_DIM': 256,   # Large embeddings
    'BATCH_SIZE': 32,       # Smaller batches for stability
    'EPOCHS': 20,           # More training
    'VALIDATION_SPLIT': 0.2,
    'RANDOM_STATE': 42,
    'USE_GLOVE': False,
    'GLOVE_PATH': None
}

# Transformer-optimized configuration
TRANSFORMER_CONFIG = {
    'DATA_PATH': "IMDB Dataset.csv",
    'OUTPUT_DIR': "./sentiment_deep_models_output",
    'CLASSICAL_MODELS_DIR': "./sentiment_student1_output",
    'MAX_FEATURES': 30000,
    'MAX_LEN': 256,         # Good for attention mechanisms
    'EMBEDDING_DIM': 100,   # Should be ivisible by number of attention heads
    'BATCH_SIZE': 32,       # Smaller for transformer memory requirements
    'EPOCHS': 15,
    'VALIDATION_SPLIT': 0.2,
    'RANDOM_STATE': 42,
    'USE_GLOVE': False,
    'GLOVE_PATH': None
}

# Memory-efficient configuration (for limited GPU memory)
CUSTOM_CONFIG = {
    'DATA_PATH': "IMDB Dataset.csv",
    'OUTPUT_DIR': "./sentiment_deep_models_output", 
    'CLASSICAL_MODELS_DIR': "./sentiment_student1_output",
    'MAX_FEATURES': 10000,
    'MAX_LEN': 200,
    'EMBEDDING_DIM': 256,
    'BATCH_SIZE': 32,
    'EPOCHS': 10,
    'VALIDATION_SPLIT': 0.2,
    'RANDOM_STATE': 42
}
