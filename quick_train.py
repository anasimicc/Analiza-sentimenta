from deep_models import DeepSentimentAnalyzer
from model_configs import (
    QUICK_TEST_CONFIG, STANDARD_CONFIG, HIGH_PERFORMANCE_CONFIG, 
    TRANSFORMER_CONFIG, CUSTOM_CONFIG, DEFAULT_CONFIG
)

def quick_train(model_name='LSTM', experiment_name='default_lstm', config=None):
    """
    Quickly train a single model
    
    Args:
        model_name: One of ['SimpleRNN', 'LSTM', 'BiLSTM', 'GRU', 'BiGRU', 'Transformer']
        experiment_name: Custom name for this run
        config: Configuration dict or config name (default: QUICK_TEST_CONFIG)
    """
    
    # Handle config parameter
    if config is None:
        config = QUICK_TEST_CONFIG
    elif isinstance(config, str):
        # If config is a string, get the corresponding config
        config_map = {
            'quick': QUICK_TEST_CONFIG,
            'standard': STANDARD_CONFIG,
            'high_performance': HIGH_PERFORMANCE_CONFIG,
            'transformer': TRANSFORMER_CONFIG,
            'custom': CUSTOM_CONFIG,
            'default': DEFAULT_CONFIG
        }
        if config.lower() in config_map:
            config = config_map[config.lower()]
        else:
            print(f"Unknown config '{config}', using QUICK_TEST_CONFIG")
            config = QUICK_TEST_CONFIG
    
    if experiment_name is None:
        experiment_name = f"quick_{model_name.lower()}"
    
    print(f"Quick training: {model_name}")
    print(f"Experiment: {experiment_name}")
    print(f"Config: {config['EPOCHS']} epochs, {config['MAX_FEATURES']} vocab, {config['MAX_LEN']} max_len")
    
    # Initialize analyzer
    analyzer = DeepSentimentAnalyzer(config=config, experiment_name=experiment_name)
    
    # Load data
    print("\nLoading and preprocessing data...")
    X_train, X_test, y_train, y_test = analyzer.load_and_preprocess_data()
    
    # Build model
    print(f"\nBuilding {model_name} model...")
    model_builders = {
        'SimpleRNN': analyzer.build_simple_rnn,
        'LSTM': lambda: analyzer.build_lstm(bidirectional=False),
        'BiLSTM': lambda: analyzer.build_lstm(bidirectional=True),
        'GRU': lambda: analyzer.build_gru(bidirectional=False),
        'BiGRU': lambda: analyzer.build_gru(bidirectional=True),
        'Transformer': analyzer.build_transformer_encoder
    }
    
    if model_name not in model_builders:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(model_builders.keys())}")
    
    model = model_builders[model_name]()
    print(f"Model architecture:")
    model.summary()
    
    # Train model
    print(f"\nTraining {model_name}...")
    trained_model, history = analyzer.train_model(model, model_name, X_train, y_train)
    
    # Evaluate model
    print(f"\nEvaluating {model_name}...")
    result = analyzer.evaluate_model(trained_model, model_name, X_test, y_test)
    
    # Plot training history
    analyzer.plot_training_history(model_name)
    
    print(f"\nQuick training completed!")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"ROC AUC: {result['auc']:.4f}")
    print(f"Results saved in: {analyzer.output_dir}")
    
    return result, analyzer

if __name__ == "__main__":
    import sys
    
    # Command line usage: python quick_train.py [model_name] [experiment_name] [config_name]
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        experiment_name = sys.argv[2] if len(sys.argv) > 2 else None
        config_name = sys.argv[3] if len(sys.argv) > 3 else 'quick'
    else:
        # Interactive mode
        print("Deep Learning Model Quick Training")
        print("=" * 40)
        
        # Model selection
        print("\nAvailable models:")
        models = ['SimpleRNN', 'LSTM', 'BiLSTM', 'GRU', 'BiGRU', 'Transformer']
        for i, model in enumerate(models, 1):
            print(f"  {i}. {model}")
        
        try:
            choice = int(input("\nChoose a model (1-6): ")) - 1
            model_name = models[choice]
        except (ValueError, IndexError):
            print("Invalid choice. Using BiLSTM as default.")
            model_name = 'BiLSTM'
        
        # Configuration selection
        print(f"\nAvailable configurations:")
        configs = [
            ('quick', 'Quick Test', 'Fast training (5 epochs, small vocab) - Good for testing'),
            ('standard', 'Standard', 'Balanced performance (10 epochs, 20k vocab) - Recommended'),
            ('high_performance', 'High Performance', 'Best results (20 epochs, 50k vocab) - Slow but accurate'),
            ('transformer', 'Transformer Optimized', 'Optimized for attention models (15 epochs, 30k vocab)'),
            ('custom', 'Custom', 'User-defined configuration')
        ]
        
        for i, (key, name, description) in enumerate(configs, 1):
            print(f"  {i}. {name}")
            print(f"     {description}")
        
        try:
            config_choice = int(input("\nChoose a configuration (1-5): ")) - 1
            config_name = configs[config_choice][0]
            print(f"Selected: {configs[config_choice][1]}")
        except (ValueError, IndexError):
            print("Invalid choice. Using Quick Test as default.")
            config_name = 'quick'
        
        # Experiment name
        default_name = f"{config_name}_{model_name.lower()}"
        experiment_name = input(f"\n📁 Experiment name (default: {default_name}): ").strip()
        if not experiment_name:
            experiment_name = default_name
        
        print(f"\nStarting training:")
        print(f"Model: {model_name}")
        print(f"Config: {config_name}")
        print(f"Experiment: {experiment_name}")
        print("-" * 40)
    
    # Run quick training
    try:
        result, analyzer = quick_train(
            model_name=model_name,
            experiment_name=experiment_name,
            config=config_name  # Use the selected configuration
        )
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
