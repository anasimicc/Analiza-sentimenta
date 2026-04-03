import argparse

from deep_models import DeepSentimentAnalyzer
from model_configs import (
    QUICK_TEST_CONFIG,
    STANDARD_CONFIG,
    HIGH_PERFORMANCE_CONFIG,
    TRANSFORMER_CONFIG,
    CUSTOM_CONFIG,
    DEFAULT_CONFIG,
)


CONFIG_MAP = {
    'quick': QUICK_TEST_CONFIG,
    'standard': STANDARD_CONFIG,
    'high_performance': HIGH_PERFORMANCE_CONFIG,
    'transformer': TRANSFORMER_CONFIG,
    'custom': CUSTOM_CONFIG,
    'default': DEFAULT_CONFIG,
}


def parse_args():
    parser = argparse.ArgumentParser(description='Train all deep sentiment models and save results.')
    parser.add_argument(
        '--config',
        default='standard',
        choices=sorted(CONFIG_MAP.keys()),
        help='Configuration preset to use for training.',
    )
    parser.add_argument(
        '--experiment',
        default=None,
        help='Experiment name used for the output folder.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = CONFIG_MAP[args.config]
    experiment_name = args.experiment or f'deep_{args.config}'

    analyzer = DeepSentimentAnalyzer(config=config, experiment_name=experiment_name)
    analyzer.run_complete_analysis()


if __name__ == '__main__':
    main()