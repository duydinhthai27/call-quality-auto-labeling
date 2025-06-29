import argparse
from multi_model_runner import run_all_models

def main():
    parser = argparse.ArgumentParser(description="Run semi-supervised learning models.")
    parser.add_argument(
        '--use-dynamic-threshold',
        action='store_true',
        help="Use a dynamic confidence threshold for active learning instead of a fixed one from config."
    )
    args = parser.parse_args()

    print("Starting model training and evaluation...")
    if args.use_dynamic_threshold:
        print("Using dynamic threshold for active learning.")
    else:
        print("Using fixed threshold for active learning.")
        
    run_all_models(use_dynamic_threshold=args.use_dynamic_threshold)

if __name__ == "__main__":
    main()
