"""Command-line interface for data processing."""

import argparse
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

from .processor import DataProcessor, ProcessingConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main entry point for data processing CLI."""
    parser = argparse.ArgumentParser(description="Process rejection detection datasets")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process a dataset")
    process_parser.add_argument("--input", type=str, required=True, help="Input file path")
    process_parser.add_argument("--output", type=str, required=True, help="Output directory")
    process_parser.add_argument("--min-prompt-length", type=int, default=10, help="Minimum prompt length")
    process_parser.add_argument("--max-prompt-length", type=int, default=1000, help="Maximum prompt length")
    process_parser.add_argument("--min-response-length", type=int, default=5, help="Minimum response length")
    process_parser.add_argument("--max-response-length", type=int, default=2000, help="Maximum response length")
    process_parser.add_argument("--remove-duplicates", action="store_true", help="Remove duplicate entries")
    process_parser.add_argument("--balance-classes", action="store_true", help="Balance classes")
    process_parser.add_argument("--validation-split", type=float, default=0.1, help="Validation split ratio")
    process_parser.add_argument("--test-split", type=float, default=0.1, help="Test split ratio")
    process_parser.add_argument("--random-seed", type=int, default=42, help="Random seed")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate a dataset")
    validate_parser.add_argument("--input", type=str, required=True, help="Input file path")
    validate_parser.add_argument("--output", type=str, help="Output file for validation report")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show dataset statistics")
    stats_parser.add_argument("--input", type=str, required=True, help="Input file path")
    
    args = parser.parse_args()
    
    if args.command == "process":
        config = ProcessingConfig(
            min_prompt_length=args.min_prompt_length,
            max_prompt_length=args.max_prompt_length,
            min_response_length=args.min_response_length,
            max_response_length=args.max_response_length,
            remove_duplicates=args.remove_duplicates,
            balance_classes=args.balance_classes,
            validation_split=args.validation_split,
            test_split=args.test_split,
            random_seed=args.random_seed
        )
        
        processor = DataProcessor(config)
        report = processor.process_dataset(args.input, args.output)
        
        print(f"Processing complete!")
        print(f"Total items: {report['total_items']}")
        print(f"Valid items: {report['valid_items']}")
        print(f"Final items: {report['final_items']}")
        print(f"Train/Val/Test: {report['train_items']}/{report['val_items']}/{report['test_items']}")
        print(f"Validation errors: {report['validation_errors']}")
    
    elif args.command == "validate":
        processor = DataProcessor()
        data = processor.load_data(args.input)
        valid_data, errors = processor.validate_data(data)
        
        print(f"Validation Results:")
        print(f"Total items: {len(data)}")
        print(f"Valid items: {len(valid_data)}")
        print(f"Errors: {len(errors)}")
        
        if errors:
            print("\nFirst 10 errors:")
            for error in errors[:10]:
                print(f"  {error}")
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write('\n'.join(errors))
            print(f"\nAll errors saved to {args.output}")
    
    elif args.command == "stats":
        processor = DataProcessor()
        data = processor.load_data(args.input)
        
        print(f"Dataset Statistics:")
        print(f"Total items: {len(data)}")
        
        # Prompt/response length stats
        prompt_lengths = [len(str(item.get('prompt', ''))) for item in data]
        response_lengths = [len(str(item.get('response', ''))) for item in data]
        
        print(f"\nPrompt lengths:")
        print(f"  Min: {min(prompt_lengths)}")
        print(f"  Max: {max(prompt_lengths)}")
        print(f"  Mean: {np.mean(prompt_lengths):.1f}")
        
        print(f"\nResponse lengths:")
        print(f"  Min: {min(response_lengths)}")
        print(f"  Max: {max(response_lengths)}")
        print(f"  Mean: {np.mean(response_lengths):.1f}")
        
        # Label distribution
        for head_name, head_config in processor.head_configs.items():
            if head_name in data[0]:
                if head_config.head_type == "multilabel":
                    all_labels = []
                    for item in data:
                        if head_name in item and isinstance(item[head_name], list):
                            all_labels.extend(item[head_name])
                    
                    from collections import Counter
                    label_counts = Counter(all_labels)
                    print(f"\n{head_name} (multilabel) distribution:")
                    for label, count in label_counts.most_common(10):
                        print(f"  {label}: {count}")
                elif head_config.head_type == "boolean":
                    # Boolean head - show distribution of each boolean field
                    print(f"\n{head_name} (boolean) distribution:")
                    boolean_fields = {}
                    for item in data:
                        if head_name in item and isinstance(item[head_name], dict):
                            for field, value in item[head_name].items():
                                if field not in boolean_fields:
                                    boolean_fields[field] = {"true": 0, "false": 0}
                                if value:
                                    boolean_fields[field]["true"] += 1
                                else:
                                    boolean_fields[field]["false"] += 1
                    
                    for field, counts in boolean_fields.items():
                        print(f"  {field}: true={counts['true']}, false={counts['false']}")
                else:
                    labels = [item.get(head_name, 'N/A') for item in data]
                    from collections import Counter
                    label_counts = Counter(labels)
                    print(f"\n{head_name} distribution:")
                    for label, count in label_counts.most_common():
                        print(f"  {label}: {count}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
