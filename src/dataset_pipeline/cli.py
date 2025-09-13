"""Command-line interface for the dataset processing pipeline."""

import argparse
import logging
import json
from pathlib import Path
from typing import List, Optional

from .config import PipelineConfig, DatasetSource, DEFAULT_DATASETS
from .pipeline import DatasetPipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main entry point for dataset pipeline CLI."""
    parser = argparse.ArgumentParser(description="Dataset processing pipeline for rejection detection")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Full pipeline command
    pipeline_parser = subparsers.add_parser("run", help="Run the complete pipeline")
    pipeline_parser.add_argument("--config", type=str, help="Path to configuration file")
    pipeline_parser.add_argument("--output-dir", type=str, default="outputs/dataset_pipeline", help="Output directory")
    pipeline_parser.add_argument("--datasets", nargs="+", help="Specific datasets to process")
    pipeline_parser.add_argument("--skip-generation", action="store_true", help="Skip response generation")
    pipeline_parser.add_argument("--skip-labeling", action="store_true", help="Skip CAI judge labeling")
    pipeline_parser.add_argument("--skip-quality", action="store_true", help="Skip quality control")
    pipeline_parser.add_argument("--parallel-workers", type=int, default=4, help="Number of parallel workers")
    pipeline_parser.add_argument("--random-seed", type=int, default=42, help="Random seed")
    
    # Ingestion command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest datasets only")
    ingest_parser.add_argument("--datasets", nargs="+", help="Specific datasets to ingest")
    ingest_parser.add_argument("--output-dir", type=str, default="outputs/dataset_pipeline/raw", help="Output directory")
    
    # Generation command
    generate_parser = subparsers.add_parser("generate", help="Generate responses only")
    generate_parser.add_argument("--input-dir", type=str, required=True, help="Input directory with ingested data")
    generate_parser.add_argument("--output-dir", type=str, default="outputs/dataset_pipeline/processed", help="Output directory")
    generate_parser.add_argument("--models", nargs="+", default=["gpt-3.5-turbo"], help="Models to use for generation")
    generate_parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    generate_parser.add_argument("--max-tokens", type=int, default=512, help="Maximum tokens to generate")
    
    # Labeling command
    label_parser = subparsers.add_parser("label", help="Label data with CAI judge only")
    label_parser.add_argument("--input-file", type=str, required=True, help="Input file with prompt-response pairs")
    label_parser.add_argument("--output-file", type=str, help="Output file for labeling results")
    label_parser.add_argument("--judge-model", type=str, default="gpt-4", help="Judge model to use")
    label_parser.add_argument("--confidence-threshold", type=float, default=0.7, help="Confidence threshold")
    label_parser.add_argument("--batch-size", type=int, default=10, help="Batch size for labeling")
    
    # Audit command
    audit_parser = subparsers.add_parser("audit", help="Audit labeled data only")
    audit_parser.add_argument("--input-file", type=str, required=True, help="Input file with labeling results")
    audit_parser.add_argument("--output-dir", type=str, default="outputs/dataset_pipeline/audit", help="Output directory")
    audit_parser.add_argument("--human-sample-rate", type=float, default=0.05, help="Human sample rate")
    audit_parser.add_argument("--min-confidence", type=float, default=0.6, help="Minimum confidence threshold")
    
    # List datasets command
    list_parser = subparsers.add_parser("list", help="List available datasets")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate pipeline configuration")
    validate_parser.add_argument("--config", type=str, help="Path to configuration file")
    
    args = parser.parse_args()
    
    if args.command == "run":
        run_pipeline(args)
    elif args.command == "ingest":
        run_ingestion(args)
    elif args.command == "generate":
        run_generation(args)
    elif args.command == "label":
        run_labeling(args)
    elif args.command == "audit":
        run_audit(args)
    elif args.command == "list":
        list_datasets()
    elif args.command == "validate":
        validate_config(args)
    else:
        parser.print_help()


def run_pipeline(args):
    """Run the complete pipeline."""
    # Load configuration
    config = load_config(args.config) if args.config else create_config_from_args(args)
    
    # Create pipeline
    pipeline = DatasetPipeline(config)
    
    # Select datasets
    datasets = select_datasets(args.datasets) if args.datasets else DEFAULT_DATASETS
    
    # Run pipeline
    result = pipeline.run_pipeline(datasets)
    
    if result.success:
        print(f"Pipeline completed successfully!")
        print(f"Total items: {result.total_items}")
        print(f"Processed items: {result.processed_items}")
        print(f"Output directory: {result.output_dir}")
        print(f"Execution time: {result.metadata['execution_time']:.2f} seconds")
    else:
        print(f"Pipeline failed: {result.metadata.get('error', 'Unknown error')}")
        exit(1)


def run_ingestion(args):
    """Run ingestion only."""
    config = PipelineConfig(output_dir=Path(args.output_dir))
    pipeline = DatasetPipeline(config)
    
    datasets = select_datasets(args.datasets) if args.datasets else DEFAULT_DATASETS
    ingested_datasets = pipeline.run_ingestion_only(datasets)
    
    print(f"Ingested {len(ingested_datasets)} datasets")
    for dataset in ingested_datasets:
        print(f"  {dataset.name}: {len(dataset.data)} items")


def run_generation(args):
    """Run generation only."""
    from .generation import GenerationConfig, ResponseGenerator
    
    config = GenerationConfig(
        models=args.models,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )
    
    generator = ResponseGenerator(config)
    
    # Load ingested data
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find JSON files with ingested data
    json_files = list(input_dir.glob("*_data.json"))
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Extract prompts
        prompts = [item["prompt"] for item in data if "prompt" in item]
        
        if prompts:
            print(f"Generating responses for {len(prompts)} prompts from {json_file.name}")
            responses = generator.generate_responses(prompts)
            
            # Save responses
            output_file = output_dir / f"{json_file.stem}_generated_responses.json"
            generator.save_generated_responses(responses, output_file)
            print(f"Saved {len(responses)} responses to {output_file}")


def run_labeling(args):
    """Run labeling only."""
    from .labeling import LabelingConfig, CAIJudge
    
    config = LabelingConfig(
        judge_model=args.judge_model,
        confidence_threshold=args.confidence_threshold,
        batch_size=args.batch_size
    )
    
    judge = CAIJudge(config)
    
    # Load data
    with open(args.input_file, 'r') as f:
        data = json.load(f)
    
    print(f"Labeling {len(data)} items")
    results = judge.label_prompts_responses(data)
    
    # Save results
    output_file = args.output_file or args.input_file.replace('.json', '_labeled.json')
    judge.save_labeling_results(results, Path(output_file))
    print(f"Saved {len(results)} labeling results to {output_file}")


def run_audit(args):
    """Run audit only."""
    from .quality import QualityConfig, QualityController
    from .labeling import LabelingResult
    
    config = QualityConfig(
        human_sample_rate=args.human_sample_rate,
        min_confidence=args.min_confidence
    )
    
    quality_controller = QualityController(config)
    
    # Load labeling results
    with open(args.input_file, 'r') as f:
        data = json.load(f)
    
    # Convert to LabelingResult objects
    results = []
    for item in data:
        result = LabelingResult(
            prompt=item["prompt"],
            response=item["response"],
            labels=item["labels"],
            confidence=item["confidence"],
            rationale=item.get("rationale"),
            judge_model=item.get("judge_model"),
            labeling_time=item.get("labeling_time"),
            metadata=item.get("metadata", {})
        )
        results.append(result)
    
    print(f"Auditing {len(results)} labeling results")
    audit_report = quality_controller.audit_dataset(results)
    
    # Save audit report
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    quality_controller.save_audit_report(audit_report, output_dir / "audit_report.json")
    
    print(f"Audit completed. Report saved to {output_dir}")
    print(f"Recommendations: {len(audit_report.recommendations)}")
    for rec in audit_report.recommendations:
        print(f"  - {rec}")


def list_datasets():
    """List available datasets."""
    print("Available datasets:")
    for dataset in DEFAULT_DATASETS:
        print(f"  {dataset.name}: {dataset.description}")
        print(f"    Source: {dataset.source_type} - {dataset.source_path}")
        print(f"    License: {dataset.license}")
        print()


def validate_config(args):
    """Validate pipeline configuration."""
    if args.config:
        try:
            config = load_config(args.config)
            print(f"Configuration loaded successfully from {args.config}")
            print(f"Datasets: {len(config.datasets)}")
            print(f"Output directory: {config.output_dir}")
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            exit(1)
    else:
        print("No configuration file specified")


def load_config(config_path: str) -> PipelineConfig:
    """Load configuration from file."""
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    # Convert to PipelineConfig object
    # This is a simplified implementation
    return PipelineConfig(**config_data)


def create_config_from_args(args) -> PipelineConfig:
    """Create configuration from command line arguments."""
    return PipelineConfig(
        output_dir=Path(args.output_dir),
        skip_generation=args.skip_generation,
        skip_labeling=args.skip_labeling,
        skip_quality_control=args.skip_quality,
        parallel_workers=args.parallel_workers,
        random_seed=args.random_seed
    )


def select_datasets(dataset_names: List[str]) -> List[DatasetSource]:
    """Select datasets by name."""
    selected = []
    available_names = {ds.name for ds in DEFAULT_DATASETS}
    
    for name in dataset_names:
        if name in available_names:
            selected.extend([ds for ds in DEFAULT_DATASETS if ds.name == name])
        else:
            print(f"Warning: Dataset '{name}' not found in available datasets")
    
    return selected


if __name__ == "__main__":
    main()
