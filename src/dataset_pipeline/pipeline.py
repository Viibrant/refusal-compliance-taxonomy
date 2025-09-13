"""Main dataset processing pipeline orchestrator."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import time
import random

from .config import PipelineConfig, DatasetSource, DEFAULT_DATASETS
from .ingestion import DatasetIngester, IngestedDataset
from .generation import ResponseGenerator, GeneratedResponse
from .labeling import CAIJudge, LabelingResult
from .quality import QualityController, AuditReport

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Container for pipeline execution result."""
    success: bool
    total_items: int
    processed_items: int
    failed_items: int
    output_dir: Path
    reports: Dict[str, Any]
    metadata: Dict[str, Any]


class DatasetPipeline:
    """Main orchestrator for the dataset processing pipeline."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.ingester = DatasetIngester()
        self.generator = ResponseGenerator(config.generation)
        self.judge = CAIJudge(config.labeling)
        self.quality_controller = QualityController(config.quality)
        
        # Set random seed for reproducibility
        random.seed(config.random_seed)
        
        # Create output directories
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        (self.config.output_dir / self.config.raw_dir).mkdir(exist_ok=True)
        (self.config.output_dir / self.config.processed_dir).mkdir(exist_ok=True)
        (self.config.output_dir / self.config.audit_dir).mkdir(exist_ok=True)
    
    def run_pipeline(self, datasets: Optional[List[DatasetSource]] = None) -> PipelineResult:
        """Run the complete dataset processing pipeline."""
        start_time = time.time()
        
        if datasets is None:
            datasets = self.config.datasets or DEFAULT_DATASETS
        
        logger.info(f"Starting pipeline with {len(datasets)} datasets")
        
        try:
            # Step 1: Ingest datasets
            ingested_datasets = self._ingest_datasets(datasets)
            
            # Step 2: Generate responses (if needed)
            if not self.config.skip_generation:
                generated_responses = self._generate_responses(ingested_datasets)
            else:
                generated_responses = []
            
            # Step 3: Label with CAI judge
            if not self.config.skip_labeling:
                labeling_results = self._label_data(ingested_datasets, generated_responses)
            else:
                labeling_results = []
            
            # Step 4: Quality control and auditing
            if not self.config.skip_quality_control and labeling_results:
                audit_report = self._audit_data(labeling_results)
            else:
                audit_report = None
            
            # Step 5: Save final processed data
            final_data = self._prepare_final_data(ingested_datasets, generated_responses, labeling_results)
            self._save_final_data(final_data)
            
            # Create pipeline result
            execution_time = time.time() - start_time
            result = PipelineResult(
                success=True,
                total_items=sum(len(ds.data) for ds in ingested_datasets),
                processed_items=len(final_data),
                failed_items=0,  # Would track failures in practice
                output_dir=self.config.output_dir,
                reports={
                    "ingestion": [ds.metadata for ds in ingested_datasets],
                    "generation": len(generated_responses),
                    "labeling": len(labeling_results),
                    "audit": audit_report.metadata if audit_report else None
                },
                metadata={
                    "execution_time": execution_time,
                    "config": self.config.__dict__,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            )
            
            logger.info(f"Pipeline completed successfully in {execution_time:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return PipelineResult(
                success=False,
                total_items=0,
                processed_items=0,
                failed_items=0,
                output_dir=self.config.output_dir,
                reports={},
                metadata={"error": str(e), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
            )
    
    def _ingest_datasets(self, datasets: List[DatasetSource]) -> List[IngestedDataset]:
        """Ingest all specified datasets."""
        ingested_datasets = []
        
        for dataset_source in datasets:
            try:
                logger.info(f"Ingesting dataset: {dataset_source.name}")
                ingested_dataset = self.ingester.ingest_dataset(dataset_source)
                ingested_datasets.append(ingested_dataset)
                
                # Save raw ingested data
                raw_output_dir = self.config.output_dir / self.config.raw_dir
                self.ingester.save_ingested_dataset(ingested_dataset, raw_output_dir)
                
                logger.info(f"Successfully ingested {len(ingested_dataset.data)} items from {dataset_source.name}")
                
            except Exception as e:
                logger.error(f"Failed to ingest dataset {dataset_source.name}: {e}")
                continue
        
        return ingested_datasets
    
    def _generate_responses(self, ingested_datasets: List[IngestedDataset]) -> List[GeneratedResponse]:
        """Generate responses for prompt-only datasets."""
        all_generated_responses = []
        
        for dataset in ingested_datasets:
            # Check if dataset has responses already
            has_responses = any("response" in item and item["response"] for item in dataset.data)
            
            if not has_responses:
                logger.info(f"Generating responses for {dataset.name}")
                
                # Extract prompts
                prompts = [item["prompt"] for item in dataset.data if "prompt" in item]
                
                # Generate responses
                generated_responses = self.generator.generate_responses(
                    prompts, 
                    self.config.generation.jailbreak_wrappers
                )
                
                all_generated_responses.extend(generated_responses)
                
                # Save generated responses
                output_file = self.config.output_dir / self.config.processed_dir / f"{dataset.name}_generated_responses.json"
                self.generator.save_generated_responses(generated_responses, output_file)
                
                logger.info(f"Generated {len(generated_responses)} responses for {dataset.name}")
        
        return all_generated_responses
    
    def _label_data(self, ingested_datasets: List[IngestedDataset], generated_responses: List[GeneratedResponse]) -> List[LabelingResult]:
        """Label all data using CAI judge."""
        all_labeling_results = []
        
        # Prepare data for labeling
        data_to_label = []
        
        # Add data from ingested datasets
        for dataset in ingested_datasets:
            for item in dataset.data:
                if "prompt" in item and "response" in item:
                    data_to_label.append({
                        "prompt": item["prompt"],
                        "response": item["response"],
                        "source": dataset.name
                    })
        
        # Add generated responses
        for response in generated_responses:
            data_to_label.append({
                "prompt": response.prompt,
                "response": response.response,
                "source": f"generated_{response.model_name}"
            })
        
        if data_to_label:
            logger.info(f"Labeling {len(data_to_label)} items with CAI judge")
            
            # Label in batches
            batch_size = self.config.labeling.batch_size
            for i in range(0, len(data_to_label), batch_size):
                batch = data_to_label[i:i + batch_size]
                batch_results = self.judge.label_prompts_responses(batch)
                all_labeling_results.extend(batch_results)
                
                logger.info(f"Labeled batch {i//batch_size + 1}/{(len(data_to_label) + batch_size - 1)//batch_size}")
            
            # Save labeling results
            output_file = self.config.output_dir / self.config.processed_dir / "labeling_results.json"
            self.judge.save_labeling_results(all_labeling_results, output_file)
            
            logger.info(f"Completed labeling {len(all_labeling_results)} items")
        
        return all_labeling_results
    
    def _audit_data(self, labeling_results: List[LabelingResult]) -> AuditReport:
        """Perform quality control and auditing."""
        logger.info("Performing quality control and auditing")
        
        audit_report = self.quality_controller.audit_dataset(labeling_results)
        
        # Save audit report
        output_file = self.config.output_dir / self.config.audit_dir / "audit_report.json"
        self.quality_controller.save_audit_report(audit_report, output_file)
        
        # Save human review samples
        if audit_report.human_review_samples:
            samples_file = self.config.output_dir / self.config.audit_dir / "human_review_samples.json"
            with open(samples_file, 'w', encoding='utf-8') as f:
                json.dump(audit_report.human_review_samples, f, indent=2, ensure_ascii=False)
        
        logger.info("Quality control and auditing completed")
        return audit_report
    
    def _prepare_final_data(self, ingested_datasets: List[IngestedDataset], 
                           generated_responses: List[GeneratedResponse], 
                           labeling_results: List[LabelingResult]) -> List[Dict[str, Any]]:
        """Prepare final processed data for training."""
        final_data = []
        
        # Convert labeling results to final format
        for result in labeling_results:
            final_item = {
                "prompt": result.prompt,
                "response": result.response,
                "head_a": result.labels.get("head_a"),
                "head_b_a": result.labels.get("head_b_a"),
                "head_b_b": result.labels.get("head_b_b"),
                "head_c": result.labels.get("head_c", []),
                "head_d": {
                    "prompt_harmful": result.labels.get("prompt_harmful", False),
                    "response_harmful": result.labels.get("response_harmful", False),
                    "response_refusal": result.labels.get("response_refusal", False)
                },
                "metadata": {
                    "confidence": result.confidence,
                    "rationale": result.rationale,
                    "judge_model": result.judge_model,
                    "labeling_time": result.labeling_time
                }
            }
            final_data.append(final_item)
        
        return final_data
    
    def _save_final_data(self, final_data: List[Dict[str, Any]]):
        """Save final processed data."""
        # Save as JSON
        output_file = self.config.output_dir / self.config.processed_dir / "final_dataset.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)
        
        # Save as JSONL for easier processing
        jsonl_file = self.config.output_dir / self.config.processed_dir / "final_dataset.jsonl"
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            for item in final_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        # Save dataset info
        info_file = self.config.output_dir / self.config.processed_dir / "dataset_info.json"
        dataset_info = {
            "total_items": len(final_data),
            "format": "multi-head classification",
            "heads": {
                "head_a": "outcome classification",
                "head_b_a": "refusal style classification", 
                "head_b_b": "compliance style classification",
                "head_c": "harm category multilabel",
                "head_d": "safety flags boolean"
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved final dataset with {len(final_data)} items")
    
    def run_ingestion_only(self, datasets: List[DatasetSource]) -> List[IngestedDataset]:
        """Run only the ingestion step."""
        return self._ingest_datasets(datasets)
    
    def run_generation_only(self, ingested_datasets: List[IngestedDataset]) -> List[GeneratedResponse]:
        """Run only the generation step."""
        return self._generate_responses(ingested_datasets)
    
    def run_labeling_only(self, data: List[Dict[str, Any]]) -> List[LabelingResult]:
        """Run only the labeling step."""
        return self.judge.label_prompts_responses(data)
    
    def run_audit_only(self, labeling_results: List[LabelingResult]) -> AuditReport:
        """Run only the audit step."""
        return self._audit_data(labeling_results)
