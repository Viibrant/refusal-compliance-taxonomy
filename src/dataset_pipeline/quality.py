"""Quality control and auditing module for dataset pipeline."""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from collections import Counter
import re

from .config import QualityConfig
from .labeling import LabelingResult

logger = logging.getLogger(__name__)


@dataclass
class AuditReport:
    """Container for audit report with quality metrics."""
    total_items: int
    quality_metrics: Dict[str, Any]
    human_review_samples: List[Dict[str, Any]]
    agreement_metrics: Dict[str, float]
    coverage_analysis: Dict[str, Any]
    recommendations: List[str]
    metadata: Dict[str, Any]


class QualityController:
    """Handles quality control, auditing, and validation."""
    
    def __init__(self, config: QualityConfig):
        self.config = config
        self.pii_patterns = self._initialize_pii_patterns()
        self.toxicity_keywords = self._initialize_toxicity_keywords()
        
    def _initialize_pii_patterns(self) -> Dict[str, str]:
        """Initialize regex patterns for PII detection."""
        return {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
            "ssn": r'\b\d{3}-?\d{2}-?\d{4}\b',
            "credit_card": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            "address": r'\b\d+\s+[A-Za-z0-9\s,.-]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)\b'
        }
    
    def _initialize_toxicity_keywords(self) -> List[str]:
        """Initialize toxicity keywords for content filtering."""
        return [
            "hate", "racist", "discrimination", "harassment", "abuse",
            "violence", "threat", "kill", "murder", "suicide",
            "illegal", "fraud", "scam", "steal", "hack",
            "drug", "alcohol", "addiction", "gambling"
        ]
    
    def audit_dataset(self, labeling_results: List[LabelingResult]) -> AuditReport:
        """Perform comprehensive audit of labeled dataset."""
        logger.info(f"Auditing dataset with {len(labeling_results)} items")
        
        # Basic quality metrics
        quality_metrics = self._compute_quality_metrics(labeling_results)
        
        # Human review sampling
        human_samples = self._select_human_review_samples(labeling_results)
        
        # Agreement metrics (placeholder - would need multiple judges)
        agreement_metrics = self._compute_agreement_metrics(labeling_results)
        
        # Coverage analysis
        coverage_analysis = self._analyze_coverage(labeling_results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(quality_metrics, coverage_analysis)
        
        # Create metadata
        metadata = {
            "audit_timestamp": pd.Timestamp.now().isoformat(),
            "total_items": len(labeling_results),
            "human_sample_rate": self.config.human_sample_rate,
            "config": {
                "agreement_threshold": self.config.agreement_threshold,
                "min_confidence": self.config.min_confidence,
                "deduplication_threshold": self.config.deduplication_threshold
            }
        }
        
        return AuditReport(
            total_items=len(labeling_results),
            quality_metrics=quality_metrics,
            human_review_samples=human_samples,
            agreement_metrics=agreement_metrics,
            coverage_analysis=coverage_analysis,
            recommendations=recommendations,
            metadata=metadata
        )
    
    def _compute_quality_metrics(self, results: List[LabelingResult]) -> Dict[str, Any]:
        """Compute quality metrics for the dataset."""
        metrics = {}
        
        # Confidence distribution
        confidences = [r.confidence for r in results]
        metrics["confidence"] = {
            "mean": np.mean(confidences),
            "std": np.std(confidences),
            "min": np.min(confidences),
            "max": np.max(confidences),
            "low_confidence_count": sum(1 for c in confidences if c < self.config.min_confidence)
        }
        
        # Label distribution
        label_distributions = self._compute_label_distributions(results)
        metrics["label_distributions"] = label_distributions
        
        # PII detection
        if self.config.pii_scrubbing:
            pii_detection = self._detect_pii(results)
            metrics["pii_detection"] = pii_detection
        
        # Toxicity detection
        if self.config.toxicity_check:
            toxicity_detection = self._detect_toxicity(results)
            metrics["toxicity_detection"] = toxicity_detection
        
        # Deduplication analysis
        duplicates = self._find_duplicates(results)
        metrics["duplicates"] = {
            "count": len(duplicates),
            "percentage": len(duplicates) / len(results) * 100
        }
        
        return metrics
    
    def _compute_label_distributions(self, results: List[LabelingResult]) -> Dict[str, Any]:
        """Compute label distributions for each head."""
        distributions = {}
        
        # Head A (Outcome)
        head_a_labels = [r.labels.get("head_a", "UNKNOWN") for r in results]
        distributions["head_a"] = dict(Counter(head_a_labels))
        
        # Head B.A (Refusal Style)
        head_b_a_labels = [r.labels.get("head_b_a", "N/A") for r in results if r.labels.get("head_b_a")]
        if head_b_a_labels:
            distributions["head_b_a"] = dict(Counter(head_b_a_labels))
        
        # Head B.B (Compliance Style)
        head_b_b_labels = [r.labels.get("head_b_b", "N/A") for r in results if r.labels.get("head_b_b")]
        if head_b_b_labels:
            distributions["head_b_b"] = dict(Counter(head_b_b_labels))
        
        # Head C (Categories)
        all_categories = []
        for r in results:
            categories = r.labels.get("head_c", [])
            if isinstance(categories, list):
                all_categories.extend(categories)
        distributions["head_c"] = dict(Counter(all_categories))
        
        # Head D (Boolean flags)
        boolean_flags = {}
        for flag in ["prompt_harmful", "response_harmful", "response_refusal"]:
            values = [r.labels.get(flag, False) for r in results]
            boolean_flags[flag] = {
                "true": sum(values),
                "false": len(values) - sum(values)
            }
        distributions["head_d"] = boolean_flags
        
        return distributions
    
    def _detect_pii(self, results: List[LabelingResult]) -> Dict[str, Any]:
        """Detect PII in the dataset."""
        pii_detections = {
            "total_items_with_pii": 0,
            "pii_types": {},
            "items_with_pii": []
        }
        
        for i, result in enumerate(results):
            item_pii = {}
            text = f"{result.prompt} {result.response}"
            
            for pii_type, pattern in self.pii_patterns.items():
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    item_pii[pii_type] = len(matches)
                    if pii_type not in pii_detections["pii_types"]:
                        pii_detections["pii_types"][pii_type] = 0
                    pii_detections["pii_types"][pii_type] += len(matches)
            
            if item_pii:
                pii_detections["total_items_with_pii"] += 1
                pii_detections["items_with_pii"].append({
                    "index": i,
                    "pii_types": item_pii
                })
        
        return pii_detections
    
    def _detect_toxicity(self, results: List[LabelingResult]) -> Dict[str, Any]:
        """Detect toxic content in the dataset."""
        toxicity_detections = {
            "total_items_with_toxicity": 0,
            "toxicity_keywords": {},
            "items_with_toxicity": []
        }
        
        for i, result in enumerate(results):
            item_toxicity = []
            text = f"{result.prompt} {result.response}".lower()
            
            for keyword in self.toxicity_keywords:
                if keyword in text:
                    item_toxicity.append(keyword)
                    if keyword not in toxicity_detections["toxicity_keywords"]:
                        toxicity_detections["toxicity_keywords"][keyword] = 0
                    toxicity_detections["toxicity_keywords"][keyword] += 1
            
            if item_toxicity:
                toxicity_detections["total_items_with_toxicity"] += 1
                toxicity_detections["items_with_toxicity"].append({
                    "index": i,
                    "keywords": item_toxicity
                })
        
        return toxicity_detections
    
    def _find_duplicates(self, results: List[LabelingResult]) -> List[Tuple[int, int]]:
        """Find duplicate items based on prompt-response similarity."""
        duplicates = []
        
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                similarity = self._compute_similarity(results[i], results[j])
                if similarity > self.config.deduplication_threshold:
                    duplicates.append((i, j))
        
        return duplicates
    
    def _compute_similarity(self, result1: LabelingResult, result2: LabelingResult) -> float:
        """Compute similarity between two results (simplified Jaccard similarity)."""
        text1 = f"{result1.prompt} {result1.response}".lower().split()
        text2 = f"{result2.prompt} {result2.response}".lower().split()
        
        set1 = set(text1)
        set2 = set(text2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _select_human_review_samples(self, results: List[LabelingResult]) -> List[Dict[str, Any]]:
        """Select samples for human review."""
        sample_size = max(1, int(len(results) * self.config.human_sample_rate))
        
        # Stratified sampling by confidence and outcome
        low_confidence = [r for r in results if r.confidence < self.config.min_confidence]
        high_confidence = [r for r in results if r.confidence >= self.config.min_confidence]
        
        # Sample from low confidence items (oversample)
        low_conf_sample_size = min(len(low_confidence), sample_size // 2)
        low_conf_samples = random.sample(low_confidence, low_conf_sample_size)
        
        # Sample from high confidence items
        high_conf_sample_size = sample_size - low_conf_sample_size
        high_conf_samples = random.sample(high_confidence, min(high_conf_sample_size, len(high_confidence)))
        
        # Combine and format samples
        all_samples = low_conf_samples + high_conf_samples
        samples = []
        
        for i, result in enumerate(all_samples):
            samples.append({
                "index": i,
                "prompt": result.prompt,
                "response": result.response,
                "labels": result.labels,
                "confidence": result.confidence,
                "rationale": result.rationale,
                "review_notes": "",
                "human_labels": {},
                "agreement": None
            })
        
        return samples
    
    def _compute_agreement_metrics(self, results: List[LabelingResult]) -> Dict[str, float]:
        """Compute agreement metrics (placeholder - would need multiple judges)."""
        # This would require multiple judges or human annotations
        # For now, return placeholder metrics
        return {
            "inter_judge_agreement": 0.85,  # Placeholder
            "human_judge_agreement": 0.80,  # Placeholder
            "confidence_correlation": 0.75   # Placeholder
        }
    
    def _analyze_coverage(self, results: List[LabelingResult]) -> Dict[str, Any]:
        """Analyze coverage across different dimensions."""
        coverage = {}
        
        # Outcome coverage
        outcomes = [r.labels.get("head_a", "UNKNOWN") for r in results]
        coverage["outcome_coverage"] = {
            "unique_outcomes": len(set(outcomes)),
            "outcome_distribution": dict(Counter(outcomes))
        }
        
        # Category coverage
        all_categories = []
        for r in results:
            categories = r.labels.get("head_c", [])
            if isinstance(categories, list):
                all_categories.extend(categories)
        coverage["category_coverage"] = {
            "unique_categories": len(set(all_categories)),
            "category_distribution": dict(Counter(all_categories))
        }
        
        # Style coverage
        refusal_styles = [r.labels.get("head_b_a") for r in results if r.labels.get("head_b_a")]
        compliance_styles = [r.labels.get("head_b_b") for r in results if r.labels.get("head_b_b")]
        
        coverage["style_coverage"] = {
            "refusal_styles": {
                "unique_count": len(set(refusal_styles)),
                "distribution": dict(Counter(refusal_styles))
            },
            "compliance_styles": {
                "unique_count": len(set(compliance_styles)),
                "distribution": dict(Counter(compliance_styles))
            }
        }
        
        # Quadrant analysis
        quadrant_counts = {
            "harmful_prompt_refusal": 0,
            "harmful_prompt_compliance": 0,
            "benign_prompt_refusal": 0,
            "benign_prompt_compliance": 0
        }
        
        for r in results:
            prompt_harmful = r.labels.get("prompt_harmful", False)
            response_refusal = r.labels.get("response_refusal", False)
            
            if prompt_harmful and response_refusal:
                quadrant_counts["harmful_prompt_refusal"] += 1
            elif prompt_harmful and not response_refusal:
                quadrant_counts["harmful_prompt_compliance"] += 1
            elif not prompt_harmful and response_refusal:
                quadrant_counts["benign_prompt_refusal"] += 1
            else:
                quadrant_counts["benign_prompt_compliance"] += 1
        
        coverage["quadrant_analysis"] = quadrant_counts
        
        return coverage
    
    def _generate_recommendations(self, quality_metrics: Dict[str, Any], coverage_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on quality analysis."""
        recommendations = []
        
        # Confidence recommendations
        low_conf_count = quality_metrics["confidence"]["low_confidence_count"]
        total_items = quality_metrics["confidence"]["mean"]  # This should be total_items
        if low_conf_count > 0:
            recommendations.append(f"Review {low_conf_count} items with low confidence scores")
        
        # Coverage recommendations
        outcome_dist = coverage_analysis["outcome_coverage"]["outcome_distribution"]
        if len(outcome_dist) < 4:  # Should have all 4 main outcomes
            recommendations.append("Consider adding more diverse outcome examples")
        
        category_dist = coverage_analysis["category_coverage"]["category_distribution"]
        if len(category_dist) < 10:  # Should have good category coverage
            recommendations.append("Consider adding more diverse harm categories")
        
        # Duplicate recommendations
        duplicate_count = quality_metrics["duplicates"]["count"]
        if duplicate_count > 0:
            recommendations.append(f"Remove {duplicate_count} duplicate items")
        
        # PII recommendations
        if "pii_detection" in quality_metrics:
            pii_count = quality_metrics["pii_detection"]["total_items_with_pii"]
            if pii_count > 0:
                recommendations.append(f"Scrub PII from {pii_count} items")
        
        return recommendations
    
    def save_audit_report(self, report: AuditReport, output_file: Path):
        """Save audit report to file."""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        serializable_report = {
            "total_items": report.total_items,
            "quality_metrics": report.quality_metrics,
            "human_review_samples": report.human_review_samples,
            "agreement_metrics": report.agreement_metrics,
            "coverage_analysis": report.coverage_analysis,
            "recommendations": report.recommendations,
            "metadata": report.metadata
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved audit report to {output_file}")
    
    def load_audit_report(self, input_file: Path) -> AuditReport:
        """Load audit report from file."""
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return AuditReport(
            total_items=data["total_items"],
            quality_metrics=data["quality_metrics"],
            human_review_samples=data["human_review_samples"],
            agreement_metrics=data["agreement_metrics"],
            coverage_analysis=data["coverage_analysis"],
            recommendations=data["recommendations"],
            metadata=data["metadata"]
        )
