"""Inference script for multi-head rejection detection model."""

import json
import sys
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional
from transformers import AutoTokenizer

from .model import MultiHeadClassifier
from .taxonomies import get_head_config, get_id_to_label_mapping


def load_trained_model(model_path: str) -> tuple[MultiHeadClassifier, AutoTokenizer]:
    """Load a trained model and tokenizer."""
    model_path = Path(model_path)
    
    # Load model config
    with open(model_path / "model_config.json", "r") as f:
        model_config = json.load(f)
    
    # Initialize model
    model = MultiHeadClassifier(
        model_name=model_config["model_name"],
        freeze_encoder=model_config["freeze_encoder"],
        head_hidden_size=model_config.get("head_hidden_size")
    )
    
    # Load model weights
    model.load_state_dict(torch.load(model_path / "pytorch_model.bin", map_location="cpu"))
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    return model, tokenizer


def predict_single(
    model: MultiHeadClassifier,
    tokenizer: AutoTokenizer,
    prompt: str,
    response: str,
    max_length: int = 512,
    return_probabilities: bool = False
) -> Dict[str, Any]:
    """Make predictions for a single prompt-response pair."""
    
    # Combine prompt and response
    text = f"{prompt} [SEP] {response}"
    
    # Tokenize
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )
    
    # Make prediction
    with torch.no_grad():
        outputs = model(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            token_type_ids=encoding.get("token_type_ids")
        )
    
    # Process outputs
    predictions = {}
    label_mappings = {}
    
    # First, get HEAD_A prediction to determine gating logic
    head_a_logits = outputs.get("head_a")
    head_a_outcome = None
    if head_a_logits is not None:
        head_a_config = get_head_config("head_a")
        head_a_id_to_label = get_id_to_label_mapping("head_a")
        head_a_probs = torch.softmax(head_a_logits, dim=-1)
        head_a_pred_idx = torch.argmax(head_a_logits, dim=-1).item()
        head_a_outcome = head_a_id_to_label[head_a_pred_idx]
    
    for head_name, logits in outputs.items():
        head_config = get_head_config(head_name)
        id_to_label = get_id_to_label_mapping(head_name)
        label_mappings[head_name] = id_to_label
        
        # Apply gating logic for style heads
        if head_name == "head_b_a" and head_a_outcome:
            # Only process head_b_a if HEAD_A predicts a REFUSAL outcome
            if not head_a_outcome.startswith("REFUSAL."):
                predictions[head_name] = {
                    "prediction": "N/A (gated - not a refusal outcome)",
                    "probability": 0.0,
                    "gated": True
                }
                continue
        elif head_name == "head_b_b" and head_a_outcome:
            # Only process head_b_b if HEAD_A predicts a COMPLY outcome
            if not head_a_outcome.startswith("COMPLY."):
                predictions[head_name] = {
                    "prediction": "N/A (gated - not a compliance outcome)",
                    "probability": 0.0,
                    "gated": True
                }
                continue
        
        if head_config.head_type == "multilabel":
            # Multi-label: use threshold
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            if return_probabilities:
                predictions[head_name] = {
                    "probabilities": probs.squeeze().tolist(),
                    "predictions": preds.squeeze().tolist(),
                    "labels": [id_to_label[i] for i in range(len(id_to_label))]
                }
            else:
                # Get active labels
                active_indices = torch.where(preds.squeeze() > 0.5)[0].tolist()
                active_labels = [id_to_label[i] for i in active_indices]
                predictions[head_name] = {
                    "predictions": active_labels,
                    "probabilities": [probs.squeeze()[i].item() for i in active_indices]
                }
        elif head_config.head_type == "boolean":
            # Boolean: each flag is independent, output all with confidence
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            boolean_results = {}
            for i, label in id_to_label.items():
                boolean_results[label] = {
                    "value": bool(preds.squeeze()[i].item()),
                    "confidence": probs.squeeze()[i].item()
                }
            
            predictions[head_name] = boolean_results
        else:
            # Single-label: use argmax
            probs = torch.softmax(logits, dim=-1)
            pred_idx = torch.argmax(logits, dim=-1).item()
            pred_label = id_to_label[pred_idx]
            
            if return_probabilities:
                predictions[head_name] = {
                    "prediction": pred_label,
                    "probability": probs.squeeze()[pred_idx].item(),
                    "all_probabilities": probs.squeeze().tolist(),
                    "all_labels": [id_to_label[i] for i in range(len(id_to_label))]
                }
            else:
                predictions[head_name] = {
                    "prediction": pred_label,
                    "probability": probs.squeeze()[pred_idx].item()
                }
    
    return {
        "predictions": predictions,
        "label_mappings": label_mappings
    }


def predict_batch(
    model: MultiHeadClassifier,
    tokenizer: AutoTokenizer,
    data: List[Dict[str, str]],
    max_length: int = 512,
    return_probabilities: bool = False
) -> List[Dict[str, Any]]:
    """Make predictions for a batch of prompt-response pairs."""
    results = []
    
    for item in data:
        prompt = item["prompt"]
        response = item["response"]
        
        result = predict_single(
            model, tokenizer, prompt, response, max_length, return_probabilities
        )
        
        results.append({
            "prompt": prompt,
            "response": response,
            **result
        })
    
    return results


def main():
    """Main inference function for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run inference with trained model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--text", type=str, help="Single text to analyze (format: 'prompt|response')")
    parser.add_argument("--input_file", type=str, help="Input file with texts")
    parser.add_argument("--output_file", type=str, help="Output file for results")
    parser.add_argument("--return_probabilities", action="store_true", help="Return full probability distributions")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    try:
        model, tokenizer = load_trained_model(args.model_path)
        print("Model loaded successfully!")
    except FileNotFoundError as e:
        print(f"Error: Model not found at {args.model_path}")
        print(f"Details: {e}")
        sys.exit(1)
    
    if args.text:
        # Single text prediction
        if "|" in args.text:
            prompt, response = args.text.split("|", 1)
        else:
            prompt = "Sample prompt"
            response = args.text
        
        print(f"\nAnalyzing:")
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        print("\n" + "="*50)
        
        result = predict_single(
            model, tokenizer, prompt, response, 
            args.max_length, args.return_probabilities
        )
        
        # Pretty print results
        for head_name, pred in result["predictions"].items():
            head_config = get_head_config(head_name)
            print(f"\n{head_name.upper()} ({head_config.head_type}):")
            
            if head_config.head_type == "multilabel":
                if pred["predictions"]:
                    print(f"  Active categories: {pred['predictions']}")
                    print(f"  Probabilities: {[f'{p:.3f}' for p in pred['probabilities']]}")
                else:
                    print("  No active categories")
            elif head_config.head_type == "boolean":
                for flag_name, flag_data in pred.items():
                    print(f"  {flag_name}: {flag_data['value']} (confidence: {flag_data['confidence']:.3f})")
            else:
                if pred.get('gated', False):
                    print(f"  {pred['prediction']}")
                else:
                    print(f"  Prediction: {pred['prediction']}")
                    print(f"  Confidence: {pred['probability']:.3f}")
    
    elif args.input_file:
        # Batch prediction
        print(f"Loading data from {args.input_file}...")
        
        with open(args.input_file, "r") as f:
            if args.input_file.endswith(".json"):
                data = json.load(f)
            else:
                # Assume CSV or text file
                data = []
                with open(args.input_file, "r") as f:
                    for line in f:
                        if "|" in line:
                            prompt, response = line.strip().split("|", 1)
                            data.append({"prompt": prompt, "response": response})
        
        print(f"Processing {len(data)} samples...")
        results = predict_batch(
            model, tokenizer, data, args.max_length, args.return_probabilities
        )
        
        # Save results
        if args.output_file:
            with open(args.output_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.output_file}")
        else:
            # Print first few results
            for i, result in enumerate(results[:3]):
                print(f"\n--- Sample {i+1} ---")
                print(f"Prompt: {result['prompt']}")
                print(f"Response: {result['response']}")
                
                for head_name, pred in result["predictions"].items():
                    head_config = get_head_config(head_name)
                    if head_config.head_type == "multilabel":
                        print(f"{head_name}: {pred['predictions']}")
                    else:
                        print(f"{head_name}: {pred['prediction']} ({pred['probability']:.3f})")
    
    else:
        print("Please provide either --text or --input_file")


if __name__ == "__main__":
    main()
