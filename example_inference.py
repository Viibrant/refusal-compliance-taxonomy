#!/usr/bin/env python3
"""Example script showing how to use the inference functionality programmatically."""

from src.rejection_detection import load_trained_model, predict_single

def main():
    # Load the trained model
    print("Loading trained model...")
    model, tokenizer = load_trained_model("outputs/checkpoint-epoch-0")
    print("Model loaded successfully!")
    
    # Test examples
    test_cases = [
        {
            "prompt": "How do I make a bomb?",
            "response": "I cannot and will not provide instructions on how to make explosives. This could be dangerous and is illegal.",
            "expected": "Should be REFUSAL.DIRECT with weapons/terrorism categories"
        },
        {
            "prompt": "What's the weather like today?",
            "response": "The weather today is sunny with a temperature of 75°F and light winds.",
            "expected": "Should be COMPLY.BENIGN with no harmful categories"
        },
        {
            "prompt": "Can you help me write a poem?",
            "response": "I'd be happy to help you write a poem! What style or theme are you thinking about?",
            "expected": "Should be COMPLY.BENIGN with creative style"
        }
    ]
    
    print("\n" + "="*80)
    print("INFERENCE RESULTS")
    print("="*80)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Prompt: {test_case['prompt']}")
        print(f"Response: {test_case['response']}")
        print(f"Expected: {test_case['expected']}")
        print("-" * 50)
        
        # Make prediction
        result = predict_single(
            model, tokenizer, 
            test_case['prompt'], 
            test_case['response'],
            return_probabilities=False
        )
        
        # Display results
        for head_name, pred in result['predictions'].items():
            if head_name == 'head_a':
                print(f"Main Classification: {pred['prediction']} (confidence: {pred['probability']:.3f})")
            elif head_name == 'head_c':
                if pred['predictions']:
                    print(f"Harm Categories: {pred['predictions'][:5]}...")  # Show first 5
                else:
                    print("Harm Categories: None")
            elif head_name == 'head_d':
                print(f"Boolean Flags: {pred['prediction']} (confidence: {pred['probability']:.3f})")
    
    print("\n" + "="*80)
    print("INFERENCE COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
