#!/usr/bin/env python3

from model_evaluator import ModelEvaluator
import json

def quick_evaluation():
    """Quick model evaluation"""
    print("🚀 Quick Model Evaluation")
    print("="*40)
    
    evaluator = ModelEvaluator()
    
    # Test with smaller sample for speed
    results = evaluator.compare_models(
        model_keys=["current", "multilingual"], 
        sample_size=20
    )
    
    # Print quick summary
    for model_key, result in results['results'].items():
        if result.get('success'):
            print(f"\n📊 {model_key}:")
            print(f"   Rating: {result.get('performance_rating', 'Unknown')}")
            print(f"   Overall: {result.get('overall_score', 0):.3f}")
            
            if 'arabic_metrics' in result:
                print(f"   Arabic: {result['arabic_metrics']['avg_similarity']:.3f}")
            
            if 'english_metrics' in result:
                print(f"   English: {result['english_metrics']['avg_similarity']:.3f}")
    
    # Show recommendation
    summary = results['summary']
    if summary['best_overall']:
        print(f"\n🏆 Recommended: {summary['best_overall']['model']}")

if __name__ == "__main__":
    quick_evaluation()
