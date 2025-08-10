#!/usr/bin/env python3
"""
Script to compare results from different Optuna optimizations.
"""

import json
from pathlib import Path
import pandas as pd

def load_threshold_info(model_name):
    """Load threshold information from JSON file."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent.parent
    model_dir = project_root / "production/model"
    threshold_file = model_dir / f"{model_name}_threshold.json"
    
    if threshold_file.exists():
        with open(threshold_file, 'r') as f:
            return json.load(f)
    else:
        return None

def compare_models():
    """Compare different model results."""
    print("=" * 80)
    print("COMPARISON OF OPTUNA OPTIMIZATION RESULTS")
    print("=" * 80)
    
    # List of models to compare
    models = [
        "xgb_sin_se_optuna",
        "xgb_fe_optuna", 
        "xgb_fe_optuna_balanced"
    ]
    
    results = {}
    
    for model in models:
        info = load_threshold_info(model)
        if info:
            results[model] = info
            print(f"✅ Loaded results for {model}")
        else:
            print(f"❌ No results found for {model}")
    
    if not results:
        print("\n❌ No model results found to compare!")
        print("Please run the Optuna optimizations first.")
        return
    
    # Create comparison table
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    
    comparison_data = []
    
    for model_name, info in results.items():
        row = {
            'Model': model_name.replace('_', ' ').title(),
            'Threshold': f"{info.get('threshold', 'N/A'):.4f}",
            'Recall (Test)': f"{info.get('recall_test', 'N/A'):.4f}",
            'Precision (Test)': f"{info.get('precision_test', 'N/A'):.4f}",
            'F2/F3 Score (Test)': f"{info.get('f2_score_test', info.get('f3_score_test', 'N/A')):.4f}",
            'Custom Score (Test)': f"{info.get('custom_score_test', 'N/A'):.4f}",
            'False Positive Rate': f"{info.get('false_positive_rate', 'N/A'):.4f}",
            'False Negative Rate': f"{info.get('false_negative_rate', 'N/A'):.4f}"
        }
        comparison_data.append(row)
    
    # Create DataFrame and display
    df = pd.DataFrame(comparison_data)
    print(df.to_string(index=False))
    
    # Detailed analysis
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS")
    print("=" * 80)
    
    for model_name, info in results.items():
        print(f"\n📊 {model_name.replace('_', ' ').title()}:")
        print(f"  - Threshold: {info.get('threshold', 'N/A'):.4f}")
        print(f"  - Recall: {info.get('recall_test', 'N/A'):.1%}")
        print(f"  - Precision: {info.get('precision_test', 'N/A'):.1%}")
        print(f"  - F2/F3 Score: {info.get('f2_score_test', info.get('f3_score_test', 'N/A')):.4f}")
        print(f"  - False Positive Rate: {info.get('false_positive_rate', 'N/A'):.1%}")
        print(f"  - False Negative Rate: {info.get('false_negative_rate', 'N/A'):.1%}")
        
        # Confusion matrix
        if 'confusion_matrix' in info:
            cm = info['confusion_matrix']
            print(f"  - Confusion Matrix:")
            print(f"    TN: {cm[0][0]}, FP: {cm[0][1]}")
            print(f"    FN: {cm[1][0]}, TP: {cm[1][1]}")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    if len(results) > 1:
        # Find best recall
        best_recall_model = max(results.items(), 
                               key=lambda x: x[1].get('recall_test', 0))
        
        # Find best precision
        best_precision_model = max(results.items(), 
                                  key=lambda x: x[1].get('precision_test', 0))
        
        # Find best F2/F3 score
        best_f_score_model = max(results.items(), 
                                key=lambda x: x[1].get('f2_score_test', x[1].get('f3_score_test', 0)))
        
        print(f"🎯 Best Recall: {best_recall_model[0]} ({best_recall_model[1].get('recall_test', 0):.1%})")
        print(f"🎯 Best Precision: {best_precision_model[0]} ({best_precision_model[1].get('precision_test', 0):.1%})")
        print(f"🎯 Best F2/F3 Score: {best_f_score_model[0]} ({best_f_score_model[1].get('f2_score_test', best_f_score_model[1].get('f3_score_test', 0)):.4f})")
        
        print(f"\n💡 Recommendation:")
        if 'balanced' in best_f_score_model[0]:
            print(f"   Use {best_f_score_model[0]} for maximum failure detection")
        else:
            print(f"   Use {best_f_score_model[0]} for balanced performance")
    
    return results

if __name__ == "__main__":
    compare_models()
