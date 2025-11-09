import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import os


def load_model_and_data(model_path, num_features):
    """Load model and prepare data for SHAP analysis.
    
    Args:
        model_path: Path to the trained model
        num_features: Number of top features to use
        
    Returns:
        clf: Trained Random Forest model
        x_test: Test features
        y_test: Test labels
        feature_names: List of feature names
    """
    
    # Load model
    clf = joblib.load(model_path)
    
    # Load feature names
    feature_importance = pd.read_csv('rf_feature_importance_cross_validation.csv')
    feature_names = feature_importance['feature'].tolist()[:num_features]
    
    # Load features
    features = pd.read_csv('results/features.csv')
    
    # Load test set (for SHAP analysis)
    test_df = pd.read_csv('test_set.csv')
    y_test = test_df['label'].to_numpy()
    test_ids = test_df['ID'].to_list()
    
    # Get test data
    x_test = []
    for id in test_ids:
        features_id = features[features['Unnamed: 0'] == id]
        features_id = features_id.drop(columns=['Unnamed: 0'])
        features_id = features_id[feature_names]
        x_test.append(features_id.to_numpy())
    x_test = np.concatenate(x_test, axis=0)
    
    print(f"✓ Loaded {len(x_test)} test samples with {num_features} features")
    
    return clf, x_test, y_test, feature_names


def create_shap_plots(model_path, num_features, max_display=20):
    """Generate comprehensive SHAP plots for a Random Forest model.
    
    Args:
        model_path: Path to the trained model
        num_features: Number of features used in the model
        max_display: Maximum number of features to display in plots
    """
    
    print('\n' + '='*80)
    print(f"Analyzing {model_path}...")
    print('='*80)
    
    model_name = Path(model_path).stem
    
    # Load model and data
    clf, x_test, y_test, feature_names = load_model_and_data(model_path, num_features)
    
    print(f"Computing SHAP values for {len(x_test)} samples...")
    
    # Create SHAP explainer (TreeExplainer is fast for tree-based models)
    explainer = shap.TreeExplainer(clf)
    
    # Compute SHAP values
    shap_values = explainer.shap_values(x_test)
    
    # For binary classification, shap_values is a list [class_0, class_1]
    # We want SHAP values for the positive class (CAD = 1)
    if isinstance(shap_values, list):
        shap_values_pos = shap_values[1]
        expected_value = explainer.expected_value[1]
    else:
        shap_values_pos = shap_values
        expected_value = explainer.expected_value
    
    print(f"✓ SHAP values computed")
    
    # Create output directory
    os.makedirs('rf_shap_plots', exist_ok=True)
    
    # ============================================
    # 1. Summary Plot (Beeswarm) - Global importance
    # ============================================
    print("Creating summary beeswarm plot...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values_pos, x_test, 
                      feature_names=feature_names,
                      max_display=max_display, 
                      show=False)
    plt.title(f'SHAP Summary Plot - {model_name}', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(f'rf_shap_plots/{model_name}_summary_beeswarm.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Summary beeswarm plot saved")
    
    # ============================================
    # 2. Summary Plot (Bar) - Mean absolute SHAP values
    # ============================================
    print("Creating summary bar plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_pos, x_test, 
                      feature_names=feature_names,
                      plot_type="bar",
                      max_display=max_display,
                      show=False)
    plt.title(f'SHAP Feature Importance - {model_name}', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(f'rf_shap_plots/{model_name}_summary_bar.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Summary bar plot saved")
    
    # ============================================
    # 3. Dependence Plots - Top 5 features
    # ============================================
    print("Creating dependence plots for top 5 features...")
    mean_abs_shap = np.abs(shap_values_pos).mean(axis=0)
    top_features_idx = np.argsort(mean_abs_shap)[::-1][:5]
    
    for i, feat_idx in enumerate(top_features_idx):
        try:
            plt.figure(figsize=(10, 6))
            # Set interaction_index=None to avoid the ambiguous array error
            shap.dependence_plot(feat_idx, shap_values_pos, x_test,
                                feature_names=feature_names,
                                interaction_index=None,
                                show=False)
            plt.title(f'SHAP Dependence Plot - {feature_names[feat_idx]}', 
                      fontsize=14, pad=20)
            plt.tight_layout()
            safe_feature_name = feature_names[feat_idx].replace('/', '_').replace(' ', '_')
            plt.savefig(f'rf_shap_plots/{model_name}_dependence_{i+1}_{safe_feature_name}.png',
                        dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"  ⚠ Warning: Could not create dependence plot for {feature_names[feat_idx]}: {str(e)}")
            plt.close()
            continue
    print(f"✓ Dependence plots saved")
    
    # ============================================
    # 4. Waterfall Plot - Example predictions
    # ============================================
    print("Creating waterfall plots for example cases...")
    
    # Get predictions
    probs = clf.predict_proba(x_test)[:, 1]
    
    # True positive with high confidence
    tp_idx = np.where((y_test == 1) & (probs > 0.8))[0]
    if len(tp_idx) > 0:
        idx = tp_idx[0]
        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(shap.Explanation(
            values=shap_values_pos[idx],
            base_values=expected_value,
            data=x_test[idx],
            feature_names=feature_names
        ), max_display=15, show=False)
        plt.title(f'True Positive Example (prob={probs[idx]:.3f})', fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(f'rf_shap_plots/{model_name}_waterfall_true_positive.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ True positive waterfall plot saved")
    
    # True negative with high confidence
    tn_idx = np.where((y_test == 0) & (probs < 0.2))[0]
    if len(tn_idx) > 0:
        idx = tn_idx[0]
        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(shap.Explanation(
            values=shap_values_pos[idx],
            base_values=expected_value,
            data=x_test[idx],
            feature_names=feature_names
        ), max_display=15, show=False)
        plt.title(f'True Negative Example (prob={probs[idx]:.3f})', fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(f'rf_shap_plots/{model_name}_waterfall_true_negative.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ True negative waterfall plot saved")
    
    # False positive (if exists)
    fp_idx = np.where((y_test == 0) & (probs > 0.8))[0]
    if len(fp_idx) > 0:
        idx = fp_idx[0]
        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(shap.Explanation(
            values=shap_values_pos[idx],
            base_values=expected_value,
            data=x_test[idx],
            feature_names=feature_names
        ), max_display=15, show=False)
        plt.title(f'False Positive Example (prob={probs[idx]:.3f})', fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(f'rf_shap_plots/{model_name}_waterfall_false_positive.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ False positive waterfall plot saved")
    
    # ============================================
    # 5. Save SHAP values for further analysis
    # ============================================
    print("Saving SHAP values and data...")
    np.save(f'rf_shap_plots/{model_name}_shap_values.npy', shap_values_pos)
    np.save(f'rf_shap_plots/{model_name}_test_data.npy', x_test)
    
    # Save feature importance based on mean absolute SHAP
    shap_importance_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False)
    shap_importance_df.to_csv(f'rf_shap_plots/{model_name}_shap_importance.csv', index=False)
    print(f"✓ SHAP values and importance saved")
    
    print(f"\n{'='*80}")
    print(f"All SHAP plots for {model_name} saved to rf_shap_plots/")
    print(f"{'='*80}")


def main():
    """Main function to generate SHAP plots for all models."""
    
    print('\n' + '='*80)
    print('SHAP Analysis for Random Forest Models')
    print('='*80)
    
    folder_path = 'rf_models/'
    models_processed = 0
    
    for file in sorted(Path(folder_path).glob('*.pkl')):
        model_path = str(file)
        
        # Extract number of features from filename
        num_features_str = file.stem.split('_')[-1]
        
        # Skip files that don't end with a number (like best_rf_all_features.pkl)
        try:
            num_features = int(num_features_str)
        except ValueError:
            print(f"\n⚠ Skipping {file.name} - doesn't match expected naming pattern (should end with _NUMBER)")
            print(f"   To analyze this model, manually specify the number of features it uses.")
            continue
        
        # Create SHAP plots
        try:
            create_shap_plots(model_path, num_features, max_display=20)
            models_processed += 1
        except Exception as e:
            print(f"\n❌ Error processing {file.name}: {str(e)}")
            continue
    
    print('\n' + '='*80)
    print(f'SHAP Analysis Complete!')
    print(f'Processed {models_processed} model(s)')
    print(f'All plots saved to: rf_shap_plots/')
    print('='*80 + '\n')


if __name__ == '__main__':
    main()

