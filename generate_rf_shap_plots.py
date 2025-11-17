import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os

def generate_shap_plots(model_name, num_features=None):
    """
    Generate SHAP plots for a specific RF model.
    
    Args:
        model_name: Name of the model file (without .pkl extension)
        num_features: Number of features to use (None for all 417 features)
    """
    print(f"\n{'='*80}")
    print(f"Processing: {model_name}")
    print(f"{'='*80}")
    
    # Load the model
    model_path = f'rf_models/{model_name}.pkl'
    model = joblib.load(model_path)
    print(f"✓ Loaded model from: {model_path}")
    print(f"Model type: {type(model)}")
    print(f"Number of features in model: {model.n_features_in_}")
    
    # Load feature names from the CSV
    all_feature_names = pd.read_csv('rf_feature_importance_cross_validation.csv')['feature'].tolist()
    
    if num_features is None:
        # Use all 417 features
        feature_names = all_feature_names
    else:
        # Use top N features
        feature_names = all_feature_names[:num_features]
    
    print(f"Number of features to use: {len(feature_names)}")
    
    # Load pre-extracted features from results/features.csv
    features_df = pd.read_csv('results/features.csv')
    
    # Load test set
    test_df = pd.read_csv('test_set.csv')
    test_ids = test_df['ID'].to_list()
    y_test = test_df['label'].to_numpy()
    
    # Get features for test set (already extracted)
    print("Loading features from test set...")
    x_test = []
    for id in test_ids:
        features_id = features_df[features_df['Unnamed: 0'] == id]
        features_id = features_id.drop(columns=['Unnamed: 0'])
        # Keep only the desired features and enforce ordering
        features_id = features_id[feature_names]
        x_test.append(features_id.to_numpy())
    
    x_test = np.concatenate(x_test, axis=0)
    print(f"Test set shape: {x_test.shape}")
    
    # Validate dimensions
    assert x_test.shape[1] == len(feature_names), f"Mismatch: x_test has {x_test.shape[1]} features but {len(feature_names)} feature names"
    assert x_test.shape[1] == model.n_features_in_, f"Mismatch: x_test has {x_test.shape[1]} features but model expects {model.n_features_in_}"
    print("✓ All dimensions match correctly")
    
    # Create SHAP explainer
    print("Creating SHAP explainer...")
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values
    print("Calculating SHAP values (this may take a while)...")
    print(f"Computing SHAP for {x_test.shape[0]} samples and {x_test.shape[1]} features...")
    shap_values = explainer.shap_values(x_test)
    
    # Handle different SHAP output formats for binary classification
    print(f"SHAP values type: {type(shap_values)}")
    if isinstance(shap_values, list):
        print(f"  SHAP values is a list with {len(shap_values)} elements (one per class)")
        shap_values = shap_values[1]  # Use positive class (CAD = 1)
        print(f"  Using class 1 (CAD positive) SHAP values")
    elif len(shap_values.shape) == 3:
        # 3D array format: (samples, features, classes)
        print(f"  SHAP values is a 3D array with shape: {shap_values.shape}")
        shap_values = shap_values[:, :, 1]  # Extract SHAP values for class 1 (CAD positive)
        print(f"  Extracted SHAP values for class 1 (CAD positive)")
    
    print(f"Final SHAP values shape: {shap_values.shape}")
    assert shap_values.shape == x_test.shape, f"SHAP shape {shap_values.shape} doesn't match data shape {x_test.shape}"
    print("✓ SHAP values computed successfully")
    
    # Save SHAP values and test data for future use
    np.save(f'rf_shap_plots/{model_name}_shap_values.npy', shap_values)
    np.save(f'rf_shap_plots/{model_name}_test_data.npy', x_test)
    print("✓ SHAP values saved to disk")
    
    # Determine max_display based on number of features
    max_display = min(20, len(feature_names))
    
    # Generate beeswarm plot with feature names
    print("Generating SHAP beeswarm plot...")
    plt.figure(figsize=(10, 12))
    shap.summary_plot(shap_values, x_test, feature_names=feature_names, plot_type="dot", show=False, max_display=max_display)
    plt.tight_layout()
    plt.savefig(f'rf_shap_plots/{model_name}_summary_beeswarm.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Beeswarm plot saved (showing top {max_display} features)")
    
    # Generate bar plot with feature names
    print("Generating SHAP bar plot...")
    plt.figure(figsize=(10, 12))
    shap.summary_plot(shap_values, x_test, feature_names=feature_names, plot_type="bar", show=False, max_display=max_display)
    plt.tight_layout()
    plt.savefig(f'rf_shap_plots/{model_name}_summary_bar.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Bar plot saved (showing top {max_display} features)")
    
    print(f"\n✓ Done with {model_name}!")
    print(f"  - rf_shap_plots/{model_name}_summary_beeswarm.png")
    print(f"  - rf_shap_plots/{model_name}_summary_bar.png")


if __name__ == '__main__':
    # Ensure output directory exists
    os.makedirs('rf_shap_plots', exist_ok=True)
    
    # List of models to process
    models_to_process = [
        ('best_rf_all_features_417', None),      # All 417 features
        ('best_rf_selected_features_25', 25),     # Top 25 features
        ('best_rf_selected_features_50', 50),     # Top 50 features
        ('best_rf_selected_features_75', 75),     # Top 75 features
        ('best_rf_selected_features_100', 100),   # Top 100 features
        ('best_rf_selected_features_150', 150),   # Top 150 features
    ]
    
    print("="*80)
    print("SHAP Plot Generation for All RF Models")
    print("="*80)
    print(f"Total models to process: {len(models_to_process)}")
    
    for i, (model_name, num_features) in enumerate(models_to_process, 1):
        print(f"\n[{i}/{len(models_to_process)}]")
        try:
            generate_shap_plots(model_name, num_features)
        except Exception as e:
            print(f"❌ Error processing {model_name}: {e}")
            continue
    
    print("\n" + "="*80)
    print("All SHAP plots generated successfully!")
    print("="*80)

