import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the feature importance data
df = pd.read_csv('rf_feature_importance_cross_validation.csv')

# Get top 10 most important features
top_10 = df.head(10)

# Create the visualization
plt.figure(figsize=(10, 6))
plt.barh(range(len(top_10)), top_10['importance'], color='steelblue')
plt.yticks(range(len(top_10)), top_10['feature'])
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Top 10 Most Important Features', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()  # Highest importance at the top
plt.tight_layout()
plt.grid(axis='x', alpha=0.3)

# Save the figure
plt.savefig('feature_importance_top10.png', dpi=300, bbox_inches='tight')
print("Figure saved as 'feature_importance_top10.png'")

# Display the plot
plt.show()

# Print the top 10 features with their importance values
print("\nTop 10 Most Important Features:")
print("=" * 50)
for idx, row in top_10.iterrows():
    print(f"{idx+1:2d}. {row['feature']:25s} {row['importance']:.6f}")

