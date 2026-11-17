import pandas as pd
import numpy as np
import scipy.stats as stats
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the Master Data
print("Loading experiment_results_MASTER.csv...")
df = pd.read_csv("experiment_results_MASTER.csv")

# 2. Calculate Run-Level Metrics (Aggregate by Method and Seed)
run_metrics = df.groupby(['method', 'seed'])[['EM', 'F1', 'hallucinated']].mean().reset_index()

# Calculate Mean, Standard Deviation, and 95% Confidence Interval across the 5 seeds
summary_stats = run_metrics.groupby('method')[['EM', 'F1', 'hallucinated']].agg(['mean', 'std'])

def calculate_95_ci(std, n=5):
    return 1.96 * (std / np.sqrt(n))

print("\n--- SUMMARY STATISTICS (Across 5 Seeds) ---")
for metric in ['EM', 'F1', 'hallucinated']:
    print(f"\nMetric: {metric}")
    for method in summary_stats.index:
        mean = summary_stats.loc[method, (metric, 'mean')]
        std = summary_stats.loc[method, (metric, 'std')]
        ci = calculate_95_ci(std)
        print(f"{method}: {mean:.4f} ± {ci:.4f} (SD: {std:.4f})")

# 3. Prepare Data for Statistical Testing
# Average each question's score across its 5 seeds, then pivot to compare methods side-by-side
q_agg = df.groupby(['method', 'question_id'])[['F1']].mean().reset_index()
pivot_df = q_agg.pivot(index='question_id', columns='method', values='F1').dropna()

# Extract arrays for the tests
prompting = pivot_df['Prompting Only']
rag_bm25 = pivot_df['RAG-BM25']
rag_dense = pivot_df['RAG-Dense']
lora = pivot_df['LoRA Fine-Tuned']

# 4. Friedman Test (Checks if at least one method is statistically different)
print("\n--- FRIEDMAN TEST ---")
stat, p_value = stats.friedmanchisquare(prompting, rag_bm25, rag_dense, lora)
print(f"Friedman Chi-Square Statistic: {stat:.4f}, p-value: {p_value:.4e}")

if p_value < 0.05:
    print("Result: Significant differences exist between the methods. Proceeding to post-hoc tests...")
    
    # Post-hoc Nemenyi test for all pairwise comparisons
    posthoc = sp.posthoc_nemenyi([prompting, rag_bm25, rag_dense, lora])
    posthoc.columns = ['Prompting', 'RAG-BM25', 'RAG-Dense', 'LoRA']
    posthoc.index = ['Prompting', 'RAG-BM25', 'RAG-Dense', 'LoRA']
    print("\nNemenyi Post-Hoc p-values:")
    print(posthoc)
    
    # 5. Wilcoxon Signed-Rank Test (Comparing the top two methods, typically Dense vs LoRA)
    print("\n--- WILCOXON SIGNED-RANK TEST (Dense vs LoRA) ---")
    w_stat, w_p = stats.wilcoxon(rag_dense, lora)
    print(f"Wilcoxon Statistic: {w_stat:.4f}, p-value: {w_p:.4e}")
    if w_p < 0.05:
        print("Result: There is a statistically significant difference between RAG-Dense and LoRA.")
    else:
        print("Result: No statistically significant difference between RAG-Dense and LoRA.")
else:
    print("Result: No significant differences found among the methods.")

# 6. Generate Bar Chart with Error Bars
print("\nGenerating charts...")
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
metrics = ['EM', 'F1', 'hallucinated']
titles = ['Exact Match (Higher is better)', 'F1 Score (Higher is better)', 'Hallucination Rate (Lower is better)']

for i, metric in enumerate(metrics):
    sns.barplot(
        x='method', y=metric, data=run_metrics, 
        ax=axes[i], capsize=.1, errorbar='sd',
        order=['Prompting Only', 'RAG-BM25', 'RAG-Dense', 'LoRA Fine-Tuned']
    )
    axes[i].set_title(titles[i], fontweight='bold')
    axes[i].set_xlabel('')
    axes[i].set_ylabel('Score')
    axes[i].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig("experiment_results_chart.png", dpi=300)
print("Charts saved as 'experiment_results_chart.png'.")