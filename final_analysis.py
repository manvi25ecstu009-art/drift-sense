import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# DRIFT-SENSE FINAL ANALYSIS
# ============================================================

RESULT_FILE = "dataset/real_localization_results.csv"
OUTPUT_DIR = "dataset/final_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print()
print("=" * 60)
print("           DRIFT-SENSE FINAL ANALYSIS")
print("=" * 60)
print()

# ------------------------------------------------------------
# LOAD RESULTS
# ------------------------------------------------------------

df = pd.read_csv(RESULT_FILE)

print("Loaded results:")
print(f"Number of pairs: {len(df)}")
print()

print("Columns found:")
print(list(df.columns))
print()

# ------------------------------------------------------------
# BASIC STATISTICS
# ------------------------------------------------------------

errors = df["error_pixels"]

mean_error = errors.mean()
median_error = errors.median()
min_error = errors.min()
max_error = errors.max()

within_5 = (errors <= 5).mean() * 100
within_10 = (errors <= 10).mean() * 100
within_20 = (errors <= 20).mean() * 100

print("=" * 60)
print("FINAL STATISTICS")
print("=" * 60)

print(f"Mean error       : {mean_error:.2f} px")
print(f"Median error     : {median_error:.2f} px")
print(f"Minimum error    : {min_error:.2f} px")
print(f"Maximum error    : {max_error:.2f} px")
print(f"Within 5 px      : {within_5:.1f}%")
print(f"Within 10 px     : {within_10:.1f}%")
print(f"Within 20 px     : {within_20:.1f}%")

# ------------------------------------------------------------
# 1. ERROR FOR EVERY IMAGE PAIR
# ------------------------------------------------------------

plt.figure(figsize=(12, 6))

x = np.arange(1, len(df) + 1)

plt.bar(x, errors)

plt.axhline(
    mean_error,
    linestyle="--",
    label=f"Mean = {mean_error:.2f}px"
)

plt.xlabel("Image Pair")
plt.ylabel("Localization Error (pixels)")
plt.title("Drift-Sense Localization Error per Image Pair")
plt.xticks(x)
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "error_per_pair.png"
    ),
    dpi=300
)

plt.close()

print()
print("Saved: error_per_pair.png")

# ------------------------------------------------------------
# 2. ERROR DISTRIBUTION
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.hist(
    errors,
    bins=10,
    edgecolor="black"
)

plt.xlabel("Localization Error (pixels)")
plt.ylabel("Number of Image Pairs")
plt.title("Distribution of Localization Error")
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "error_distribution.png"
    ),
    dpi=300
)

plt.close()

print("Saved: error_distribution.png")

# ------------------------------------------------------------
# 3. ACCURACY THRESHOLDS
# ------------------------------------------------------------

thresholds = [
    "≤ 5 px",
    "≤ 10 px",
    "≤ 20 px"
]

percentages = [
    within_5,
    within_10,
    within_20
]

plt.figure(figsize=(8, 6))

plt.bar(
    thresholds,
    percentages
)

plt.xlabel("Localization Accuracy Threshold")
plt.ylabel("Percentage of Image Pairs (%)")
plt.title("Drift-Sense Localization Accuracy")

for i, value in enumerate(percentages):

    plt.text(
        i,
        value + 1,
        f"{value:.1f}%",
        ha="center"
    )

plt.ylim(
    0,
    100
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "accuracy_thresholds.png"
    ),
    dpi=300
)

plt.close()

print("Saved: accuracy_thresholds.png")

# ------------------------------------------------------------
# 4. WORST CASES
# ------------------------------------------------------------

worst = df.sort_values(
    "error_pixels",
    ascending=False
).head(10)

print()
print("=" * 60)
print("TOP 10 WORST CASES")
print("=" * 60)

for _, row in worst.iterrows():

    print(
        f"{row['filename']} : "
        f"{row['error_pixels']:.2f} px"
    )

worst.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "worst_cases.csv"
    ),
    index=False
)

print()
print("Saved: worst_cases.csv")

# ------------------------------------------------------------
# 5. BEST CASES
# ------------------------------------------------------------

best = df.sort_values(
    "error_pixels",
    ascending=True
).head(10)

print()
print("=" * 60)
print("TOP 10 BEST CASES")
print("=" * 60)

for _, row in best.iterrows():

    print(
        f"{row['filename']} : "
        f"{row['error_pixels']:.2f} px"
    )

best.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "best_cases.csv"
    ),
    index=False
)

print()
print("Saved: best_cases.csv")

# ------------------------------------------------------------
# 6. SAVE SUMMARY
# ------------------------------------------------------------

summary = pd.DataFrame({
    "Metric": [
        "Number of image pairs",
        "Mean error (px)",
        "Median error (px)",
        "Minimum error (px)",
        "Maximum error (px)",
        "Within 5 px (%)",
        "Within 10 px (%)",
        "Within 20 px (%)"
    ],
    "Value": [
        len(df),
        mean_error,
        median_error,
        min_error,
        max_error,
        within_5,
        within_10,
        within_20
    ]
})

summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "final_summary.csv"
    ),
    index=False
)

print()
print("Saved: final_summary.csv")

# ------------------------------------------------------------
# DONE
# ------------------------------------------------------------

print()
print("=" * 60)
print("             ANALYSIS COMPLETE")
print("=" * 60)

print()
print("All files saved in:")
print(OUTPUT_DIR)
print()