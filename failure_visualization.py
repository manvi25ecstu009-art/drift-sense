import cv2
import pandas as pd
import os

# ============================================================
# DRIFT-SENSE FAILURE VISUALIZATION
# ============================================================

RESULT_FILE = "dataset/real_localization_results.csv"

SEARCH_DIR = "dataset/search"

OUTPUT_DIR = "dataset/final_analysis/failure_visualizations"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD RESULTS
# ============================================================

df = pd.read_csv(RESULT_FILE)

# Get 5 worst cases
worst = df.sort_values(
    "error_pixels",
    ascending=False
).head(5)

print()
print("=" * 60)
print("       DRIFT-SENSE FAILURE VISUALIZATION")
print("=" * 60)
print()

# ============================================================
# PROCESS EACH FAILURE
# ============================================================

for _, row in worst.iterrows():

    filename = row["filename"]

    true_x = int(row["true_x"])
    true_y = int(row["true_y"])

    predicted_x = int(row["predicted_x"])
    predicted_y = int(row["predicted_y"])

    error = float(row["error_pixels"])

    # Convert reference_021.png -> search_021.png
    search_filename = filename.replace(
        "reference_",
        "search_"
    )

    search_path = os.path.join(
        SEARCH_DIR,
        search_filename
    )

    image = cv2.imread(search_path)

    if image is None:
        print(
            f"Could not read: {search_path}"
        )
        continue

    # ========================================================
    # GROUND TRUTH
    # ========================================================

    cv2.circle(
        image,
        (true_x, true_y),
        8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        image,
        "GROUND TRUTH",
        (true_x + 10, true_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
        cv2.LINE_AA
    )

    # ========================================================
    # PREDICTION
    # ========================================================

    cv2.circle(
        image,
        (predicted_x, predicted_y),
        8,
        (0, 0, 255),
        2
    )

    cv2.putText(
        image,
        "PREDICTION",
        (predicted_x + 10, predicted_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1,
        cv2.LINE_AA
    )

    # ========================================================
    # ERROR LINE
    # ========================================================

    cv2.line(
        image,
        (true_x, true_y),
        (predicted_x, predicted_y),
        (255, 0, 0),
        2
    )

    # ========================================================
    # RESULT INFORMATION
    # ========================================================

    text = (
        f"{filename} | Error = {error:.2f} px"
    )

    cv2.putText(
        image,
        text,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        image,
        text,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA
    )

    # ========================================================
    # SAVE IMAGE
    # ========================================================

    output_name = filename.replace(
        ".png",
        "_failure.png"
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        output_name
    )

    cv2.imwrite(
        output_path,
        image
    )

    print(
        f"{filename}: error = {error:.2f}px"
    )

    print(
        f"Saved: {output_path}"
    )

# ============================================================
# DONE
# ============================================================

print()
print("=" * 60)
print("       FAILURE VISUALIZATION COMPLETE")
print("=" * 60)
print()