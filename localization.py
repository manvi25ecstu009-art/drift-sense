import os
import csv
import cv2
import numpy as np


# ============================================================
# PATHS
# ============================================================

DATASET_DIR = "dataset"

REFERENCE_DIR = os.path.join(
    DATASET_DIR,
    "reference"
)

SEARCH_DIR = os.path.join(
    DATASET_DIR,
    "search"
)

GROUND_TRUTH_FILE = os.path.join(
    DATASET_DIR,
    "ground_truth.csv"
)

RESULT_FILE = os.path.join(
    DATASET_DIR,
    "real_localization_results.csv"
)


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

def load_ground_truth():

    ground_truth = {}

    with open(
        GROUND_TRUTH_FILE,
        "r",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            reference_file = os.path.basename(
                row["reference_file"]
            )

            true_x = int(
                float(row["x"])
            )

            true_y = int(
                float(row["y"])
            )

            ground_truth[
                reference_file
            ] = (
                true_x,
                true_y
            )

    return ground_truth


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):

    if image is None:
        return None

    gray = image.copy()

    # Normalize contrast
    gray = cv2.normalize(
        gray,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return gray


# ============================================================
# EDGE IMAGE
# ============================================================

def get_edges(image):

    return cv2.Canny(
        image,
        50,
        150
    )


# ============================================================
# ROTATE IMAGE
# ============================================================

def rotate_image(
    image,
    angle
):

    h, w = image.shape[:2]

    center = (
        w / 2,
        h / 2
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    rotated = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

    return rotated


# ============================================================
# SCALE REFERENCE
# ============================================================

def scale_image(
    image,
    scale
):

    h, w = image.shape[:2]

    new_w = max(
        10,
        int(w * scale)
    )

    new_h = max(
        10,
        int(h * scale)
    )

    return cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR
    )


# ============================================================
# TEMPLATE MATCHING
# ============================================================

def template_match(
    reference,
    search
):

    result = cv2.matchTemplate(
        search,
        reference,
        cv2.TM_CCOEFF_NORMED
    )

    _, max_value, _, max_location = cv2.minMaxLoc(
        result
    )

    return (
        max_location[0],
        max_location[1],
        max_value
    )


# ============================================================
# EDGE MATCHING
# ============================================================

def edge_match(
    reference,
    search,
    x,
    y
):

    rh, rw = reference.shape

    sh, sw = search.shape

    if (
        x < 0
        or y < 0
        or x + rw > sw
        or y + rh > sh
    ):
        return 0.0

    search_patch = search[
        y:y + rh,
        x:x + rw
    ]

    reference_edges = get_edges(
        reference
    )

    search_edges = get_edges(
        search_patch
    )

    if (
        reference_edges.size == 0
        or search_edges.size == 0
    ):
        return 0.0

    reference_edges = (
        reference_edges > 0
    ).astype(
        np.float32
    )

    search_edges = (
        search_edges > 0
    ).astype(
        np.float32
    )

    intersection = np.sum(
        reference_edges * search_edges
    )

    reference_count = np.sum(
        reference_edges
    )

    if reference_count == 0:
        return 0.0

    return float(
        intersection
        / reference_count
    )


# ============================================================
# STRUCTURAL VERIFICATION
# ============================================================

def structural_score(
    reference,
    search,
    x,
    y
):

    rh, rw = reference.shape

    sh, sw = search.shape

    if (
        x < 0
        or y < 0
        or x + rw > sw
        or y + rh > sh
    ):
        return 0.0

    patch = search[
        y:y + rh,
        x:x + rw
    ]

    if patch.shape != reference.shape:
        return 0.0

    # Compare edge structure
    ref_edges = get_edges(
        reference
    )

    patch_edges = get_edges(
        patch
    )

    ref_edges = (
        ref_edges > 0
    ).astype(
        np.float32
    )

    patch_edges = (
        patch_edges > 0
    ).astype(
        np.float32
    )

    ref_norm = np.linalg.norm(
        ref_edges
    )

    patch_norm = np.linalg.norm(
        patch_edges
    )

    if (
        ref_norm == 0
        or patch_norm == 0
    ):
        return 0.0

    score = np.sum(
        ref_edges * patch_edges
    ) / (
        ref_norm * patch_norm
    )

    return float(score)


# ============================================================
# DRIFT-SENSE PREDICTION
# ============================================================

def predict_location(
    reference,
    search
):

    reference = preprocess_image(
        reference
    )

    search = preprocess_image(
        search
    )

    best_prediction = None

    # --------------------------------------------------------
    # Multiple scales
    # --------------------------------------------------------

    scales = [
        0.90,
        0.95,
        1.00,
        1.05,
        1.10
    ]

    # --------------------------------------------------------
    # Multiple rotations
    # --------------------------------------------------------

    angles = [
        -8,
        -6,
        -4,
        -2,
        0,
        2,
        4,
        6,
        8
    ]

    for scale in scales:

        scaled_reference = scale_image(
            reference,
            scale
        )

        for angle in angles:

            rotated_reference = rotate_image(
                scaled_reference,
                angle
            )

            rh, rw = (
                rotated_reference.shape
            )

            sh, sw = search.shape

            if (
                rh >= sh
                or rw >= sw
            ):
                continue

            # ------------------------------------------------
            # Template matching
            # ------------------------------------------------

            x, y, template_score = template_match(
                rotated_reference,
                search
            )

            # ------------------------------------------------
            # Edge score
            # ------------------------------------------------

            edge_score = edge_match(
                rotated_reference,
                search,
                x,
                y
            )

            # ------------------------------------------------
            # Structural score
            # ------------------------------------------------

            structural = structural_score(
                rotated_reference,
                search,
                x,
                y
            )

            # ------------------------------------------------
            # Combined score
            # ------------------------------------------------

            combined_score = (
                0.50 * template_score
                + 0.25 * edge_score
                + 0.25 * structural
            )

            prediction = {
                "x": x,
                "y": y,
                "score": combined_score,
                "scale": scale,
                "angle": angle
            }

            if (
                best_prediction is None
                or combined_score
                > best_prediction["score"]
            ):
                best_prediction = prediction

    return best_prediction


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print(
        "       DRIFT-SENSE ROBUST REAL LOCALIZATION"
    )
    print("=" * 60)
    print()

    # --------------------------------------------------------
    # Load ground truth
    # --------------------------------------------------------

    ground_truth = load_ground_truth()

    print(
        f"Loaded {len(ground_truth)} "
        f"ground-truth entries."
    )

    print(
        "Prediction is performed using images only."
    )

    print(
        "Ground truth is used only for evaluation."
    )

    print()

    # ========================================================
    # IMPORTANT FIX
    # ========================================================
    #
    # DO NOT scan the reference directory.
    #
    # Use ONLY the filenames that appear in
    # ground_truth.csv.
    #
    # This guarantees exactly 30 test pairs.
    # ========================================================

    reference_files = sorted(
        ground_truth.keys()
    )

    print(
        f"Processing exactly "
        f"{len(reference_files)} "
        f"ground-truth pairs."
    )

    print()

    results = []

    errors = []

    within_5 = 0

    within_10 = 0

    # ========================================================
    # PROCESS EACH PAIR
    # ========================================================

    for index, filename in enumerate(
        reference_files,
        start=1
    ):

        print(
            f"Processing pair "
            f"{index}/{len(reference_files)}..."
        )

        # ----------------------------------------------------
        # Reference filename
        # ----------------------------------------------------

        reference_path = os.path.join(
            REFERENCE_DIR,
            filename
        )

        # ----------------------------------------------------
        # Search filename
        #
        # reference_001.png
        #       ↓
        # search_001.png
        # ----------------------------------------------------

        search_filename = filename.replace(
            "reference_",
            "search_"
        )

        search_path = os.path.join(
            SEARCH_DIR,
            search_filename
        )

        # ----------------------------------------------------
        # Read images
        # ----------------------------------------------------

        reference = cv2.imread(
            reference_path,
            cv2.IMREAD_GRAYSCALE
        )

        search = cv2.imread(
            search_path,
            cv2.IMREAD_GRAYSCALE
        )

        if reference is None:

            print(
                "  ERROR: Could not read reference:"
            )

            print(
                f"  {reference_path}"
            )

            continue

        if search is None:

            print(
                "  ERROR: Could not read search:"
            )

            print(
                f"  {search_path}"
            )

            continue

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = predict_location(
            reference,
            search
        )

        if prediction is None:

            print(
                "  ERROR: No prediction."
            )

            continue

        predicted_x = int(
            prediction["x"]
        )

        predicted_y = int(
            prediction["y"]
        )

        # ----------------------------------------------------
        # Ground truth
        # ----------------------------------------------------

        true_x, true_y = ground_truth[
            filename
        ]

        # ----------------------------------------------------
        # Error
        # ----------------------------------------------------

        error = np.sqrt(
            (
                predicted_x
                - true_x
            ) ** 2
            +
            (
                predicted_y
                - true_y
            ) ** 2
        )

        errors.append(
            error
        )

        if error <= 5:
            within_5 += 1

        if error <= 10:
            within_10 += 1

        # ----------------------------------------------------
        # Print prediction
        # ----------------------------------------------------

        print(
            f"  True="
            f"({true_x},{true_y}) "
            f"Predicted="
            f"({predicted_x},{predicted_y}) "
            f"Error="
            f"{error:.2f}px "
            f"Score="
            f"{prediction['score']:.3f}"
        )

        print(
            f"  Scale="
            f"{prediction['scale']:.2f} "
            f"Angle="
            f"{prediction['angle']}"
        )

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        results.append(
            [
                filename,
                true_x,
                true_y,
                predicted_x,
                predicted_y,
                error,
                prediction["score"],
                prediction["scale"],
                prediction["angle"]
            ]
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    result_directory = os.path.dirname(
        RESULT_FILE
    )

    if result_directory:

        os.makedirs(
            result_directory,
            exist_ok=True
        )

    with open(
        RESULT_FILE,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow(
            [
                "filename",
                "true_x",
                "true_y",
                "predicted_x",
                "predicted_y",
                "error_pixels",
                "score",
                "scale",
                "angle"
            ]
        )

        writer.writerows(
            results
        )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print()

    print(
        "=" * 60
    )

    print(
        "                 RESULTS"
    )

    print(
        "=" * 60
    )

    if len(errors) > 0:

        errors_np = np.array(
            errors,
            dtype=np.float32
        )

        mean_error = np.mean(
            errors_np
        )

        median_error = np.median(
            errors_np
        )

        min_error = np.min(
            errors_np
        )

        max_error = np.max(
            errors_np
        )

        percentage_5 = (
            100.0
            * within_5
            / len(errors)
        )

        percentage_10 = (
            100.0
            * within_10
            / len(errors)
        )

        within_20 = np.sum(
            errors_np <= 20
        )

        percentage_20 = (
            100.0
            * within_20
            / len(errors)
        )

        print(
            f"Mean localization error: "
            f"{mean_error:.2f} pixels"
        )

        print(
            f"Median localization error: "
            f"{median_error:.2f} pixels"
        )

        print(
            f"Minimum error: "
            f"{min_error:.2f} pixels"
        )

        print(
            f"Maximum error: "
            f"{max_error:.2f} pixels"
        )

        print(
            f"Within 5 pixels: "
            f"{percentage_5:.1f}%"
        )

        print(
            f"Within 10 pixels: "
            f"{percentage_10:.1f}%"
        )

        print(
            f"Within 20 pixels: "
            f"{percentage_20:.1f}%"
        )

    else:

        print(
            "No valid ground-truth comparisons."
        )

    print()

    print(
        "Results saved to:"
    )

    print(
        RESULT_FILE
    )

    print(
        "=" * 60
    )

    print()


# ============================================================
# START PROGRAM
# ============================================================

if __name__ == "__main__":
    main()