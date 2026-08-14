import os
import cv2
import csv
import numpy as np

# ============================================================
# REAL SEM DATASET GENERATOR
# Reference = yellow-box region
# Search = entire SEM image
# ============================================================

IMAGE_PATH = r"C:\Users\manvi\OneDrive - The Northcap university\Desktop\PROJECTS\SEMICON\drift_sense\sem_980_cleaned.png"

DATASET = "dataset"
REFERENCE_DIR = os.path.join(DATASET, "reference")
SEARCH_DIR = os.path.join(DATASET, "search")

os.makedirs(REFERENCE_DIR, exist_ok=True)
os.makedirs(SEARCH_DIR, exist_ok=True)

# ------------------------------------------------------------
# 1. Read the real SEM image
# ------------------------------------------------------------

image = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("ERROR: Could not find", IMAGE_PATH)
    exit()

height, width = image.shape

print("SEM image size:", width, "x", height)

# ------------------------------------------------------------
# 2. Yellow-box coordinates
# ------------------------------------------------------------
# Approximate coordinates of the yellow rectangle
# in the original SEM image.

X1 = 133
Y1 = 130

X2 = 267
Y2 = 345

reference = image[Y1:Y2, X1:X2]

ref_height, ref_width = reference.shape

print("Reference size:", ref_width, "x", ref_height)
print("Ground truth:", X1, Y1)

# Save clean reference
cv2.imwrite(
    os.path.join(REFERENCE_DIR, "reference_clean.png"),
    reference
)

# Random generator
rng = np.random.default_rng(2026)


# ============================================================
# FUNCTIONS
# ============================================================

def add_noise(img, sigma):

    noise = rng.normal(
        0,
        sigma,
        img.shape
    ).astype(np.float32)

    result = img.astype(np.float32) + noise

    return np.clip(result, 0, 255).astype(np.uint8)


def apply_blur(img, kernel):

    if kernel == 1:
        return img

    return cv2.GaussianBlur(
        img,
        (kernel, kernel),
        0
    )


def change_contrast(img, contrast, brightness):

    result = (
        img.astype(np.float32) * contrast
        + brightness
    )

    return np.clip(
        result,
        0,
        255
    ).astype(np.uint8)


def edge_brightening(img, strength):

    edges = cv2.Canny(
        img,
        40,
        120
    )

    kernel = np.ones(
        (2, 2),
        np.uint8
    )

    edges = cv2.dilate(
        edges,
        kernel,
        iterations=1
    )

    result = img.astype(np.float32)

    result[edges > 0] += strength

    return np.clip(
        result,
        0,
        255
    ).astype(np.uint8)


def transform_image(img, angle, scale):

    center = (
        width / 2,
        height / 2
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        scale
    )

    transformed = cv2.warpAffine(
        img,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

    return transformed, matrix


def transform_point(x, y, matrix):

    point = np.array(
        [x, y, 1],
        dtype=np.float32
    )

    result = matrix @ point

    return result[0], result[1]


def transform_bbox(x, y, w, h, matrix):

    corners = np.array([
        [x,     y,     1],
        [x + w, y,     1],
        [x,     y + h, 1],
        [x + w, y + h, 1]
    ], dtype=np.float32).T

    transformed = matrix @ corners

    xs = transformed[0]
    ys = transformed[1]

    new_x = int(round(xs.min()))
    new_y = int(round(ys.min()))

    new_w = int(round(xs.max() - xs.min()))
    new_h = int(round(ys.max() - ys.min()))

    return new_x, new_y, new_w, new_h


# ============================================================
# 3. Generate 30 test cases
# ============================================================

rows = []

for i in range(1, 31):

    # --------------------------------------------------------
    # Difficulty
    # --------------------------------------------------------

    if i <= 10:

        difficulty = "Easy"

        angle = rng.uniform(-2, 2)

        scale = rng.uniform(
            0.97,
            1.03
        )

        noise_sigma = rng.uniform(
            8,
            14
        )

        blur_kernel = 1

        edge_strength = rng.uniform(
            6,
            12
        )

    elif i <= 20:

        difficulty = "Medium"

        angle = rng.uniform(-5, 5)

        scale = rng.uniform(
            0.94,
            1.06
        )

        noise_sigma = rng.uniform(
            14,
            22
        )

        blur_kernel = 3

        edge_strength = rng.uniform(
            10,
            18
        )

    else:

        difficulty = "Hard"

        angle = rng.uniform(-8, 8)

        scale = rng.uniform(
            0.90,
            1.10
        )

        noise_sigma = rng.uniform(
            22,
            32
        )

        blur_kernel = 5

        edge_strength = rng.uniform(
            15,
            25
        )

    # --------------------------------------------------------
    # 4. Transform the WHOLE search image
    # --------------------------------------------------------

    search, matrix = transform_image(
        image,
        angle,
        scale
    )

    # --------------------------------------------------------
    # 5. Calculate NEW ground truth
    # --------------------------------------------------------

    new_x, new_y, new_w, new_h = transform_bbox(
        X1,
        Y1,
        ref_width,
        ref_height,
        matrix
    )

    # --------------------------------------------------------
    # 6. Apply search-image degradation
    # --------------------------------------------------------

    search = apply_blur(
        search,
        blur_kernel
    )

    search = change_contrast(
        search,
        rng.uniform(0.90, 1.10),
        rng.uniform(-10, 10)
    )

    search = edge_brightening(
        search,
        edge_strength
    )

    # IMPORTANT:
    # Every image gets a NEW noise pattern.
    search = add_noise(
        search,
        noise_sigma
    )

    # --------------------------------------------------------
    # 7. Create independent reference noise
    # --------------------------------------------------------

    reference_variant = add_noise(
        reference,
        rng.uniform(2, 6)
    )

    # --------------------------------------------------------
    # 8. Save files
    # --------------------------------------------------------

    reference_name = (
        f"reference_{i:03d}.png"
    )

    search_name = (
        f"search_{i:03d}.png"
    )

    cv2.imwrite(
        os.path.join(
            REFERENCE_DIR,
            reference_name
        ),
        reference_variant
    )

    cv2.imwrite(
        os.path.join(
            SEARCH_DIR,
            search_name
        ),
        search
    )

    # --------------------------------------------------------
    # 9. Save ground truth
    # --------------------------------------------------------

    rows.append({
        "image_id": i,
        "reference_file":
            f"reference/{reference_name}",

        "search_file":
            f"search/{search_name}",

        "x": new_x,
        "y": new_y,

        "difficulty":
            difficulty,

        "rotation":
            round(float(angle), 3),

        "scale":
            round(float(scale), 3),

        "noise":
            round(float(noise_sigma), 3),

        "blur":
            blur_kernel,

        "edge_brightening":
            round(float(edge_strength), 3)
    })


# ============================================================
# 10. Save ground-truth CSV
# ============================================================

csv_path = os.path.join(
    DATASET,
    "ground_truth.csv"
)

with open(
    csv_path,
    "w",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=rows[0].keys()
    )

    writer.writeheader()

    writer.writerows(rows)


print()
print("=" * 60)
print("REAL SEM DATASET COMPLETE")
print("=" * 60)

print("Pairs:", len(rows))
print("Easy: 10")
print("Medium: 10")
print("Hard: 10")

print()
print("Reference:")
print("  Yellow-box region")

print()
print("Search:")
print("  Entire SEM image")

print()
print("Variations:")
print("  Noise")
print("  Blur")
print("  Rotation")
print("  Scaling")
print("  Contrast")
print("  Brightness")
print("  Edge brightening")

print()
print("Ground truth:")
print(" ", csv_path)

print("=" * 60)