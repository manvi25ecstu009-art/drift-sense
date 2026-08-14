import os
import cv2
import csv
import random
import numpy as np


# ============================================================
# DRIFT-SENSE HARD DATASET GENERATOR
# ============================================================

NUM_PAIRS = 30

IMAGE_SIZE = 1000
REFERENCE_SIZE = 100

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

PREVIEW_DIR = os.path.join(
    DATASET_DIR,
    "previews"
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(REFERENCE_DIR, exist_ok=True)
os.makedirs(SEARCH_DIR, exist_ok=True)
os.makedirs(PREVIEW_DIR, exist_ok=True)


# ============================================================
# RANDOM SEED
# ============================================================

random.seed(42)
np.random.seed(42)


# ============================================================
# HELPER: CLIP IMAGE
# ============================================================

def clip_image(image):

    return np.clip(
        image,
        0,
        255
    ).astype(np.uint8)


# ============================================================
# EDGE BRIGHTENING
# ============================================================

def edge_brightening(image, strength=0.25):

    image_float = image.astype(np.float32)

    # Sobel edges
    gx = cv2.Sobel(
        image_float,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        image_float,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    magnitude = cv2.magnitude(
        gx,
        gy
    )

    # Normalize
    maximum = magnitude.max()

    if maximum > 0:

        magnitude = (
            magnitude / maximum
        )

    # Add brighter edges
    enhanced = (
        image_float
        +
        strength * 255 * magnitude
    )

    return clip_image(enhanced)


# ============================================================
# INDEPENDENT SENSOR NOISE
# ============================================================

def add_sensor_noise(
    image,
    gaussian_sigma,
    poisson_strength,
    salt_probability=0.0
):

    image_float = image.astype(
        np.float32
    )

    # --------------------------------------------------------
    # Gaussian sensor noise
    # --------------------------------------------------------

    gaussian_noise = np.random.normal(
        0,
        gaussian_sigma,
        image.shape
    )

    noisy = (
        image_float
        +
        gaussian_noise
    )

    # --------------------------------------------------------
    # Poisson-like shot noise
    # --------------------------------------------------------

    normalized = np.clip(
        noisy / 255.0,
        0,
        1
    )

    poisson_noise = (
        np.random.poisson(
            normalized
            * poisson_strength
        )
        / poisson_strength
        - normalized
    )

    noisy = (
        normalized
        +
        poisson_noise
    ) * 255.0

    # --------------------------------------------------------
    # Occasional sensor hot pixels
    # --------------------------------------------------------

    if salt_probability > 0:

        mask = np.random.random(
            image.shape
        ) < salt_probability

        noisy[mask] = 255

    return clip_image(noisy)


# ============================================================
# RANDOM BLUR
# ============================================================

def apply_random_blur(
    image,
    sigma
):

    if sigma <= 0:

        return image

    return cv2.GaussianBlur(
        image,
        (0, 0),
        sigmaX=sigma
    )


# ============================================================
# RANDOM CONTRAST / BRIGHTNESS
# ============================================================

def change_contrast_brightness(
    image,
    contrast,
    brightness
):

    result = (
        image.astype(np.float32)
        * contrast
        +
        brightness
    )

    return clip_image(result)


# ============================================================
# CREATE DRAM BASE PATTERN
# ============================================================

def create_dram_pattern(
    size,
    pitch,
    line_width,
    dot_radius
):

    image = np.zeros(
        (size, size),
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # Horizontal word lines
    # --------------------------------------------------------

    offset = random.randint(
        0,
        pitch - 1
    )

    y = offset

    while y < size:

        cv2.line(
            image,
            (0, int(y)),
            (size - 1, int(y)),
            210,
            int(line_width)
        )

        y += pitch

    # --------------------------------------------------------
    # Vertical bit lines
    # --------------------------------------------------------

    offset_x = random.randint(
        0,
        pitch - 1
    )

    x = offset_x

    while x < size:

        cv2.line(
            image,
            (int(x), 0),
            (int(x), size - 1),
            210,
            int(line_width)
        )

        x += pitch

    # --------------------------------------------------------
    # Contact / via dots
    # --------------------------------------------------------

    y_values = list(
        range(
            offset,
            size,
            pitch
        )
    )

    x_values = list(
        range(
            offset_x,
            size,
            pitch
        )
    )

    for y in y_values:

        for x in x_values:

            cv2.circle(
                image,
                (int(x), int(y)),
                int(dot_radius),
                245,
                -1
            )

    return image


# ============================================================
# CREATE FINFET PATTERN
# ============================================================

def create_finfet_pattern(
    size,
    fin_pitch,
    fin_width,
    gate_positions
):

    image = np.zeros(
        (size, size),
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # Vertical fins
    # --------------------------------------------------------

    offset = random.randint(
        0,
        fin_pitch - 1
    )

    x = offset

    while x < size:

        cv2.line(
            image,
            (int(x), 0),
            (int(x), size - 1),
            210,
            int(fin_width)
        )

        x += fin_pitch

    # --------------------------------------------------------
    # Horizontal gate bars
    # --------------------------------------------------------

    for y in gate_positions:

        if 0 <= y < size:

            cv2.line(
                image,
                (0, int(y)),
                (size - 1, int(y)),
                245,
                3
            )

    # --------------------------------------------------------
    # Small gate crossing highlights
    # --------------------------------------------------------

    for y in gate_positions:

        x = offset

        while x < size:

            cv2.circle(
                image,
                (int(x), int(y)),
                3,
                255,
                -1
            )

            x += fin_pitch

    return image


# ============================================================
# CREATE REFERENCE PATTERN
# ============================================================

def create_reference(
    architecture
):

    if architecture == "DRAM":

        # Reference is deliberately dense
        pitch = random.randint(
            18,
            23
        )

        line_width = random.choice(
            [2, 3, 4]
        )

        dot_radius = random.choice(
            [3, 4, 5]
        )

        image = create_dram_pattern(
            REFERENCE_SIZE,
            pitch,
            line_width,
            dot_radius
        )

    else:

        fin_pitch = random.randint(
            10,
            14
        )

        fin_width = random.choice(
            [2, 3]
        )

        gate_positions = [
            random.randint(30, 40),
            random.randint(60, 70)
        ]

        image = create_finfet_pattern(
            REFERENCE_SIZE,
            fin_pitch,
            fin_width,
            gate_positions
        )

    # --------------------------------------------------------
    # Edge brightening
    # --------------------------------------------------------

    image = edge_brightening(
        image,
        strength=random.uniform(
            0.15,
            0.30
        )
    )

    # --------------------------------------------------------
    # Reference blur
    # --------------------------------------------------------

    image = apply_random_blur(
        image,
        random.uniform(
            0.15,
            0.55
        )
    )

    # --------------------------------------------------------
    # Reference contrast
    # --------------------------------------------------------

    image = change_contrast_brightness(
        image,
        random.uniform(
            0.90,
            1.10
        ),
        random.uniform(
            -5,
            5
        )
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Independent reference noise
    # --------------------------------------------------------

    image = add_sensor_noise(
        image,
        gaussian_sigma=random.uniform(
            2.0,
            5.0
        ),
        poisson_strength=random.uniform(
            30,
            70
        ),
        salt_probability=0.00005
    )

    return image


# ============================================================
# TRANSFORM TARGET
# ============================================================

def transform_target(
    reference,
    rotation,
    scale
):

    h, w = reference.shape

    center = (
        w / 2,
        h / 2
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        rotation,
        scale
    )

    transformed = cv2.warpAffine(
        reference,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

    return transformed


# ============================================================
# CREATE LARGE SEARCH BACKGROUND
# ============================================================

def create_search_background(
    architecture,
    difficulty
):

    search = np.zeros(
        (IMAGE_SIZE, IMAGE_SIZE),
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # Different background parameters
    # --------------------------------------------------------

    if architecture == "DRAM":

        if difficulty == "easy":

            pitch = random.randint(
                90,
                120
            )

        elif difficulty == "medium":

            pitch = random.randint(
                105,
                150
            )

        else:

            # Highly periodic difficult case
            pitch = random.choice(
                [100, 110, 120]
            )

        line_width = random.choice(
            [2, 3, 4]
        )

        offset_x = random.randint(
            0,
            pitch - 1
        )

        offset_y = random.randint(
            0,
            pitch - 1
        )

        # ----------------------------------------------------
        # Horizontal lines
        # ----------------------------------------------------

        y = offset_y

        while y < IMAGE_SIZE:

            cv2.line(
                search,
                (0, int(y)),
                (IMAGE_SIZE - 1, int(y)),
                random.randint(170, 215),
                line_width
            )

            y += pitch

        # ----------------------------------------------------
        # Vertical lines
        # ----------------------------------------------------

        x = offset_x

        while x < IMAGE_SIZE:

            cv2.line(
                search,
                (int(x), 0),
                (int(x), IMAGE_SIZE - 1),
                random.randint(170, 215),
                line_width
            )

            x += pitch

        # ----------------------------------------------------
        # Background contact dots
        # ----------------------------------------------------

        if difficulty != "hard":

            y = offset_y

            while y < IMAGE_SIZE:

                x = offset_x

                while x < IMAGE_SIZE:

                    cv2.circle(
                        search,
                        (int(x), int(y)),
                        2,
                        235,
                        -1
                    )

                    x += pitch

                y += pitch

    else:

        # ----------------------------------------------------
        # FinFET background
        # ----------------------------------------------------

        fin_pitch = random.randint(
            25,
            45
        )

        fin_width = random.choice(
            [2, 3, 4]
        )

        offset = random.randint(
            0,
            fin_pitch - 1
        )

        x = offset

        while x < IMAGE_SIZE:

            cv2.line(
                search,
                (int(x), 0),
                (int(x), IMAGE_SIZE - 1),
                random.randint(175, 215),
                fin_width
            )

            x += fin_pitch

        # Several gate bars
        gate1 = random.randint(
            250,
            400
        )

        gate2 = random.randint(
            550,
            750
        )

        cv2.line(
            search,
            (0, gate1),
            (IMAGE_SIZE - 1, gate1),
            230,
            3
        )

        cv2.line(
            search,
            (0, gate2),
            (IMAGE_SIZE - 1, gate2),
            230,
            3
        )

    return search


# ============================================================
# INSERT TARGET INTO SEARCH IMAGE
# ============================================================

def insert_target(
    search,
    target,
    x,
    y,
    alpha
):

    h, w = target.shape

    target_area = search[
        y:y + h,
        x:x + w
    ].astype(np.float32)

    target_float = target.astype(
        np.float32
    )

    # --------------------------------------------------------
    # Blend target with background
    # --------------------------------------------------------

    blended = (
        (1 - alpha)
        * target_area
        +
        alpha
        * target_float
    )

    search[
        y:y + h,
        x:x + w
    ] = clip_image(
        blended
    )

    return search


# ============================================================
# CREATE SEARCH IMAGE
# ============================================================

def create_search_image(
    reference,
    architecture,
    difficulty
):

    # --------------------------------------------------------
    # Create background
    # --------------------------------------------------------

    search = create_search_background(
        architecture,
        difficulty
    )

    # --------------------------------------------------------
    # Transform target
    #
    # IMPORTANT:
    # This is DIFFERENT from reference capture.
    # --------------------------------------------------------

    if difficulty == "easy":

        rotation = random.uniform(
            -1.0,
            1.0
        )

        scale = random.uniform(
            0.98,
            1.02
        )

    elif difficulty == "medium":

        rotation = random.uniform(
            -3.0,
            3.0
        )

        scale = random.uniform(
            0.94,
            1.06
        )

    else:

        rotation = random.uniform(
            -5.0,
            5.0
        )

        scale = random.uniform(
            0.90,
            1.10
        )

    target = transform_target(
        reference,
        rotation,
        scale
    )

    # --------------------------------------------------------
    # Extra search blur
    # --------------------------------------------------------

    target = apply_random_blur(
        target,
        random.uniform(
            0.20,
            0.80
        )
    )

    # --------------------------------------------------------
    # Edge enhancement
    # --------------------------------------------------------

    target = edge_brightening(
        target,
        strength=random.uniform(
            0.10,
            0.25
        )
    )

    # --------------------------------------------------------
    # Random position
    #
    # Keep target completely inside image
    # --------------------------------------------------------

    x = random.randint(
        20,
        IMAGE_SIZE - REFERENCE_SIZE - 20
    )

    y = random.randint(
        20,
        IMAGE_SIZE - REFERENCE_SIZE - 20
    )

    # --------------------------------------------------------
    # Target visibility
    # --------------------------------------------------------

    if difficulty == "easy":

        alpha = random.uniform(
            0.80,
            0.95
        )

    elif difficulty == "medium":

        alpha = random.uniform(
            0.60,
            0.82
        )

    else:

        alpha = random.uniform(
            0.45,
            0.70
        )

    # --------------------------------------------------------
    # Insert target
    # --------------------------------------------------------

    search = insert_target(
        search,
        target,
        x,
        y,
        alpha
    )

    # --------------------------------------------------------
    # Apply edge brightening to entire search
    # --------------------------------------------------------

    search = edge_brightening(
        search,
        strength=random.uniform(
            0.10,
            0.20
        )
    )

    # --------------------------------------------------------
    # Search image blur
    # --------------------------------------------------------

    if difficulty == "easy":

        blur_sigma = random.uniform(
            0.20,
            0.50
        )

    elif difficulty == "medium":

        blur_sigma = random.uniform(
            0.30,
            0.70
        )

    else:

        blur_sigma = random.uniform(
            0.40,
            1.00
        )

    search = apply_random_blur(
        search,
        blur_sigma
    )

    # --------------------------------------------------------
    # Search contrast / brightness
    # --------------------------------------------------------

    search = change_contrast_brightness(
        search,
        random.uniform(
            0.85,
            1.10
        ),
        random.uniform(
            -10,
            10
        )
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # SEARCH GETS ITS OWN INDEPENDENT NOISE
    #
    # It is NOT reused from reference.
    # --------------------------------------------------------

    if difficulty == "easy":

        gaussian_sigma = random.uniform(
            5,
            9
        )

        poisson_strength = random.uniform(
            20,
            50
        )

    elif difficulty == "medium":

        gaussian_sigma = random.uniform(
            8,
            14
        )

        poisson_strength = random.uniform(
            15,
            40
        )

    else:

        gaussian_sigma = random.uniform(
            12,
            20
        )

        poisson_strength = random.uniform(
            10,
            30
        )

    search = add_sensor_noise(
        search,
        gaussian_sigma,
        poisson_strength,
        salt_probability=0.0001
    )

    return search, x, y, rotation, scale


# ============================================================
# CREATE PREVIEW
# ============================================================

def create_preview(
    reference,
    search,
    x,
    y,
    image_id,
    architecture,
    difficulty
):

    # Make copies
    reference_color = cv2.cvtColor(
        reference,
        cv2.COLOR_GRAY2BGR
    )

    search_color = cv2.cvtColor(
        search,
        cv2.COLOR_GRAY2BGR
    )

    # Ground truth rectangle
    cv2.rectangle(
        search_color,
        (x, y),
        (
            x + REFERENCE_SIZE,
            y + REFERENCE_SIZE
        ),
        (0, 0, 255),
        2
    )

    # --------------------------------------------------------
    # Resize reference for display
    # --------------------------------------------------------

    reference_display = cv2.resize(
        reference_color,
        (400, 400),
        interpolation=cv2.INTER_NEAREST
    )

    # --------------------------------------------------------
    # Resize search for display
    # --------------------------------------------------------

    search_display = cv2.resize(
        search_color,
        (500, 500),
        interpolation=cv2.INTER_AREA
    )

    # --------------------------------------------------------
    # Create canvas
    # --------------------------------------------------------

    canvas = np.ones(
        (600, 950, 3),
        dtype=np.uint8
    ) * 255

    canvas[
        100:500,
        50:450
    ] = reference_display

    canvas[
        50:550,
        450:950
    ] = search_display

    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------

    cv2.putText(
        canvas,
        f"Reference - 100 x 100",
        (90, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2
    )

    cv2.putText(
        canvas,
        f"Search - 1000 x 1000",
        (590, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2
    )

    cv2.putText(
        canvas,
        f"{architecture} | {difficulty}",
        (590, 555),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2
    )

    cv2.putText(
        canvas,
        f"Ground Truth = ({x}, {y})",
        (590, 585),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        2
    )

    filename = os.path.join(
        PREVIEW_DIR,
        f"pair_{image_id:02d}_preview.png"
    )

    cv2.imwrite(
        filename,
        canvas
    )


# ============================================================
# MAIN DATASET GENERATION
# ============================================================

print()
print("==============================================")
print("     DRIFT-SENSE HARD DATASET GENERATOR")
print("==============================================")
print()

rows = []


for image_id in range(
    1,
    NUM_PAIRS + 1
):

    print(
        f"Generating pair {image_id:02d}"
    )

    # --------------------------------------------------------
    # Alternate architectures
    # --------------------------------------------------------

    if image_id % 2 == 0:

        architecture = "FinFET"

    else:

        architecture = "DRAM"

    # --------------------------------------------------------
    # Difficulty distribution
    # --------------------------------------------------------

    if image_id <= 10:

        difficulty = "easy"

    elif image_id <= 20:

        difficulty = "medium"

    else:

        difficulty = "hard"

    # --------------------------------------------------------
    # Generate reference
    # --------------------------------------------------------

    reference = create_reference(
        architecture
    )

    # --------------------------------------------------------
    # Generate search
    # --------------------------------------------------------

    search, x, y, rotation, scale = (
        create_search_image(
            reference,
            architecture,
            difficulty
        )
    )

    # --------------------------------------------------------
    # File names
    # --------------------------------------------------------

    reference_filename = (
        f"reference_{image_id:03d}.png"
    )

    search_filename = (
        f"search_{image_id:03d}.png"
    )

    reference_path = os.path.join(
        REFERENCE_DIR,
        reference_filename
    )

    search_path = os.path.join(
        SEARCH_DIR,
        search_filename
    )

    # --------------------------------------------------------
    # Save images
    # --------------------------------------------------------

    cv2.imwrite(
        reference_path,
        reference
    )

    cv2.imwrite(
        search_path,
        search
    )

    # --------------------------------------------------------
    # Preview first few
    # --------------------------------------------------------

    if image_id <= 5:

        create_preview(
            reference,
            search,
            x,
            y,
            image_id,
            architecture,
            difficulty
        )

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    rows.append({

        "image_id": image_id,

        "reference_file": reference_path,

        "search_file": search_path,

        "x": x,

        "y": y,

        "architecture": architecture,

        "difficulty": difficulty,

        "rotation": rotation,

        "scale": scale

    })


# ============================================================
# WRITE CSV
# ============================================================

with open(
    GROUND_TRUTH_FILE,
    "w",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "image_id",
            "reference_file",
            "search_file",
            "x",
            "y",
            "architecture",
            "difficulty",
            "rotation",
            "scale"
        ]
    )

    writer.writeheader()

    writer.writerows(
        rows
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

print()
print("==============================================")
print("      HARD DATASET GENERATION COMPLETE")
print("==============================================")

print(
    f"Total image pairs: {NUM_PAIRS}"
)

print()
print(
    "Reference images:"
)

print(
    REFERENCE_DIR
)

print()
print(
    "Search images:"
)

print(
    SEARCH_DIR
)

print()
print(
    "Ground truth:"
)

print(
    GROUND_TRUTH_FILE
)

print()
print(
    "Preview images:"
)

print(
    PREVIEW_DIR
)

print()
print("Architecture:")
print("DRAM + FinFET")

print()
print("Difficulty:")
print("10 Easy + 10 Medium + 10 Hard")

print()
print("Independent reference/search noise: YES")
print("Rotation variation: YES")
print("Scale variation: YES")
print("Blur variation: YES")
print("Contrast variation: YES")
print("SEM edge brightening: YES")
print("Periodic background: YES")

print()
print("==============================================")