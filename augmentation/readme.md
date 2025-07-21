# Dataset Augmentation Tool (PyQt5)

A simple and user-friendly GUI tool built with **Python** and **PyQt5** for dataset augmentation, specifically designed for datasets with **images and labels** (AABB / OBB format). This tool provides rotation, brightness adjustment, and blur augmentation, and automatically splits your dataset into **train** and **valid** folders.

---

## Features

* Support for **OBB (Oriented Bounding Box)** and **AABB (Axis-Aligned Bounding Box)** label formats
* Auto-detection of label format
* Image rotation with configurable angle range and step
* Brightness adjustment (darker or brighter)
* Gaussian blur augmentation
* Train/Validation split with adjustable ratio
* Live preview for brightness and blur effects
* Multi-threading to avoid GUI freezing during processing

---

## Directory Structure

**Input Folder** (must contain these subfolders):

```
input_folder/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── labels/
    ├── image1.txt
    ├── image2.txt
    └── ...
```

**Output Folder (after running augmentation):**

```
output_folder/
├── train/
│   ├── images/
│   └── labels/
└── valid/
    ├── images/
    └── labels/
```

---

## How to Use

1. **Select Input Folder:** Should contain `images/` and `labels/`.
2. **Select Output Folder:** Where the augmented dataset will be saved.
3. **Dataset Format:** Auto-detect or manually choose between OBB / AABB.
4. **Rotation Settings:**

   * Angle Range: min/max (-180 \~ 180)
   * Step: e.g., 120 for three rotations (-180, -60, 60, 180)
5. **Brightness:** Adjust using the slider (0.5 \~ 1.5 factor).
6. **Blur:** Choose kernel size (0 \~ 15) via slider.
7. **Train Ratio:** Percentage split for training data.
8. **Live Preview:** Preview brightness and blur effects on selected image.
9. **Start Augmentation:** Confirm and run.

---

## Requirements

```bash
pip install PyQt5 opencv-python numpy
```

---

## Running the App

```bash
python your_script_name.py
```

---

## Output Examples

* Original image
* Rotated images by specified angles
* Brightness-adjusted image
* Blurred image

All outputs saved in `train/` and `valid/` according to the specified split.

---

## Notes

* **OBB format:** `class_id x1 y1 x2 y2 x3 y3 x4 y4`
* **AABB format:** `class_id cx cy w h` (all normalized between 0\~1)
* Input folders must follow the required structure.
* The program will automatically create `Augmentation/` if output is not specified.

---

## License

Free to use for academic and non-commercial purposes.
