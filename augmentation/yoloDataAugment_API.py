import os
import cv2
import numpy as np
import random
import shutil

from PyQt5.QtCore import QThread, pyqtSignal

########################################
# A. Label Parse (AABB / OBB)
########################################

def parse_label_line_aabb(line: str):
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cls = parts[0]
    x_center = float(parts[1])
    y_center = float(parts[2])
    w = float(parts[3])
    h = float(parts[4])
    return cls, x_center, y_center, w, h

def parse_label_line_obb(line: str):
    parts = line.strip().split()
    if len(parts) != 9:  # class + 8 個值
        return None
    cls = parts[0]
    coords = [float(v) for v in parts[1:]]  # x1,y1,x2,y2,x3,y3,x4,y4
    return cls, coords

########################################
# B. AABB 的 corners <-> YOLO
########################################

def corners_to_yolo_fmt(xmin, ymin, xmax, ymax, img_w, img_h):
    xmin = max(0, min(xmin, img_w - 1))
    xmax = max(0, min(xmax, img_w - 1))
    ymin = max(0, min(ymin, img_h - 1))
    ymax = max(0, min(ymax, img_h - 1))

    bw = xmax - xmin
    bh = ymax - ymin
    cx = xmin + bw / 2.0
    cy = ymin + bh / 2.0

    if img_w < 1 or img_h < 1:
        return None
    if bw <= 0 or bh <= 0:
        return None

    cx_norm = cx / img_w
    cy_norm = cy / img_h
    w_norm = bw / img_w
    h_norm = bh / img_h

    return (cx_norm, cy_norm, w_norm, h_norm)

def yolo_to_corners_aabb(x_center, y_center, w, h, img_w, img_h):
    x_center_px = x_center * img_w
    y_center_px = y_center * img_h
    half_w = (w * img_w) / 2.0
    half_h = (h * img_h) / 2.0
    xmin = x_center_px - half_w
    xmax = x_center_px + half_w
    ymin = y_center_px - half_h
    ymax = y_center_px + half_h
    return (xmin, ymin, xmax, ymax)

########################################
# C. OBB (8 點) 處理
########################################

def obb_to_pixel_coords(coords, img_w, img_h):
    pixel_coords = []
    for i in range(0, 8, 2):
        x = coords[i] * img_w
        y = coords[i+1] * img_h
        pixel_coords.append((x, y))
    return pixel_coords

def pixel_coords_to_obb_format(pixel_coords, img_w, img_h):
    norm_coords = []
    for (px, py) in pixel_coords:
        nx = px / img_w
        ny = py / img_h
        norm_coords.append(nx)
        norm_coords.append(ny)
    return norm_coords

########################################
# D. 仿射輔助 (Shear / Rotate)
########################################

def apply_affine_to_point(x, y, M):
    new_pt = np.dot(M, np.array([x, y, 1], dtype=np.float32))
    return new_pt[0], new_pt[1]

########################################
# E. Shear / Crop Labels for AABB
########################################

def shear_labels_aabb(label_lines, shear_factor, old_w, old_h, new_w, new_h):
    # 仿射矩陣
    M = np.float32([
        [1, shear_factor, 0],
        [0, 1,           0]
    ])

    new_label_lines = []
    for line in label_lines:
        parsed = parse_label_line_aabb(line)
        if not parsed:
            continue
        cls, x_center, y_center, w, h = parsed

        xmin, ymin, xmax, ymax = yolo_to_corners_aabb(x_center, y_center, w, h, old_w, old_h)

        pts = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        transformed_pts = []
        for (px, py) in pts:
            x_new, y_new = apply_affine_to_point(px, py, M)
            transformed_pts.append((x_new, y_new))

        xs = [p[0] for p in transformed_pts]
        ys = [p[1] for p in transformed_pts]
        new_xmin = min(xs)
        new_xmax = max(xs)
        new_ymin = min(ys)
        new_ymax = max(ys)

        yolo_bbox = corners_to_yolo_fmt(new_xmin, new_ymin, new_xmax, new_ymax, new_w, old_h)
        if yolo_bbox is not None:
            cx_norm, cy_norm, w_norm, h_norm = yolo_bbox
            new_label_lines.append(f"{cls} {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}")

    return new_label_lines

def crop_labels_aabb(label_lines, scale, offset_x, offset_y, old_w, old_h):
    new_label_lines = []
    for line in label_lines:
        parsed = parse_label_line_aabb(line)
        if not parsed:
            continue
        cls, x_center, y_center, w, h = parsed

        xmin, ymin, xmax, ymax = yolo_to_corners_aabb(x_center, y_center, w, h, old_w, old_h)

        xmin_resized = xmin * scale
        xmax_resized = xmax * scale
        ymin_resized = ymin * scale
        ymax_resized = ymax * scale

        if scale > 1.0:
            # 從放大後的影像中「裁切」(負 offset)
            xmin_final = xmin_resized - offset_x
            xmax_final = xmax_resized - offset_x
            ymin_final = ymin_resized - offset_y
            ymax_final = ymax_resized - offset_y
        else:
            # 若是縮小，則貼到 canvas (正 offset)
            xmin_final = xmin_resized + offset_x
            xmax_final = xmax_resized + offset_x
            ymin_final = ymin_resized + offset_y
            ymax_final = ymax_resized + offset_y

        bbox = corners_to_yolo_fmt(xmin_final, ymin_final, xmax_final, ymax_final, old_w, old_h)
        if bbox is not None:
            cx_norm, cy_norm, w_norm, h_norm = bbox
            new_label_lines.append(f"{cls} {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}")
    return new_label_lines

########################################
# F. Shear / Crop Labels for OBB (8點)
########################################

def shear_labels_obb(label_lines, shear_factor, old_w, old_h, new_w, new_h):
    M = np.float32([
        [1, shear_factor, 0],
        [0, 1,           0]
    ])

    new_label_lines = []
    for line in label_lines:
        parsed = parse_label_line_obb(line)
        if not parsed:
            continue
        cls, coords = parsed

        pixel_pts = obb_to_pixel_coords(coords, old_w, old_h)

        transformed_pts = []
        for (px, py) in pixel_pts:
            x_new, y_new = apply_affine_to_point(px, py, M)
            transformed_pts.append((x_new, y_new))

        final_pts = []
        for (x_, y_) in transformed_pts:
            xx = max(0, min(x_, new_w - 1))
            yy = max(0, min(y_, old_h - 1))
            final_pts.append((xx, yy))

        norm_coords = []
        for (xx, yy) in final_pts:
            nx = xx / new_w
            ny = yy / old_h
            norm_coords.append(nx)
            norm_coords.append(ny)

        formatted = " ".join(f"{v:.6f}" for v in norm_coords)
        new_label_lines.append(f"{cls} {formatted}")

    return new_label_lines

def crop_labels_obb(label_lines, scale, offset_x, offset_y, old_w, old_h):
    new_label_lines = []
    for line in label_lines:
        parsed = parse_label_line_obb(line)
        if not parsed:
            continue
        cls, coords = parsed

        pixel_pts = obb_to_pixel_coords(coords, old_w, old_h)

        pts_resized = [(x * scale, y * scale) for (x, y) in pixel_pts]

        final_pts = []
        if scale > 1.0:
            for (rx, ry) in pts_resized:
                fx = rx - offset_x
                fy = ry - offset_y
                final_pts.append((fx, fy))
        else:
            for (rx, ry) in pts_resized:
                fx = rx + offset_x
                fy = ry + offset_y
                final_pts.append((fx, fy))

        clamped_pts = []
        for (fx, fy) in final_pts:
            cx = max(0, min(fx, old_w - 1))
            cy = max(0, min(fy, old_h - 1))
            clamped_pts.append((cx, cy))

        norm_coords = pixel_coords_to_obb_format(clamped_pts, old_w, old_h)
        formatted = " ".join(f"{v:.6f}" for v in norm_coords)
        new_label_lines.append(f"{cls} {formatted}")

    return new_label_lines

########################################
# G. 統一對外: Shear / Crop Labels
########################################

def shear_labels(label_lines, shear_factor, old_w, old_h, new_w, new_h, label_format="aabb"):
    if label_format.lower() == "aabb":
        return shear_labels_aabb(label_lines, shear_factor, old_w, old_h, new_w, new_h)
    elif label_format.lower() == "obb":
        return shear_labels_obb(label_lines, shear_factor, old_w, old_h, new_w, new_h)
    else:
        raise ValueError(f"未知的標籤格式: {label_format}")

def crop_labels(label_lines, scale, offset_x, offset_y, old_w, old_h, label_format="aabb"):
    if label_format.lower() == "aabb":
        return crop_labels_aabb(label_lines, scale, offset_x, offset_y, old_w, old_h)
    elif label_format.lower() == "obb":
        return crop_labels_obb(label_lines, scale, offset_x, offset_y, old_w, old_h)
    else:
        raise ValueError(f"未知的標籤格式: {label_format}")

########################################
# H. 其他增強函式
########################################

def rotate_image(image, angle):
    """
    回傳: (旋轉後的影像, 仿射矩陣 M, 旋轉後影像的寬, 旋轉後影像的高)
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos_ = abs(M[0, 0])
    sin_ = abs(M[0, 1])
    nW = int(h * sin_ + w * cos_)
    nH = int(h * cos_ + w * sin_)

    # 修正平移
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]

    rotated = cv2.warpAffine(image, M, (nW, nH))
    return rotated, M, nW, nH

def random_brightness(image, max_delta=0.15):
    factor = 1.0 + random.uniform(-max_delta, max_delta)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] *= factor
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    hsv = hsv.astype(np.uint8)
    bright_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bright_img

def random_blur(image, max_kernel=3):
    if max_kernel <= 0:
        return image
    k = random.randint(1, max_kernel)
    if k % 2 == 0:
        k += 1
    blur_img = cv2.GaussianBlur(image, (k, k), 0)
    return blur_img

def random_shear(image, max_shear=0.2):
    h, w = image.shape[:2]
    shear_factor = random.uniform(-max_shear, max_shear)
    new_w = int(w + abs(shear_factor * h))
    M = np.float32([
        [1, shear_factor, 0],
        [0, 1,           0]
    ])
    sheared = cv2.warpAffine(image, M, (new_w, h))
    return sheared, M, new_w, h, shear_factor

def random_center_crop_zoom(image, max_ratio=0.3):
    h, w = image.shape[:2]
    scale = 1.0 + random.uniform(-max_ratio, max_ratio)
    if scale <= 0:
        scale = 0.01
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 輸出影像仍保持 (w, h) 大小
    if abs(scale - 1.0) < 1e-6:
        return resized, scale, 0, 0

    if scale > 1.0:
        # 從放大後的圖中取中央區域
        start_x = (new_w - w) // 2
        start_y = (new_h - h) // 2
        cropped = resized[start_y:start_y + h, start_x:start_x + w]
        return cropped, scale, start_x, start_y
    else:
        # 貼到 canvas
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        offset_x = (w - new_w) // 2
        offset_y = (h - new_h) // 2
        canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized
        return canvas, scale, offset_x, offset_y

########################################
# I. Label 讀寫 & 複製檔案
########################################

def copy_labels(label_path):
    if not os.path.exists(label_path):
        return []
    with open(label_path, 'r') as f:
        lines = f.read().splitlines()
    return lines

def save_labels(label_path, lines):
    with open(label_path, 'w') as f:
        for line in lines:
            f.write(f"{line}\n")

def copy_file(src, dst):
    shutil.copy2(src, dst)

########################################
# === 新增: 1) 計算增強數量的函式
########################################

def calculate_augmentation_count(rotate_step, brightness_factor, blur_kernel_size, shear_ratio, crop_ratio):
    aug_count = 0

    # Rotate
    if rotate_step > 0:
        # angles = 0, rotate_step, 2*rotate_step, ..., < 360
        possible_angles = list(range(0, 360, rotate_step))
        rotate_count = max(0, len(possible_angles) - 1)  # 排除 angle=0
        aug_count += rotate_count

    # Brightness
    if brightness_factor > 0:
        aug_count += 1

    # Blur
    if blur_kernel_size > 0:
        aug_count += 1

    # Shear
    if shear_ratio > 0:
        aug_count += 1

    # Crop
    if crop_ratio > 0:
        aug_count += 1

    return aug_count

########################################
# === 新增: 2) 自動判斷標籤格式的函式
########################################

def auto_detect_label_format(labels_dir):
    if not os.path.isdir(labels_dir):
        return 'aabb'  # 預設

    label_files = [f for f in os.listdir(labels_dir) if f.lower().endswith('.txt')]
    for lf in label_files:
        path = os.path.join(labels_dir, lf)
        if os.path.getsize(path) == 0:
            continue  # 空檔案，跳過
        with open(path, 'r') as f:
            first_line = f.readline().strip()
            if not first_line:
                continue
            parts = first_line.split()
            if len(parts) == 5:
                return 'aabb'
            elif len(parts) == 9:
                return 'obb'
            else:
                # 參數長度不符合 => 可能是損壞檔或意外格式
                continue
    return 'aabb'

########################################
# === 新增: 3) Rotate Labels
########################################

def rotate_labels_aabb(label_lines, M, old_w, old_h, new_w, new_h):
    """
    旋轉 AABB 標籤：
    1. 先將 YOLO bbox 轉為 corners (xmin, ymin, xmax, ymax)
    2. 用旋轉仿射矩陣 M 轉換每個 corner
    3. clamp 到 [0, new_w-1] x [0, new_h-1]
    4. 再轉回 YOLO 格式
    """
    new_label_lines = []
    for line in label_lines:
        parsed = parse_label_line_aabb(line)
        if not parsed:
            continue
        cls, x_center, y_center, w, h = parsed

        xmin, ymin, xmax, ymax = yolo_to_corners_aabb(x_center, y_center, w, h, old_w, old_h)

        corners = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        transformed_pts = []
        for (px, py) in corners:
            x_new, y_new = apply_affine_to_point(px, py, M)
            transformed_pts.append((x_new, y_new))

        xs = [p[0] for p in transformed_pts]
        ys = [p[1] for p in transformed_pts]
        xmin_t, xmax_t = min(xs), max(xs)
        ymin_t, ymax_t = min(ys), max(ys)

        # clamp
        xmin_clamped = max(0, min(xmin_t, new_w - 1))
        xmax_clamped = max(0, min(xmax_t, new_w - 1))
        ymin_clamped = max(0, min(ymin_t, new_h - 1))
        ymax_clamped = max(0, min(ymax_t, new_h - 1))

        bbox = corners_to_yolo_fmt(xmin_clamped, ymin_clamped, xmax_clamped, ymax_clamped, new_w, new_h)
        if bbox is not None:
            cx_norm, cy_norm, w_norm, h_norm = bbox
            new_label_lines.append(f"{cls} {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}")
    return new_label_lines

def rotate_labels_obb(label_lines, M, old_w, old_h, new_w, new_h):
    """
    旋轉 OBB 標籤：
    1. 將 OBB(8點)轉成 pixel coords
    2. 用旋轉仿射矩陣 M 轉換
    3. clamp
    4. 轉回 normalized coords
    """
    new_label_lines = []
    for line in label_lines:
        parsed = parse_label_line_obb(line)
        if not parsed:
            continue
        cls, coords = parsed

        pixel_pts = obb_to_pixel_coords(coords, old_w, old_h)

        transformed_pts = []
        for (px, py) in pixel_pts:
            x_new, y_new = apply_affine_to_point(px, py, M)
            transformed_pts.append((x_new, y_new))

        clamped_pts = []
        for (xx, yy) in transformed_pts:
            xx_clamped = max(0, min(xx, new_w - 1))
            yy_clamped = max(0, min(yy, new_h - 1))
            clamped_pts.append((xx_clamped, yy_clamped))

        norm_coords = pixel_coords_to_obb_format(clamped_pts, new_w, new_h)
        formatted = " ".join(f"{v:.6f}" for v in norm_coords)
        new_label_lines.append(f"{cls} {formatted}")
    return new_label_lines

def rotate_labels(label_lines, M, old_w, old_h, new_w, new_h, label_format="aabb"):
    if label_format.lower() == "aabb":
        return rotate_labels_aabb(label_lines, M, old_w, old_h, new_w, new_h)
    elif label_format.lower() == "obb":
        return rotate_labels_obb(label_lines, M, old_w, old_h, new_w, new_h)
    else:
        raise ValueError(f"未知的標籤格式: {label_format}")

########################################
# J. Dataset Split + Augment (核心)
########################################

def split_and_augment_dataset(
    input_images_dir,
    input_labels_dir,
    output_base_dir,
    rotate_step=0,
    brightness_factor=0.0,
    blur_kernel_size=0,
    shear_ratio=0.0,
    crop_ratio=0.0,
    progress_callback=None,
    label_format='auto'
):
    # 1) 若 label_format=='auto' => 自動判斷
    if label_format == 'auto':
        detected = auto_detect_label_format(input_labels_dir)
        if progress_callback:
            progress_callback(f"自動偵測標籤格式: {detected}")
        label_format = detected  # 接下來都用這個

    train_images_dir = os.path.join(output_base_dir, 'train', 'images')
    train_labels_dir = os.path.join(output_base_dir, 'train', 'labels')
    valid_images_dir = os.path.join(output_base_dir, 'valid', 'images')
    valid_labels_dir = os.path.join(output_base_dir, 'valid', 'labels')

    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(valid_images_dir, exist_ok=True)
    os.makedirs(valid_labels_dir, exist_ok=True)

    image_files = [
        f for f in os.listdir(input_images_dir)
        if f.lower().endswith(('.jpg', '.png', '.jpeg'))
    ]
    total_images = len(image_files)
    if progress_callback:
        progress_callback(f"Found {total_images} images in '{input_images_dir}'.")

    if total_images == 0:
        if progress_callback:
            progress_callback("No images found. Abort.")
        return

    # 隨機打亂 & split
    random.shuffle(image_files)
    split_index = total_images // 2
    train_files = image_files[:split_index]
    valid_files = image_files[split_index:]

    if progress_callback:
        progress_callback(f"Split: Train={len(train_files)}, Valid={len(valid_files)}")

    # 先計算「每張圖的增強數量」，再計算「總增強數量」
    aug_count_per_image = calculate_augmentation_count(
        rotate_step,
        brightness_factor,
        blur_kernel_size,
        shear_ratio,
        crop_ratio
    )
    total_aug_images = len(train_files) * aug_count_per_image
    if progress_callback:
        progress_callback(f"預計將額外產生 {total_aug_images} 張增強影像（訓練集部分）")

    # ---- valid: 只複製 ----
    for vf in valid_files:
        src_img_path = os.path.join(input_images_dir, vf)
        dst_img_path = os.path.join(valid_images_dir, vf)
        copy_file(src_img_path, dst_img_path)

        label_path = os.path.splitext(vf)[0] + '.txt'
        src_lbl_path = os.path.join(input_labels_dir, label_path)
        if os.path.exists(src_lbl_path):
            dst_lbl_path = os.path.join(valid_labels_dir, label_path)
            copy_file(src_lbl_path, dst_lbl_path)

    # ---- train: 原圖 + 單獨套用各增強 ----
    augmented_items = []
    for tf in train_files:
        src_img_path = os.path.join(input_images_dir, tf)
        image = cv2.imread(src_img_path)
        if image is None:
            continue
        h, w = image.shape[:2]
        base_name = os.path.splitext(tf)[0]

        # 複製原圖 & 原標籤
        dst_img_path = os.path.join(train_images_dir, tf)
        copy_file(src_img_path, dst_img_path)

        label_file = base_name + '.txt'
        src_lbl_path = os.path.join(input_labels_dir, label_file)
        label_lines = copy_labels(src_lbl_path)
        if os.path.exists(src_lbl_path):
            dst_lbl_path = os.path.join(train_labels_dir, label_file)
            copy_file(src_lbl_path, dst_lbl_path)

        # (1) Rotate
        if rotate_step > 0:
            angles = list(range(0, 360, rotate_step))
            angles = [a for a in angles if a != 0]  # 排除 angle=0
            for ang in angles:
                rotated_img, M_rot, new_w_, new_h_ = rotate_image(image, ang)
                out_img_name = f"{base_name}_rotate_{ang}.jpg"
                out_lbl_name = f"{base_name}_rotate_{ang}.txt"

                # 旋轉標籤
                new_label_lines = rotate_labels(label_lines, M_rot, w, h, new_w_, new_h_, label_format=label_format)
                augmented_items.append((rotated_img, out_img_name, new_label_lines, out_lbl_name))

        # (2) Brightness
        if brightness_factor > 0:
            bright_img = random_brightness(image, brightness_factor)
            out_img_name = f"{base_name}_bright.jpg"
            out_lbl_name = f"{base_name}_bright.txt"
            # 亮度不影響標籤
            new_label_lines = label_lines
            augmented_items.append((bright_img, out_img_name, new_label_lines, out_lbl_name))

        # (3) Blur
        if blur_kernel_size > 0:
            blur_img = random_blur(image, blur_kernel_size)
            out_img_name = f"{base_name}_blur.jpg"
            out_lbl_name = f"{base_name}_blur.txt"
            # 模糊不影響標籤
            new_label_lines = label_lines
            augmented_items.append((blur_img, out_img_name, new_label_lines, out_lbl_name))

        # (4) Shear
        if shear_ratio > 0:
            sheared_img, M_shear, new_w_shear, new_h_shear, shear_factor = random_shear(image, shear_ratio)
            out_img_name = f"{base_name}_shear.jpg"
            out_lbl_name = f"{base_name}_shear.txt"

            # 正確更新標籤
            new_label_lines = shear_labels(label_lines, shear_factor, w, h, new_w_shear, new_h_shear, label_format=label_format)
            augmented_items.append((sheared_img, out_img_name, new_label_lines, out_lbl_name))

        # (5) Crop
        if crop_ratio > 0:
            cz_img, scale, offset_x, offset_y = random_center_crop_zoom(image, crop_ratio)
            out_img_name = f"{base_name}_crop.jpg"
            out_lbl_name = f"{base_name}_crop.txt"

            new_label_lines = crop_labels(label_lines, scale, offset_x, offset_y, w, h, label_format=label_format)
            augmented_items.append((cz_img, out_img_name, new_label_lines, out_lbl_name))

    if progress_callback:
        progress_callback(f"Total augmented items (train): {len(augmented_items)}")

    # --- 寫出增強檔 ---
    for aug_img, aug_img_name, aug_lbls, aug_lbl_name in augmented_items:
        out_img_path = os.path.join(train_images_dir, aug_img_name)
        out_lbl_path = os.path.join(train_labels_dir, aug_lbl_name)
        cv2.imwrite(out_img_path, aug_img)
        save_labels(out_lbl_path, aug_lbls)

    if progress_callback:
        progress_callback("Augmentation done.")

########################################
# K. QThread
########################################

class AugmentationThread(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(
        self,
        input_images_dir,
        input_labels_dir,
        output_base_dir,
        rotate_step,
        brightness_factor,
        blur_kernel_size,
        shear_ratio,
        crop_ratio,
        label_format='auto'
    ):
        super().__init__()
        self.input_images_dir = input_images_dir
        self.input_labels_dir = input_labels_dir
        self.output_base_dir = output_base_dir
        self.rotate_step = rotate_step
        self.brightness_factor = brightness_factor
        self.blur_kernel_size = blur_kernel_size
        self.shear_ratio = shear_ratio
        self.crop_ratio = crop_ratio
        self.label_format = label_format

    def run(self):
        try:
            self.progress.emit("數據增強開始...")
            split_and_augment_dataset(
                input_images_dir=self.input_images_dir,
                input_labels_dir=self.input_labels_dir,
                output_base_dir=self.output_base_dir,
                rotate_step=self.rotate_step,
                brightness_factor=self.brightness_factor,
                blur_kernel_size=self.blur_kernel_size,
                shear_ratio=self.shear_ratio,
                crop_ratio=self.crop_ratio,
                progress_callback=self.progress.emit,
                label_format=self.label_format
            )
            self.progress.emit("數據增強完成。")
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))
