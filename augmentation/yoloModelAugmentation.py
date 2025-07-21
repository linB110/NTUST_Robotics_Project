import sys
import os
import cv2
import numpy as np
import random
import shutil

from PyQt5.QtWidgets import (
    QApplication, QVBoxLayout, QLabel, QLineEdit,
    QPushButton, QSlider, QFileDialog, QWidget, QHBoxLayout, QMessageBox, QGroupBox, QComboBox, QSpinBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5 import QtGui


def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(matrix[0, 0])
    sin = np.abs(matrix[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    matrix[0, 2] += (nW / 2) - center[0]
    matrix[1, 2] += (nH / 2) - center[1]
    rotated = cv2.warpAffine(image, matrix, (nW, nH))
    return rotated, matrix


def rotate_point(x, y, matrix):
    new_x = matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]
    new_y = matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]
    return new_x, new_y


def rotate_labels(label_path, matrix, w, h, nW, nH):
    """
    OBB 格式 (class_id x1 y1 x2 y2 x3 y3 x4 y4)。
    """
    if not os.path.exists(label_path):
        print(f"Warning: Label file {label_path} does not exist.")
        return []

    with open(label_path, 'r') as file:
        lines = file.readlines()

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        class_id = parts[0]
        coords = list(map(float, parts[1:]))
        new_coords = []
        # 逐點旋轉
        for i in range(0, len(coords), 2):
            x_norm, y_norm = coords[i], coords[i + 1]
            x = x_norm * w
            y = y_norm * h
            new_x, new_y = rotate_point(x, y, matrix)
            new_x_norm = new_x / nW
            new_y_norm = new_y / nH
            # 限制範圍到 [0, 1]
            new_x_norm = min(max(new_x_norm, 0), 1)
            new_y_norm = min(max(new_y_norm, 0), 1)
            new_coords.extend([new_x_norm, new_y_norm])
        new_line = ' '.join([class_id] + list(map(str, new_coords)))
        new_lines.append(new_line)

    return new_lines


def rotate_labels_aabb(label_path, matrix, w, h, nW, nH):
    """
    AABB 格式 (class_id cx cy w h)，旋轉後重新計算 AABB。
    """
    if not os.path.exists(label_path):
        print(f"Warning: Label file {label_path} does not exist.")
        return []

    with open(label_path, 'r') as file:
        lines = file.readlines()

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        class_id = parts[0]
        cx, cy, bw, bh = map(float, parts[1:5])

        # 將中心點及寬高轉為絕對座標
        abs_cx = cx * w
        abs_cy = cy * h
        abs_w = bw * w
        abs_h = bh * h

        # 計算四個角點（左上、右上、右下、左下）
        points = [
            (abs_cx - abs_w / 2, abs_cy - abs_h / 2),  # 左上
            (abs_cx + abs_w / 2, abs_cy - abs_h / 2),  # 右上
            (abs_cx + abs_w / 2, abs_cy + abs_h / 2),  # 右下
            (abs_cx - abs_w / 2, abs_cy + abs_h / 2)   # 左下
        ]

        # 旋轉每個點
        rotated_points = [rotate_point(px, py, matrix) for px, py in points]

        # 限制旋轉後的點在影像範圍內 [0, nW-1]、[0, nH-1]
        rotated_points = [
            (min(max(0, px), nW - 1), min(max(0, py), nH - 1))
            for px, py in rotated_points
        ]

        # 根據旋轉後的點集計算新的 AABB
        xs, ys = zip(*rotated_points)
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # 計算新的中心點和寬高，並進行歸一化
        new_cx = ((min_x + max_x) / 2) / nW
        new_cy = ((min_y + max_y) / 2) / nH
        new_bw = (max_x - min_x) / nW
        new_bh = (max_y - min_y) / nH

        # 限制範圍 [0, 1]
        new_cx = min(max(new_cx, 0), 1)
        new_cy = min(max(new_cy, 0), 1)
        new_bw = min(max(new_bw, 0), 1)
        new_bh = min(max(new_bh, 0), 1)

        # 如果寬或高 <= 0，則跳過
        if new_bw <= 0 or new_bh <= 0:
            continue

        # 新的標籤行
        new_line = f"{class_id} {new_cx:.6f} {new_cy:.6f} {new_bw:.6f} {new_bh:.6f}"
        new_lines.append(new_line)

    return new_lines


def save_labels(label_path, lines):
    with open(label_path, 'w') as file:
        for line in lines:
            file.write(f"{line}\n")


def adjust_brightness(image, brightness_factor):
    """
    調整影像亮度：brightness_factor < 1.0 時變暗, > 1.0 時變亮
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 2] = hsv[:, :, 2] * brightness_factor
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
    hsv = np.array(hsv, dtype=np.uint8)
    image_bright = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return image_bright


def apply_blur(image, blur_kernel_size):
    """
    以指定的 kernel size 對影像做 Gaussian Blur
    """
    if blur_kernel_size > 0:
        if blur_kernel_size % 2 == 0:
            blur_kernel_size += 1
        image_blur = cv2.GaussianBlur(image, (blur_kernel_size, blur_kernel_size), 0)
        return image_blur
    return image


def copy_file(src, dst, progress_callback=None):
    """
    複製檔案，若需要可在這裡加上進度回呼。
    """
    try:
        shutil.copy2(src, dst)
        if progress_callback:
            progress_callback(f"Copied file: {src} -> {dst}")
    except Exception as e:
        if progress_callback:
            progress_callback(f"Error copying file {src} to {dst}: {e}")


#################################
# 自動偵測 Format 函式
#################################
def detect_dataset_format(labels_dir):
    """
    偵測標籤檔案的格式:
      - 若其中一行(例如第一行)有 5 個參數(含 classId)，則推斷為 AABB
      - 若有 9 個參數(含 classId)，則推斷為 OBB
      - 若均不符合或無標籤檔案，回傳 None
    """
    label_files = [f for f in os.listdir(labels_dir) if f.lower().endswith('.txt')]
    if not label_files:
        return None  # 沒有任何標籤檔案

    first_label_file_path = os.path.join(labels_dir, label_files[0])
    with open(first_label_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # 跳過空行
            parts = line.split()
            # AABB => class_id cx cy w h -> parts = 5
            # OBB  => class_id x1 y1 x2 y2 x3 y3 x4 y4 -> parts = 9
            if len(parts) == 5:
                return "AABB"
            elif len(parts) == 9:
                return "OBB"
            # 如果都不符合，也可繼續下一行或直接回傳 None
        return None

def get_rotation_angles(min_angle, max_angle, step):
    if min_angle > max_angle:
        min_angle, max_angle = max_angle, min_angle
    return list(range(min_angle, max_angle + 1, step))


def split_and_augment_dataset(
    input_images_dir,
    input_labels_dir,
    output_base_dir,
    step=120,
    brightness_factor=1.0,
    blur_kernel_size=0,
    dataset_format="OBB",
    min_angle=-180,
    max_angle=180,
    train_ratio=50,
    progress_callback=None
):
    rotate_func = rotate_labels if dataset_format == "OBB" else rotate_labels_aabb
    identity_matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)

    os.makedirs(output_base_dir, exist_ok=True)
    images = [f for f in os.listdir(input_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total = len(images)
    if total == 0:
        if progress_callback:
            progress_callback("沒有找到圖片，結束處理。")
        return

    # ✅ 先分割 train/valid
    random.shuffle(images)
    train_count = int(total * (train_ratio / 100.0))
    train_files = images[:train_count]
    valid_files = images[train_count:]

    def ensure_dirs(base):
        os.makedirs(os.path.join(base, "images"), exist_ok=True)
        os.makedirs(os.path.join(base, "labels"), exist_ok=True)

    def augment_and_save(file_list, split_name):
        split_base = os.path.join(output_base_dir, split_name)
        ensure_dirs(split_base)

        for fname in file_list:
            img_path = os.path.join(input_images_dir, fname)
            label_path = os.path.join(input_labels_dir, os.path.splitext(fname)[0] + ".txt")
            img = cv2.imread(img_path)
            if img is None:
                continue

            (h, w) = img.shape[:2]
            base_name = os.path.splitext(fname)[0]

            def save(img_data, lbl_data, img_name_suffix):
                img_name = f"{base_name}{img_name_suffix}.jpg"
                lbl_name = f"{base_name}{img_name_suffix}.txt"
                img_save_path = os.path.join(split_base, "images", img_name)
                lbl_save_path = os.path.join(split_base, "labels", lbl_name)
                cv2.imwrite(img_save_path, img_data)
                save_labels(lbl_save_path, lbl_data)
                if progress_callback:
                    progress_callback(f"[{split_name}] 儲存圖像: {img_save_path}")

            # 原圖
            label_lines = rotate_func(label_path, identity_matrix, w, h, w, h)
            if label_lines:
                save(img.copy(), label_lines, "")

            # 旋轉
            angles = get_rotation_angles(min_angle, max_angle, step)
            for angle in angles:
                rotated_img, matrix = rotate_image(img, angle)
                nH, nW = rotated_img.shape[:2]
                new_labels = rotate_func(label_path, matrix, w, h, nW, nH)
                if new_labels:
                    save(rotated_img, new_labels, f"_rot{angle}")

            # 亮度
            bright_img = adjust_brightness(img, brightness_factor)
            bright_labels = rotate_func(label_path, identity_matrix, w, h, w, h)
            if bright_labels:
                save(bright_img, bright_labels, "_bright")

            # 模糊
            if blur_kernel_size > 0:
                blur_img = apply_blur(img, blur_kernel_size)
                blur_labels = rotate_func(label_path, identity_matrix, w, h, w, h)
                if blur_labels:
                    save(blur_img, blur_labels, "_blur")

            if progress_callback:
                progress_callback(f"[{split_name}] 處理完成: {fname}")

    augment_and_save(train_files, "train")
    augment_and_save(valid_files, "valid")

    if progress_callback:
        progress_callback(f"訓練集: {len(train_files)} 張圖像，驗證集: {len(valid_files)} 張圖像")
        progress_callback("數據增強與分割完成。")


class AugmentationThread(QThread):
    """
    用於後台執行數據增強，以免卡住主介面。
    """
    progress = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(
        self,
        input_images_dir,
        input_labels_dir,
        output_base_dir,
        step,
        brightness_factor,
        blur_kernel_size,
        dataset_format,
        min_angle,
        max_angle,
        train_ratio
    ):
        super().__init__()
        self.input_images_dir = input_images_dir
        self.input_labels_dir = input_labels_dir
        self.output_base_dir = output_base_dir
        self.step = step
        self.brightness_factor = brightness_factor
        self.blur_kernel_size = blur_kernel_size
        self.dataset_format = dataset_format
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.train_ratio = train_ratio

    def run(self):
        try:
            self.progress.emit("數據增強開始...")
            split_and_augment_dataset(
                input_images_dir=self.input_images_dir,
                input_labels_dir=self.input_labels_dir,
                output_base_dir=self.output_base_dir,
                step=self.step,
                brightness_factor=self.brightness_factor,
                blur_kernel_size=self.blur_kernel_size,
                dataset_format=self.dataset_format,
                min_angle=self.min_angle,
                max_angle=self.max_angle,
                train_ratio=self.train_ratio,
                progress_callback=self.progress.emit
            )
            self.progress.emit("數據增強完成。")
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))



class AugmentationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.selected_input_folder = None
        self.selected_output_folder = None
        self.setAcceptDrops(True)
        self.augmentation_started = False
        self.total_images = 0
        self.preview_image_path = None
        self.original_preview_image = None
        self.processed_preview_image = None
        print("應用程式初始化完成")

    def initUI(self):
        self.setWindowTitle("Dataset Augmentation Tool")
        self.setGeometry(100, 100, 1000, 800)

        layout = QVBoxLayout()

        # 輸入資料夾選擇區域
        input_layout = QVBoxLayout()
        input_label = QLabel("選擇輸入資料夾（包含 'images' 和 'labels' 子資料夾）")
        input_label.setAlignment(Qt.AlignCenter)
        input_label.setStyleSheet("border: 2px dashed #aaaaaa; padding: 20px;")
        self.input_folder_label = input_label
        input_layout.addWidget(self.input_folder_label)

        input_browse_button = QPushButton("瀏覽輸入資料夾")
        input_browse_button.clicked.connect(self.browse_input_folder)
        input_layout.addWidget(input_browse_button)
        layout.addLayout(input_layout)

        # 輸出資料夾選擇區域
        output_layout = QVBoxLayout()
        output_label = QLabel("選擇輸出資料夾")
        output_label.setAlignment(Qt.AlignCenter)
        output_label.setStyleSheet("border: 2px dashed #aaaaaa; padding: 20px;")
        self.output_folder_label = output_label
        output_layout.addWidget(output_label)

        output_browse_button = QPushButton("瀏覽輸出資料夾")
        output_browse_button.clicked.connect(self.browse_output_folder)
        output_layout.addWidget(output_browse_button)
        layout.addLayout(output_layout)

        # 資料集格式選擇
        format_layout = QHBoxLayout()
        format_label = QLabel("資料集格式:")
        self.format_combobox = QComboBox()
        self.format_combobox.addItems(["OBB", "AABB"])  # 預設先放兩個選項
        format_layout.addWidget(format_label)
        format_layout.addWidget(self.format_combobox)
        layout.addLayout(format_layout)

        # 旋轉角度增量 - 用 QSpinBox 取代 QLineEdit
        step_label = QLabel("旋轉角度增量 (1~360):")
        self.spin_box = QSpinBox()
        self.spin_box.setRange(1, 360)    # 最小值=1, 最大值=360
        self.spin_box.setValue(120)       # 預設值
        self.spin_box.setSingleStep(1)    # 每次按箭頭時的步長，可自行調整
        layout.addWidget(step_label)
        layout.addWidget(self.spin_box)
        
        # 旋轉角度範圍
        angle_range_layout = QHBoxLayout()
        angle_range_layout.addWidget(QLabel("旋轉角度範圍 (min/max):"))

        self.min_angle_spin = QSpinBox()
        self.min_angle_spin.setRange(-180, 180)
        self.min_angle_spin.setValue(0)
        self.min_angle_spin.setToolTip("最小旋轉角度，範圍 -180 到 180")
        angle_range_layout.addWidget(self.min_angle_spin)

        self.max_angle_spin = QSpinBox()
        self.max_angle_spin.setRange(-180, 180)
        self.max_angle_spin.setValue(180)
        self.max_angle_spin.setToolTip("最大旋轉角度，範圍 -180 到 180")
        angle_range_layout.addWidget(self.max_angle_spin)

        layout.addLayout(angle_range_layout)

        # 訓練集比例
        partition_layout = QHBoxLayout()
        partition_layout.addWidget(QLabel("訓練集比例 (%):"))

        self.train_ratio_spin = QSpinBox()
        self.train_ratio_spin.setRange(1, 99)
        self.train_ratio_spin.setValue(50)
        self.train_ratio_spin.setToolTip("介於 1 到 99 之間的訓練比例，剩餘即為驗證集比例")
        partition_layout.addWidget(self.train_ratio_spin)

        layout.addLayout(partition_layout)


        # 亮度調整設置
        brightness_layout = QHBoxLayout()
        brightness_label = QLabel("亮度調整:")
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(50, 150)  # 0.5 到 1.5
        self.brightness_slider.setValue(100)      # 預設值為 1.0
        self.brightness_slider.valueChanged.connect(self.update_brightness_label)
        self.brightness_value = QLabel("1.00")
        brightness_layout.addWidget(brightness_label)
        brightness_layout.addWidget(self.brightness_slider)
        brightness_layout.addWidget(self.brightness_value)
        layout.addLayout(brightness_layout)

        # 模糊調整設置
        blur_layout = QHBoxLayout()
        blur_label = QLabel("模糊調整:")
        self.blur_slider = QSlider(Qt.Horizontal)
        self.blur_slider.setRange(0, 15)
        self.blur_slider.setValue(0)
        self.blur_slider.setTickInterval(1)
        self.blur_slider.setTickPosition(QSlider.TicksBelow)
        self.blur_slider.valueChanged.connect(self.update_blur_label)
        self.blur_value = QLabel("0")
        blur_layout.addWidget(blur_label)
        blur_layout.addWidget(self.blur_slider)
        blur_layout.addWidget(self.blur_value)
        layout.addLayout(blur_layout)

        # 增強後影像數量顯示
        count_layout = QVBoxLayout()
        self.train_count_label = QLabel("預計訓練集影像數: -")
        self.valid_count_label = QLabel("預計驗證集影像數: -")
        count_layout.addWidget(self.train_count_label)
        count_layout.addWidget(self.valid_count_label)
        layout.addLayout(count_layout)

        # 預覽區域
        preview_group = QGroupBox("亮度與模糊調整預覽")
        preview_layout = QHBoxLayout()

        self.original_preview_label = QLabel("原始圖像預覽")
        self.original_preview_label.setAlignment(Qt.AlignCenter)
        self.original_preview_label.setFixedSize(400, 400)
        self.original_preview_label.setStyleSheet("border: 1px solid black;")
        preview_layout.addWidget(self.original_preview_label)

        self.processed_preview_label = QLabel("調整後圖像預覽")
        self.processed_preview_label.setAlignment(Qt.AlignCenter)
        self.processed_preview_label.setFixedSize(400, 400)
        self.processed_preview_label.setStyleSheet("border: 1px solid black;")
        preview_layout.addWidget(self.processed_preview_label)

        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)

        # 預覽選擇按鈕
        preview_button = QPushButton("選擇預覽圖像")
        preview_button.clicked.connect(self.select_preview_image)
        layout.addWidget(preview_button)

        # 執行按鈕
        execute_button = QPushButton("執行數據增強")
        execute_button.clicked.connect(self.confirm_augmentation)
        layout.addWidget(execute_button)

        # 狀態顯示
        self.status_label = QLabel("狀態: 等待操作")
        layout.addWidget(self.status_label)

        self.setLayout(layout)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            folder_path = urls[0].toLocalFile()
            if os.path.isdir(folder_path):
                if 'images' in os.listdir(folder_path) and 'labels' in os.listdir(folder_path):
                    self.selected_input_folder = folder_path
                    self.input_folder_label.setText(f"輸入資料夾: {folder_path}")
                    self.status_label.setText("狀態: 已選擇輸入資料夾")
                    self.calculate_total_images()
                    self.auto_detect_format()  # 拖曳時也做自動檢測
                else:
                    self.selected_output_folder = folder_path
                    self.output_folder_label.setText(f"輸出資料夾: {folder_path}")
                    self.status_label.setText("狀態: 已選擇輸出資料夾")
            else:
                self.status_label.setText("狀態: 請拖放有效的資料夾")

    def browse_input_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "選擇輸入資料夾")
        if folder_path:
            if 'images' in os.listdir(folder_path) and 'labels' in os.listdir(folder_path):
                self.selected_input_folder = folder_path
                self.input_folder_label.setText(f"輸入資料夾: {folder_path}")
                self.status_label.setText("狀態: 已選擇輸入資料夾")
                self.calculate_total_images()
                self.auto_detect_format()  # 手動瀏覽後呼叫自動檢測
            else:
                QMessageBox.warning(self, "警告", "選擇的資料夾不包含 'images' 和 'labels' 子資料夾！")

    def browse_output_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "選擇輸出資料夾")
        if folder_path:
            self.selected_output_folder = folder_path
            self.output_folder_label.setText(f"輸出資料夾: {folder_path}")
            self.status_label.setText("狀態: 已選擇輸出資料夾")
    
    ##############################################
    # 自動偵測標籤格式 (AABB/OBB)，並設定 combobox
    ##############################################
    def auto_detect_format(self):
        """
        自動偵測標籤格式 (AABB/OBB)，並設定 combobox。
        """
        if not self.selected_input_folder:
            return

        labels_dir = os.path.join(self.selected_input_folder, "labels")
        if not os.path.exists(labels_dir):
            return

        detected_format = detect_dataset_format(labels_dir)
        if detected_format == "AABB":
            self.format_combobox.setCurrentText("AABB")
            self.status_label.setText("狀態: 偵測到 AABB 格式，已自動設定")
        elif detected_format == "OBB":
            self.format_combobox.setCurrentText("OBB")
            self.status_label.setText("狀態: 偵測到 OBB 格式，已自動設定")
        else:
            self.status_label.setText("狀態: 無法自動判斷，請手動選擇 Format")

    def calculate_total_images(self):
        if not self.selected_input_folder:
            return

        images_dir = os.path.join(self.selected_input_folder, "images")
        if os.path.exists(images_dir):
            self.total_images = len(
                [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
            )
            self.update_image_counts()
        else:
            self.total_images = 0
            self.update_image_counts()

    def update_brightness_label(self):
        value = self.brightness_slider.value() / 100
        self.brightness_value.setText(f"{value:.2f}")
        self.update_image_counts()
        self.update_preview()

    def update_blur_label(self):
        value = self.blur_slider.value()
        self.blur_value.setText(f"{value}")
        self.update_image_counts()
        self.update_preview()

    def update_image_counts(self):
        """
        根據使用者設定，估算最終增強總數，並依訓練比例預估 train/valid 數量。
        """
        if self.total_images == 0:
            self.train_count_label.setText("預計訓練集影像數: -")
            self.valid_count_label.setText("預計驗證集影像數: -")
            return

        try:
            step = self.spin_box.value()
            min_angle = self.min_angle_spin.value()
            max_angle = self.max_angle_spin.value()
            train_ratio = self.train_ratio_spin.value()
            blur_kernel_size = self.blur_slider.value()

            if min_angle > max_angle:
                min_angle, max_angle = max_angle, min_angle

            # 計算旋轉 augment 次數
            rotation_count = ((max_angle - min_angle) // step) + 1 if step > 0 else 0
            brightness = 1  # 固定會套用一次亮度
            blur = 1 if blur_kernel_size > 0 else 0

            # 每張圖像會有：原圖 + 旋轉 + 亮度 + 模糊
            per_image_output = 1 + rotation_count + brightness + blur
            total_augmented_images = self.total_images * per_image_output

            # 用比例切分
            train_count = int(total_augmented_images * (train_ratio / 100.0))
            valid_count = total_augmented_images - train_count

            self.train_count_label.setText(f"預計訓練集影像數: {train_count}")
            self.valid_count_label.setText(f"預計驗證集影像數: {valid_count}")

        except Exception as e:
            self.train_count_label.setText("預計訓練集影像數: -")
            self.valid_count_label.setText("預計驗證集影像數: -")
            print(f"更新影像數量時出現錯誤: {e}")

    def confirm_augmentation(self):
        if not self.selected_input_folder:
            self.status_label.setText("狀態: 請先選擇輸入資料夾")
            QMessageBox.warning(self, "警告", "請先選擇包含 'images' 和 'labels' 子資料夾的輸入資料夾！")
            return

        reply = QMessageBox.question(
            self, "確認", "是否開始數據增強？", QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            print("用戶確認開始數據增強")
            self.run_augmentation()
        else:
            print("用戶取消數據增強")

    def run_augmentation(self):
        if self.augmentation_started:
            QMessageBox.warning(self, "警告", "增強已經開始，請勿重複操作！")
            return

        # 改用 spin_box 取得旋轉角度
        step = self.spin_box.value()
        brightness_factor = self.brightness_slider.value() / 100
        blur_kernel_size = self.blur_slider.value()
        min_angle = self.min_angle_spin.value()
        max_angle = self.max_angle_spin.value()
        train_ratio = self.train_ratio_spin.value()

        input_images_dir = os.path.join(self.selected_input_folder, "images")
        input_labels_dir = os.path.join(self.selected_input_folder, "labels")

        if self.selected_output_folder:
            output_base_dir = self.selected_output_folder
        else:
            # 沒有指定輸出資料夾時，就在輸入資料夾下建立一個 "Augmentation" 資料夾
            output_base_dir = os.path.join(self.selected_input_folder, "Augmentation")

        if not os.path.exists(input_images_dir) or not os.path.exists(input_labels_dir):
            self.status_label.setText("狀態: 無效的資料夾結構")
            QMessageBox.critical(self, "錯誤", "資料夾結構無效，請確認包含 'images' 和 'labels' 資料夾！")
            return

        dataset_format = self.format_combobox.currentText()

        self.status_label.setText("狀態: 正在進行數據增強...")
        self.augmentation_started = True

        self.augmentation_thread = AugmentationThread(
            input_images_dir=input_images_dir,
            input_labels_dir=input_labels_dir,
            output_base_dir=output_base_dir,
            step=step,
            brightness_factor=brightness_factor,
            blur_kernel_size=blur_kernel_size,
            dataset_format=dataset_format,
            min_angle=min_angle,
            max_angle=max_angle,
            train_ratio=train_ratio
        )

        self.augmentation_thread.progress.connect(self.update_status)
        self.augmentation_thread.finished.connect(self.augmentation_finished)
        self.augmentation_thread.error.connect(self.augmentation_error)
        self.augmentation_thread.start()

    def update_status(self, message):
        self.status_label.setText(f"狀態: {message}")

    def augmentation_finished(self):
        self.status_label.setText("狀態: 數據增強完成")
        QMessageBox.information(self, "完成", "數據增強完成！\n已自動分割有效驗證集至 valid 資料夾。")
        self.augmentation_started = False

    def augmentation_error(self, error_message):
        self.status_label.setText(f"狀態: 發生錯誤 - {error_message}")
        QMessageBox.critical(self, "錯誤", f"發生錯誤: {error_message}")
        self.augmentation_started = False

    def select_preview_image(self):
        if not self.selected_input_folder:
            QMessageBox.warning(self, "警告", "請先選擇輸入資料夾！")
            return

        images_dir = os.path.join(self.selected_input_folder, "images")
        if not os.path.exists(images_dir):
            QMessageBox.warning(self, "警告", "輸入資料夾中沒有 'images' 子資料夾！")
            return

        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if not image_files:
            QMessageBox.warning(self, "警告", "輸入資料夾中沒有找到圖像文件！")
            return

        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(
            self, "選擇預覽圖像", images_dir, "Image Files (*.png *.jpg *.jpeg)", options=options
        )
        if file_path:
            self.preview_image_path = file_path
            self.original_preview_image = cv2.imread(file_path)
            if self.original_preview_image is None:
                QMessageBox.critical(self, "錯誤", "無法讀取選擇的圖像！")
                return
            self.display_image(self.original_preview_label, self.original_preview_image)
            self.update_preview()

    def display_image(self, label, image):
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QtGui.QImage(image.data, width, height, bytes_per_line, QtGui.QImage.Format_BGR888)
        pixmap = QtGui.QPixmap.fromImage(q_image)
        pixmap = pixmap.scaled(label.width(), label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pixmap)

    def update_preview(self):
        if self.original_preview_image is None:
            return

        brightness_factor = self.brightness_slider.value() / 100
        blur_kernel_size = self.blur_slider.value()

        # 調整亮度
        preview_image = adjust_brightness(self.original_preview_image, brightness_factor)
        # 模糊處理
        if blur_kernel_size > 0:
            if blur_kernel_size % 2 == 0:
                blur_kernel_size += 1
            preview_image = apply_blur(preview_image, blur_kernel_size)

        self.processed_preview_image = preview_image
        self.display_image(self.processed_preview_label, self.processed_preview_image)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AugmentationApp()
    window.show()
    sys.exit(app.exec_())
