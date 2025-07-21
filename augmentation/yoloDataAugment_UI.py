import sys
import os
import cv2
import random

from PyQt5.QtWidgets import (
    QApplication, QVBoxLayout, QLabel, QPushButton, QSlider, QFileDialog,
    QWidget, QHBoxLayout, QMessageBox, QGroupBox, QComboBox
)
from PyQt5.QtCore import Qt
from PyQt5 import QtGui

# 請確保此匯入名稱對應您實際檔名
from yoloDataAugment_API import (
    AugmentationThread,
    rotate_image,
    random_brightness,
    random_blur,
    random_shear,
    random_center_crop_zoom
)

def is_valid_input_folder(folder_path: str) -> bool:
    if not os.path.isdir(folder_path):
        return False
    contents = os.listdir(folder_path)
    return ('images' in contents) and ('labels' in contents)

def count_images_in_directory(images_dir: str) -> int:
    if not os.path.exists(images_dir):
        return 0
    image_files = [
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]
    return len(image_files)

class AugmentationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

        self.selected_input_folder = None
        self.selected_output_folder = None
        self.setAcceptDrops(True)

        self.augmentation_started = False
        self.total_images = 0

        # 預覽用
        self.preview_image_path = None
        self.original_preview_image = None
        self.processed_preview_image = None

        # 預設 label_format 為 AABB；若需要 OBB, 可以改成 'obb'
        self.label_format = "aabb"

        print("應用程式初始化完成")

    def initUI(self):
        self.setWindowTitle("Dataset Augmentation Tool (Single-Aug per Image)")
        self.setGeometry(100, 100, 1000, 800)

        layout = QVBoxLayout()

        # -------- 輸入資料夾選擇 --------
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

        # -------- 輸出資料夾選擇 --------
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

        # -------- Label Format 下拉選單 --------
        format_layout = QHBoxLayout()
        format_label = QLabel("Label Format:")
        self.format_combo = QComboBox()
        self.format_combo.addItems(["aabb", "obb"])  # AABB or OBB
        self.format_combo.currentTextChanged.connect(self.on_format_changed)
        format_layout.addWidget(format_label)
        format_layout.addWidget(self.format_combo)
        layout.addLayout(format_layout)

        # -------- 旋轉角度增量 (rotate_step) --------
        rotate_layout = QHBoxLayout()
        rotate_label = QLabel("Rotate Step (0 => skip):")
        self.rotate_slider = QSlider(Qt.Horizontal)
        self.rotate_slider.setRange(0, 180)
        self.rotate_slider.setValue(0)
        self.rotate_slider.setTickPosition(QSlider.TicksBelow)
        self.rotate_slider.valueChanged.connect(self.on_rotate_changed)
        self.rotate_value_label = QLabel("0")
        rotate_layout.addWidget(rotate_label)
        rotate_layout.addWidget(self.rotate_slider)
        rotate_layout.addWidget(self.rotate_value_label)
        layout.addLayout(rotate_layout)

        # -------- Brightness (0 => skip) --------
        brightness_layout = QHBoxLayout()
        brightness_label = QLabel("Brightness (0 => skip) (Max=0.50):")
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(0, 50)
        self.brightness_slider.setValue(0)
        self.brightness_slider.setTickPosition(QSlider.TicksBelow)
        self.brightness_slider.valueChanged.connect(self.on_brightness_changed)
        self.brightness_value_label = QLabel("0.00")
        brightness_layout.addWidget(brightness_label)
        brightness_layout.addWidget(self.brightness_slider)
        brightness_layout.addWidget(self.brightness_value_label)
        layout.addLayout(brightness_layout)

        # -------- Blur (0 => skip) --------
        blur_layout = QHBoxLayout()
        blur_label = QLabel("Blur Kernel (0 => skip):")
        self.blur_slider = QSlider(Qt.Horizontal)
        self.blur_slider.setRange(0, 15)
        self.blur_slider.setValue(0)
        self.blur_slider.setTickPosition(QSlider.TicksBelow)
        self.blur_slider.valueChanged.connect(self.on_blur_changed)
        self.blur_value_label = QLabel("0")
        blur_layout.addWidget(blur_label)
        blur_layout.addWidget(self.blur_slider)
        blur_layout.addWidget(self.blur_value_label)
        layout.addLayout(blur_layout)

        # -------- Shear (0 => skip) --------
        shear_layout = QHBoxLayout()
        shear_label = QLabel("Shear ratio (0 => skip) (Max=1.0):")
        self.shear_slider = QSlider(Qt.Horizontal)
        self.shear_slider.setRange(0, 100)
        self.shear_slider.setValue(0)
        self.shear_slider.setTickPosition(QSlider.TicksBelow)
        self.shear_slider.valueChanged.connect(self.on_shear_changed)
        self.shear_value_label = QLabel("0.00")
        shear_layout.addWidget(shear_label)
        shear_layout.addWidget(self.shear_slider)
        shear_layout.addWidget(self.shear_value_label)
        layout.addLayout(shear_layout)

        # -------- Crop (0 => skip) --------
        crop_layout = QHBoxLayout()
        crop_label = QLabel("Crop/Zoom ratio (0 => skip) (Max=1.0):")
        self.crop_slider = QSlider(Qt.Horizontal)
        self.crop_slider.setRange(0, 100)
        self.crop_slider.setValue(0)
        self.crop_slider.setTickPosition(QSlider.TicksBelow)
        self.crop_slider.valueChanged.connect(self.on_crop_changed)
        self.crop_value_label = QLabel("0.00")
        crop_layout.addWidget(crop_label)
        crop_layout.addWidget(self.crop_slider)
        crop_layout.addWidget(self.crop_value_label)
        layout.addLayout(crop_layout)

        # -------- 預計輸出影像數 --------
        count_layout = QVBoxLayout()
        self.train_count_label = QLabel("預計訓練集影像數: -")
        self.valid_count_label = QLabel("預計驗證集影像數: -")
        count_layout.addWidget(self.train_count_label)
        count_layout.addWidget(self.valid_count_label)
        layout.addLayout(count_layout)

        # -------- 預覽區域 --------
        preview_group = QGroupBox("預覽")
        preview_layout = QHBoxLayout()

        self.original_preview_label = QLabel("原圖")
        self.original_preview_label.setAlignment(Qt.AlignCenter)
        self.original_preview_label.setFixedSize(400, 400)
        self.original_preview_label.setStyleSheet("border: 1px solid black;")
        preview_layout.addWidget(self.original_preview_label)

        self.processed_preview_label = QLabel("增強後")
        self.processed_preview_label.setAlignment(Qt.AlignCenter)
        self.processed_preview_label.setFixedSize(400, 400)
        self.processed_preview_label.setStyleSheet("border: 1px solid black;")
        preview_layout.addWidget(self.processed_preview_label)

        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)

        # -------- 預覽選擇按鈕 --------
        preview_button = QPushButton("選擇預覽圖像")
        preview_button.clicked.connect(self.select_preview_image)
        layout.addWidget(preview_button)

        # -------- 執行按鈕 --------
        run_button = QPushButton("執行數據增強")
        run_button.clicked.connect(self.confirm_augmentation)
        layout.addWidget(run_button)

        # -------- 狀態顯示 --------
        self.status_label = QLabel("狀態: 等待操作")
        layout.addWidget(self.status_label)

        self.setLayout(layout)

    # ------------------ Label Format 下拉 ------------------
    def on_format_changed(self, text):
        self.label_format = text  # 'aabb' or 'obb'

    # ------------------ Slider 事件 ------------------
    def on_rotate_changed(self):
        val = self.rotate_slider.value()
        self.rotate_value_label.setText(str(val))
        self.update_preview()
        self.update_image_counts()

    def on_brightness_changed(self):
        val = self.brightness_slider.value() / 100.0
        self.brightness_value_label.setText(f"{val:.2f}")
        self.update_preview()
        self.update_image_counts()

    def on_blur_changed(self):
        val = self.blur_slider.value()
        self.blur_value_label.setText(str(val))
        self.update_preview()
        self.update_image_counts()

    def on_shear_changed(self):
        val = self.shear_slider.value() / 100.0
        self.shear_value_label.setText(f"{val:.2f}")
        self.update_preview()
        self.update_image_counts()

    def on_crop_changed(self):
        val = self.crop_slider.value() / 100.0
        self.crop_value_label.setText(f"{val:.2f}")
        self.update_preview()
        self.update_image_counts()

    # ------------------ Drag & Drop ------------------
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            folder_path = urls[0].toLocalFile()
            if os.path.isdir(folder_path):
                if is_valid_input_folder(folder_path):
                    self.selected_input_folder = folder_path
                    self.input_folder_label.setText(f"輸入資料夾: {folder_path}")
                    self.status_label.setText("狀態: 已選擇輸入資料夾")
                    self.calculate_total_images()
                else:
                    self.selected_output_folder = folder_path
                    self.output_folder_label.setText(f"輸出資料夾: {folder_path}")
                    self.status_label.setText("狀態: 已選擇輸出資料夾")
            else:
                self.status_label.setText("狀態: 請拖放有效的資料夾")

    # ------------------ 瀏覽資料夾 ------------------
    def browse_input_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "選擇輸入資料夾")
        if folder_path:
            if is_valid_input_folder(folder_path):
                self.selected_input_folder = folder_path
                self.input_folder_label.setText(f"輸入資料夾: {folder_path}")
                self.status_label.setText("狀態: 已選擇輸入資料夾")
                self.calculate_total_images()
            else:
                QMessageBox.warning(self, "警告", "選擇的資料夾不包含 'images' 和 'labels' 子資料夾！")

    def browse_output_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "選擇輸出資料夾")
        if folder_path:
            self.selected_output_folder = folder_path
            self.output_folder_label.setText(f"輸出資料夾: {folder_path}")
            self.status_label.setText("狀態: 已選擇輸出資料夾")

    def calculate_total_images(self):
        if not self.selected_input_folder:
            return
        images_dir = os.path.join(self.selected_input_folder, "images")
        self.total_images = count_images_in_directory(images_dir)
        self.update_image_counts()

    def update_image_counts(self):
        """
        根據目前的 slider 設定，粗估最終產生的影像數量。
        注意：這裡只是大概估算，與實際程序中計算的方式(已封裝在 API 裡)可能略有不同。
        """
        if self.total_images == 0:
            self.train_count_label.setText("預計訓練集影像數: -")
            self.valid_count_label.setText("預計驗證集影像數: -")
            return

        train_count = self.total_images // 2
        valid_count = self.total_images - train_count

        # rotate
        rotate_step = self.rotate_slider.value()
        if rotate_step > 0:
            # angles = 0, rotate_step, 2*rotate_step, ..., <360
            # 實際產生 rotate_num = len(angles) - 1
            possible_angles = list(range(0, 360, rotate_step))
            rotate_num = max(0, len(possible_angles) - 1)
        else:
            rotate_num = 0

        brightness_num = 1 if self.brightness_slider.value() > 0 else 0
        blur_num = 1 if self.blur_slider.value() > 0 else 0
        shear_num = 1 if self.shear_slider.value() > 0 else 0
        crop_num = 1 if self.crop_slider.value() > 0 else 0

        # 單張圖：原圖 + rotate_num + brightness + blur + shear + crop
        total_per_image = 1 + rotate_num + brightness_num + blur_num + shear_num + crop_num

        train_aug = train_count * total_per_image
        self.train_count_label.setText(f"預計訓練集影像數: {train_aug}")
        self.valid_count_label.setText(f"預計驗證集影像數: {valid_count}")

    # ------------------ 使用者確認後，開始增強 ------------------
    def confirm_augmentation(self):
        if not self.selected_input_folder:
            QMessageBox.warning(self, "警告", "尚未選擇輸入資料夾！")
            return

        reply = QMessageBox.question(self, "確認", "是否開始數據增強？", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.run_augmentation()

    def run_augmentation(self):
        if self.augmentation_started:
            QMessageBox.warning(self, "警告", "增強已經開始，請勿重複操作！")
            return

        rotate_step = self.rotate_slider.value()
        brightness_factor = self.brightness_slider.value() / 100.0
        blur_kernel_size = self.blur_slider.value()
        shear_ratio = self.shear_slider.value() / 100.0
        crop_ratio = self.crop_slider.value() / 100.0

        if not self.selected_input_folder:
            QMessageBox.warning(self, "警告", "尚未選擇輸入資料夾！")
            return
        images_dir = os.path.join(self.selected_input_folder, "images")
        labels_dir = os.path.join(self.selected_input_folder, "labels")
        if not (os.path.exists(images_dir) and os.path.exists(labels_dir)):
            QMessageBox.critical(self, "錯誤", "輸入資料夾結構無效（無 images/labels）")
            return

        if self.selected_output_folder:
            output_dir = self.selected_output_folder
        else:
            output_dir = os.path.join(self.selected_input_folder, "Augmentation")

        self.status_label.setText("狀態: 正在增強...")
        self.augmentation_started = True

        # 啟動多執行緒
        self.augmentation_thread = AugmentationThread(
            input_images_dir=images_dir,
            input_labels_dir=labels_dir,
            output_base_dir=output_dir,
            rotate_step=rotate_step,
            brightness_factor=brightness_factor,
            blur_kernel_size=blur_kernel_size,
            shear_ratio=shear_ratio,
            crop_ratio=crop_ratio,
            label_format=self.label_format
        )
        self.augmentation_thread.progress.connect(self.on_progress)
        self.augmentation_thread.finished.connect(self.on_finished)
        self.augmentation_thread.error.connect(self.on_error)
        self.augmentation_thread.start()

    def on_progress(self, msg):
        self.status_label.setText(f"狀態: {msg}")

    def on_finished(self):
        self.status_label.setText("狀態: 數據增強完成")
        QMessageBox.information(self, "完成", "數據增強完成！")
        self.augmentation_started = False

    def on_error(self, err_msg):
        self.status_label.setText(f"狀態: 發生錯誤 - {err_msg}")
        QMessageBox.critical(self, "錯誤", f"發生錯誤: {err_msg}")
        self.augmentation_started = False

    # ------------------ 預覽 ------------------
    def select_preview_image(self):
        if not self.selected_input_folder:
            QMessageBox.warning(self, "警告", "請先選擇輸入資料夾！")
            return
        images_dir = os.path.join(self.selected_input_folder, "images")
        if not os.path.exists(images_dir):
            QMessageBox.warning(self, "警告", "輸入資料夾中無 images！")
            return
        image_files = [
            f for f in os.listdir(images_dir)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ]
        if not image_files:
            QMessageBox.warning(self, "警告", "沒有圖像文件！")
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "選擇預覽圖像", images_dir, "Images (*.jpg *.png *.jpeg)"
        )
        if file_path:
            img = cv2.imread(file_path)
            if img is None:
                QMessageBox.critical(self, "錯誤", "無法讀取選擇的圖像！")
                return
            self.preview_image_path = file_path
            self.original_preview_image = img
            self.display_image(self.original_preview_label, img)
            self.update_preview()

    def display_image(self, label, img):
        if img is None:
            return
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = rgb_img.shape
        bytes_per_line = c * w
        convert_to_Qt_format = QtGui.QImage(
            rgb_img.data,
            w,
            h,
            bytes_per_line,
            QtGui.QImage.Format_RGB888
        )
        pixmap = QtGui.QPixmap.fromImage(convert_to_Qt_format)
        pixmap = pixmap.scaled(label.width(), label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pixmap)

    def update_preview(self):
        """
        這裡是 "疊加" 預覽，與實際增強方式(每種增強單獨套用)並不相同。
        目的是讓使用者快速在界面上看出「若全部都套用」時的視覺效果大約如何。
        """
        if self.original_preview_image is None:
            return
        img = self.original_preview_image.copy()

        # (1) rotate: 先拿 slider.value() 當作一個角度
        rot_step = self.rotate_slider.value()
        if rot_step > 0:
            # 直接用 rot_step 作為 angle 預覽
            rotated, M, newW, newH = rotate_image(img, rot_step)
            img = rotated

        # (2) brightness
        bri = self.brightness_slider.value() / 100.0
        if bri > 0:
            img = random_brightness(img, bri)

        # (3) blur
        blur_k = self.blur_slider.value()
        if blur_k > 0:
            img = random_blur(img, blur_k)

        # (4) shear
        shear_val = self.shear_slider.value() / 100.0
        if shear_val > 0:
            sheared, M_shear, new_w_, new_h_, shear_factor = random_shear(img, shear_val)
            img = sheared

        # (5) crop
        crop_val = self.crop_slider.value() / 100.0
        if crop_val > 0:
            cz_img, _, _, _ = random_center_crop_zoom(img, crop_val)
            img = cz_img

        self.processed_preview_image = img
        self.display_image(self.processed_preview_label, img)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AugmentationApp()
    window.show()
    sys.exit(app.exec_())
