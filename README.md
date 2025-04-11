# NTUST_Robotics_Project
Lab project using robotic arm and computer vision and deep learning model

# 🤖 AI-Powered Robotic Sorting System

本專題專注於將人工智慧結合機械手臂與電腦視覺，實作一套可分類、夾取與堆疊物件的智能系統。

This project integrates AI, computer vision, and robotics to build a smart system capable of object detection, classification, and stacking using a robotic arm and a depth camera.

---

## 🎯 Features / 系統特色

- ✅ 物件偵測與分類（使用 YOLOv8 實例分割）
- ✅ 三維深度偵測（RealSense D435i）
- ✅ 手眼協調與空間轉換（two-stage PnP）
- ✅ 可動作指令控制（物件分類、指定夾取）
- ✅ 碰撞防護與 Z 軸距離限制 (depth control)

---

## 🧰 Technologies Used

| 類別 | 技術 |
|------|------|
| Programming | Python 3 |
| CV / AI     | YOLOv8n-seg, OpenCV, Roboflow |
| Hardware    | RealSense D435i, 自動化手臂（六軸） |
| Control     | Perspective-n-Point, Matrix Transforms, TCP transmission protocol |
| Platform    | Windows |
| IDE         | VS code |

---

## 🧪 Demo Highlights

- 🎯 實例分割與方向判別  
- 🎯 三維空間夾取 & 多物體分類  
- 🎯 顏色排序堆疊任務實作


## 🧠 Skills & Knowledge Gained

- 深度學習模型訓練與部署
- 相機與座標轉換（PnP, Homogenius matrix manipulation, image processing）
- 三維點雲處理與碰撞偵測 (implementation of depth camera)
- 系統整合：視覺 + 控制 + 動作
- 團隊協作與流程設計（含軟硬整合）

## 👥 Members
1. Lin Huang Ting
2. Shi Jun Kai
3. Huang Hsin Hua

NTUST mechanical engineering academic project
