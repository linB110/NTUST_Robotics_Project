# 🤖 AI-Powered Robotic Sorting System

A full-stack robotics system integrating AI-powered computer vision and robotic manipulation for object detection, classification, and autonomous sorting tasks.  
This project demonstrates a robust pipeline from perception to control, enabling precise object pick-and-place and intelligent stacking based on visual attributes.

[📺 **Demo Video1 (YouTube)**](https://www.youtube.com/watch?v=j3vBnhzgqhY)
[📺 **Demo Video2 (YouTube)**](https://www.youtube.com/watch?v=uE8dl_GGoJs)

---

## 🎯 Key Features
- **Instance Segmentation & Classification** using YOLOv8-seg for robust object detection and class-wise sorting.
- **3D Spatial Localization** via RealSense D435i depth sensing and two-stage PnP for accurate hand-eye coordination.
- **Dynamic Manipulation Planning** for category-based grasping, placement, and stacking with spatial constraints.
- **Collision Avoidance & Z-axis Control** utilizing depth estimation and safety thresholds.

---

## 🛠️ System Architecture Overview
```plaintext
Depth Camera (D435i)  -->  Object Detection (YOLOv8n-seg)  -->  3D Pose Estimation (PnP)
   |                        |                                     |
Point Cloud               Class Label                          Target Pose (TCP)
   |                        |                                     |
Collision Check        Action Decision                     Robotic Arm Control (TCP socket)
   |_____________________|____________________________________|
                             Control Loop
```

---

## 🧰 Technologies
| Category   | Tools / Frameworks                    |
|------------|----------------------------------------|
| Programming | Python 3                               |
| Computer Vision / AI | YOLOv8n-seg, OpenCV, Roboflow   |
| Hardware    | Intel RealSense D435i, 6-DOF Robotic Arm |
| Control Algorithms | Perspective-n-Point (PnP), Matrix Transforms |
| Platform    | Windows OS                              |
| IDE         | Visual Studio Code                      |
| data transmission         | TCP protocol               |


---

## 🧪 Core Demonstrations
- **Instance Segmentation** for object recognition & pose estimation  
- **3D Grasping & Manipulation** with precise spatial transforms  
- **Category-based Sorting & Stacking** for complex task pipelines
- **data augmentation for more generalize model** 
---

---

## 🧠 Skills & Technical Contributions
- Deep learning deployment on edge hardware (YOLOv8 segmentation)
- Hand-eye calibration, PnP-based pose estimation
- Point cloud processing for collision checking and grasp planning
- System integration across computer vision, motion control, and robotics
- Multi-disciplinary teamwork in robotics software & hardware integration

---

## 👥 Team Members
| Name           | Role                               |
|----------------|------------------------------------|
| Lin Huang-Ting | Vision & Control Integration |
| Shi Jun-Kai    | Robotics Arm Control, Hardware Setup       |
| Huang Hsin-Hua | AI Model Training, Dataset Preparation     |

> **Supervised by**: Advanced Intelligent Robot and Automation Lab, Department of Mechanical Engineering, NTUST

---

## 📌 Project Summary
This project bridges AI vision with robotic manipulation through effective system design and control strategies. It showcases practical applications of deep learning, sensor fusion, and robotics for real-world object handling tasks in manufacturing environments.

---

---

## 🚀 Future Work
- Enhance grasp planning with reinforcement learning
- Integrate a more real-time feedback loop for dynamic environments
- Deploy on embedded platforms (ROS2 + Jetson)

---

# 🔑 Why This Matters (For Applications)
This repository reflects strong practical skills in:
- **Vision-based robotic control**
- **AI perception integration**
- **System design & safety protocols in robotics**
It demonstrates readiness for advanced robotics research, industrial automation R&D, or graduate-level academic work in robotics and AI.
