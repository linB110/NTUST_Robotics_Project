# 🤖 AI-Powered Robotic Sorting System

An integrated full-stack robotics system that combines AI-based vision robotic manipulation for autonomous sorting, stacking, and pick-and-place operations. This project showcases an end-to-end pipeline from perception to action, designed for industrial-like object sorting tasks.

[📺 **Demo Video 1**](https://www.youtube.com/watch?v=j3vBnhzgqhY)
[📺 **Demo Video 2**](https://www.youtube.com/watch?v=uE8dl_GGoJs)

---

## 🚀 Project Highlights

* 🎨 **YOLOv8n-seg-based Instance Segmentation** for object classification and region-aware detection
* 🌍 **3D Localization & PnP Pose Estimation** using RealSense D435i depth sensing
* ⚖️ **Category-based Grasp Planning & Autonomous Stacking** with height tracking and collision avoidance
* 🔧 **Full Pipeline Integration** from visual input to real-world robot control (TCP socket communication)

---


### 🔹 Key Responsibilities:

* Built **instance segmentation system** using **YOLOv8-seg**, including dataset annotation and training pipeline
* Designed **PnP-based pose estimation pipeline** for converting 2D bounding boxes to 3D spatial transforms
* Integrated **TCP communication** between PC and robotic manipulator
* Led **system-level integration** across depth sensing, grasp planning, and robotic execution
* Created visual debugging tools and control flow logging

---

## 📖 Technical Summary

This system bridges AI-based object recognition with robotic motion execution through a modular architecture. It includes camera calibration, real-time depth processing, object detection, pose estimation, and robotic grasping control, all synchronized via a central loop.

### 🌐 System Architecture

```plaintext
Depth Camera (D435i) ➔ Object Detection (YOLOv8n-seg) ➔ 3D Pose Estimation (PnP)
    |                        |                                  |
Point Cloud              Class Label                       Target Pose (TCP)
    |                        |                                  |
Collision Check        Action Decision                Robotic Arm Control (TCP socket)
    |______________________|__________________________________|
                            Control Loop
```

---

## 🛠️ Technologies Used

| Component         | Tools / Frameworks                           |
| ----------------- | -------------------------------------------- |
| Programming       | Python 3                                     |
| AI / CV           | YOLOv8n-seg, OpenCV, Roboflow                |
| Depth & Pose      | Intel RealSense D435i, PnP, matrix transform |
| Robotic Control   | 6-DOF Arm, TCP socket                        |
| Platform          | Windows 10                                   |
| Data Augmentation | Custom pipeline (OBB format)                 |

---

## 🔮 Core Demonstrations

* 🔍 **Real-time Instance Segmentation** for classification and grasping
* 📍 **PnP Estimation** of object pose relative to camera frame
* 🛋 **3D Grasp Execution** based on height, category, and position
* 🌀 **Category-based Sorting & Stack Placement** with dynamic height planning
* 💡 **Visual Feedback Loop** with Z-axis safety validation

---

## 🧪 Engineering Challenges & Solutions

* ❗ **YOLOv8-seg integration and OBB augmentation**

  * ✅ Built modular `augmentation/` tool with UI, API, and command-line entry

* ❗ **PnP accuracy under noise from depth sensor**

  * ✅ Refined camera intrinsic calibration and used chessboard images (`calibration/`, `image/` folders)

* ❗ **Real-time TCP latency during robot command loop**

  * ✅ Designed non-blocking communication loop with simple ACK strategy for robot feedback

---

## 🌐 Folder Structure Overview

```
NTUST_Robotics_Project/
├── augmentation/      # Data augmentation UI, API for YOLOv8 OBB format
├── calibration/       # Intrinsic calibration scripts and data
├── image/             # Sample chessboard images for calibration
├── model/             # Trained YOLOv8 models
├── project_report/    # Final PDF report (9 pages)
├── source/
│   ├── robot_control            # TCP socket control and point transmission
│   ├── pnp_pose_estimation      # 3D transform logic via PnP metohd
│   ├── yolo img/video_test      # System integration test for yolo model
│   ├── training/                # Custom model training pipeline

```

---

## 🙋‍ Team & Roles

| Name           | Responsibility                       |
| -------------- | ------------------------------------ |
| Huang-Ting Lin | Vision pipeline, Control Integration |
| Shi Jun-Kai    | Hardware setup, Arm control logic    |
| Huang Hsin-Hua | Model training, Dataset preparation  |

> **Advisor**: Prof Chi-Yu, Lin
> **Lab** : Advanced Intelligent Robot & Automation Lab, NTUST

---

## 📚 Learning Outcomes

* Learned to deploy and optimize **YOLOv8 segmentation** on limited dataset
* Built end-to-end **robotic control loop** using 3D spatial data and PnP
* Gained hands-on experience with **sensor fusion** and **pose estimation**
* Understood practical aspects of **robotics system safety** and real-world calibration
* A basic knowledge built in computer vision and robotics

---

## 🚀 Future Improvements

* Integrate **reinforcement learning** for adaptive grasp planning
* Add **closed-loop visual feedback** for placement correction
* Port to **embedded deployment** (e.g., Jetson + ROS2)

---

## 🔑 Why This Project Matters (For Research/Applications)

This project reflects:

* ✅ Strong foundation in **vision-based robotic control**
* ✅ Experience in **multi-modal system integration**
* ✅ Application of AI in **industrial robotics contexts**

Demonstrates readiness for graduate research, robotics R\&D, or academic study in autonomous systems, computer vision, and human-robot collaboration.

---

Pull requests and collaborations are welcomed!
