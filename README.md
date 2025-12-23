Real-Time Face Recognition System Using Webcam


1. Project Overview

This project implements a real-time face recognition system using a webcam to detect and identify multiple individuals simultaneously. The system is capable of recognizing at least 5 known individuals and displaying their names with bounding boxes in real time.

Purpose

This project was developed as part of Computer Vision Intern Assignment 2 to demonstrate understanding of:

I.Computer vision fundamentals
II.Face detection and recognition
III.Real-time video processing
IV.Clean, modular, and well-documented code practices

Key Features

I.Real-time face detection from webcam
II.Recognition of 5+ individuals in a single frame
III.Bounding boxes with name labels
IV.Handles unknown faces
V.CPU-optimized (no GPU required)
VI.Modular and well-documented code

2. Approach and Methodology

The system follows a two-stage pipeline:

Stage 1: Face Encoding (Offline)

I.Load images from dataset folders
II.Validate images (exactly one face per image)
III.Detect faces using HOG (Histogram of Oriented Gradients)
IV.Extract 128-D facial embeddings using a pre-trained deep learning model
V.Store encodings in encodings.pickle

Stage 2: Real-Time Recognition

I.Capture webcam frames using OpenCV
II.Resize frames (45%) for performance
III.Detect multiple faces per frame using HOG
IV.Compare embeddings using Euclidean distance
V.Apply threshold (TOLERANCE = 0.48)
VI.Label recognized faces, otherwise mark as Unknown
VII.Draw bounding boxes and names in real time

Technical Choices

I.Face Detection: HOG (CPU-efficient)
II.Recognition: Distance-based embedding matching
III.Multi-face support: Yes (5+ faces per frame)

3. Setup Instructions
3.1 Prerequisites

I.OS: Windows / macOS / Linux
II.Python: 3.9 (recommended)
III.Hardware: Webcam, minimum 4GB RAM
IV.IDE: VS Code or Anaconda

3.2 Environment Setup
I. Option A: Using Anaconda (Recommended)

conda create -n face_recog python=3.9
conda activate face_recog

II. Option B: Using VS Code (With venv)

i.Open VS Code
ii.Open your project folder (face-recognition-webcam)
iii.Open Terminal → New Terminal

python -m venv face_recog


Activate environment:

i.Windows

face_recog\Scripts\activate


ii.macOS / Linux

source face_recog/bin/activate

3.3 Install Dependencies
Recommended Method
pip install -r requirements.txt

requirements.txt
face-recognition==1.3.0
opencv-python==4.8.0.74
numpy==1.24.3

3.4 Verify Installation
python -c "import face_recognition, cv2, numpy; print('✓ All libraries installed successfully')"

4. How to Run the Application
4.1 Project Structure
face-recognition-webcam/
├── dataset/
│   ├── Person1/
│   ├── Person2/
│   ├── Person3/
│   ├── Person4/
│   └── Person5/
├── demo/
│   ├── screenshots
│   │       ├──screenshot1
│   │       ├──screenshot2
│   │       ├──screenshot3
│   │       ├──screenshot4
│   │       ├──screenshot5
│   └── demo_video.mp4
├── encode_faces.py
├── recognize_faces.py
├── requirements.txt
├── .gitignore
└── README.md

4.2 Step-by-Step Execution
Step 1: Prepare Dataset

I.Minimum 5 people
II.3–10 images per person
III.One face per image
IV.Clear, well-lit images

Step 2: Encode Faces
python encode_faces.py


Expected output:

[INFO] Encoding complete. Total people encoded: 5

Step 3: Run Face Recognition
python recognize_faces.py


Controls:

Press Q to quit

4.3 Configuration (Optional)
TOLERANCE = 0.48
RESIZE_SCALE = 0.45
DETECTION_MODEL = "hog"

5. Dataset Description

I.Organized by person name
II.One folder per individual
III.Dataset excluded from GitHub using .gitignore
IV.Created manually by the user

6. Assumptions and Limitations
Assumptions

I.Webcam available at index 0
II.Adequate lighting
III.Clear frontal faces
IV.Limitations
V.Performance drops in low light
VI.Cannot recognize unseen individuals
VII.CPU-based (15–25 FPS)

7. Demo Evidence

I.Demo video and screenshots are provided in the demo/ folder:
II.Single-face recognition
III.Unknown face handling





