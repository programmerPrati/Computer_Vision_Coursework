# Computer Vision Coursework

This repository contains a series of projects developed for my Computer Vision coursework. These assignments cover the end-to-end vision pipeline, from basic image manipulation to advanced stereo matching and 3D geometry.

## Projects Overview

### Proj 1: Images as Functions
**Concept:** Understanding images as discrete 2D functions $f(x, y)$ where the value at each coordinate represents intensity or color.
* **Implementation:** Explored basic image processing techniques including convolution, filtering, and image gradients.
* **Key Learning:** Mastering the mathematical representation of images to perform noise reduction and edge detection.

### Proj 2: Hough Transform
**Concept:** A feature extraction technique used to isolate shapes within an image by performing a voting procedure in a parameter space.
* **Implementation:** Developed a system to detect lines and shapes (circles) by transforming edge-detected pixels into the **Hough Space**.
* **Key Learning:** Understanding how to find global patterns (like roads or structural boundaries) from local edge data.



### Proj 3: Camera Parameter & Fundamental Matrix Estimation
**Concept:** Bridging the gap between 2D images and the 3D world through **Epipolar Geometry**.
* **Implementation:** Estimated the **Intrinsic and Extrinsic camera parameters** to understand how a camera maps 3D points to a 2D plane.
  * Computed the **Fundamental Matrix**, which relates corresponding points between two different images of the same scene.
* **Key Learning:** Using the 8-point algorithm and RANSAC to handle outliers in point correspondence.



### Proj 4: Stereo Matching
**Concept:** Estimating depth from a pair of rectified images by calculating the "disparity" between matching pixels.
* **Implementation:** Implemented block-matching algorithms to find correspondences along scanlines.
  * Generated **Disparity Maps** where pixel intensity represents the distance from the camera.
* **Key Learning:** Understanding the trade-offs between window size, computational speed, and depth accuracy.



---

## Technical Skills & Core Concepts

* **Mathematical Modeling:** Applying linear algebra, multi-view geometry, and calculus to solve complex 3D reconstruction and geometric vision problems.
* **Computer Vision Frameworks:**
    * **Geometric Vision:** Camera Calibration, Fundamental Matrix Estimation, and Epipolar Geometry.
    * **Spatial Analysis:** Stereo Matching, Disparity Mapping, and 3D perception logic.
    * **Feature Extraction:** Hough Transform for shape detection and image gradient analysis.
* **Optimization & Robust Estimation:** Utilizing **RANSAC**, Least-Squares estimation, and custom loss functions to handle outliers and ensure model integrity.
* **Programming & Tools:** Advanced **Python** implementation using NumPy for matrix manipulation and vectorized operations.


Setup files should be included with each project
