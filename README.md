# CUDA HSV Chroma Keying 

This project implements real-time green screen removal using CUDA and HSV color masking. It allows you to remove a green background from live webcam input using CUDA acceleration for real-time performance.

## Features

- CUDA-based chroma keying (green screen removal)
- HSV-based color masking for precise background segmentation
- Real-time performance with GPU acceleration
- Works with most webcams using OpenCV

## Files

- `main.cu`: Main CUDA code for chroma keying
- `CMakeLists.txt`: Build configuration (optional)
- `.gitignore`: Standard ignore list for CUDA/OpenCV binaries

## How to Run

1. **Make sure you have:**
   - CUDA Toolkit 12.5+ installed
   - OpenCV 4.8.0 prebuilt (VC16) SDK set up at `C:/opencv/`
   - A webcam connected

2. **Compile with:**
   ```bash
   "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin\nvcc.exe" main.cu -o chromaimg.exe ^
   -I"C:\opencv\build\include" ^
   -L"C:\opencv\build\x64\vc16\lib" -lopencv_world4120
   ```
   > Adjust the paths above to match your OpenCV installation.


3. **Run the executable:**
   ```bash
   ./chromaimg.exe
   ```

## Improvements Over Previous Stage

- Switched to HSV color space for robust green detection
- Improved background segmentation accuracy
- Eliminated artifacts on similar-colored clothing and lighting
- Achieved smoother edge handling in real-time feed

## Notes

- You may tweak HSV threshold ranges in the code to adapt to different lighting/green screen setups.
- Compatible with most USB webcams recognized by OpenCV.

---
Created by Blastermania
