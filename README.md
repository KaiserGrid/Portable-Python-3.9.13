# Portable Python 3.9.13 (Windows)

An all-in-one, portable Python 3.9.13 environment for Windows. It comes pre-bundled with popular libraries for machine learning, computer vision, and general tooling—no installation required.

![Python 3.9.13](https://img.shields.io/badge/Python-3.9.13-blue)
![Platform: Windows](https://img.shields.io/badge/Platform-Windows-blue)
![Libraries: ML/CV](https://img.shields.io/badge/Libraries-ML%2FCV-purple)
![Zero-Install](https://img.shields.io/badge/Setup-Zero--Install-green)

---

## ⚡ Quick Start

1.  **Download and Unzip** the portable bundle.
2.  **Run** `python.exe` for a console environment or `pythonw.exe` for a windowed application without a console.
3.  **Verify** the installation by running the following commands in the terminal:
    ```bash
    python --version
    # Expected output: 3.9.13
    python -c "import cv2, dlib, numpy; print('All good!')"
    ```
You are now ready to run scripts using the pre-installed libraries.

---

## 📦 What's Inside

This package is designed to be a complete, self-contained environment.

### **Core Components**
* **Python Runtime:** `python.exe`, `pythonw.exe`, `python39.dll`, `python3.dll`, etc.
* **Standard Extensions:** A comprehensive set of `.pyd` files (`_ssl.pyd`, `_sqlite3.pyd`, etc.) for full functionality.

### **Pre-installed Libraries**
The `Lib/site-packages/` directory includes a curated set of popular libraries:
* `dlib`
* `OpenCV` (`cv2`)
* `numpy`
* `CMake` (tooling for builds)
* `FaceNet`
* `tkinter-embed`

### **Face Utilities**
This bundle includes a set of scripts for face detection and recognition:
* `face_logger.py`: A utility for logging face detections.
* `face_logger_cli.py`: A command-line wrapper for easier interaction with `face_logger.py`.
* `logs.csv`: A sample output file for detected faces.

---

## ⚙️ Usage

### **Running Python Scripts**
* **From a terminal:** Navigate to the folder and run `.\python.exe your_script.py`
* **For a GUI app:** Use `.\pythonw.exe your_gui_script.py` to run without a console window.

### **Installing Additional Packages (Optional)**
If the `Scripts` folder is included, you can use `pip`:
```bash
.\python.exe -m pip install <package_name>
```
For offline installations, you can download `.whl` files and install them with `pip`, or manually place the package directory into `Lib/site-packages/`.

**Note:** This is a portable environment. It is not recommended to upgrade the core Python version.

### **Using the Face Logging Utilities**
The `face_logger.py` script is a powerful tool for face detection and logging. Here’s a typical usage example:
```bash
.\python.exe face_logger.py --input 0 --output logs.csv
```
This command will use your default webcam (`--input 0`) to detect faces and append the results to `logs.csv`. For more options, use the `--help` flag or inspect the source code.

---

## 📁 Directory Layout
```
/
├── python.exe
├── pythonw.exe
├── python39.dll
├── Lib/
│   └── site-packages/  (dlib, opencv, numpy, etc.)
├── face_logger.py
├── face_logger_cli.py
├── logs.csv
├── vcruntime140.dll
├── libssl-1_1.dll
├── LICENSE.txt
└── README.md (this file)
```

---

## 🖥️ System Requirements
* **OS:** Windows 10/11 (64-bit)
* **CPU:** Must support SSE2.
* **Webcam:** Required for camera-based features.
* **Disk Space:** Sufficient space for the bundle and any generated logs.

---

## 🔧 Tips and Troubleshooting

* **`DLL load failed` or `module not found` errors:** Ensure all DLLs and the Python executables remain in the same directory. Do not move files individually.
* **`OpenCV unable to open camera`:**
    * Try different camera indices (e.g., `--input 0` or `--input 1`).
    * Make sure no other application is using your webcam.
    * Consider updating your camera drivers.
* **`pip` installation failures:**
    * Ensure you are using wheels that match `cp39` (Python 3.9) and `win_amd64` (64-bit Windows).
    * For a completely offline approach, copy the package directory directly into `Lib/site-packages`.

---

## ⚖️ Security and Licensing
* This project is subject to the license in `LICENSE.txt`.
* Third-party libraries like dlib, OpenCV, and numpy have their own licenses. Be sure to comply with them for any redistribution or commercial use.

---

## 📝 Changelog & Credits
* **Credits:** Maintained by KaiserGrid.
* **Changelog:** Refer to the repository commits for the latest updates.

This bundle is designed for simplicity and portability. Keep the folder structure intact for the best experience.
