# Edge AI Vegetable Classifier — Based on XIAO ESP32S3 Sense

A real-time vegetable classification system running fully on-device using a **Seeed XIAO ESP32S3 Sense** microcontroller with an OV2640 camera and a 128×64 OLED display. Two quantised TFLite CNNs (Low‑96 and High‑160) are deployed via **Edge Impulse**, with an adaptive dual-model mode that escalates from the fast low-res model to the more accurate high-res model only when confidence is low.

**Classifies:** Avocado · Green Pepper · Tomato · Unknown

---

## Repository Structure

```
grad-project/
│
├── ESP32/                          # Arduino firmware for the XIAO ESP32S3
│   ├── HW_Testing/                 # Hardware bring-up tests
│   ├── Data Collection/            # Image capture & preprocessing
│   ├── Classification_Model_Low_96/
│   ├── Classification_Model_High_160/
│   └── Adaptive Classification_BUGS/ #This is Adaptive Classification Combined Model it contains both High and Low Model but is hase bugs
│
├── NN/                             # Python training pipeline
│   ├── dataset/                    # Resized training images + labels
│   ├── scripts/                    # Train and test scripts
│   ├── exports/                    # INT8 TFLite model files
│   └── firmware/                   # Edge Impulse Arduino library zips
└── Report/                         # Figures, charts, and reference images
```

---

## Code Blocks — What Is Where

### 1. `ESP32/HW_Testing/` — Hardware Bring-Up

These are the very first sketches used to verify that each hardware component works in isolation before any ML inference.

| File | Language | What it does |
|------|----------|--------------|
| `OLED_Test/OLED_Test.ino` | Arduino (C++) | Initialises the SSD1306 128×64 OLED over I2C and prints a test message. Used to confirm the display is wired and addressed correctly (`0x3C`). |
| `CameraWebServer/CameraWebServer.ino` | Arduino (C++) | Streams live MJPEG video from the OV2640 camera over Wi-Fi to a browser. Used to verify the camera, PSRAM, and network stack all work on the XIAO ESP32S3. Also includes `board_config.h` (pin defines) and `camera_pins.h` (multi-board pin table). |
| `TF_Test/test_tf.py` | Python | Sanity-checks that TensorFlow is installed correctly on the host PC by running a trivial add operation (`tf.add(5, 3)`). |

---

### 2. `ESP32/Data Collection/` — Training Data Capture & Preprocessing

This stage captures raw images from the device and prepares them for training.

| File | Language | What it does |
|------|----------|--------------|
| `26_Collect_Images_XIAO_ESP32S3/26_Collect_Images_XIAO_ESP32S3.ino` | Arduino (C++) | Runs a lightweight HTTP image-collection server (from the EloquentEsp32cam library). Browse to the board's IP address in a browser, then tap to capture and label images live. Saves frames at XGA resolution (1024×768). |
| `Image-size-reducer-script/image_crop.py` | Python | Post-processes the raw captured images: centre-crops each image to a square, then resizes to **96×96** (or **160×160** if `TARGET_SIZE` is changed) using bilinear interpolation. Outputs to a new `Training_<SIZE>/` folder, preserving the class subfolder layout. |
| `Image-size-reducer-script/Training/` | Images | The **raw (unresized)** captured JPEG images, organised into four class subfolders: `Avocado/`, `Pepper_Green/`, `Tomato/`, `Unknown/`. |

---

### 3. `NN/` — Neural Network Training Pipeline

The model is trained on the **host PC** and exported as a quantised TFLite file.

| File/Folder | Language | What it does |
|-------------|----------|--------------|
| `dataset/labels.txt` | Text | Lists the four class names in training order: `Avocado`, `Pepper_Green`, `Tomato`, `Unknown`. Must match the label order used in inference code. |
| `dataset/Training/` | Images | **Resized** training images (produced by `image_crop.py`), organised into the same four class subfolders. These are fed directly into Keras. |
| `scripts/train_two_models.py` | Python (TensorFlow/Keras) | Trains **two separate tiny CNNs** in one run — one at 96×96 (`low`) and one at 160×160 (`high`). Each model is a custom MobileNet-style stack: `Conv2D(8) → Conv2D(16) → DepthwiseConv2D → GlobalAvgPool → Dense(32) → Softmax`. Applies data augmentation (random brightness/contrast), exports each model to INT8 TFLite via full-integer post-training quantisation, and saves to `exports/`. |
| `scripts/webcam_test_tflite.py` | Python (OpenCV + TFLite) | Loads a `.tflite` model and runs it live on the **host webcam** to validate the model before flashing to hardware. Applies the same INT8 dequantisation logic as the embedded firmware, and overlays the predicted label and confidence on the video feed. Includes EMA smoothing and an "Unknown" rejection gate. |
| `exports/model_low_96_int8.tflite` | TFLite binary | The **Low model** — trained at 96×96, INT8 quantised (~8.9 KB). Optimised for speed. |
| `exports/model_high_160_int8.tflite` | TFLite binary | The **High model** — trained at 160×160, INT8 quantised (~8.9 KB). Optimised for accuracy. |
| `firmware/ei-tilak_model_low_98-arduino-1.0.4.zip` | ZIP | Edge Impulse Arduino library wrapping the Low model, ready to install in the Arduino IDE via *Sketch → Include Library → Add .ZIP Library*. |
| `firmware/ei-tilak_model_high_160-arduino-1.0.1.zip` | ZIP | Edge Impulse Arduino library wrapping the High model — same install process. |

---

### 4. `ESP32/Classification_Model_Low_96/` — Production Inference (Low Model)

Two production sketches that run the **96×96 Low model** for continuous classification.

| File | Language | What it does |
|------|----------|--------------|
| `Classification_OLED/Classification_OLED.ino` | Arduino (C++) | Captures a frame from the OV2640 camera, runs the Edge Impulse Low‑96 classifier, and displays the top-1 label, confidence (%), and inference latency (ms) on the OLED screen. Shows `L` for Low model mode. |
| `Classification_SerialMonitor/Classification_SerialMonitor.ino` | Arduino (C++) | Same inference pipeline as the OLED variant, but outputs results to the **Arduino Serial Monitor** instead of the display. Useful for debugging confidence scores and timing. |

---

### 5. `ESP32/Classification_Model_High_160/` — Production Inference (High Model)

Two production sketches that run the **160×160 High model** for higher-accuracy classification.

| File | Language | What it does |
|------|----------|--------------|
| `Classification_OLED_160/Classification_OLED_160.ino` | Arduino (C++) | Same as the Low OLED sketch but uses the `Tilak_model_High_160_inferencing` library. OLED shows `H` for High model mode, with label, F1 confidence %, and latency. Camera frame is captured at 320×240 and centre-cropped to 160×160 before inference. |
| `Classification_SerialMonitor_160/Classification_SerialMonitor_160.ino` | Arduino (C++) | High‑160 inference with Serial Monitor output only. Based on the standard Edge Impulse camera example template. |

---

### 6. `ESP32/Adaptive Classification_BUGS/` — Adaptive Dual-Model Classifier *(Work in Progress)*

An experimental sketch that dynamically switches between the two models based on confidence.

| File | Language | What it does |
|------|----------|--------------|
| `AdaptiveClassification/AdaptiveClassification.ino` | Arduino (C++) | Runs the **Low‑96** model every frame. If the top-1 confidence drops below `LOW_CONF_THRESH` (0.70), the same captured frame is re-run through the **High‑160** model ("escalation"). The device stays in High mode until it achieves `HIGH_STABLE_THRESH` consecutive confident results or `HIGH_TIMEOUT_MS` elapses, then reverts to Low. OLED shows current model (`L`/`H`), label, confidence, and latency. Requires the **merged** `Tilak_models_Adaptive_inferencing` library (the zip in `Adaptive Classification_BUGS/`). ⚠️ Known bugs — not production-ready. |
| `Readme.txt` | Text | Empty placeholder. |
| `Tilak_models_Adaptive_inferencing-arduino-1.0.0.zip` | ZIP | The merged Edge Impulse library containing **both** models for the adaptive sketch. |

---

### 7. `Report/` — Report Assets

Contains all figures and media used in the project report. Not executable code.

| Folder | Contents |
|--------|----------|
| `Important Charts/` | System pipeline diagram and CNN architecture chart |
| `Model Output/High_160/` & `Model Output/Low_96/` | Edge Impulse screenshots: feature explorer, confusion matrix, performance metrics, and raw class activation maps |
| `Reference Images/Hardware/` | Photos of the PCB (front/back), OLED test photo, portability test, and a demo output video |
| `Reference Images/` | Data collection reference image and OpenCV webcam test screenshot |
| `Tilak Raval 21208713_Proposal.pdf` | Original project proposal |

---

## Recommended Flash Order

1. **Verify hardware** → flash `HW_Testing/OLED_Test` and `HW_Testing/CameraWebServer`
2. **Collect images** → flash `Data Collection/26_Collect_Images_XIAO_ESP32S3`
3. **Preprocess** → run `image_crop.py` on the host
4. **Train** → run `NN/scripts/train_two_models.py`
5. **Validate on PC** → run `NN/scripts/webcam_test_tflite.py`
6. **Test on device** → flash `feasibility_test_v2`
7. **Deploy** → install the EI library zip and flash the appropriate `Classification_*` sketch

---

## Dependencies

**Arduino libraries:** `EloquentEsp32cam`, `ArduTFLite`, `Adafruit SSD1306`, `Adafruit GFX`, `Edge Impulse inferencing library` (installed from the `.zip` files in `NN/firmware/`)

**Python packages:** `tensorflow`, `opencv-python`, `numpy`, `Pillow`

**Board:** Seeed XIAO ESP32S3 Sense — set *PSRAM: OPI PSRAM* and *Partition Scheme: Huge APP (3MB No OTA)* in Arduino IDE Tools menu.

**Acknowledgment:** This README file was written with the assistance of Microsoft Copilot.
