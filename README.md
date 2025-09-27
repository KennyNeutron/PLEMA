# PLEMA: Prompt Lung Examination via Machine Automation for Tuberculosis

## Overview

PLEMA (**Pathogen Locator Evaluation for Mycobacterium Analysis**) is a research-driven system designed to assist in the **early detection of Tuberculosis (TB)**.  
By integrating **robotics, digital microscopy, and AI-powered analysis (YOLO via Roboflow)**, PLEMA enhances diagnostic accuracy, reduces manual errors, and accelerates TB detection in both clinical and low-resource settings.

This repository documents the development of PLEMA, from research foundations to implementation and testing.

---

## Abstract

In Kidapawan City, one of the most pressing health concerns is **tuberculosis (TB)**, a life-threatening disease caused by _Mycobacterium tuberculosis (Mtb)_.  
Traditional diagnostic procedures remain highly manual, which introduces delays, increases human error, and reduces efficiency.  
**PLEMA (Pathogen Locator Evaluation for Mycobacterium Analysis)** addresses these challenges through an **automated robotic platform** for sputum smear microscopy, integrated with **AI-powered YOLO detection models trained via Roboflow**.  
By combining affordable hardware (ESP32, digital microscope, optics) with AI-based analysis, PLEMA is designed to provide a **scalable, accurate, and accessible TB diagnostic tool**.

---

## Objectives

- Automate sputum smear microscopy for TB diagnosis.
- Compare PLEMA’s diagnostic accuracy, sensitivity, and specificity with traditional microscopy.
- Evaluate cost-effectiveness of AI-powered microscopy vs. existing methods.
- Provide a **scalable solution** for under-resourced healthcare settings.

---

## Methodology

- **Hardware:** ESP32 microcontroller, SVBONY SV109 digital microscope, 100× oil immersion objective lens, condenser, LED light source, 3D-printed chassis, automated XY stage.
- **Software:** AI-powered detection pipeline using **YOLO models trained via Roboflow**.
- **Treatments:**
  - Treatment 1 – 5 mL spot sputum
  - Treatment 2 – 5 mL early-morning sputum
  - Treatment 3 – 5 mL second spot sputum
  - **Control:** Standard diagnostic method

Results will be validated by comparing automated system outputs with human laboratory technologist analysis.

---

## Features

- 🫁 **AI-Based TB Detection** using YOLO + Roboflow datasets.
- ⚙️ **Automated Microscopy System** integrated with ESP32 and digital camera.
- 📊 **Diagnostic Accuracy Metrics**: Sensitivity, Specificity, Accuracy, and Processing Time.
- 🌍 **Designed for Low-Resource Settings** with scalable and cost-effective hardware.

---

## Tech Stack

- **Programming Language:** Python
- **ML Framework:** YOLO (Roboflow)
- **Microcontroller:** ESP32
- **Tools:** OpenCV, NumPy, Pandas, Matplotlib
- **Deployment (Planned):** Flask / FastAPI

---

## Roadmap

- [ ] Data collection & annotation via Roboflow
- [ ] Model training & baseline testing
- [ ] Integration with ESP32 + digital microscope
- [ ] Pilot testing in laboratory conditions
- [ ] Deployment in clinics and rural healthcare facilities

---

## References

This project is based on research titled:  
**“PLEMA: Pathogen Locator Evaluation for Mycobacterium Analysis”** (Kidapawan City National High School, 2025).

For detailed literature, see the [Research Paper](./docs/plema-final-paper.docx).

---

## Authors

- Oasay, Angela Marie B.
- Labrado, Jellian Pauline B.

Research Adviser: **Josephine G. Verdeblanco**  
Coordinator: **Lita L. Gapuz**  
OIC: **Glady E. Pagunsan**

---

## License

This project is released under the **MIT License** unless otherwise specified.
