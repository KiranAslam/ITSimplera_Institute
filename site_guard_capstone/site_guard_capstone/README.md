# Real-Time PPE Compliance Detection Using YOLOv8 and Multi-Object Tracking

**IT Simplera Institute — AI/ML Internship Program — Final Capstone (Milestone 2)**
**Submitted by:** Kiran &nbsp;|&nbsp; **Submission Date:** 21 August 2026

**Keywords:** YOLOv8 · ByteTrack Multi-Object Tracking · Class-Balanced Dataset Fusion · Flask Deployment

---

## 1. Objective

Non-compliance with Personal Protective Equipment (PPE) — helmets, safety vests, gloves, and goggles — remains a leading cause of preventable injuries in industrial and hazardous work environments. Manual safety inspection is slow and inconsistent, and cannot cover every worker across every zone at every moment. This project addresses that gap with an automated, camera-based PPE compliance monitoring system that detects workers in real time, identifies which required protective items each individual is or isn't wearing, and logs violations on a per-person basis — reducing reliance on manual patrols and enabling faster intervention before an incident occurs.

## 2. Selected Dataset

Rather than using a single off-the-shelf PPE dataset, a composite dataset was constructed by merging three public YOLO-format sources, since no single public dataset covered the required class list with balanced representation:

| Source | Role |
|---|---|
| Ultralytics "Construction-PPE" (`github.com/ultralytics/assets`) | Base dataset — ~1,416 images, 11 classes |
| Safety-Vests (Roboflow Universe, v14) | Supplement — additional no-vest violation examples |
| PPE-v2 (Roboflow Universe, ayvu, v2) | Supplement — extra helmet/gloves/goggles/vest/none/Person examples via class-ID remapping |

After merging, the boots / no-boots classes were dropped as out of scope, producing a final 10-class label space: **helmet, gloves, vest, goggles, none, Person, no_helmet, no_goggle, no_gloves, no_vest**. The final split consists of **2,828 training**, **263 validation**, and **249 test** images.

## 3. Literature Review Summary

| Paper | Approach Summarized |
|---|---|
| Ferdous & Ahsan (2022) — *PeerJ Computer Science* | Introduces the CHVG dataset; benchmarks anchor-free YOLOX variants for hardhats, vests, and safety glasses. |
| Otgonbold et al. (2022) — *Sensors* | Extends SHEL5K to 5,000+ images; shows dataset quality/class balance strongly affects helmet-detection accuracy. |
| Barlybayev et al. (2024) — *Cogent Engineering* | Compares YOLOv8n/s/m/l/x, finding larger variants more reliable for person/vest classes at the cost of speed. |
| Ahmad & Rahimi (2025) — *J. Safety Science and Resilience* | Proposes the 17-class SH17 dataset; YOLOv9-e exceeds 70.9% PPE-detection accuracy. |
| Li et al. (2025) — *Frontiers in Built Environment* | Adds a DilateFormer attention mechanism to YOLOv5s for small-object (helmet) detection in cluttered scenes. |

Across these papers, three gaps stand out: class imbalance is addressed almost entirely through loss weighting rather than at the data level; detectors are evaluated frame-by-frame rather than as a continuous tracking pipeline; and accuracy is reported per-detection rather than per-person, overstating real-world reliability. This project addresses all three directly.

## 4. Data Preprocessing & Class Rebalancing

Each source's class IDs were remapped into a unified label space prior to merging. Severe class imbalance was found in the raw merge — several violation classes (*goggles*, *no_helmet*, *no_goggle*) had fewer than 500 boxes. Per-class counts were corrected via an upper bound of 1,500 boxes/class and a protected minimum of 1,200 boxes/class, with under-represented classes reinforced across iterative targeted merge rounds. Image/label pairing was verified per split and orphaned files removed.

**Fig. 1** — Per-class box counts before and after rebalancing (`class_distribution.png` in the `paper/` folder). The most severely under-represented classes in the raw merge — *goggles* (427), *no_goggle* (337), *no_helmet* (400) — were all brought above the 1,200-box protected minimum.

## 5. Baseline Model & Experimental Design

The original plan called for three YOLOv8 variants (nano, small, medium). Due to the project's time constraints, only **YOLOv8s** was fully trained and evaluated for this submission — trained for 60 epochs at image size 640, AdamW optimizer, cosine learning-rate schedule, class-weighted loss (`cls=0.7`), fixed seed (42) for reproducibility, on a single NVIDIA T4 GPU (Google Colab).

> **Limitation (time-constrained):** YOLOv8n and YOLOv8m were planned as lighter/heavier baselines to quantify the accuracy-vs-speed trade-off, but were not trained within the available time. This is the primary open item for extending this work.

## 6. Evaluation & Results

### 6.1 Overall Test-Set Performance

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|---|---|---|---|---|
| YOLOv8s (60 epochs) | 0.738 | 0.402 | 0.835 | 0.668 |

### 6.2 Per-Class Results

| Class | Precision | Recall | mAP@0.5 |
|---|---|---|---|
| helmet | 0.907 | 0.437 | 0.584 |
| gloves | 0.884 | 0.617 | 0.636 |
| vest | 0.904 | 0.556 | 0.710 |
| goggles | 0.718 | 0.936 | 0.805 |
| none | 0.759 | 0.617 | 0.672 |
| Person | 0.858 | 0.640 | 0.742 |
| no_helmet | 0.888 | 0.867 | 0.898 |
| no_goggle | 0.754 | 0.597 | 0.703 |
| no_gloves | 0.867 | 0.700 | 0.836 |
| no_vest | 0.810 | 0.714 | 0.795 |

> The four dedicated violation classes averaged **0.808 mAP@0.5** — close to or higher than the presence classes they pair with — confirming the rebalancing step (Section 4) brought minority classes to a competitive accuracy level. One notable exception: *helmet* shows the highest precision (0.907) but lowest recall (0.437) of any class, a conservative failure mode worth further investigation.

## 7. Proposed Improvement: Persistent Tracking & Deduplicated Logging

Published PPE detectors are typically evaluated on isolated frames, which doesn't reflect how compliance is judged on-site — per person, continuously, over time. This project adds a **ByteTrack-based multi-object tracking layer** on top of the trained detector so each person/object keeps a stable track identity across frames. Violations are logged **once per unique (track ID, missing-item) pair**, preventing duplicate/flooded alerts from repeated per-frame detections.

> **Limitation (time-constrained):** The tracking-and-deduplication layer is implemented and running in the deployed dashboard, but its quantitative accuracy has not yet been independently verified against manually reviewed ground truth on a test video — this evaluation is an immediate next step.

## 8. Deployment

The trained YOLOv8s model is deployed as a lightweight Flask web dashboard (no authentication, single-user local demo). The application captures live webcam frames, runs YOLOv8 inference with persistent ByteTrack tracking, and overlays bounding boxes color-coded by compliance status (red = violation, green = compliant) with the assigned track ID. A side panel polls two JSON endpoints to display a live, deduplicated violation log table and summary statistics (unique people/objects tracked, total violations, per-class breakdown). All violations are also persisted to a CSV log file. Deployment code, a `requirements.txt`, and setup instructions are included in this repository and run locally without additional services.

## 9. Conclusion & Future Work

This project delivered a real-time PPE compliance detection system combining a composite, class-rebalanced dataset with a YOLOv8s detector and a ByteTrack-based tracking layer for per-person, deduplicated violation logging, deployed as a live Flask dashboard. The rebalanced baseline reached an mAP@0.5 of 0.738, with the four violation classes averaging a competitive 0.808 mAP@0.5 — evidence that correcting class imbalance at the data level is an effective strategy for this problem. Immediate next steps include training the planned YOLOv8n/m baselines, quantitatively verifying the tracking layer's violation-logging accuracy against ground truth, and extending the system to multiple simultaneous camera feeds with per-zone violation attribution.

---

## Repository Structure & Setup

See [`README.md`](./README.md) *(deployment/setup instructions)* and [`paper/paper.pdf`](./paper/paper.pdf) *(full IEEE-format research paper)* for run instructions, dataset download links, and reproducibility details.