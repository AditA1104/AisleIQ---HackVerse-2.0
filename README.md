# AisleIQ---HackVerse 2.0
The Invisible Shop Assistant

> **Bringing E-Commerce Level Analytics to Physical Brick-and-Mortar Retail Using Existing CCTV Infrastructure.**

[![Track: AI for Business Transformation](https://img.shields.io/badge/Hackathon_Track-AI_for_Business_Transformation-blue.svg)](#)
[![Framework: LangChain](https://img.shields.io/badge/GenAI-LangChain-121011.svg)](https://www.langchain.com/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://aisleiq-dashboard.streamlit.app)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

**Live Demo:** [https://aisleiq.streamlit.app](https://aisleiq-dashboard.streamlit.app)

AisleIQ transforms standard, legacy CCTV video streams into actionable, e-commerce-style retail analytics for physical store owners without requiring any hardware upgrades.
By combining real-time computer vision tracking with Generative AI, AisleIQ identifies customer dwell behavior, generates automated staff alerts, and delivers actionable spatial heatmaps to optimize store revenue.

---

## 📌 Problem Statement

E-commerce giants like Amazon track every click, hover time, and abandoned cart in real time. Physical store owners, on the other hand, operate **completely blind**. They record thousands of hours of CCTV footage daily, but only use it retroactively after theft occurs. 

**AisleIQ** turns passive security cameras into active, real-time revenue drivers with **Zero Hardware Cost**.

---

## 🚀 Key Features

### 1. 🧠 Dual-Vector Intent Engine (Custom Friction Logic)
Unlike basic dwell-time counters that trigger false alarms when customers read text messages, AisleIQ evaluates two simultaneous vectors:
* **Pacing Ratio ($R = D / N$):** Total distance traveled ($D$) divided by net displacement ($N$).
* **Friction Score ($T \times R$):** Dwell time ($T$) multiplied by pacing ratio.

This mathematically distinguishes between a **Relaxed Browser** (low motion, passive) and **Active Hesitation / Choice Paralysis** (high pacing/dwelling, high friction).

### 2. 🚨 GenAI Floor Staff Alerts (Slack Integration)
When high friction or confusion ($>45\text{s}$ lingering) is detected, AisleIQ routes context to a LLM via LangChain to generate concise, 1-sentence urgent notifications dispatched directly to floor staff via Slack webhooks.

### 3. 🗺️ Dynamic Traffic Heatmaps
Generates spatial heatmaps showing high-density engagement zones ("red zones"), enabling store managers to optimize product placement and charge premium rates for high-visibility shelf space.

### 4. 📊 Store Manager Command Dashboard
A Streamlit dashboard providing real-time video analytics, active customer tracking, live alert logs, and daily trend summaries.

---

## 🏗️ System Architecture

             +-----------------------------------+
             |      Existing CCTV Camera /       |
             |          Video Feed               |
             +-----------------------------------+
                               |
                               v
             +-----------------------------------+
             |    Computer Vision & Tracking     |
             |    (YOLO / Centroid Tracking)     |
             +-----------------------------------+
                               |
                               v
             +-----------------------------------+
             |    Dual-Vector Intent Engine      |
             |  (Pacing Ratio & Friction Score)  |
             +-----------------------------------+
                               |
                     [ High Friction / Threshold ]
                               |
                               v
             +-----------------------------------+
             |      LangChain GenAI Engine       |
             |  (Context-Aware Alert Synthesizer)|
             +-----------------------------------+
                               |
              +----------------+----------------+
              |                                 |
              v                                 v
    +------------------------------+  +------------------------------+
    |   Instant Slack Alerts       |  |   Streamlit Manager UI       |
    |   (Floor Staff Mobile)       |  |   (Analytics & Live Heatmap) |
    +------------------------------+  +------------------------------+
---
👥 Team & Contributions:

- Dhanush: OpenCV Configuration, Video Processing.

- Pooja: Friction Matrix & Logic, Centroid Tracking, Heatmap Configuration.

- Mridul: Streamlit Frontend Dashboard & UI/UX Analytics.

- Adit: GenAI Alert Synthesis, LangChain Pipeline & Slack Integration.

📜 License

Distributed under the MIT License. See LICENSE for more information.
