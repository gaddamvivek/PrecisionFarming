---
title: SmartFarming
emoji: 🌱
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
---

# SmartFarming — AI-Powered Precision Agriculture

Smart Farming is an IoT and ML-based system that helps farmers make data-driven decisions on crop selection and fertilizer usage. This project originated from a startup focused on precision agriculture using LoRaWAN sensors and machine learning.

## Features

- **Crop Recommendation** — Input soil NPK values, temperature, humidity, pH, and rainfall. Six ML models (Random Forest, XGBoost, SVM, Decision Tree, Naive Bayes, Logistic Regression) each predict the best crop. A majority vote gives the final recommendation.
- **Fertilizer Recommendation** — Input your crop type and current soil NPK readings. The system compares against ideal values and recommends corrective action.

## ML Models

| Model | Algorithm |
|---|---|
| Random Forest | Ensemble (bagging) |
| XGBoost | Ensemble (boosting) |
| SVM | Support Vector Machine |
| Decision Tree | Tree-based |
| Naive Bayes | Probabilistic |
| Logistic Regression | Linear |

All models trained on the [Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) (2200 samples, 22 crops).

## Tech Stack

- **Backend:** Python, Flask
- **ML:** scikit-learn, XGBoost
- **Frontend:** HTML, CSS, JavaScript
- **Deployment:** Docker on Hugging Face Spaces

## Background

This project is part of a larger Precision Farming system that integrates:
- LoRaWAN IoT sensors (NPK, pH, moisture, humidity)
- ThingsMatee cloud gateway
- Automated SMS alerts via Twilio
- Scheduled ML inference pipeline

See [PrecisionFarming](https://github.com/gaddamvivek/PrecisionFarming) for the full IoT automation codebase.
