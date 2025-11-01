# AICTE-1-Sugguna-Akhil
# Electric Vehicle (EV) Cost Prediction

**Technical Domain:** Generative AI

## 🚗 Introduction

### 1. EV Cost Prediction
Predicting the cost of an Electric Vehicle (EV) based on its specifications helps manufacturers and customers make data-driven decisions. Machine Learning (ML) models analyze factors like battery capacity, motor power, range, brand, and model year to forecast EV prices. Algorithms such as Random Forest, XGBoost, and Linear Regression can provide accurate estimates of EV pricing.

### 2. Generative AI in Automotive Applications
Generative AI tools like ChatGPT, DALL·E, Gemini, and DeepSeek can create new text, images, and insights from existing data. In automotive domains, these tools can:
- Build chatbots that explain model predictions.
- Generate synthetic data for incomplete datasets.
- Create intelligent reports and visualizations automatically.

### 3. Combining EV Cost Prediction with Generative AI
Integrating ML for price forecasting and Generative AI for explanations makes EV insights more accessible. A GPT-powered chatbot can answer EV-related queries, while image models (e.g., DALL·E) can generate EV visualizations and comparisons.

## ❓ Problem Statement
Many EV buyers find it difficult to understand price variations between different electric vehicles. Traditional listings do not clearly show how specifications like battery capacity, motor power, or vehicle type influence the final cost.

**This project aims to:**
1. Develop a Machine Learning model that predicts EV prices using technical specifications.  
2. Integrate a Generative AI chatbot that explains these predictions in natural language.  
3. Build a Streamlit-based web interface for interactive user input and instant EV cost prediction.

## ⚙️ Requirements
**Data Inputs:**
- Battery Capacity (kWh)
- Motor Power (kW)
- Range (km)
- Vehicle Weight (kg)
- Brand and Model Type
- Model Year

**Outputs:**
- Predicted Price (in INR or USD)
- Feature Importance (key factors affecting cost)
- AI-based Explanations through Chatbot

**Software & Tools:**
- Programming: Python  
- ML Libraries: scikit-learn, XGBoost, LightGBM  
- Generative AI APIs: OpenAI, Gemini, DeepSeek  
- Frontend Framework: Streamlit  
- Visualization: Matplotlib, Plotly, Seaborn  

## 📊 Datasets
| No. | Dataset Name | Source Link |
|-----|---------------|--------------|
| 1 | Electric Vehicle Dataset 2024 | https://www.kaggle.com/datasets/vanillatyy1/electric-vehicle-dataset |
| 2 | Electric Car Prices Dataset | https://www.kaggle.com/datasets/nelgiriyewithana/electric-car-price-dataset |
| 3 | Electric Vehicle Specs & Prices | https://www.kaggle.com/datasets/borisethefuture/electric-vehicle-specifications-and-prices |
| 4 | Global EV Data Explorer | https://www.iea.org/data-and-statistics/data-product/global-ev-data-explorer |

## 🧠 Summary
This project integrates Machine Learning and Generative AI to build an intelligent Electric Vehicle Cost Prediction System. The ML model estimates the price of EVs using their specifications, while the Generative AI chatbot provides explanations and insights. Through a Streamlit interface, users can input vehicle details, view predicted prices, and understand how each factor affects cost — making EV analysis more interactive and insightful.
