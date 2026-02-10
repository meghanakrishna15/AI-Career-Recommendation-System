# 🎯 AI Career Recommendation System

An AI-powered web application that recommends suitable career paths based on a user's skills and interests using Machine Learning.

---

## 🚀 Project Overview

Choosing the right career can be confusing for students and professionals.  
This system uses **Machine Learning** to analyze skill scores and interest areas, then predicts the most suitable career with a confidence percentage.

---

## 🧠 Features

- Skill-based career prediction
- Machine Learning model using Random Forest
- Confidence percentage for predictions
- Interactive web app built with Streamlit
- Clean and user-friendly UI

---

## 🛠️ Tech Stack

- **Programming Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn  
- **Machine Learning Model:** Random Forest Classifier  
- **Web Framework:** Streamlit  
- **IDE:** VS Code  

---

## 📊 Machine Learning Details

- Encoded categorical features using Label Encoding  
- Trained using Random Forest Classifier  
- Achieved **high accuracy on structured dataset**  
- Used probability scores to calculate prediction confidence  

---

## 📁 Project Structure

AI_Career_Recommendation/
│
├── data/
│ └── career_data.csv
│
├── model/
│ ├── career_model.pkl
│ ├── interest_encoder.pkl
│ ├── career_encoder.pkl
│ └── accuracy.pkl
│
├── train_model.py
├── app.py
└── README.md

## ▶️ How to Run the Project

1. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn streamlit

2. Train the model:
   python train_model.py
   
3. Run the web app:
   streamlit run app.py
   
⚠️ Disclaimer
This project is built for educational purposes only and should not be considered as professional career advice.

👩‍💻 Author
Meghana Krishna
Aspiring Data Scientist | Python & Machine Learning Enthusiast