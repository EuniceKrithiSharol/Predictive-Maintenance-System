# ⚙️ Predictive Maintenance System

A Machine Learning-based predictive maintenance application that analyzes equipment sensor data to predict potential machine failure and maintenance requirements.

---

## 🚀 Project Overview

Predictive maintenance uses Machine Learning and sensor data to identify potential equipment failures before they occur.

This project analyzes machine operating conditions and predicts the probability of machine failure based on multiple sensor features.

The goal is to support proactive maintenance and reduce unexpected equipment downtime.

---

## 🧠 Machine Learning Approach

This project uses supervised Machine Learning.

The model learns patterns between machine operating conditions and historical failure outcomes.

### Input Features

- Temperature
- Vibration
- Pressure
- Rotation Speed
- Operating Hours
- Humidity

### Prediction

The system predicts whether the machine is:

- Operating Normally
- At Risk of Failure

---

## 🤖 Model Used

### Random Forest Classifier

Random Forest is an ensemble Machine Learning algorithm that combines multiple decision trees.

It is useful for:

- Classification
- Feature importance analysis
- Non-linear relationships
- Complex sensor data

---

## ✨ Features

- Machine sensor analysis
- Machine failure prediction
- Failure probability
- Interactive maintenance dashboard
- Sensor visualization
- Feature importance analysis
- Individual machine analysis
- CSV upload
- Batch machine predictions
- Maintenance risk identification

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest
- Plotly
- Streamlit

---

## 📁 Project Structure

```text
Predictive-Maintenance-System/
│
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── app.py
│
├── data/
│   └── README.md
│
├── src/
│   └── predictor.py
│
├── models/
│   └── README.md
│
├── notebooks/
│   └── README.md
│
└── reports/
    └── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Predictive-Maintenance-System.git
```

Move into the project directory:

```bash
cd Predictive-Maintenance-System
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 🔄 System Workflow

```text
Machine Sensors
      ↓
Sensor Data Collection
      ↓
Feature Processing
      ↓
Machine Learning Model
      ↓
Failure Probability
      ↓
Normal Operation / Maintenance Required
```

---

## 📤 CSV Input Format

Custom machine sensor datasets should contain:

```text
Temperature
Vibration
Pressure
Rotation_Speed
Operating_Hours
Humidity
```

The application predicts maintenance risk for each machine record.

---

## 💡 Real-World Applications

Predictive maintenance can be used in:

- Manufacturing
- Industrial automation
- Automotive systems
- Aviation
- Energy systems
- Smart factories
- Industrial IoT
- Heavy machinery

---

## 🔮 Future Improvements

- Real IoT sensor integration
- Time-series prediction
- Remaining Useful Life prediction
- Deep Learning models
- XGBoost models
- Real-time monitoring
- Cloud deployment
- Automated maintenance alerts
- Equipment health scoring

---

## 👩‍💻 Author

Developed as part of an Artificial Intelligence, Machine Learning and Data Analytics portfolio.
