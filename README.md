# Univariate Linear Regression

## 📌 Overview
This project implements **Univariate Linear Regression** to predict **revenue** based on **population size**. The dataset consists of:  
- **Population size** (in 10,000s)  
- **Revenue** (in $10,000s)  

A **Linear Regression model** is trained to predict revenue based on the given population size. The implementation is written in **Python** and includes visualization of the results.

---

## 🔹 Linear Regression Model
The model follows the equation:

\[
Y = \theta_0 + \theta_1 X
\]

Where:  
- \( Y \) = Predicted Revenue  
- \( X \) = Population size  
- \( \theta_0 \), \( \theta_1 \) = Model parameters  

The **cost function** is minimized using **Gradient Descent** to optimize model performance.

---

## 📊 Visualizations

### 📈 Linear Regression Model
The trained model fitting the dataset:  
![Linear Regression](Revenue-for-the-population/Linear_Regression_model.png?raw=true "Linear Regression Model")

---

### 🌄 Contour Plot  
The **contour plot** visualizes how the cost function behaves for different values of model parameters. The global minimum represents the optimal values of \(\theta_0\) and \(\theta_1\):  
![Contour Plot](Revenue-for-the-population/contour_plot.png?raw=true "Contour plot")

---

### 📉 Cost Function  
The cost function curve shows how the error decreases as training progresses:  
![Cost function](Revenue-for-the-population/cost_function.png?raw=true "Cost Function")

---

## 🚀 Installation & Setup
To run the project locally:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/meghasv09/LinearRegression.git
   cd LinearRegression
2. **Run the script:**
   ```bash
   python linear_regression.py
