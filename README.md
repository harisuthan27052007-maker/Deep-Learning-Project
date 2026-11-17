# Linear Regression: Food Truck Profit Prediction

An implementation of Univariate Linear Regression from scratch using Python and NumPy to predict the profit of a food truck based on city population. 

* **Reference:** Coursera Machine Learning course by Andrew Ng (Exercise 1).

---

## 📋 Overview

Suppose you are the CEO of a restaurant franchise considering different cities for opening a new outlet. The chain already has trucks in various cities, and you have data for profits and populations from those cities. This project uses gradient descent to fit a linear model to the dataset to predict potential profits in new cities[cite: 1].

## 📊 Dataset

The dataset (`data/ex1data1.txt`) contains 97 records with two variables[cite: 1]:
1. **Population:** Population of a city (in 10,000s)[cite: 1].
2. **Profit:** Profit of a food truck in that city (in $10,000s)[cite: 1].

### Summary Statistics
| Metric | Population | Profit |
| :--- | :--- | :--- |
| **Count** | 97[cite: 1] | 97[cite: 1] |
| **Mean** | 8.1598[cite: 1] | 5.8391[cite: 1] |
| **Std Dev** | 3.8699[cite: 1] | 5.5103[cite: 1] |
| **Min** | 5.0269[cite: 1] | -2.6807[cite: 1] |
| **Max** | 22.2030[cite: 1] | 24.1470[cite: 1] |

---

## ⚙️ Methodology & Results

1. **Data Preprocessing:** Added an intercept term (a column of `1`s) to the feature matrix $X$ and converted the Pandas DataFrame into NumPy matrices[cite: 1].
2. **Cost Function:** Implemented the Mean Squared Error (MSE) cost function $J(\theta)$[cite: 1]:
   $$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$$
   * **Initial Cost (at $\theta = [0, 0]$):** `32.0727`[cite: 1]
3. **Gradient Descent:** Ran batch gradient descent with the following hyperparameters[cite: 1]:
   * **Learning Rate ($\alpha$):** `0.01`[cite: 1]
   * **Iterations:** `1000`[cite: 1]
4. **Final Model Parameters & Cost:**
   * **Optimal $\theta$:** `[-3.2414, 1.1273]`[cite: 1]
   * **Minimized Cost:** `4.5160`[cite: 1]

---

## 🛠️ Tech Stack & Prerequisites

To run this notebook locally, you need Python 3 along with the following libraries[cite: 1]:
* `numpy`[cite: 1]
* `pandas`[cite: 1]
* `matplotlib`[cite: 1]
* `jupyter`

### Installation
```bash
pip install numpy pandas matplotlib jupyter
