# neural-network-challenge-2

# Employee Attrition Analysis
This project analyzes employee attrition using a neural network model to predict employee attrition and department based on various features.

##Project Overview
File: attrition.ipynb
Purpose: Predict employee attrition and department using neural networks.
Dataset: Attrition Data

##Requirements
pandas
numpy
scikit-learn
tensorflow
Install the required libraries using pip:

bash
pip install pandas numpy scikit-learn tensorflow


##Steps
1. Data Import and Exploration: Load and inspect the dataset.
2. Data Preparation: Select features, encode categorical data, and scale numeric data.
3. Model Creation: Build a neural network with shared layers and two output branches for department and attrition.
4. Model Training: Train the model on the training data.
5. Model Evaluation: Evaluate and print the model's accuracy.
   
##Key Points
- Metrics: Accuracy is used but may not be ideal for imbalanced data; consider precision, recall, F1 score, or AUC.
- Activation Function: Softmax is used for output layers to handle multi-class classification.
- Improvements: Feature engineering, handling class imbalance, hyperparameter tuning, regularization, exploring different architectures, ensemble methods, data augmentation, and advanced optimization algorithms.
  
##How to Run
Clone the repository.
Open attrition.ipynb file 
Install the required libraries.
Open attrition.ipynb in Jupyter Notebook.
Run the cells sequentially to execute the analysis.
