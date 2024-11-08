# Data Mining: Predicting Income Classification

## Introduction

This project applies and evaluates multiple classification algorithms on the **Adult Income dataset**, a common dataset for income prediction tasks. The objective is to compare the performance of three models—**Random Forest**, **K-Nearest Neighbors (KNN)**, and **LSTM (Long Short-Term Memory)**—on predicting whether an individual's income exceeds a specified threshold based on demographic factors.

## Algorithms Overview

- **Random Forest**: An ensemble method that creates multiple decision trees and merges them to produce more accurate and stable predictions.
- **K-Nearest Neighbors (KNN)**: A simple, instance-based learning algorithm that classifies data points based on the majority class of their nearest neighbors.
- **LSTM**: A type of recurrent neural network (RNN) particularly suited for sequential data and capable of capturing long-term dependencies in data.

## Dataset

The **Adult Income dataset** contains demographic information such as age, education, occupation, and work hours, along with the target income class label. The goal is to classify each individual as earning either above or below the specified income threshold based on these features.

Link : https://archive.ics.uci.edu/dataset/2/adult

## Requirements

To run this project, you will need the following Python packages:

- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `tensorflow`

You can install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Usage

1. **Load the Dataset**: The script loads and preprocesses the dataset, including handling missing values, encoding categorical variables, and scaling numerical features.
   
2. **Train and Evaluate Models**:
   - Each model is trained using a train-test split.
   - Model performance is evaluated using metrics such as accuracy, precision, recall, F1 score, AUC, and Brier score.

3. **Comparing Results**: The script provides a comparison of model performance metrics to help identify the most suitable model for this classification task.

### Steps to Run
**For `.ipynb` (Jupyter Notebook):** 
1. Install Packages: Run `pip install -r requirements.txt` to install all required packages.  
2. Open the Jupyter Notebook in the project folder.
4. Run All Cells: In the Jupyter Notebook, go to the Cell menu and select Run All to execute all code cells.  
5. Review the Results: Check the output under each cell for the results.  
**For `.py` (Python Script):**
1. Navigate to Folder: Open project folder and open CMD from there. 
2. Run Script: Execute the script by running `python Varshith_Jagula_FinalTermProj.py`.  
3. Review the Results: Check the terminal output for results and messages.

### Example Usage:

1. After loading the dataset, the script will automatically split the data and train each model.
2. The output will display the metrics for each model on the test set, including accuracy, precision, recall, F1 score, and AUC.


### Expected Output

1. **Manual Calculation of Performance Metrics**:
   - Confusion matrix values (True Positives, True Negatives, False Positives, False Negatives) for each fold.
     
   - Calculations for other performance metrics using formulas:

-	**Precision:** Measures how many of the predicted positive cases are actually positive.
-	**Recall (True Positive Rate):** Measures how well the model identifies actual positive cases.
-	**True Negative Rate (Specificity):** Measures how well the model identifies actual negative cases.
-	**False Positive Rate (FPR):** Measures the proportion of actual negatives incorrectly predicted as positive.
-	**False Negative Rate (FNR):** Measures the proportion of actual positives incorrectly predicted as negative.
-	**Balanced Accuracy (BACC):** Provides a balanced view of accuracy across classes, especially useful for imbalanced datasets.
-	**True Skill Statistic (TSS):** Assesses the model's ability to distinguish between classes.
-	**Heidke Skill Score (HSS):** Compares model performance to random chance.
-	**F1 Score:** Balances Precision and Recall, offering a combined view of the two metrics.
-	**Accuracy:** Overall correctness of the model’s predictions across both positive and negative cases.
-	**Error Rate:** Proportion of incorrect predictions made by the model.
-	**Sklearn Accuracy:** Standard accuracy measure calculated using sklearn’s function.
-	**AUC (Area Under the Curve)**: Reflects the model’s ability to distinguish between classes, with higher values indicating better performance.
-	**Brier Score:** Evaluates the accuracy of probability predictions, with lower values indicating more accurate probability estimates. 


2. **10-Fold Cross Validation**:
   - Performance metrics for each of the 10 folds, presented in a table.
   - Average metrics across all folds, with calculations for each metric listed above.

3. **Tabular Summary**:
   - Consolidated table with metrics for each algorithm across the 10 folds.
   - Summary table showing the average performance metrics for each algorithm, enabling straightforward comparison.


### Results

From the analysis, **Random Forest** consistently demonstrates higher accuracy and AUC compared to **KNN** and **LSTM**, making it a suitable choice for this classification task. **LSTM**, while generally effective for sequential data, did not significantly outperform traditional models here, and **KNN** had lower accuracy and F1 scores.

### Conclusion

For income classification on this dataset, Random Forest provides a reliable balance of accuracy and interpretability, while KNN may be better suited for simpler or smaller datasets. LSTM's performance suggests it is not as optimal for non-sequential data in this context.
