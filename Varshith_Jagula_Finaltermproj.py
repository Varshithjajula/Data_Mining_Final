# %% [markdown]
# # Adult Income Prediction: A Comparative Analysis of Classification Models
# - By Varshith Jajula
# - NJIT UCID: VJ252

# %% [markdown]
# ## Importing Libararies

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, roc_auc_score, brier_score_loss, confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# %% [markdown]
# ## Loading the data

# %%
adult_df = pd.read_csv( "adult.data", sep=",", header = None)

# %% [markdown]
# The dataset used for this project is the Adult Income dataset, available at: Adult - UCI Machine Learning Repository. 

# %% [markdown]
# Setting the Column Names

# %%
column_names = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]
adult_df.columns = column_names


# %% [markdown]
# # Understanding the Data
# 

# %%
adult_df.head()
print(adult_df['income'].unique())


# %% [markdown]
# The Target Variable has 2 values [' <=50K' ' >50K'].

# %%
for col in ['workclass', 'occupation', 'native-country']:
    adult_df[col].fillna(adult_df[col].mode()[0], inplace=True)

# %%
adult_df.isnull().sum()
adult_df.info()

# %%
adult_df['income'] = adult_df['income'].map({' <=50K': 0, ' >50K': 1})
adult_df.head()

# %%
income_counts = adult_df['income'].value_counts()
income_percentages = (income_counts / income_counts.sum()) * 100
fig, ax = plt.subplots()
income_counts.plot(kind='bar', ax=ax)
for i, count in enumerate(income_counts):
    ax.text(i, count + 500, f'{income_percentages[i]:.2f}%', ha='center', fontsize=10)

plt.xlabel('Income (0 = <=50K, 1 = >50K)')
plt.ylabel('Count')
plt.title('Income Distribution with Percentages')
plt.show()


# %% [markdown]
# The Distrubution of Target Variable

# %% [markdown]
# ## Data preprocessing

# %%
# Separating Target and Features from the DataFrame
features = adult_df.iloc[:, :-1]
Y = adult_df.iloc[:, -1]

# %%
# Checking for Categorical Variables
features_complete= features.select_dtypes('object').apply(pd.Series.nunique, axis = 0)

# %%
features_complete

# %%
# Label Encoding
le = LabelEncoder()
le_count = 0
encoded_columns = []
for col in features:
    if features[col].dtype == 'object':
        if len(list(features[col].unique())) <= 2:
            le.fit(features[col])
            features[col] = le.transform(features[col])
            le_count += 1
            encoded_columns.append(col)

print('%d columns were label encoded: %s' % (le_count, encoded_columns))

# %%
features_complete = pd.get_dummies(features)
features_complete.head()

# %% [markdown]
# By applying label encoding to columns with fewer than two levels, the total
# number of columns was reduced, streamlining the model for improved efficiency and performance.
# 
# 
# 
# 
# 
# 
# 

# %%
X = features_complete

# %% [markdown]
# ## Standarization

# %%
scaler = StandardScaler()
X[['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']] = scaler.fit_transform(X[['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']])



# %% [markdown]
# Here, we are standardizing  numeric columns in the dataset X using StandardScaler. The numeric features are transformed to have a mean of 0 and standard deviation of 1, which can improve the performance of many machine learning algorithms by ensuring all features are on a similar scale.

# %% [markdown]
# ## Splitting the data for training and testing

# %%
# Splitting the data in 70-30 Ratio
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Y_train shape:", Y_train.shape)
print("Y_test shape:", Y_test.shape)


# %% [markdown]
# # Model Building
# 1) Random Forest
# 
# 2) KNN
# 
# 3) LSTM

# %%
from sklearn.metrics import confusion_matrix, accuracy_score

def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    # Core metrics
    TPR = TP / (TP + FN) if (TP + FN) != 0 else 0  # True Positive Rate (Recall)
    TNR = TN / (TN + FP) if (TN + FP) != 0 else 0  # True Negative Rate
    FPR = FP / (TN + FP) if (TN + FP) != 0 else 0  # False Positive Rate
    FNR = FN / (TP + FN) if (TP + FN) != 0 else 0  # False Negative Rate
    BACC = (TPR + TNR) / 2  # Balanced Accuracy
    TSS = TPR - FPR  # True Skill Statistic
    HSS = 2 * (TP * TN - FP * FN) / ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN)) if ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN)) != 0 else 0  # Heidke Skill Score
    Precision = TP / (TP + FP) if (TP + FP) != 0 else 0  # Precision
    F1_measure = 2 * (Precision * TPR) / (Precision + TPR) if (Precision + TPR) != 0 else 0  # F1 Score
    Accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0  # Calculated Accuracy
    Error_rate = (FP + FN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0  # Error Rate
    sklearn_accuracy = accuracy_score(y_true, y_pred)  # Accuracy by sklearn

    return {
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'TPR': TPR,
        'TNR': TNR,
        'FPR': FPR,
        'FNR': FNR,
        'BACC': BACC,
        'TSS': TSS,
        'HSS': HSS,
        'Precision': Precision,
        'F1_measure': F1_measure,
        'Calculated_Accuracy': Accuracy,
        'Error_rate': Error_rate,
        'Sklearn_Accuracy': sklearn_accuracy
    }


# %%
knn_parameters = {"n_neighbors": [5, 7, 9, 11, 13, 15, 17, 19, 21]}
knn_model = KNeighborsClassifier()
knn_cv = GridSearchCV(knn_model, knn_parameters, cv=10, n_jobs=-1)
knn_cv.fit(X_train, Y_train)
print("\nBest Parameters for KNN based on GridSearchCV: ", knn_cv.best_params_)
print('\n')
best_n_neighbors = knn_cv.best_params_['n_neighbors']


# %% [markdown]
# ### Random Forest

# %%
# Params for Random Forest
param_dist_rf = {
    "n_estimators": [50, 100, 150],
    "min_samples_leaf": [1, 2, 4],
    "criterion": ['gini', 'entropy']
}

rf_classifier = RandomForestClassifier()

#RandomizedSearchCV
random_search_rf = RandomizedSearchCV(
    estimator=rf_classifier,
    param_distributions=param_dist_rf,
    n_iter=10,  
    cv=10,
    n_jobs=-1,
    random_state=42
)

# Model Fitting 
random_search_rf.fit(X_train, Y_train)
best_rf_params = random_search_rf.best_params_
print("\nBest Parameters for Random Forest based on RandomizedSearchCV: ", best_rf_params)

# %%
rf_model = RandomForestClassifier(**best_rf_params, random_state=42)
rf_metrics = []
skf = StratifiedKFold(n_splits=10)

# Metric Calculation For Each Fold 
for fold, (train_index, val_index) in enumerate(skf.split(X_train, Y_train), start=1):
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = Y_train.iloc[train_index], Y_train.iloc[val_index]

    rf_model.fit(X_train_fold, y_train_fold)
    y_val_pred = rf_model.predict(X_val_fold)
    y_val_prob = rf_model.predict_proba(X_val_fold)[:, 1]
    metrics = calculate_metrics(y_val_fold, y_val_pred)
    metrics['AUC'] = roc_auc_score(y_val_fold, y_val_prob)
    metrics['Brier_Score'] = brier_score_loss(y_val_fold, y_val_prob)
    rf_metrics.append(metrics)

# Average Calcualtion
rf_metrics_df = pd.DataFrame(rf_metrics)
average_rf_metrics = rf_metrics_df.mean().to_frame(name='Average')




# %%
print("\nRandom Forest Metrics for Each Fold:\n", rf_metrics_df.T)


# %%
print("\nRandom Forest Average Metrics across all folds:\n", average_rf_metrics)

# %% [markdown]
# ### KNN Model
# 

# %%
# KNN Model Initialization
knn_model = KNeighborsClassifier(n_neighbors=best_n_neighbors)

knn_metrics = []

for fold, (train_index, val_index) in enumerate(skf.split(X_train, Y_train), start=1):
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = Y_train.iloc[train_index], Y_train.iloc[val_index]
    knn_model.fit(X_train_fold, y_train_fold)
    y_val_pred = knn_model.predict(X_val_fold)
    y_val_prob = knn_model.predict_proba(X_val_fold)[:, 1]
    metrics = calculate_metrics(y_val_fold, y_val_pred)
    metrics['AUC'] = roc_auc_score(y_val_fold, y_val_prob)
    metrics['Brier_Score'] = brier_score_loss(y_val_fold, y_val_prob)

    knn_metrics.append(metrics)
knn_metrics_df = pd.DataFrame(knn_metrics)
average_knn_metrics = knn_metrics_df.mean().to_frame(name='Average')




# %%
print("\nKNN Metrics for Each Fold:\n", knn_metrics_df.T)


# %%
print("\nKNN Average Metrics across all folds:\n", average_knn_metrics)

# %% [markdown]
# ### LSTM
#   

# %%
# Reshaping
X_train_reshaped = X_train.values.astype(np.float32).reshape((X_train.shape[0], 1, X_train.shape[1]))
Y_train_reshaped = Y_train.values.astype(np.int32)  

def create_lstm_model():
    model = Sequential()
    model.add(LSTM(50, input_shape=(1, X_train_reshaped.shape[2]), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

lstm_metrics = []

for fold, (train_index, val_index) in enumerate(skf.split(X_train_reshaped, Y_train_reshaped), start=1):
    X_train_fold, X_val_fold = X_train_reshaped[train_index], X_train_reshaped[val_index]
    y_train_fold, y_val_fold = Y_train_reshaped[train_index], Y_train_reshaped[val_index]

    lstm_model = create_lstm_model()
    history = lstm_model.fit(X_train_fold, y_train_fold, epochs=10, batch_size=32, verbose=0)
    y_val_prob = lstm_model.predict(X_val_fold)
    y_val_pred = (y_val_prob > 0.5).astype("int32")
    metrics = calculate_metrics(y_val_fold, y_val_pred)
    metrics['AUC'] = roc_auc_score(y_val_fold, y_val_prob)
    metrics['Brier_Score'] = brier_score_loss(y_val_fold, y_val_prob)

    lstm_metrics.append(metrics)
lstm_metrics_df = pd.DataFrame(lstm_metrics)
average_lstm_metrics = lstm_metrics_df.mean().to_frame(name='Average')

# %%
print("\nLSTM Metrics for Each Fold:\n", lstm_metrics_df.T)

# %%
print("\nLSTM Average Metrics across all folds:\n", average_lstm_metrics)

# %% [markdown]
# ### Making Predictions

# %%
lstm_model_final = create_lstm_model()
lstm_model_final.fit(X_train_reshaped, Y_train, epochs=10, batch_size=32, verbose=0)

# Reshape test data to match input shape expected by LSTM model
X_test_reshaped = X_test.values.astype(np.float32).reshape((X_test.shape[0], 1, X_test.shape[1]))

y_test_prob = lstm_model_final.predict(X_test_reshaped)

y_test_pred = (y_test_prob > 0.5).astype("int32")

test_metrics = calculate_metrics(Y_test, y_test_pred)

test_metrics['AUC'] = roc_auc_score(Y_test, y_test_prob)
test_metrics['Brier_Score'] = brier_score_loss(Y_test, y_test_prob)

test_metrics_df = pd.DataFrame(test_metrics, index=[0]).T  
test_metrics_df.columns = ["Value"]  
print("\nLSTM Metrics on Test Set:\n", test_metrics_df)

# %%
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(Y_test, y_test_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--') 
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve For LSTM')
plt.legend(loc='lower right')
plt.grid()
plt.show()

# %%
rf_model.fit(X_train, Y_train)

y_test_pred = rf_model.predict(X_test)
test_metrics = calculate_metrics(Y_test, y_test_pred)
y_test_prob = rf_model.predict_proba(X_test)[:, 1]
test_metrics['AUC'] = roc_auc_score(Y_test, y_test_prob)
test_metrics['Brier_Score'] = brier_score_loss(Y_test, y_test_prob)
test_metrics_df = pd.DataFrame(test_metrics, index=[0]).T  
test_metrics_df.columns = ["Value"]  
print("\n RF Metrics on Test Set:\n", test_metrics_df)

# %%
fpr, tpr, threshods = roc_curve(Y_test, y_test_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--') 
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve For Random Forest')
plt.legend(loc='lower right')
plt.grid()
plt.show()

# %%
knn_model.fit(X_train, Y_train)
y_test_pred_knn = knn_model.predict(X_test)
test_metrics_knn = calculate_metrics(Y_test, y_test_pred_knn)
y_test_prob_knn = knn_model.predict_proba(X_test)[:, 1]
test_metrics_knn['AUC'] = roc_auc_score(Y_test, y_test_prob_knn)
test_metrics_knn['Brier_Score'] = brier_score_loss(Y_test, y_test_prob_knn)
test_metrics_knn_df = pd.DataFrame(test_metrics_knn, index=[0]).T
test_metrics_knn_df.columns = ["Value"]
print("\nKNN Metrics on Test Set:\n", test_metrics_knn_df)

# %%
fpr, tpr, thresholds = roc_curve(Y_test, y_test_prob_knn)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--') 
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve For KNN')
plt.legend(loc='lower right')
plt.grid()
plt.show()


