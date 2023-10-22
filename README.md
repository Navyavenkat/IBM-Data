# IBM-z-datathon-23
# INTRODUCTION
    "Scan and Nutrizoe"-Empower the health and medicine awrareness to the user. 
# Problem
   Due to human errors we are facing lot of issues regarding the intake of medicine.

# Approach:
   Instead of consulting professionals everytime we can knew the in formation in the "NUSCA" app.
# Process
# Step 1: Data Preparation
Collect current data for medicine information.

# Step 2: Create a dataset
By the existing dataset and some kind of additional features  to add.

# Step 3: Use of QR 
Build a app and scanning the QR code for details about medicine.

# Step 4: Data Input
Fetching the data from scanner and suggesting the balanced diet as per the user convinence.

# Step 5: Training
Train the model to learn from the historical data.

# Step 6: Prediction
 It is used to predict the current user query to solution in suggestion

# Code

# Data preprocessing

```
  import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


df=pd.read_csv('Churn_Modelling.csv')
df


df.isnull().sum()


df.duplicated()


df.drop('RowNumber',axis=1,inplace=True)
df.drop('CustomerId',axis=1,inplace=True)
df.drop('Age',axis=1,inplace=True)
df.drop('Gender',axis=1,inplace=True)
df.drop('Surname',axis=1,inplace=True)
df.drop('Geography',axis=1,inplace=True)
df


ms=MinMaxScaler()
df2=pd.DataFrame(ms.fit_transform(df))
df2


X=df2.iloc[:,:-1].values
X


y=df2.iloc[:,-1].values
y


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train)
print("X_train: ",len(X_train))
print(X_test)
print("Size of X_test: ",len(X_test))

```
