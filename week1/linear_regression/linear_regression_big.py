import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
def floor_sep(floor):
    parts = floor.split(' out of ')
    room_floor = parts[0].strip()
    total_floor = parts[1].strip() if len(parts) > 1 else "0"
    if room_floor == "Ground":
        room_floor = 0
    elif room_floor == "Lower Basement":
        room_floor = -2
    elif room_floor == "Upper Basement":
        room_floor = -1
    else:
        room_floor = int(room_floor)
    total_floor = int(total_floor)
    return room_floor, total_floor
df=pd.read_csv("soc2025/SOC-2025/week1/linear_regression/house_rent_dataset.csv")
df=df.drop(columns=["Posted On","Area Locality","Point of Contact"])
df['Room Floor'], df['Total Floors'] = zip(*df['Floor'].apply(floor_sep))
df=df.drop(columns=["Floor"])
df=pd.get_dummies(df, columns=["Area Type","City","Furnishing Status","Tenant Preferred"])
x=df.drop(columns=["Rent"])
y=df["Rent"]
x=x.fillna(0).astype(float).values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
alpha = 1e-3
epochs = 10000
m,n= x_train.shape
w=np.zeros((n,1))
b=0
def compute_cost(X, y, w, b):
    predictions = X.dot(w) + b
    error = predictions - y
    return np.mean(error ** 2) / 2
def compute_gradient(x, y, w, b):
    m,n = x.shape
    dj_dw = np.zeros((n,1))
    dj_db = 0.0
    for i in range(m):
        error=(np.dot(x[i],w)+b)-y[i]
        for j in range(n):
            dj_dw[j]+=error*x[i,j]
        dj_db+=error
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db
def gradient_descent(x, y, w, b, alpha, iters):
    m = x.shape[0]
    for i in range(iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db
        if i % 100 == 0:
            cost = compute_cost(x, y, w, b)
            print(f"Iteration {i}: Cost {cost}")
    return w, b
mean = np.mean(x_train, axis=0)
std = np.std(x_train, axis=0)
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std
w, b = gradient_descent(x_train, y_train.values.reshape(-1, 1), w, b, alpha, epochs)
y_pred = np.dot(x_test, w) + b
y_test = y_test.values.reshape(-1, 1)
mse=np.mean((y_pred-y_test)**2)
print("Mean Squared Error on Test Set:", mse)
for i, col in enumerate(df.drop(columns=["Rent"]).columns):
    print(f"Weight for {col}: {w[i][0]}")
print("Bias term:", b)