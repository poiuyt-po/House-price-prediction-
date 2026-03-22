import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Data
data = {
    "size": [500, 1000, 1500, 2000, 2500],
    "price": [10, 20, 30, 40, 50]
}

df = pd.DataFrame(data)

# Input / Output
X = df[["size"]]
y = df["price"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
pred = model.predict([[1200]])
print("Predicted price:", pred)
