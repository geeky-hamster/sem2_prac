from sklearn.linear_model import LinearRegression
import pandas as pd

# Sample dataset
data = {
    'Area': [1000, 1500, 1800, 2400],
    'Bedrooms': [2, 3, 3, 4],
    'Bathrooms': [2, 2, 3, 3],
    'Price': [3500000, 5000000, 6500000, 9000000]
}

df = pd.DataFrame(data)

# Features and target
X = df[['Area', 'Bedrooms', 'Bathrooms']].values
y = df['Price']

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict new house price for a 2000 sq ft, 3 Bed, 2 Bath house
prediction = model.predict([[2000, 3, 2]])
print("Predicted Price: ₹", round(prediction[0], 2))


