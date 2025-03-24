# prompt_templates.py

BLOG_IDEA_PROMPT = """
You are a creative content strategist specializing in generating engaging blog post ideas.

I need {num_ideas} blog post ideas for a content creator in the {niche} niche. 
The ideas should be {include_outline} and have a {tone} tone.

For each idea:
1. Provide a catchy, SEO-friendly title
2. Write a brief description of the concept (2-3 sentences)
3. If outlines are requested, include a 5-7 point outline with key sections

Make sure the ideas are:
- Trending and timely for current interests
- Specific enough to be actionable
- Designed to engage the target audience
- Unique and not generic

Format each idea clearly with numbers and proper spacing for readability.
RESPOND ONLY WITH THE BLOG IDEAS AND NO OTHER TEXT.
"""

import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Connect to the SQLite database (same as the FastAPI backend)
DATABASE_URL = "sqlite:///./inventory.db"
engine = create_engine(DATABASE_URL)
conn = sqlite3.connect("inventory.db")
cursor = conn.cursor()

# Fetch inventory data from the database
query = """
SELECT id, name, quantity, price FROM items
"""
df = pd.read_sql(query, conn)

# Generate synthetic time-based data (since DB has no date field)
df['date'] = pd.date_range(start='1/1/2023', periods=len(df), freq='D')
df['sales'] = np.random.randint(50, 500, len(df))  # Simulated sales data
df['lead_time'] = np.random.randint(2, 10, len(df))  # Simulated lead time
df['seasonality_factor'] = np.random.uniform(0.5, 1.5, len(df))  # Simulated seasonality

# Feature selection
X = df[['sales', 'quantity', 'lead_time', 'seasonality_factor']]
y = df['sales'].shift(-1).fillna(method='ffill')  # Predict next day's sales

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train AI model for demand forecasting
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")

# Function to predict future inventory trends
def predict_inventory_trend(sales, stock_level, lead_time, seasonality_factor):
    prediction = model.predict([[sales, stock_level, lead_time, seasonality_factor]])
    return f"Predicted Sales for Next Period: {prediction[0]:.2f}"

# Example Usage
print(predict_inventory_trend(200, 500, 5, 1.2))

# Close database connection
conn.close()
