import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('heart.csv')

# Replace 'target' with the actual target column if needed
if 'target' not in df.columns:
    print("Available columns:", df.columns)
    raise Exception("Check the target column name")

X = df.drop('target', axis=1)
y = df['target']

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model
pickle.dump((model, scaler), open('model.pkl', 'wb'))
print("Model trained and saved as model.pkl")
