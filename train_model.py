import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load dataset
df = pd.read_csv('heart.csv')

# Print columns
print("Columns:", df.columns)

# Drop unnecessary columns
df.drop(['id', 'dataset'], axis=1, inplace=True)

# Drop rows with missing values
df.dropna(inplace=True)

# Encode categorical columns if needed
if df['sex'].dtype == object:
    df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})

# Convert target to binary
df['num'] = df['num'].apply(lambda x: 1 if x > 0 else 0)

# Separate features and target
X = df.drop('num', axis=1)
y = df['num']

# Encode other categorical columns if they contain text
for col in X.columns:
    if X[col].dtype == object:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Calculate accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("📊 Accuracy on test set:", accuracy)


# Save model and scaler
pickle.dump((model, scaler), open('model.pkl', 'wb'))
print("✅ Model trained and saved as model.pkl")

from sklearn.metrics import accuracy_score

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy of the model: {accuracy:.2f}")

