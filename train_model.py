import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
df = pd.read_csv("data/career_data.csv")

# Encode categorical data
interest_encoder = LabelEncoder()
career_encoder = LabelEncoder()

df["interest"] = interest_encoder.fit_transform(df["interest"])
df["career"] = career_encoder.fit_transform(df["career"])

X = df.drop("career", axis=1)
y = df["career"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Save everything
pickle.dump(model, open("model/career_model.pkl", "wb"))
pickle.dump(interest_encoder, open("model/interest_encoder.pkl", "wb"))
pickle.dump(career_encoder, open("model/career_encoder.pkl", "wb"))
pickle.dump(accuracy, open("model/accuracy.pkl", "wb"))

print("✅ Model trained successfully!")
print(f"📊 Model Accuracy: {accuracy*100:.2f}%")
