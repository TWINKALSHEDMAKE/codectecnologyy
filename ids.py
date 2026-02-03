import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ==============================
# 1. Load Dataset
# ==============================
# Dataset ka naam same rakha hai jo tumhare paas hai
data = pd.read_csv(
    "dataset/kddcup.data_10_percent_corrected",
    header=None
)

# ==============================
# 2. Features & Label
# ==============================
X = data.iloc[:, :-1]   # saare columns except last
y = data.iloc[:, -1]    # last column = attack label

# ==============================
# 3. Categorical → Numeric
# ==============================
X = pd.get_dummies(X)

# 🔴 IMPORTANT FIX (tumhara error yahin se aa raha tha)
X.columns = X.columns.astype(str)
X = X.astype(float)

# ==============================
# 4. Train-Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==============================
# 5. Model (Random Forest)
# ==============================
model = RandomForestClassifier(
    n_estimators=30,
    random_state=42
)

# ==============================
# 6. Train Model
# ==============================
model.fit(X_train, y_train)

# ==============================
# 7. Prediction & Accuracy
# ==============================
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print("IDS Model Accuracy:", accuracy)
