import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
import pickle

print("Loading dataset...")
df = pd.read_csv("Crop_recommendation.csv")

# Encode labels
le = LabelEncoder()
y  = le.fit_transform(df['label'])

# Feature columns — order MUST match the order used in app.py predict()
FEATURE_COLS = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
X = df[FEATURE_COLS]

print(f"\nFeature ranges in dataset:")
for col in X.columns:
    print(f"  {col:15s}: {X[col].min():.1f} – {X[col].max():.1f}  (mean={X[col].mean():.1f})")

print(f"\nTraining label distribution (22 crops):")
print(df['label'].value_counts().to_string())

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining RandomForest model...")
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Evaluate
test_acc  = model.score(X_test, y_test)
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"\n✅ Test Accuracy:  {test_acc * 100:.2f}%")
print(f"✅ CV Accuracy:    {cv_scores.mean() * 100:.2f}% ± {cv_scores.std() * 100:.2f}%")

# ── SANITY CHECK: verify model does NOT always predict muskmelon ──
print("\n🔍 Sanity check — predictions for typical Indian city inputs:")
test_cases = [
    # city hint,  N,   P,   K,  temp,  hum,   ph,  rain(annual)
    ("Hyderabad", 60,  40,  40, 28.0,  68,   7.2,  812),
    ("Kolkata",   70,  50,  45, 29.0,  82,   6.4, 1600),
    ("Punjab",    70,  60,  55, 18.0,  55,   7.0,  720),
    ("Kerala",    75,  55,  58, 27.0,  88,   5.7, 3100),
    ("Rajasthan", 38,  28,  25, 33.0,  40,   8.2,  280),
]
for label, n, p, k, t, h, ph_val, r in test_cases:
    feat = np.array([[n, p, k, t, h, ph_val, r]])
    probs = model.predict_proba(feat)[0]
    top3  = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:3]
    preds = ", ".join([f"{le.classes_[i]}({p*100:.0f}%)" for i, p in top3])
    print(f"  {label:12s}: {preds}")

# Save
with open("xgb_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print(f"\n🎉 Saved: xgb_model.pkl and encoder.pkl")
print(f"   Crops supported ({len(le.classes_)}): {list(le.classes_)}")
print(f"   Feature order: {FEATURE_COLS}")
