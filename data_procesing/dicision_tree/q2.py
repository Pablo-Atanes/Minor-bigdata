import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report
import numpy as np

# 1. JSON inladen
with open('data/questions/q2_lifestyle_patterns.json') as f:
    data = json.load(f)
df = pd.DataFrame(data)

# 2. Variabelen & Balanceren
features = ['Smoker', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump'] 
X = df[features]
y = df['Diabetes_binary']

df_gezond = df[df['Diabetes_binary'] == 0]
df_diabetes = df[df['Diabetes_binary'] == 1]
df_gezond_under = df_gezond.sample(len(df_diabetes), random_state=42)
df_balanced = pd.concat([df_gezond_under, df_diabetes], axis=0)

X_balanced = df_balanced[features]
y_balanced = df_balanced['Diabetes_binary']

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=42
)

# 4. Model trainen
# We gebruiken een max_depth van 4 om de combinaties goed te zien
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

# 5. De Patronen (Beslisregels)
print("--- Lifestyle Combinaties & Diabetes Risico ---")
# De tree laat zien welke factoren samen 'class 1' (diabetes) triggeren
tree_rules = export_text(model, feature_names=features)
print(tree_rules)

# 6. Feature Importance
importance = pd.DataFrame({'Factor': features, 'Belangrijkheid': model.feature_importances_})
print("\n--- Impact per Factor ---")
print(importance.sort_values(by='Belangrijkheid', ascending=False))

# 7. Model Evaluatie
y_pred = model.predict(X_test)
print("\n--- Rapport ---")
print(classification_report(y_test, y_pred))

# Conclusie: De resultaten van deze voorspelling en de nauwkeurigheid. Zijn erg vergelijkbaar met die van onderzoeksvraag 4. 
# Hier heb je nog een datapunt erbij maar zoals in de resultaten te zien gebruikt het de variabele "veggies" eigenlijk niet.