import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report, confusion_matrix

# 1. JSON inladen
with open('data/questions/q1_fruit_diabetes.json') as f:
    data = json.load(f)
df = pd.DataFrame(data)

# 2. Variabelen & Balanceren
features = ['Fruits'] 
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

# 4. Model trainen (Decision Tree)
# Omdat we maar 1 feature hebben, houden we de boom ondiep
model = DecisionTreeClassifier(max_depth=2, random_state=42)
model.fit(X_train, y_train)

# 5. Beslisregels en Analyse
print("--- Beslisregel voor Fruit ---")
tree_rules = export_text(model, feature_names=features)
print(tree_rules)

# 6. Statistiek: Hoe vaak komt fruit voor per groep?
analyse = df_balanced.groupby('Diabetes_binary')['Fruits'].mean() * 100
print("\n--- Percentage mensen dat dagelijks fruit eet ---")
print(f"Gezonde groep:  {analyse[0]:.2f}%")
print(f"Diabetes groep: {analyse[1]:.2f}%")

# 7. Evaluatie
y_pred = model.predict(X_test)
print("\n--- Model Prestaties ---")
print(classification_report(y_test, y_pred))

# Conclusie: net zoals naive bayes is het dicision tree model niet in staat om hier een voorspelling voor te doen.