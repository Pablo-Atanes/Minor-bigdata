import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# 1. Data inladen
with open('data/questions/q5_cholesterol_bloodpressure.json') as f:
    data = json.load(f)
df = pd.DataFrame(data)

# 2. Features (we kunnen de interactie-term 'HighBP_x_HighChol' weghalen 
# omdat de boom zelf de interactie ontdekt, maar we laten hem voor nu staan)
features = ['HighBP', 'HighChol', 'HighBP_x_HighChol'] 
X = df[features]
y = df['Diabetes_binary']

# 3. Balanceren (Under-sampling)
df_diabetes = df[df['Diabetes_binary'] == 1]
df_gezond = df[df['Diabetes_binary'] == 0].sample(len(df_diabetes), random_state=42)
df_balanced = pd.concat([df_gezond, df_diabetes])

X_bal = df_balanced[features]
y_bal = df_balanced['Diabetes_binary']

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.2, random_state=42)

# 5. Model trainen
# We houden de boom ondiep (max_depth=3) zodat hij leesbaar blijft
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# 6. De Boom Visualiseren (Tekstueel)
tree_rules = export_text(model, feature_names=features)
print("--- Beslisboom Structuur ---")
print(tree_rules)

# 7. Evaluatie
y_pred = model.predict(X_test)
print("\n--- Model Prestaties ---")
print(classification_report(y_test, y_pred))

# # 8. Visuele weergave (Optioneel, opent een scherm)
# plt.figure(figsize=(12,8))
# plot_tree(model, feature_names=features, class_names=['Gezond', 'Diabetes'], filled=True)
# plt.show()


scenarios = [
    [0, 0, 0], # Geen van beide
    [1, 0, 0], # Alleen HighBP
    [0, 1, 0], # Alleen HighChol
    [1, 1, 1]  # Beide (inclusief interactie-term)
]

# Bereken de kans (probability) voor elk scenario
probabilities = model.predict_proba(scenarios)

print("--- Kans op Diabetes per scenario ---")
print(f"Geen van beide:          {probabilities[0][1]*100:.1f}%")
print(f"Alleen Hoge Bloeddruk:   {probabilities[1][1]*100:.1f}%")
print(f"Alleen Hoog Cholesterol: {probabilities[2][1]*100:.1f}%")
print(f"BEIDE:                   {probabilities[3][1]*100:.1f}%")