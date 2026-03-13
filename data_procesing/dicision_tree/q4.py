import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn import tree
import matplotlib.pyplot as plt

# 1. Data inladen en voorbereiden
with open('data/questions/q4_cumulative_lifestyle.json') as f:
    data = json.load(f)
df = pd.DataFrame(data)

# Verwijder dubbele 'PhysActivity' in je lijst
features = ['Unhealthy_Lifestyle_Score', 'Fruits',  'Smoker', 'PhysActivity', 'HvyAlcoholConsump'] 
X = df[features]
y = df['Diabetes_binary']

# 2. Data balanceren (Under-sampling)
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

# 4. Decision Tree Model
# We beperken de diepte (max_depth) om het model leesbaar te houden voor de onderzoeksvraag
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

# 5. Visualisatie van de interacties
print("--- Beslisregels (Interactie tussen factoren) ---")
tree_rules = export_text(model, feature_names=features)
print(tree_rules)

# 6. Feature Importance (Welke factor weegt het zwaarst?)
importance = pd.DataFrame({'Factor': features, 'Belangrijkheid': model.feature_importances_})
print("\n--- Belangrijkste Voorspellers ---")
print(importance.sort_values(by='Belangrijkheid', ascending=False))

# 7. Model Evaluatie
score = model.score(X_test, y_test)
print(f"\nModel Nauwkeurigheid: {score:.2f}")

# Optioneel: Visualiseer de boom (als je een UI hebt)
# plt.figure(figsize=(20,10))
# tree.plot_tree(model, feature_names=features, class_names=['Gezond', 'Diabetes'], filled=True)
# plt.show()

# conclusie: Doordat 2 van de lifestyle factoren bijna geen invloed hebben op het hebben van diabetes. 
# is de lifestyle scrore ook geen goede indicatie. 
