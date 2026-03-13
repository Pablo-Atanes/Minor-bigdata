import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report, accuracy_score

# 1. Volledige dataset inladen
# Zorg dat je hier de file inlaadt die ALLES bevat (lifestyle, socio, etc.)
with open('data/questions/full_set.json') as f: 
    data = json.load(f)
df = pd.DataFrame(data)

# 2. Voorbereiden (Alle mogelijke features)
X = df.drop('Diabetes_binary', axis=1)
y = df['Diabetes_binary']

# Data balanceren voor een eerlijke score
df_gezond = df[df['Diabetes_binary'] == 0]
df_diabetes = df[df['Diabetes_binary'] == 1]
df_balanced = pd.concat([df_gezond.sample(len(df_diabetes), random_state=42), df_diabetes])

X_bal = df_balanced.drop('Diabetes_binary', axis=1)
y_bal = df_balanced['Diabetes_binary']

X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.2, random_state=42)

# 3. Stepwise Selection (RFE)
# We gebruiken een Random Forest om de belangrijkste features te bepalen
estimator = RandomForestClassifier(n_estimators=100, random_state=42)
# We selecteren de 10 beste features (pas dit aantal aan naar wens)
selector = RFE(estimator, n_features_to_select=22, step=1)
selector = selector.fit(X_train, y_train)

# 4. Welke features zijn gekozen?
selected_features = X_train.columns[selector.support_]
print("--- Geselecteerde Features via Stepwise Methode ---")
print(selected_features.tolist())

# 5. Model trainen met de beste features
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

final_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
final_model.fit(X_train_selected, y_train)

# 6. Resultaat
y_pred = final_model.predict(X_test_selected)
print(f"\nMaximale Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\n--- Eindverslag ---")
print(classification_report(y_test, y_pred))

# Als het model zelf de 10 beste variabele kan pakken komt hij op de volgende lijst en haalt hij een nauwkeurigheid van 73%.
# ['Fruits', 'HighBP', 'HighChol', 'GenHlth', 'Age', 'Education', 'Income', 'MentHlth', 'PhysHlth', 'BMI']

# Maximale Accuracy: 0.7342

# --- Eindverslag ---
#              precision    recall  f1-score   support

#           0       0.75      0.70      0.72      7012
#           1       0.72      0.77      0.74      7027

#     accuracy                           0.73     14039
#    macro avg       0.74      0.73      0.73     14039
# weighted avg       0.74      0.73      0.73     14039

# bij 15 features komt het model tot de volgende resultaten:

# Maximale Accuracy: 0.7373

# --- Eindverslag ---
#               precision    recall  f1-score   support

#            0       0.76      0.70      0.73      7012
#            1       0.72      0.77      0.75      7027

#     accuracy                           0.74     14039
#    macro avg       0.74      0.74      0.74     14039
# weighted avg       0.74      0.74      0.74     14039

# met alle variabelen komt het tot het volgende resultaat:
# Maximale Accuracy: 0.7401

# --- Eindverslag ---
#               precision    recall  f1-score   support

#            0       0.76      0.70      0.73      7012
#            1       0.72      0.78      0.75      7027

#     accuracy                           0.74     14039
#    macro avg       0.74      0.74      0.74     14039
# weighted avg       0.74      0.74      0.74     14039

# Conclusie: het hoogste resultaat dat ik kan halen is 
