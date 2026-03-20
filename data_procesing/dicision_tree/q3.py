import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report, confusion_matrix

def run_q3():

    # 1. JSON inladen
    with open('data/questions/q3_socioeconomic.json') as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    # 2. Variabelen en Balanceren
    features = ['Income', 'Education'] 
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
    # We zetten max_depth op 3 om een heldere, interpreteerbare boom te krijgen
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    # 5. Resultaten en Beslispaden
    print("--- De Beslisregels van het Model ---")
    # Dit laat zien hoe het model keuzes maakt op basis van de data
    tree_rules = export_text(model, feature_names=features)
    print(tree_rules)

    # 6. Feature Importance
    importance = pd.DataFrame({'Factor': features, 'Belangrijkheid': model.feature_importances_})
    print("\n--- Impact Analyse ---")
    print(importance.sort_values(by='Belangrijkheid', ascending=False))

    # 7. Evaluatie
    y_pred = model.predict(X_test)
    print("\n--- Model Prestaties ---")
    print(classification_report(y_test, y_pred))

    # Conclusie: Op basis van de uitslag van de dicision tree is het inkomen een veel betere voorspeller van het opleidingsniveau.
    # en met een naukeurigheid van 58% is het beter dan het gokken op 50/50.
    # Ook ligt de split van het inkomen niet direct in het midden maar tussen 6 en 7. 
    # Dit zegt dat je volgens dat als je minder dan 50.000 euro verdiend je direct in de diabetes groep word geplaatst.