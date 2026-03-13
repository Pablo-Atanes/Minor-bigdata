import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def q2_process_nb():
    # 1. JSON inladen
    with open('data/questions/q2_lifestyle_patterns.json') as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    # 2. Variabelen definiëren
    # We gebruiken nu een lijst met meerdere lifestyle kenmerken
    features = ['Smoker', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump'] 
    X = df[features]
    y = df['Diabetes_binary']

    # Check op data types en missende waarden
    print("--- Data Info ---")
    print(df[features + ['Diabetes_binary']].info())

    # 3. Data balanceren (Under-sampling)
    # Dit is cruciaal om te voorkomen dat het model alleen de meerderheid (gezonde mensen) voorspelt.
    df_gezond = df[df['Diabetes_binary'] == 0]
    df_diabetes = df[df['Diabetes_binary'] == 1]

    # Maak de gezonde groep even groot als de diabetes groep
    df_gezond_under = df_gezond.sample(len(df_diabetes), random_state=42)
    df_balanced = pd.concat([df_gezond_under, df_diabetes], axis=0)

    # Variabelen opnieuw toewijzen op basis van de GEBALANCEERDE dataset
    X_balanced = df_balanced[features]
    y_balanced = df_balanced['Diabetes_binary']

    # 4. Train/Test Split
    # We trainen op 80% van de gebalanceerde data en testen op de overige 20%
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.2, random_state=42
    )

    # 5. Model trainen
    # BernoulliNB is ideaal voor binaire (0/1) features
    model = BernoulliNB()
    model.fit(X_train, y_train)

    # 6. Resultaten en Kansberekening
    # We halen de logaritmische kansen op en zetten deze om naar normale percentages
    probs = model.feature_log_prob_
    probabilities = np.exp(probs)

    # Maak een gebundeld overzicht van de kansen per lifestyle factor
    # probabilities[0] zijn de kansen voor de gezonde groep, [1] voor de diabetes groep
    chance_df = pd.DataFrame({
        'Lifestyle Factor': features,
        'Kans bij Gezond (%)': np.round(probabilities[0] * 100, 2),
        'Kans bij Diabetes (%)': np.round(probabilities[1] * 100, 2)
    })

    # Voeg een verschilkolom toe om de impact te zien
    chance_df['Verschil (pp)'] = np.round(chance_df['Kans bij Diabetes (%)'] - chance_df['Kans bij Gezond (%)'], 2)

    # 7. Model Evaluatie
    y_pred = model.predict(X_test)

    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(y_test, y_pred))

    print("\n--- Uitgebreid Rapport ---")
    print(classification_report(y_test, y_pred))

    print("\n--- Lifestyle Patroon Analyse ---")
    print(chance_df.to_string(index=False))


# conclussie:  het model kan moeilijk inschatten of een persoon wel of geen diabetes
# Maar aan de kansberekeningen is te zien dat een gezonde levensstyle wel degelijk helpt met het voorkomen van diabetes
