import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def q3_process_nb():
    # 1. JSON inladen
    with open('data/questions/q3_socioeconomic.json') as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    # 2. Variabelen definiëren
    # We gebruiken nu een lijst met meerdere lifestyle kenmerken
    features = ['Income', 'Education'] 
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


    model = MultinomialNB()
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

    # 1. Groepeer de data op basis van diabetes status en bereken het gemiddelde
    analyse = df_balanced.groupby('Diabetes_binary')[['Income', 'Education']].mean()

    # 2. Maak het leesbaar
    print("\n--- Sociaal-Economische Analyse (Gemiddelden) ---")
    print(f"Gemiddeld inkomensniveau (1-8):")
    print(f"  - Gezonde groep:  {analyse.loc[0, 'Income']:.2f}")
    print(f"  - Diabetes groep: {analyse.loc[1, 'Income']:.2f}")

    print(f"\nGemiddeld educatieniveau (1-6):")
    print(f"  - Gezonde groep:  {analyse.loc[0, 'Education']:.2f}")
    print(f"  - Diabetes groep: {analyse.loc[1, 'Education']:.2f}")

    # 3. Bereken het procentuele verschil
    income_diff = ((analyse.loc[1, 'Income'] - analyse.loc[0, 'Income']) / analyse.loc[0, 'Income']) * 100
    print(f"\nConclusie: De diabetesgroep heeft een gemiddeld inkomen dat {income_diff:.2f}% lager/hoger ligt.")

    educatie_diff = ((analyse.loc[1, 'Education'] - analyse.loc[0, 'Education']) / analyse.loc[0, 'Education']) * 100
    print(f"\nConclusie: De diabetesgroep heeft een gemiddeld {educatie_diff:.2f}% lagere/hogere opleiding.")


# conclussie:  met een gemiddelde score van 57% goed zitten er dus wel degelijk verschillen
# dit is ook terug te zien in het in de kansberekeningen waar je wel degelijk zien dat een lager inkomen en 
# een lagere opleiding een hogere kans op diabetes geeft


