import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def q4_process_nb():
    # 1. JSON inladen
    with open('data/questions/q4_cumulative_lifestyle.json') as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    # 2. Variabelen definiëren
    # We gebruiken nu een lijst met meerdere lifestyle kenmerken
    features = ['Unhealthy_Lifestyle_Score', 'Fruits', 'Smoker', 'PhysActivity', 'PhysActivity', 'HvyAlcoholConsump'] 
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
    analyse = df_balanced.groupby('Diabetes_binary')[['Unhealthy_Lifestyle_Score']].mean()

    # 3. Bereken het procentuele verschil
    unhealthy_lifestyle_diff = ((analyse.loc[1, 'Unhealthy_Lifestyle_Score'] - analyse.loc[0, 'Unhealthy_Lifestyle_Score']) / analyse.loc[0, 'Unhealthy_Lifestyle_Score']) * 100
    print(f"\nConclusie: De diabetesgroep heeft een gemiddeld lifestyle score dat {unhealthy_lifestyle_diff:.2f}% lager/hoger ligt.")

    # Bereken het percentage mensen met diabetes per lifestyle score
    score_impact = df_balanced.groupby('Unhealthy_Lifestyle_Score')['Diabetes_binary'].value_counts(normalize=True).unstack() * 100

    # Hernoem kolommen voor de duidelijkheid
    score_impact.columns = ['Gezond (%)', 'Diabetes (%)']

    print("\n--- Verband: Lifestyle Score vs Diabetes Kans ---")
    print(score_impact.round(2))

# Conclusie trekken op basis van de data
# correlation = df_balanced['Unhealthy_Lifestyle_Score'].corr(df_balanced['Diabetes_binary'])
# print(f"\nDe correlatie-coëfficiënt tussen de score en diabetes is: {correlation:.4f}")

# conclussie:  
# eerst wilde ik alleen met de 'Unhealthy_Lifestyle_Score' werken, 
# dit gaf echter niet de gewenste resultaten omdat het model te weinig onderschied kon maken
# toen ik de velden toevoegde waar de Unhealthy_Lifestyle_Score op is gebaseerd kon het model wel voorspellingen doen
# en zat het gemiddeld 57% van de gevallen goed
# vervolgens heb ik in beeld gebracht wat de kans op diabetes is per score en daaruit blijkt dat als je alleen naar de score kijkt 
# dat er helemaal geen verband te trekken is.
# de conclussie die ik hieruit kan trekken is is dat een aantal velden die in de unhealthy lifestyle score zijn opgenomen niet goed werken
# en als het model ze dus los van elkaar ook heeft dat het toch vaker een goede voorspelling kan doen dan een foute.



