import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report, confusion_matrix

def q5_process_nb():
    # 1. Data inladen
    with open('data/questions/q5_cholesterol_bloodpressure.json') as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    # 2. Features voor de medische analyse
    features = ['HighBP', 'HighChol', 'HighBP_x_HighChol'] 
    X = df[features]
    y = df['Diabetes_binary']

    # 3. Balanceren (Under-sampling) voor eerlijke kansberekening
    df_diabetes = df[df['Diabetes_binary'] == 1]
    df_gezond = df[df['Diabetes_binary'] == 0].sample(len(df_diabetes), random_state=42)
    df_balanced = pd.concat([df_gezond, df_diabetes])

    X_bal = df_balanced[features]
    y_bal = df_balanced['Diabetes_binary']

    # 4. Model Training
    X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.2, random_state=42)
    model = BernoulliNB()
    model.fit(X_train, y_train)

    # 5. Synergie Analyse: Kansen per groep berekenen
    # We kijken hoe vaak Diabetes voorkomt bij elke specifieke combinatie
    synergie_check = df_balanced.groupby(['HighBP', 'HighChol'])['Diabetes_binary'].mean() * 100

    print("--- Risico Analyse (Kans op Diabetes per combinatie) ---")
    print(f"Geen van beide:        {synergie_check[0,0]:.2f}%")
    print(f"Alleen Hoge Bloeddruk: {synergie_check[1,0]:.2f}%")
    print(f"Alleen Hoog Cholesterol: {synergie_check[0,1]:.2f}%")
    print(f"BEIDE (Interactie):    {synergie_check[1,1]:.2f}%")

    # 6. Model Evaluatie
    y_pred = model.predict(X_test)
    print("\n--- Model Prestaties ---")
    print(classification_report(y_test, y_pred))

    # 7. Relatieve Risico Factor (Odds)
    risk_both = synergie_check[1,1]
    risk_none = synergie_check[0,0]
    ratio = risk_both / risk_none

    print(f"\nConclusie: Iemand met zowel HighBP als HighChol heeft een {ratio:.2f}x hogere kans")
    print("op diabetes dan iemand met gezonde waarden in deze dataset.")