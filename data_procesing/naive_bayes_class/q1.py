import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB # Beter voor 0/1 data zoals Fruits
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


def nb_process_q1():
    # JSON inladen vanuit de json die alleen de data bevat voor deze onderzoeksvraag
    with open('data/questions/q1_fruit_diabetes.json') as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    # De feature die voor deze onderzoeksvraag van belang is
    features = ['Fruits'] 
    X = df[features]

    # de target die voor alle onderzoeks vragen centraal zal staan 
    y = df['Diabetes_binary']

    # print vooral als er mogelijk nog fouten en nulwaarden in de dataset zouden zitten
    print(X.info())

    # hier word de data opgesplitst in 2 groepen, de wel en de niet diabetes
    # dit is belangrijk omdat je zo naar de verschillen kunt kijken binnen de 2 groepen van hoevaak het eten van fruit voorkomt
    df_gezond = df[df['Diabetes_binary'] == 0]
    df_diabetes = df[df['Diabetes_binary'] == 1]

    # zorg dat je een even grote groep van wel en niet diabetes krijgt
    # Dit heb ik gedaan door een random groep van de lengte van de diabetes groep te pakken in de gezonde groep
    # hierdoor gaan het algoritme op zoek naar andere variabele die het verschil kunnen maken
    # de test word op die manier nauwkeuriger uitgevoerd
    df_gezond_under = df_gezond.sample(len(df_diabetes), random_state=42)

    # Voeg de nieuwe groep gezond weer bij diabetes groep
    df_balanced = pd.concat([df_gezond_under, df_diabetes], axis=0)

    X_balanced = df_balanced[features]
    y_balanced = df_balanced['Diabetes_binary']

    # hier is ingesteld wat de train en de test split word, 20% van de dataset word bewaard om later mee te testen
    # de overige 80% word gebruikt voor het trainen van het model
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

    # het model dat het best werkt met naive bayes is bernoulli
    model = BernoulliNB()

    # hier word het model getrained
    model.fit(X_train, y_train)

    # hier worden de resultaten omgezet in percentages zodat er een kans percentage uit komt
    probs = model.feature_log_prob_
    probabilities = np.exp(probs)

    # create the predictions
    y_pred = model.predict(X_test)

    # Een tabel die laat zien waar het model de mist in ging
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # geeft je precies de scores die het heeft behaald met de voorspellingen
    print("\nUitgebreid rapport:")
    print(classification_report(y_test, y_pred))

    print(f"Kans op fruit eten als je GEZOND bent: {probabilities[0][0]*100:.2f}%")
    print(f"Kans op fruit eten als je DIABETES hebt: {probabilities[1][0]*100:.2f}%")


# Conclussie:   naive bayes is niet in staat om te voorspellen of iemand diabetes heeft op basis van het eten van fruit
