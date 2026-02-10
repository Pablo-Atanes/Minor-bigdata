import numpy as np
import pandas as pd
import statsmodels.api as sm


TARGET = "Diabetes_binary"


def _run_logistic(df, predictors, title):
    """Voer een logistische regressie uit en toon resultaten.

    Parameters:
        df: DataFrame met data
        predictors: lijst van predictorvariabelen
        title: titel voor de output
    Returns:
        Het gefitte model (GLMResultsWrapper)
    """
    X = sm.add_constant(df[predictors])
    y = df[TARGET]
    model = sm.Logit(y, X).fit(disp=0)

    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    print(f"  Type: {'Enkelvoudige' if len(predictors) == 1 else 'Meervoudige'} logistische regressie")
    print(f"  Observaties: {int(model.nobs):,}")
    print(f"  Pseudo R-squared: {model.prsquared:.4f}")
    print(f"  Log-Likelihood: {model.llf:.1f}")
    print(f"  AIC: {model.aic:.1f}")

    # Coefficienten tabel
    results = pd.DataFrame({
        "Coefficient": model.params,
        "Std. Error": model.bse,
        "z-waarde": model.tvalues,
        "p-waarde": model.pvalues,
        "Odds Ratio": np.exp(model.params),
        "OR 95% CI laag": np.exp(model.conf_int()[0]),
        "OR 95% CI hoog": np.exp(model.conf_int()[1]),
    })
    print(f"\n{results.to_string()}\n")

    # Significantie per predictor
    for pred in predictors:
        coef = model.params[pred]
        p = model.pvalues[pred]
        odds = np.exp(coef)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."

        if p < 0.05:
            richting = "verhoogt" if coef > 0 else "verlaagt"
            print(f"  {pred}: {richting} de kans op diabetes (OR={odds:.3f}, p={p:.4f}) {sig}")
        else:
            print(f"  {pred}: geen significant effect (OR={odds:.3f}, p={p:.4f}) {sig}")

    print()
    return model


# -- Regressies per onderzoeksvraag --

def regression_q1(q1):
    """Q1: Fruit Consumptie & Diabetes — 1 enkelvoudige regressie."""
    print("\n" + "#"*70)
    print("  Q1: FRUIT CONSUMPTIE & DIABETES")
    print("#"*70)

    model = _run_logistic(q1, ["Fruits"],
                          "Enkelvoudig: Fruits → Diabetes")

    odds = np.exp(model.params["Fruits"])
    p = model.pvalues["Fruits"]
    print("  Interpretatie:")
    if p < 0.05:
        if odds < 1:
            pct = (1 - odds) * 100
            print(f"  Dagelijks fruit eten verlaagt de kans op diabetes met {pct:.1f}%")
            print(f"  (odds ratio = {odds:.3f}, p = {p:.4f}).")
        else:
            pct = (odds - 1) * 100
            print(f"  Dagelijks fruit eten verhoogt de kans op diabetes met {pct:.1f}%")
            print(f"  (odds ratio = {odds:.3f}, p = {p:.4f}).")
    else:
        print(f"  Geen significant verband gevonden (p = {p:.4f}).")
    print()
    return {"enkelvoudig_fruits": model}


def regression_q2(q2):
    """Q2: Lifestyle Patronen & Diabetes — 5 enkelvoudig + 1 meervoudig."""
    print("\n" + "#"*70)
    print("  Q2: LIFESTYLE PATRONEN & DIABETES")
    print("#"*70)

    factors = ["Smoker", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump"]
    models = {}

    # 5 enkelvoudige regressies
    for f in factors:
        m = _run_logistic(q2, [f], f"Enkelvoudig: {f} → Diabetes")
        models[f"enkelvoudig_{f}"] = m

    # 1 meervoudige regressie
    m = _run_logistic(q2, factors,
                      "Meervoudig: Alle lifestyle factors → Diabetes")
    models["meervoudig"] = m

    print("  Interpretatie meervoudige regressie:")
    print("  Wanneer alle lifestyle factoren tegelijk worden meegenomen:")
    for f in factors:
        odds = np.exp(m.params[f])
        p = m.pvalues[f]
        sig = "significant" if p < 0.05 else "niet significant"
        richting = "risicoverhogend" if odds > 1 else "beschermend"
        print(f"    - {f}: OR={odds:.3f} ({richting}, {sig}, p={p:.4f})")
    print()
    return models


def regression_q3(q3):
    """Q3: Socio-economische Status & Diabetes — 2 enkelvoudig + 1 meervoudig."""
    print("\n" + "#"*70)
    print("  Q3: SOCIO-ECONOMISCHE STATUS & DIABETES")
    print("#"*70)

    models = {}

    # Enkelvoudig: Income
    m = _run_logistic(q3, ["Income"],
                      "Enkelvoudig: Income → Diabetes")
    models["enkelvoudig_income"] = m

    odds_inc = np.exp(m.params["Income"])
    print("  Interpretatie:")
    print(f"  Per stap omhoog in inkomenscategorie verandert de odds op diabetes")
    print(f"  met factor {odds_inc:.3f} (= {(odds_inc-1)*100:+.1f}% per categorie).")
    print()

    # Enkelvoudig: Education
    m = _run_logistic(q3, ["Education"],
                      "Enkelvoudig: Education → Diabetes")
    models["enkelvoudig_education"] = m

    odds_edu = np.exp(m.params["Education"])
    print("  Interpretatie:")
    print(f"  Per stap omhoog in opleidingsniveau verandert de odds op diabetes")
    print(f"  met factor {odds_edu:.3f} (= {(odds_edu-1)*100:+.1f}% per niveau).")
    print()

    # Meervoudig: Income + Education
    m = _run_logistic(q3, ["Income", "Education"],
                      "Meervoudig: Income + Education → Diabetes")
    models["meervoudig"] = m

    print("  Interpretatie meervoudige regressie:")
    print("  Wanneer inkomen en opleiding tegelijk worden meegenomen:")
    for var in ["Income", "Education"]:
        odds = np.exp(m.params[var])
        p = m.pvalues[var]
        sig = "significant" if p < 0.05 else "niet significant"
        print(f"    - {var}: OR={odds:.3f} ({sig}, p={p:.4f})")
    print()
    return models


def regression_q4(q4):
    """Q4: Cumulatief Effect Lifestyle — 1 enkelvoudig + 1 meervoudig."""
    print("\n" + "#"*70)
    print("  Q4: CUMULATIEF EFFECT LIFESTYLE FACTORS")
    print("#"*70)

    models = {}

    # Enkelvoudig: Unhealthy_Lifestyle_Score
    m = _run_logistic(q4, ["Unhealthy_Lifestyle_Score"],
                      "Enkelvoudig: Unhealthy Lifestyle Score → Diabetes")
    models["enkelvoudig_score"] = m

    odds_score = np.exp(m.params["Unhealthy_Lifestyle_Score"])
    print("  Interpretatie:")
    print(f"  Per punt stijging in de Unhealthy Lifestyle Score (0-4)")
    print(f"  stijgt de odds op diabetes met factor {odds_score:.3f}")
    print(f"  (= +{(odds_score-1)*100:.1f}% per extra risicofactor).")
    print()

    # Meervoudig: individuele factoren
    indiv = ["Smoker", "PhysActivity", "Fruits", "HvyAlcoholConsump"]
    m = _run_logistic(q4, indiv,
                      "Meervoudig: Individuele lifestyle factors → Diabetes")
    models["meervoudig"] = m

    print("  Interpretatie meervoudige regressie:")
    print("  Door de individuele factoren los te modelleren zien we welke")
    print("  het sterkst bijdragen aan het cumulatieve risico:")
    for f in indiv:
        odds = np.exp(m.params[f])
        p = m.pvalues[f]
        sig = "significant" if p < 0.05 else "niet significant"
        richting = "risicoverhogend" if odds > 1 else "beschermend"
        print(f"    - {f}: OR={odds:.3f} ({richting}, {sig}, p={p:.4f})")
    print()
    return models


def regression_q5(q5):
    """Q5: Cholesterol + Bloeddruk & Diabetes — 2 enkelvoudig + 1 meervoudig."""
    print("\n" + "#"*70)
    print("  Q5: CHOLESTEROL + BLOEDDRUK & DIABETES")
    print("#"*70)

    models = {}

    # Enkelvoudig: HighBP
    m = _run_logistic(q5, ["HighBP"],
                      "Enkelvoudig: HighBP → Diabetes")
    models["enkelvoudig_highbp"] = m

    odds_bp = np.exp(m.params["HighBP"])
    print("  Interpretatie:")
    print(f"  Hoge bloeddruk verhoogt de odds op diabetes met factor {odds_bp:.3f}")
    print(f"  (= +{(odds_bp-1)*100:.1f}%).")
    print()

    # Enkelvoudig: HighChol
    m = _run_logistic(q5, ["HighChol"],
                      "Enkelvoudig: HighChol → Diabetes")
    models["enkelvoudig_highchol"] = m

    odds_chol = np.exp(m.params["HighChol"])
    print("  Interpretatie:")
    print(f"  Hoog cholesterol verhoogt de odds op diabetes met factor {odds_chol:.3f}")
    print(f"  (= +{(odds_chol-1)*100:.1f}%).")
    print()

    # Meervoudig: HighBP + HighChol + interactie
    m = _run_logistic(q5, ["HighBP", "HighChol", "HighBP_x_HighChol"],
                      "Meervoudig: HighBP + HighChol + Interactie → Diabetes")
    models["meervoudig"] = m

    p_inter = m.pvalues["HighBP_x_HighChol"]
    odds_inter = np.exp(m.params["HighBP_x_HighChol"])
    print("  Interpretatie meervoudige regressie met interactieterm:")
    if p_inter < 0.05:
        if odds_inter > 1:
            print(f"  De interactieterm is significant (p={p_inter:.4f}, OR={odds_inter:.3f}).")
            print("  Dit bevestigt een synergistisch effect: de combinatie van hoge bloeddruk")
            print("  en hoog cholesterol verhoogt het diabetes risico meer dan de som van")
            print("  de individuele effecten.")
        else:
            print(f"  De interactieterm is significant (p={p_inter:.4f}, OR={odds_inter:.3f}).")
            print("  Het gecombineerde effect is kleiner dan verwacht op basis van de")
            print("  individuele effecten (sub-additief).")
    else:
        print(f"  De interactieterm is niet significant (p={p_inter:.4f}, OR={odds_inter:.3f}).")
        print("  Er is geen bewijs voor een synergistisch effect. De effecten van hoge")
        print("  bloeddruk en hoog cholesterol op diabetes zijn onafhankelijk (additief).")
    print()
    return models


def regression_all(questions):
    """Voer alle regressieanalyses uit voor alle onderzoeksvragen."""
    all_models = {}
    all_models["q1"] = regression_q1(questions["q1_fruit_diabetes"])
    all_models["q2"] = regression_q2(questions["q2_lifestyle_patterns"])
    all_models["q3"] = regression_q3(questions["q3_socioeconomic"])
    all_models["q4"] = regression_q4(questions["q4_cumulative_lifestyle"])
    all_models["q5"] = regression_q5(questions["q5_cholesterol_bloodpressure"])
    return all_models
