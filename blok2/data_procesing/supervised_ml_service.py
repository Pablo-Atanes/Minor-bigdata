"""
Gesuperviseerde machine-learning analyses voor Stap 10.

Naast de twee ongesuperviseerde technieken (PCA en clusteranalyse in
`ml_analysis_service`) passen we hier de vier *gesuperviseerde* technieken uit
de toegestane lijst toe: Decision Tree, Naive Bayes, SVM en een neuraal
netwerk (MLP). Alle vier worden op exact DEZELFDE onderzoeksvraag losgelaten,
zodat we ze eerlijk met elkaar kunnen vergelijken en in de eindconclusie de
beste kunnen aanwijzen.

  Onderzoeksvraag (classificatie):
    "Kunnen we, op basis van het maandrendement van de overige assets in
     dezelfde maand, voorspellen of MSFT die maand STIJGT of DAALT?"

Dit is een binaire classificatie: het label is 1 (stijging) of 0 (daling).
De features zijn de maandrendementen van de 19 andere assets. We gebruiken
een tijdreeks-correcte train/test-split (eerste 80% trainen, laatste 20%
testen, geen shuffle) zodat we niet 'in de toekomst kijken'.

Hergebruikt de dataloading van `ml_analysis_service` zodat we precies
dezelfde opgeschoonde maandelijkse portfolio's gebruiken als in de rest van
het project.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

try:
    from IPython.display import display, Markdown
except ImportError:
    display = print
    Markdown = lambda x: x

from blok2.data_procesing.ml_analysis_service import _load_assets_monthly, _ASSET_GROUP


# De asset waarvan we de richting voorspellen. MSFT is een logische keuze:
# het is een liquide, breed gevolgd aandeel dat in Stap 6/7 al centraal stond.
_TARGET = 'MSFT'


def _build_classification_data(target=_TARGET):
    """
    Bouwt de feature-matrix X en het binaire label y voor de classificatie.

    - y = 1 als het maandrendement van `target` > 0 (stijging), anders 0.
    - X = de maandrendementen van alle OVERIGE assets in dezelfde maand.

    Returnt (X_df, y_series, feature_names). Rijen met NaN (de eerste maand
    na pct_change) vallen weg.
    """
    monthly = _load_assets_monthly()
    returns = monthly.pct_change().dropna()

    y = (returns[target] > 0).astype(int)
    X = returns.drop(columns=[target])

    return X, y, list(X.columns)


def _split_train_test(X, y, test_size=0.2):
    """
    Tijdreeks-correcte split: de eerste (1 - test_size) als train, de rest
    als test. Geen shuffle -- we mogen niet met toekomstige maanden trainen.
    """
    n = len(X)
    n_test = max(1, int(round(n * test_size)))
    n_train = n - n_test
    X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
    y_train, y_test = y.iloc[:n_train], y.iloc[n_train:]
    return X_train, X_test, y_train, y_test


def _make_models():
    """
    De vier gesuperviseerde technieken uit de toegestane lijst, elk met
    nette, verdedigbare standaardinstellingen voor een kleine dataset.

    SVM en het neuraal netwerk zijn gevoelig voor de schaal van de features,
    daarom standaardiseren we de input voor die twee (zie `run_classifiers`).
    """
    return {
        'Decision Tree': DecisionTreeClassifier(max_depth=3, random_state=42),
        'Naive Bayes': GaussianNB(),
        'SVM': SVC(kernel='rbf', C=1.0, random_state=42),
        'Neuraal netwerk': MLPClassifier(
            hidden_layer_sizes=(16, 8),
            max_iter=2000,
            random_state=42,
        ),
    }


# Modellen die op gestandaardiseerde features moeten draaien.
_NEEDS_SCALING = {'SVM', 'Neuraal netwerk'}


def run_classifiers(target=_TARGET, test_size=0.2):
    """
    Traint alle vier de classifiers op dezelfde train/test-split en verzamelt
    hun resultaten in een vergelijkbare structuur.

    Returnt een dict met o.a. een resultaten-DataFrame (accuracy/F1 per model),
    de getrainde modellen, de confusion matrices en de baseline (de accuracy
    die je haalt door simpelweg altijd de meest voorkomende klasse te gokken).
    """
    X, y, feature_names = _build_classification_data(target)
    X_train, X_test, y_train, y_test = _split_train_test(X, y, test_size)

    # Standaardiseren op basis van ALLEEN de train-set (geen data leakage).
    scaler = StandardScaler().fit(X_train.values)
    X_train_s = scaler.transform(X_train.values)
    X_test_s = scaler.transform(X_test.values)

    # Baseline: altijd de meest voorkomende klasse in de train-set gokken.
    majority_class = int(y_train.mode().iloc[0])
    baseline_acc = accuracy_score(y_test, np.full(len(y_test), majority_class))

    models = _make_models()
    rows = []
    fitted = {}
    confusions = {}

    for name, model in models.items():
        if name in _NEEDS_SCALING:
            Xtr, Xte = X_train_s, X_test_s
        else:
            Xtr, Xte = X_train.values, X_test.values

        model.fit(Xtr, y_train)
        pred = model.predict(Xte)

        rows.append({
            'Techniek': name,
            'Accuracy (test)': accuracy_score(y_test, pred),
            'F1 (test)': f1_score(y_test, pred, zero_division=0),
            'Accuracy (train)': accuracy_score(y_train, model.predict(Xtr)),
        })
        fitted[name] = model
        confusions[name] = confusion_matrix(y_test, pred)

    results_df = pd.DataFrame(rows).set_index('Techniek')
    results_df = results_df.sort_values('Accuracy (test)', ascending=False)

    return {
        'results': results_df,
        'models': fitted,
        'confusions': confusions,
        'feature_names': feature_names,
        'baseline_acc': baseline_acc,
        'majority_class': majority_class,
        'n_train': len(y_train),
        'n_test': len(y_test),
        'target': target,
        'X_train': X_train,
        'y_train': y_train,
        'class_balance': dict(y.value_counts().sort_index()),
    }


def _plot_model_comparison(result):
    """Staafdiagram van de test-accuracy per techniek + baseline-lijn."""
    res = result['results']
    plt.figure(figsize=(9, 5))
    colors = sns.color_palette('tab10', len(res))
    plt.bar(res.index, res['Accuracy (test)'] * 100, color=colors, alpha=0.85)
    plt.axhline(result['baseline_acc'] * 100, color='red', linestyle='--',
                label=f"Baseline (altijd klasse {result['majority_class']}): "
                      f"{result['baseline_acc']*100:.1f}%")
    plt.ylabel('Test-accuracy (%)')
    plt.ylim(0, 100)
    plt.title(f"Vergelijking classifiers — richting {result['target']} voorspellen")
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()


def _plot_confusion_matrices(result):
    """Confusion matrix per techniek, in een raster naast elkaar."""
    names = list(result['confusions'])
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 3.6))
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        sns.heatmap(result['confusions'][name], annot=True, fmt='d', cmap='Blues',
                    cbar=False, ax=ax,
                    xticklabels=['Daling', 'Stijging'],
                    yticklabels=['Daling', 'Stijging'])
        ax.set_title(name, fontsize=10)
        ax.set_xlabel('Voorspeld')
        ax.set_ylabel('Werkelijk')
    plt.suptitle('Confusion matrices op de test-set')
    plt.tight_layout()
    plt.show()


def _plot_decision_tree(result):
    """Tekent de (ondiepe) decision tree zodat de splits leesbaar zijn."""
    model = result['models']['Decision Tree']
    plt.figure(figsize=(16, 7))
    plot_tree(model, feature_names=result['feature_names'],
              class_names=['Daling', 'Stijging'], filled=True, rounded=True,
              fontsize=8)
    plt.title('Decision Tree (max_depth=3) — welke assets sturen de richting?')
    plt.tight_layout()
    plt.show()


def show_supervised_analysis(target=_TARGET):
    """
    Toont de volledige gesuperviseerde analyse in de notebook: uitleg, de vier
    getrainde modellen, een vergelijkingsgrafiek, confusion matrices, de
    decision tree, en een conclusie die de best presterende techniek aanwijst.
    """
    display(Markdown(
        f"## Technieken 3-6 — Gesuperviseerde classificatie\n\n"
        f"**Onderzoeksvraag:** Kunnen we, op basis van het maandrendement van de "
        f"overige assets in dezelfde maand, voorspellen of **{target}** die maand "
        f"*stijgt* of *daalt*?\n\n"
        f"Dit is een binaire classificatie (label 1 = stijging, 0 = daling). We "
        f"passen alle vier de resterende toegestane technieken toe — **Decision "
        f"Tree, Naive Bayes, SVM en een neuraal netwerk** — op exact dezelfde "
        f"train/test-split, zodat we ze eerlijk kunnen vergelijken. De split is "
        f"tijdreeks-correct: we trainen op de eerste ~80% van de maanden en "
        f"testen op de laatste ~20% (geen shuffle, dus geen kijken in de "
        f"toekomst). SVM en het neuraal netwerk krijgen gestandaardiseerde "
        f"features, omdat die technieken schaalgevoelig zijn."
    ))

    result = run_classifiers(target)

    bal = result['class_balance']
    print(f"Doel-asset: {target}")
    print(f"Aantal maanden: train = {result['n_train']}, test = {result['n_test']}")
    print(f"Klassenverdeling (hele reeks): daling = {bal.get(0, 0)}, "
          f"stijging = {bal.get(1, 0)}")
    print(f"Baseline-accuracy (altijd klasse {result['majority_class']} gokken): "
          f"{result['baseline_acc']*100:.1f}%")

    display(Markdown("### Resultaten per techniek (gesorteerd op test-accuracy)"))
    display(result['results'].style.format({
        'Accuracy (test)': '{:.1%}',
        'F1 (test)': '{:.3f}',
        'Accuracy (train)': '{:.1%}',
    }).background_gradient(cmap='Greens', subset=['Accuracy (test)']))

    _plot_model_comparison(result)
    _plot_confusion_matrices(result)
    _plot_decision_tree(result)

    # Conclusie: wijs de beste techniek aan en zet die af tegen de baseline.
    res = result['results']
    best_name = res.index[0]
    best_acc = res.iloc[0]['Accuracy (test)']
    baseline = result['baseline_acc']
    beats_baseline = best_acc > baseline

    if beats_baseline:
        oordeel = (
            f"**{best_name}** presteert met **{best_acc*100:.1f}%** test-accuracy "
            f"het best en komt **boven** de baseline van {baseline*100:.1f}% uit. "
            f"Er zit dus enig voorspellend signaal in de gezamenlijke "
            f"asset-rendementen."
        )
    else:
        oordeel = (
            f"De best presterende techniek is **{best_name}** "
            f"({best_acc*100:.1f}% test-accuracy), maar geen enkel model komt "
            f"betekenisvol **boven** de baseline van {baseline*100:.1f}% uit. "
            f"De maandrichting van {target} is met deze features dus nauwelijks "
            f"betrouwbaar te voorspellen — wat zelf ook een waardevolle uitkomst "
            f"is: de markt is op maandbasis grotendeels efficiënt."
        )

    display(Markdown(
        f"### Conclusie gesuperviseerde technieken\n\n"
        f"- We hebben **vier** classificatietechnieken op dezelfde onderzoeksvraag "
        f"losgelaten en op een eerlijke, tijdreeks-correcte test-set vergeleken.\n"
        f"- {oordeel}\n"
        f"- De **Decision Tree** is daarbij het best te interpreteren: in de boom "
        f"hierboven is direct zichtbaar welke assets als eerste worden gebruikt om "
        f"de richting van {target} te splitsen. SVM en het neuraal netwerk zijn "
        f"krachtiger maar werken als 'black box'.\n"
        f"- De accuracy op de **train**-set ligt voor de flexibele modellen "
        f"(Decision Tree, neuraal netwerk) hoger dan op de test-set: een teken van "
        f"lichte overfitting, wat bij {result['n_train']} trainmaanden te "
        f"verwachten is."
    ))

    return result


if __name__ == "__main__":
    r = run_classifiers()
    print("Baseline acc:", round(r['baseline_acc'], 3))
    print(r['results'])
