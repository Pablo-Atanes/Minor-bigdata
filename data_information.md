# Diabetes Health Indicators Dataset - Data Structure (Point System)

## Dataset Bron

- **Survey:** CDC BRFSS (Behavioral Risk Factor Surveillance System) 2015
- **Type:** Telephone survey (landline + mobile)
- **Sample:** 253,680 respondenten (adults 18+)
- **Structuur:** Alle variabelen gebruiken een numeriek punt/categorie systeem

---

## Complete Variable Coding Schema

### Target Variable

| Variable | Code | Betekenis |
|---|---|---|
| Diabetes_binary | 0 | No diabetes |
| Diabetes_binary | 1 | Prediabetes or Diabetes |

---

## Demographic Variables

### 1. Age (`_AGE_G`)

13-punt schaal - Leeftijdscategorieen

| Code | Leeftijdsgroep |
|---|---|
| 1 | 18-24 years |
| 2 | 25-29 years |
| 3 | 30-34 years |
| 4 | 35-39 years |
| 5 | 40-44 years |
| 6 | 45-49 years |
| 7 | 50-54 years |
| 8 | 55-59 years |
| 9 | 60-64 years |
| 10 | 65-69 years |
| 11 | 70-74 years |
| 12 | 75-79 years |
| 13 | 80+ years |

> Voorbeeld: Als Age = 1, dan is de persoon 18-24 jaar oud.

### 2. Education (`_EDUCAG`)

6-punt schaal - Opleidingsniveau

| Code | Opleidingsniveau |
|---|---|
| 1 | Never attended school or only kindergarten |
| 2 | Grades 1-8 (Elementary) |
| 3 | Grades 9-11 (Some high school) |
| 4 | Grade 12 or GED (High school graduate) |
| 5 | College 1-3 years (Some college or technical school) |
| 6 | College 4+ years (College graduate) |

Simplificatie vaak gebruikt:

- 1-3 = Did not graduate high school
- 4 = High school graduate
- 5 = Some college
- 6 = College graduate

### 3. Income (`_INCOMG`)

8-punt schaal - Jaarlijks huishoudinkomen

| Code | Inkomenscategorie |
|---|---|
| 1 | Less than $10,000 |
| 2 | $10,000 to < $15,000 |
| 3 | $15,000 to < $20,000 |
| 4 | $20,000 to < $25,000 |
| 5 | $25,000 to < $35,000 |
| 6 | $35,000 to < $50,000 |
| 7 | $50,000 to < $75,000 |
| 8 | $75,000 or more |

> Voorbeeld: Als Income = 8, dan verdient het huishouden >= $75,000 per jaar.

### 4. Sex

Binaire variabele

| Code | Geslacht |
|---|---|
| 0 | Female |
| 1 | Male |

---

## Health Status Variables

### 5. GenHlth (General Health)

5-punt Likert schaal - Zelf-gerapporteerde algemene gezondheid

| Code | Gezondheidsstatus |
|---|---|
| 1 | Excellent |
| 2 | Very good |
| 3 | Good |
| 4 | Fair |
| 5 | Poor |

> Interpretatie: Hogere score = slechtere gezondheid

### 6. MentHlth (Mental Health)

Continue variabele (0-30)

- Aantal dagen in de afgelopen 30 dagen dat mentale gezondheid niet goed was
- 0 = geen slechte dagen
- 30 = alle 30 dagen waren slecht
- Gemeten in hele dagen

### 7. PhysHlth (Physical Health)

Continue variabele (0-30)

- Aantal dagen in de afgelopen 30 dagen dat fysieke gezondheid niet goed was
- 0 = geen slechte dagen
- 30 = alle 30 dagen waren slecht

---

## Medical Conditions (Binary 0/1)

| Variable | Code | Betekenis |
|---|---|---|
| HighBP | 0 | No high blood pressure |
| HighBP | 1 | Has high blood pressure |
| HighChol | 0 | No high cholesterol |
| HighChol | 1 | Has high cholesterol |
| CholCheck | 0 | No cholesterol check in past 5 years |
| CholCheck | 1 | Cholesterol checked in past 5 years |
| Stroke | 0 | Never had a stroke |
| Stroke | 1 | Had a stroke |
| HeartDiseaseorAttack | 0 | No coronary heart disease or MI |
| HeartDiseaseorAttack | 1 | Has CHD or had MI |

> Voorbeeld: HighBP = 1 betekent de persoon heeft hoge bloeddruk.

---

## Lifestyle Factors (Binary 0/1)

| Variable | Code | Betekenis |
|---|---|---|
| Smoker | 0 | No (smoked < 100 cigarettes lifetime) |
| Smoker | 1 | Yes (smoked >= 100 cigarettes lifetime) |
| PhysActivity | 0 | No physical activity in past 30 days |
| PhysActivity | 1 | Physical activity in past 30 days |
| Fruits | 0 | Consume fruit < 1 time per day |
| Fruits | 1 | Consume fruit >= 1 time per day |
| Veggies | 0 | Consume vegetables < 1 time per day |
| Veggies | 1 | Consume vegetables >= 1 time per day |
| HvyAlcoholConsump | 0 | No heavy drinking |
| HvyAlcoholConsump | 1 | Heavy drinker (Men: >14 drinks/week, Women: >7 drinks/week) |

---

## Healthcare Access (Binary 0/1)

| Variable | Code | Betekenis |
|---|---|---|
| AnyHealthcare | 0 | No healthcare coverage |
| AnyHealthcare | 1 | Has healthcare coverage |
| NoDocbcCost | 0 | Could afford to see doctor |
| NoDocbcCost | 1 | Could NOT see doctor due to cost in past 12 months |
| DiffWalk | 0 | No difficulty walking/climbing stairs |
| DiffWalk | 1 | Has difficulty walking/climbing stairs |

---

## Continuous Variable

### BMI (Body Mass Index)

Continue variabele - Numerieke waarde

- Berekend uit zelf-gerapporteerde lengte en gewicht
- Range: typisch 12-98
- < 18.5 = Underweight
- 18.5-24.9 = Normal weight
- 25.0-29.9 = Overweight
- >= 30.0 = Obese

> Voorbeeld: BMI = 32 betekent de persoon is obese.

---

## Example Data Records

### Voorbeeld 1: Gezonde jonge persoon

| Variable | Waarde | Betekenis |
|---|---|---|
| Diabetes_binary | 0 | - |
| Age | 2 | 25-29 jaar |
| Education | 6 | College graduate |
| Income | 8 | $75,000+ |
| Sex | 0 | Female |
| BMI | 22 | Normal weight |
| GenHlth | 2 | Very good |
| HighBP | 0 | No |
| HighChol | 0 | No |
| Smoker | 0 | No |
| PhysActivity | 1 | Yes |
| Fruits | 1 | Yes, daily |
| Veggies | 1 | Yes, daily |

> Interpretatie: Gezonde jonge vrouw, hoog opgeleid, actieve lifestyle

### Voorbeeld 2: Oudere persoon met diabetes

| Variable | Waarde | Betekenis |
|---|---|---|
| Diabetes_binary | 1 | - |
| Age | 11 | 70-74 jaar |
| Education | 4 | High school graduate |
| Income | 4 | $20,000-$25,000 |
| Sex | 1 | Male |
| BMI | 33 | Obese |
| GenHlth | 4 | Fair |
| HighBP | 1 | Yes |
| HighChol | 1 | Yes |
| HeartDiseaseorAttack | 1 | Yes |
| Smoker | 1 | Yes |
| PhysActivity | 0 | No |
| NoDocbcCost | 1 | Can't afford doctor |

> Interpretatie: Oudere man met diabetes, meerdere comorbiditeiten, lager inkomen, ongezonde lifestyle

---

## Key Insights over de Data Structuur

### Waarom dit "Point System" belangrijk is

**Ordinale Variabelen** (volgorde heeft betekenis):

- Age: 1 -> 13 (jong -> oud)
- GenHlth: 1 -> 5 (excellent -> poor)
- Education: 1 -> 6 (laag -> hoog)
- Income: 1 -> 8 (laag -> hoog)

**Binaire Variabelen** (0/1):

- Alle medische condities
- Alle lifestyle factors
- Sex, Healthcare access

**Continue Variabelen:**

- BMI (numeriek)
- MentHlth (0-30 dagen)
- PhysHlth (0-30 dagen)

### Implicaties voor Machine Learning

**Feature Scaling nodig voor:**

- BMI (range 12-98)
- MentHlth, PhysHlth (0-30)
- Age, Income, Education (verschillende schalen)

**Geen scaling nodig voor:**

- Binary variables (already 0/1)

**Encoding opties:**

- Ordinale variabelen kunnen numeriek blijven (Age, Education, Income, GenHlth)
- Of: One-Hot Encoding voor betere model performance

---

## Praktisch Voorbeeld voor Analyse

```python
# Voorbeeld: Data interpreteren
if row['Age'] == 13 and row['BMI'] > 30 and row['HighBP'] == 1:
    print("High-risk patient: Elderly (80+), Obese, Hypertension")

# Voorbeeld: Risk score berekenen
risk_score = (
    row['HighBP'] +           # +1 if high BP
    row['HighChol'] +         # +1 if high cholesterol
    (row['BMI'] > 30) * 1 +   # +1 if obese
    (row['Age'] > 9) * 1 +    # +1 if 60+
    (row['GenHlth'] > 3) * 1  # +1 if fair/poor health
)
# Risk score ranges 0-5
```
