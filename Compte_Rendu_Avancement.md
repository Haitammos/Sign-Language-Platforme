# Compte Rendu d'Avancement — Projet de Fin d'Études

## Reconnaissance de la Langue des Signes en Temps Réel avec Intelligence Artificielle

---

**Étudiant :** [Votre Nom Complet]  
**Filière :** [Votre Filière]  
**Encadrant :** [Nom du Professeur]  
**Date :** Avril 2026  

---

## 1. Description du Projet

Ce projet consiste à développer une **plateforme web intelligente** capable de reconnaître la **langue des signes américaine (ASL)** en temps réel à l'aide de la caméra, et de la traduire en texte et en parole dans **4 langues** (Anglais, Français, Espagnol, Arabe).

Le système utilise une architecture **« Dual-Brain »** (Double Cerveau IA) :
- **Cerveau 1 — Random Forest** : Reconnaissance des signes statiques (lettres A-Z, chiffres 0-9, espace, point) avec **100% de précision**.
- **Cerveau 2 — LSTM (Réseau de neurones récurrents)** : Reconnaissance des gestes dynamiques (17 mots/phrases) avec **98.7% de précision**.

---

## 2. Technologies Utilisées

| Technologie | Rôle |
|---|---|
| **Python 3.10** | Langage principal |
| **TensorFlow / Keras** | Modèle LSTM pour la reconnaissance de gestes dynamiques |
| **Scikit-Learn** | Modèle Random Forest pour la reconnaissance de lettres statiques |
| **MediaPipe Holistic** | Extraction de 1662 points clés (mains, corps, visage) en temps réel |
| **OpenCV** | Capture et traitement du flux vidéo de la caméra |
| **Flask** | Serveur web backend pour la plateforme |
| **HTML / CSS / JavaScript** | Interface web professionnelle (frontend) |
| **Lucide Icons** | Icônes SVG professionnelles |
| **Google TTS (gTTS)** | Synthèse vocale dans les 4 langues |
| **Deep Translator** | Traduction automatique multi-langue |

---

## 3. Fonctionnalités Réalisées

### 3.1. Détection en Temps Réel
- Détection de **38 signes statiques** (A-Z, 0-9, espace, point) via Random Forest
- Détection de **17 mots/gestes dynamiques** via LSTM :
  - hello, thanks, I love you, hi, my, name, is, what, how, you, please, sorry, help, yes, no, good, bad
- Construction automatique de **phrases complètes** mot par mot
- Exemple : L'utilisateur peut signer "Hi my name is" en temps réel

### 3.2. Traduction Multi-Langue
- Traduction instantanée en **4 langues** : Anglais, Français, Espagnol, Arabe
- Synthèse vocale native dans chaque langue via Google TTS

### 3.3. Plateforme Web Professionnelle
- **Page d'accueil** : Présentation de l'architecture et des statistiques
- **Page d'apprentissage** : 55 leçons structurées avec suivi de progression (style Coursera)
- **Page de leçon individuelle** : Image de référence + validation du geste par caméra en temps réel
- **Page de pratique libre** : Mode texte libre avec boutons Reset, Delete, Pause, Speak
- **Design professionnel** : Interface sombre moderne, icônes SVG, responsive

### 3.4. Pipeline d'Entraînement Complet
- Script de collecte de données pour les lettres (`collectImgs.py`)
- Script de collecte de données pour les mots (`collect_words.py`)
- Script de création du dataset (`createDataset.py`)
- Script d'entraînement du Random Forest (`trainClassifier.py`)
- Script d'entraînement du LSTM (`train_lstm.py`)

---

## 4. Architecture Technique

```
┌─────────────────────────────────────────────────────┐
│                   CAMÉRA (DroidCam)                 │
│                        │                            │
│                 MediaPipe Holistic                   │
│            (1662 points clés extraits)              │
│                    │           │                     │
│        ┌───────────┘           └───────────┐        │
│        ▼                                   ▼        │
│  ┌──────────────┐                ┌──────────────┐   │
│  │  Cerveau 1   │                │  Cerveau 2   │   │
│  │ Random Forest│                │    LSTM      │   │
│  │ (model.p)    │                │ (action.h5)  │   │
│  │              │                │              │   │
│  │ Lettres A-Z  │                │ 17 mots      │   │
│  │ Chiffres 0-9 │                │ dynamiques   │   │
│  │ Espace, Point│                │              │   │
│  │ Précision:   │                │ Précision:   │   │
│  │   100%       │                │   98.7%      │   │
│  └──────┬───────┘                └──────┬───────┘   │
│         └──────────────┬───────────────┘            │
│                        ▼                            │
│              Construction de Phrase                  │
│                        │                            │
│              ┌─────────┼───────────┐                │
│              ▼         ▼           ▼                │
│         Traduction   Texte    Synthèse Vocale       │
│         (4 langues)  Affiché    (gTTS)              │
│                                                     │
│              ┌─────────────────────┐                │
│              │   Plateforme Web    │                │
│              │      (Flask)        │                │
│              └─────────────────────┘                │
└─────────────────────────────────────────────────────┘
```

---

## 5. Résultats

| Modèle | Données d'entraînement | Précision |
|---|---|---|
| Random Forest (lettres) | 38 classes × 300 images = 11,400 images | **100%** |
| LSTM (mots/gestes) | 17 classes × 30 séquences × 30 frames = 15,300 frames | **98.7%** |

---

## 6. Difficultés Rencontrées et Solutions

| Difficulté | Solution |
|---|---|
| Caméra DroidCam de mauvaise qualité | Réduction des seuils de détection MediaPipe (0.5 → 0.3) |
| Première tentative LSTM : 7.8% de précision | Correction : normalisation Z-score, activation tanh, Dropout, BatchNorm, learning rate réduit → 98.7% |
| Détection du geste "Hello" échouait | Abaissement du seuil de confiance spécifique (0.85 → 0.55) |
| Gel de la caméra pendant la traduction | Traduction déplacée vers des threads asynchrones |

---

## 7. Captures d'Écran

> [INSÉRER ICI vos captures d'écran de la plateforme web]
>
> Suggestions :
> 1. Page d'accueil (http://localhost:5000)
> 2. Page d'apprentissage avec la grille des leçons
> 3. Page de leçon avec la validation par caméra
> 4. Page de pratique avec une phrase détectée

---

## 8. Prochaines Étapes

- [ ] Optimisation de la détection en temps réel (réduction des faux positifs LSTM)
- [ ] Tests en conditions réelles (éclairage varié, différents utilisateurs)
- [ ] Déploiement sur GitHub avec documentation README
- [ ] Préparation de la soutenance finale

---

## 9. Conclusion

Le projet est dans un état d'avancement très avancé. Les deux modèles d'IA sont entraînés et fonctionnels avec des précisions de 100% et 98.7%. La plateforme web est entièrement développée avec une interface professionnelle. Les prochaines étapes se concentrent sur l'optimisation de la détection en temps réel et la préparation de la présentation finale.
