import pandas as pd
import numpy as np
import re
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
import os

# Configurer les styles des graphiques
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ==============================================
# 1. CONFIGURATION INTELLIGENTE POUR LE FRANÇAIS IVOIRIEN
# ==============================================

class AnalyseurIvoirienIntelligent:
    """Analyseur spécialisé pour le français ivoirien"""
    
    def __init__(self):
        self.lexique = self._charger_lexique_ivoirien()
        self.patterns_sarcasme = self._charger_patterns_sarcasme()
        self.expressions_locales = self._charger_expressions_locales()
        
    def _charger_lexique_ivoirien(self):
        """Lexique spécialisé pour le contexte ivoirien"""
        return {
            # ========== POSITIFS FORTS ==========
            "félicitations": 0.85, "félicitation": 0.85, "féliciter": 0.8,
            "bravo": 0.75, "excellent": 0.9, "exceptionnel": 0.85,
            "parfait": 0.85, "impeccable": 0.8, "remarquable": 0.8,
            "super": 0.7, "génial": 0.75, "fantastique": 0.8,
            "merveilleux": 0.8, "formidable": 0.75,
            
            # ========== POSITIFS MODÉRÉS ==========
            "bon": 0.6, "bonne": 0.6, "bien": 0.6,
            "agréable": 0.55, "sympathique": 0.5, "aimable": 0.5,
            "utile": 0.55, "efficace": 0.6, "pratique": 0.5,
            "satisfaisant": 0.55, "convenable": 0.5,
            
            # ========== REMERCIEMENTS ==========
            "merci": 0.4, "remercier": 0.4, "remerciement": 0.4,
            "gratitude": 0.5, "reconnaissance": 0.45,
            
            # ========== EXPRESSIONS LOCALES POSITIVES ==========
            "mécanique": 0.35,  # confiance/simplicité
            "c'est bon": 0.3, "c'est bien": 0.35,
            "prado": 0.4, "pr ado": 0.4,  # Référence politique
            "ado": 0.3, "alassane": 0.3, "ouattara": 0.3,
            "choco": 0.45, "chocosite": 0.45,  # Supporters
            "rhdp": 0.25,  # Parti politique
            
            # ========== NÉGATIFS FORTS ==========
            "honte": -0.85, "honteux": -0.85, "honteuse": -0.85,
            "dégout": -0.8, "dégoût": -0.8, "dégoutant": -0.85,
            "écœurant": -0.8, "révoltant": -0.75,
            "horrible": -0.9, "atroce": -0.85, "affreux": -0.8,
            "catastrophe": -0.8, "désastre": -0.85,
            
            # ========== INSULTES ET AGRESSIONS ==========
            "imbécile": -0.9, "idiot": -0.85, "stupide": -0.8,
            "crétin": -0.85, "abruti": -0.9, "débile": -0.85,
            
            # ========== CRIMES ET MALVERSATIONS ==========
            "escroc": -0.9, "escroquerie": -0.85, "arnaque": -0.85,
            "voleur": -0.9, "vol": -0.85, "corrompu": -0.9,
            "corruption": -0.85, "tricher": -0.8, "fraude": -0.85,
            "bandit": -0.85, "criminel": -0.9,
            
            # ========== MENSONGES ET TRAHISONS ==========
            "menteur": -0.8, "mensonge": -0.75, "tromper": -0.7,
            "traître": -0.85, "trahison": -0.8,
            "hypocrite": -0.75, "hypocrisie": -0.7,
            
            # ========== EXPRESSIONS LOCALES NÉGATIVES ==========
            "tchê": -0.6, "tche": -0.6,  # frustration
            "saya": -0.7,  # mourir
            "walaye": -0.55,  # exclamation négative
            "palabre": -0.1,  # discussion interminable
            
            # ========== CRITIQUES POLITIQUES ==========
            "dictateur": -0.85, "dictature": -0.9,
            "autocrate": -0.8, "tyran": -0.85,
            "incompétent": -0.7, "incompétence": -0.7,
            "nul": -0.65, "nulle": -0.65,
            "inutile": -0.6,
            
            # ========== PROBLÈMES ET DIFFICULTÉS ==========
            "problème": -0.4, "difficulté": -0.35,
            "erreur": -0.45, "faute": -0.5,
            "échec": -0.6, "raté": -0.55,
            "cher": -0.3, "coûteux": -0.35,
            "pauvre": -0.4, "pauvreté": -0.5,
            
            # ========== NEUTRES CONTEXTUELS ==========
            "gouvernement": 0.0, "président": 0.0, "ministre": 0.0,
            "politique": 0.0, "politicien": 0.0,
            "élection": 0.0, "vote": 0.0, "voter": 0.0,
            "démocratie": 0.1, "république": 0.1,
            "discussion": 0.0, "débat": 0.0,
            "information": 0.05, "info": 0.05,
            "question": 0.0, "demande": 0.0,
            "avis": 0.0, "opinion": 0.0,
        }
    
    def _charger_patterns_sarcasme(self):
        """Patterns pour détecter le sarcasme et l'ironie"""
        return [
            (r"merci (?:beaucoup|infiniment|énormément).* (?:mais|pourtant|cependant|sauf que)", -0.85),
            (r"félicitations.* (?:mais|pourtant|cependant|sauf que|alors que)", -0.8),
            (r"bravo.* (?:mais|pourtant|cependant|sauf que)", -0.75),
            (r"je (?:vous |)remercie.* (?:force|armes|violence|mentir|tricher|voler)", -0.9),
            (r"longue vie.* (?:dictature|corruption|voleur|escroc|menteur)", -0.85),
            (r"excellent.* (?:problème|échec|catastrophe|désastre)", -0.8),
            (r"belle initiative.* (?:mais|sauf|pourtant|cependant)", -0.7),
            (r"super idée.* (?:mais|sauf|pourtant)", -0.65),
            (r"génial.* (?:par contre|en revanche|mais)", -0.6),
        ]
    
    def _charger_expressions_locales(self):
        """Expressions locales avec leur signification"""
        return {
            "tchê": "frustration",
            "walaye": "exclamation négative", 
            "saya": "mourir",
            "mécanique": "simplicité/confiance",
            "palabre": "discussion longue",
            "ado": "Alassane Dramane Ouattara",
            "choco": "supporteur RHDP",
            "prado": "Président ADO",
        }
    
    def analyser_texte(self, texte):
        """Analyse complète d'un texte"""
        if not isinstance(texte, str) or not texte.strip():
            return self._resultat_vide()
        
        texte_lower = texte.lower()
        
        # 1. Analyse TextBlob
        blob = TextBlob(texte)
        score_textblob = blob.sentiment.polarity
        subjectivite = blob.sentiment.subjectivity
        
        # 2. Analyse lexicale
        score_lexical, mots_significatifs = self._analyser_lexicalement(texte_lower)
        
        # 3. Détection sarcasme
        score_sarcasme = self._detecter_sarcasme(texte_lower)
        
        # 4. Détection expressions locales
        expressions_trouvees = self._detecter_expressions_locales(texte_lower)
        
        # 5. Fusion intelligente
        if score_sarcasme < -0.3:
            poids = {"textblob": 0.1, "lexical": 0.7, "sarcasme": 0.2}
        elif expressions_trouvees:
            poids = {"textblob": 0.2, "lexical": 0.7, "sarcasme": 0.1}
        else:
            poids = {"textblob": 0.3, "lexical": 0.6, "sarcasme": 0.1}
        
        # Score final
        score_final = (
            score_textblob * poids["textblob"] +
            score_lexical * poids["lexical"] +
            score_sarcasme * poids["sarcasme"]
        )
        
        # Ajustement pour sarcasme
        if score_sarcasme < -0.4 and score_final > -0.2:
            score_final = max(score_final * -1, -0.6)
        
        # Détermination du sentiment
        if score_sarcasme < -0.5:
            sentiment = "NÉGATIF (sarcasme)"
        elif score_final > 0.2:
            sentiment = "POSITIF"
        elif score_final < -0.2:
            sentiment = "NÉGATIF"
        elif -0.1 <= score_final <= 0.1:
            sentiment = "NEUTRE"
        else:
            sentiment = "MIXTE"
        
        # Confiance
        confiance = min(
            abs(score_lexical) * 0.6 +
            subjectivite * 0.3 +
            abs(score_sarcasme) * 0.1,
            1.0
        )
        
        return {
            "texte_original": texte[:200],
            "score_final": round(score_final, 4),
            "sentiment": sentiment,
            "confiance": round(confiance, 4),
            "score_sarcasme": round(score_sarcasme, 4),
            "score_textblob": round(score_textblob, 4),
            "score_lexical": round(score_lexical, 4),
            "expressions_locales": expressions_trouvees,
            "sarcasme_detecte": score_sarcasme < -0.3,
        }
    
    def _analyser_lexicalement(self, texte):
        """Analyse lexicale avancée"""
        mots = re.findall(r'\b\w+\b', texte)
        score_total = 0
        mots_significatifs = []
        negation_active = False
        
        for mot in mots:
            # Gestion des négations
            if mot in ["pas", "non", "jamais", "aucun", "rien"]:
                negation_active = True
                continue
            
            # Vérifier le mot dans le lexique
            if mot in self.lexique:
                score = self.lexique[mot]
                
                # Appliquer négation si active
                if negation_active:
                    score = -score * 0.8
                    negation_active = False
                
                score_total += score
                mots_significatifs.append({"mot": mot, "score": round(score, 3)})
            else:
                negation_active = False
        
        # Normalisation
        if mots_significatifs:
            score_moyen = score_total / len(mots_significatifs)
        else:
            score_moyen = 0
        
        return score_moyen, mots_significatifs
    
    def _detecter_sarcasme(self, texte):
        """Détection de sarcasme"""
        score_sarcasme = 0.0
        
        # Vérification des patterns
        for pattern, score in self.patterns_sarcasme:
            if re.search(pattern, texte, re.IGNORECASE):
                score_sarcasme += score
        
        return max(min(score_sarcasme, 0), -1.0)
    
    def _detecter_expressions_locales(self, texte):
        """Détection des expressions locales"""
        expressions_trouvees = []
        for expr, signification in self.expressions_locales.items():
            if expr in texte:
                expressions_trouvees.append({"expression": expr, "signification": signification})
        return expressions_trouvees
    
    def _resultat_vide(self):
        """Résultat pour texte vide"""
        return {
            "texte_original": "",
            "score_final": 0,
            "sentiment": "NEUTRE",
            "confiance": 0,
            "score_sarcasme": 0,
            "score_textblob": 0,
            "score_lexical": 0,
            "expressions_locales": [],
            "sarcasme_detecte": False,
        }

# ==============================================
# 2. FONCTIONS DE VISUALISATION
# ==============================================

def creer_graphiques(df, dossier_output="graphiques"):
    """Créer tous les graphiques d'analyse"""
    
    # Créer le dossier de sortie
    os.makedirs(dossier_output, exist_ok=True)
    
    # 1. Distribution des sentiments (Camembert)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Distribution des sentiments
    sentiment_counts = df['sentiment'].value_counts()
    colors = ['#4CAF50', '#F44336', '#FFC107', '#9E9E9E']  # Vert, Rouge, Jaune, Gris
    
    axes[0].pie(sentiment_counts.values, labels=sentiment_counts.index, 
                autopct='%1.1f%%', colors=colors[:len(sentiment_counts)],
                startangle=90)
    axes[0].set_title('📊 Distribution des Sentiments', fontsize=14, fontweight='bold')
    axes[0].axis('equal')
    
    # 2. Histogramme des scores
    axes[1].hist(df['score_final'], bins=30, color='#2196F3', edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Neutre')
    axes[1].set_xlabel('Score de Sentiment', fontsize=12)
    axes[1].set_ylabel('Nombre de Commentaires', fontsize=12)
    axes[1].set_title('📈 Distribution des Scores', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. Boxplot par sentiment
    sentiment_data = []
    sentiment_labels = []
    for sentiment in df['sentiment'].unique():
        sentiment_data.append(df[df['sentiment'] == sentiment]['score_final'])
        sentiment_labels.append(sentiment)
    
    bp = axes[2].boxplot(sentiment_data, labels=sentiment_labels, patch_artist=True)
    
    # Colorier les boxplots
    colors_box = ['lightgreen', 'lightcoral', 'lightyellow', 'lightgrey']
    for patch, color in zip(bp['boxes'], colors_box[:len(sentiment_labels)]):
        patch.set_facecolor(color)
    
    axes[2].set_ylabel('Score', fontsize=12)
    axes[2].set_title('📦 Boxplot par Catégorie', fontsize=14, fontweight='bold')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(True, alpha=0.3)
    
    # 4. Diagramme en barres horizontales (top mots)
    if 'details' in df.columns and df['details'].notna().any():
        # Extraire les mots les plus fréquents (simplifié)
        all_text = ' '.join(df['texte_original'].dropna().astype(str).tolist())
        words = re.findall(r'\b\w+\b', all_text.lower())
        word_freq = pd.Series(words).value_counts().head(15)
        
        axes[3].barh(range(len(word_freq)), word_freq.values, color='#FF9800')
        axes[3].set_yticks(range(len(word_freq)))
        axes[3].set_yticklabels(word_freq.index)
        axes[3].set_xlabel('Fréquence', fontsize=12)
        axes[3].set_title('🏷️ Mots les Plus Fréquents', fontsize=14, fontweight='bold')
        axes[3].invert_yaxis()
    
    # 5. Graphique de densité
    from scipy import stats
    
    for sentiment in df['sentiment'].unique():
        subset = df[df['sentiment'] == sentiment]['score_final']
        if len(subset) > 1:
            kde = stats.gaussian_kde(subset)
            x_vals = np.linspace(subset.min(), subset.max(), 100)
            axes[4].plot(x_vals, kde(x_vals), label=sentiment, linewidth=2)
    
    axes[4].set_xlabel('Score', fontsize=12)
    axes[4].set_ylabel('Densité', fontsize=12)
    axes[4].set_title('📉 Densité des Scores par Sentiment', fontsize=14, fontweight='bold')
    axes[4].legend()
    axes[4].grid(True, alpha=0.3)
    
    # 6. Graphique de confiance vs score
    scatter = axes[5].scatter(df['score_final'], df['confiance'], 
                              c=df['score_final'], cmap='coolwarm', 
                              alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    axes[5].axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    axes[5].axhline(y=0.5, color='green', linestyle='--', linewidth=1, alpha=0.5)
    axes[5].set_xlabel('Score de Sentiment', fontsize=12)
    axes[5].set_ylabel('Confiance', fontsize=12)
    axes[5].set_title('🎯 Score vs Confiance', fontsize=14, fontweight='bold')
    axes[5].grid(True, alpha=0.3)
    
    plt.colorbar(scatter, ax=axes[5], label='Score')
    
    plt.tight_layout()
    plt.savefig(f'{dossier_output}/analyse_complete.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Graphique supplémentaire : Évolution temporelle (si index temporel)
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    
    # Créer un index temporel fictif si non disponible
    df['index_temporel'] = range(len(df))
    
    # Moyenne mobile des scores
    df['score_moyen_mobile'] = df['score_final'].rolling(window=10, min_periods=1).mean()
    
    ax2.plot(df['index_temporel'], df['score_moyen_mobile'], 
             color='#2196F3', linewidth=2.5, label='Moyenne Mobile (fenêtre=10)')
    ax2.scatter(df['index_temporel'], df['score_final'], 
                c=df['score_final'], cmap='RdYlBu', alpha=0.3, s=20)
    
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Neutre')
    ax2.fill_between(df['index_temporel'], df['score_moyen_mobile'], 0, 
                     where=(df['score_moyen_mobile'] >= 0), 
                     color='green', alpha=0.2, label='Zone Positive')
    ax2.fill_between(df['index_temporel'], df['score_moyen_mobile'], 0, 
                     where=(df['score_moyen_mobile'] < 0), 
                     color='red', alpha=0.2, label='Zone Négative')
    
    ax2.set_xlabel('Commentaire (ordre chronologique)', fontsize=12)
    ax2.set_ylabel('Score de Sentiment', fontsize=12)
    ax2.set_title('📈 Évolution des Sentiments', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{dossier_output}/evolution_sentiments.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Graphique des sarcasmes détectés
    if 'sarcasme_detecte' in df.columns:
        sarcasme_counts = df['sarcasme_detecte'].value_counts()
        
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        bars = ax3.bar(['Non', 'Oui'], sarcasme_counts.values, 
                       color=['#4CAF50', '#FF9800'], edgecolor='black')
        
        ax3.set_ylabel('Nombre de Commentaires', fontsize=12)
        ax3.set_title('😏 Sarcasmes Détectés', fontsize=14, fontweight='bold')
        
        # Ajouter les valeurs sur les barres
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{dossier_output}/sarcasmes.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    # Graphique comparatif si étiquettes manuelles disponibles
    if 'New_Sentiment' in df.columns:
        fig4, axes4 = plt.subplots(1, 2, figsize=(14, 6))
        
        # Distribution des sentiments automatiques
        auto_counts = df['sentiment'].value_counts()
        axes4[0].bar(auto_counts.index, auto_counts.values, color='#2196F3', alpha=0.7)
        axes4[0].set_title('🏷️ Sentiments Automatiques', fontsize=12, fontweight='bold')
        axes4[0].set_ylabel('Nombre')
        axes4[0].tick_params(axis='x', rotation=45)
        
        # Distribution des sentiments manuels
        manual_counts = df['New_Sentiment'].value_counts()
        axes4[1].bar(manual_counts.index, manual_counts.values, color='#4CAF50', alpha=0.7)
        axes4[1].set_title('✍️ Sentiments Manuels', fontsize=12, fontweight='bold')
        axes4[1].set_ylabel('Nombre')
        axes4[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{dossier_output}/comparaison_sentiments.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Matrice de confusion (simplifiée)
        from sklearn.metrics import confusion_matrix
        
        # Simplifier les catégories pour la comparaison
        df['sentiment_simple'] = df['sentiment'].apply(lambda x: 'POSITIF' if 'POSITIF' in x else 
                                                      'NÉGATIF' if 'NÉGATIF' in x else 'NEUTRE')
        df['New_Sentiment_simple'] = df['New_Sentiment'].apply(lambda x: 'POSITIF' if 'Positif' in str(x) else 
                                                              'NÉGATIF' if 'Négatif' in str(x) else 'NEUTRE')
        
        cm = confusion_matrix(df['New_Sentiment_simple'], df['sentiment_simple'], 
                            labels=['POSITIF', 'NÉGATIF', 'NEUTRE'])
        
        fig5, ax5 = plt.subplots(figsize=(8, 6))
        im = ax5.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax5.figure.colorbar(im, ax=ax5)
        
        ax5.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=['POSITIF', 'NÉGATIF', 'NEUTRE'],
               yticklabels=['POSITIF', 'NÉGATIF', 'NEUTRE'],
               title='🎯 Matrice de Confusion',
               ylabel='Réel (Manuel)',
               xlabel='Prédit (Automatique)')
        
        # Ajouter les valeurs dans les cases
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax5.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.savefig(f'{dossier_output}/matrice_confusion.png', dpi=150, bbox_inches='tight')
        plt.show()

# ==============================================
# 3. ANALYSEUR DE FICHIER COMPLET
# ==============================================

def analyser_fichier_excel(chemin_fichier):
    """Analyser un fichier Excel complet"""
    
    print("🤖 SYSTÈME D'ANALYSE DES SENTIMENTS AVEC GRAPHIQUES")
    print("   Version : Visualisation Avancée")
    print("=" * 70)
    
    try:
        # Charger le fichier
        print(f"\n📂 Chargement du fichier...")
        df = pd.read_excel(chemin_fichier)
        print(f"   ✅ Fichier chargé : {len(df)} commentaires")
        
        # Identifier la colonne de texte
        colonne_texte = 'Text_Clean' if 'Text_Clean' in df.columns else df.columns[0]
        print(f"   📝 Colonne analysée : '{colonne_texte}'")
        
        # Initialiser l'analyseur
        analyseur = AnalyseurIvoirienIntelligent()
        
        # Analyser chaque commentaire
        print("\n🔍 Analyse en cours...")
        resultats = []
        
        for idx, row in df.iterrows():
            texte = row[colonne_texte]
            
            if pd.isna(texte):
                resultat = {
                    "texte_original": "",
                    "score_final": 0,
                    "sentiment": "NEUTRE",
                    "confiance": 0,
                    "sarcasme_detecte": False,
                }
            else:
                resultat = analyseur.analyser_texte(str(texte))
            
            resultats.append(resultat)
            
            # Afficher progression
            if (idx + 1) % 50 == 0:
                print(f"   ✓ {idx + 1}/{len(df)} commentaires analysés")
        
        print("✅ Analyse terminée !")
        
        # Créer DataFrame de résultats
        df_resultats = pd.DataFrame(resultats)
        
        # Fusionner avec original
        df_final = pd.concat([df.reset_index(drop=True), df_resultats], axis=1)
        
        # Statistiques
        print("\n" + "=" * 70)
        print("📊 STATISTIQUES DÉTAILLÉES")
        print("=" * 70)
        
        stats = df_final['sentiment'].value_counts()
        for sentiment, count in stats.items():
            pourcentage = (count / len(df_final)) * 100
            print(f"   {sentiment}: {count} ({pourcentage:.1f}%)")
        
        # Scores moyens
        print(f"\n   📈 Score moyen: {df_final['score_final'].mean():.3f}")
        print(f"   🎯 Score médian: {df_final['score_final'].median():.3f}")
        print(f"   📏 Écart-type: {df_final['score_final'].std():.3f}")
        
        # Sarcasme détecté
        sarcasme_count = df_final['sarcasme_detecte'].sum()
        if sarcasme_count > 0:
            pourcentage_sarcasme = (sarcasme_count / len(df_final)) * 100
            print(f"   😏 Sarcasme détecté: {sarcasme_count} ({pourcentage_sarcasme:.1f}%)")
        
        # Top mots positifs/négatifs
        print("\n   🔍 Analyse lexicale:")
        print(f"      Score lexical moyen: {df_final['score_lexical'].mean():.3f}")
        print(f"      Score TextBlob moyen: {df_final['score_textblob'].mean():.3f}")
        
        # Comparaison avec étiquettes manuelles si disponible
        if 'New_Sentiment' in df_final.columns:
            print("\n" + "=" * 70)
            print("🎯 COMPARAISON AVEC ÉTIQUETTES MANUELLES")
            print("=" * 70)
            
            # Simplifier les catégories pour comparaison
            def simplifier_sentiment(s):
                if isinstance(s, str):
                    if 'POSITIF' in s:
                        return 'POSITIF'
                    elif 'NÉGATIF' in s:
                        return 'NÉGATIF'
                    else:
                        return 'NEUTRE'
                return 'NEUTRE'
            
            df_final['sentiment_simple'] = df_final['sentiment'].apply(simplifier_sentiment)
            df_final['New_Sentiment_simple'] = df_final['New_Sentiment'].apply(
                lambda x: 'POSITIF' if 'Positif' in str(x) else 
                         'NÉGATIF' if 'Négatif' in str(x) else 'NEUTRE'
            )
            
            matches = (df_final['sentiment_simple'] == df_final['New_Sentiment_simple'])
            match_rate = matches.mean() * 100
            
            print(f"   ✅ Correspondance exacte: {match_rate:.1f}%")
            print(f"   📊 Nombre de correspondances: {matches.sum()}/{len(df_final)}")
            
            # Matrice de confusion détaillée
            from sklearn.metrics import classification_report
            print("\n   📋 Rapport de classification:")
            print(classification_report(
                df_final['New_Sentiment_simple'], 
                df_final['sentiment_simple'],
                target_names=['POSITIF', 'NÉGATIF', 'NEUTRE']
            ))
        
        # Sauvegarder les résultats
        chemin_sortie = chemin_fichier.replace('.xlsx', '_analyse_detaille.xlsx')
        df_final.to_excel(chemin_sortie, index=False)
        print(f"\n💾 Résultats détaillés sauvegardés dans: {chemin_sortie}")
        
        # Créer les graphiques
        print("\n🎨 Génération des graphiques...")
        creer_graphiques(df_final, dossier_output="graphiques_sentiment")
        print("✅ Graphiques sauvegardés dans le dossier 'graphiques_sentiment'")
        
        # Afficher un résumé exécutif
        print("\n" + "=" * 70)
        print("📋 RÉSUMÉ EXÉCUTIF")
        print("=" * 70)
        
        # Top 5 commentaires positifs
        print("\n🏆 TOP 5 COMMENTAIRES POSITIFS:")
        top_positifs = df_final.nlargest(5, 'score_final')
        for i, (idx, row) in enumerate(top_positifs.iterrows(), 1):
            texte = row['texte_original'][:60] + "..." if len(row['texte_original']) > 60 else row['texte_original']
            print(f"   {i}. Score: {row['score_final']:.3f} - Confiance: {row['confiance']:.1%}")
            print(f"      \"{texte}\"")
        
        # Top 5 commentaires négatifs
        print("\n💢 TOP 5 COMMENTAIRES NÉGATIFS:")
        top_negatifs = df_final.nsmallest(5, 'score_final')
        for i, (idx, row) in enumerate(top_negatifs.iterrows(), 1):
            texte = row['texte_original'][:60] + "..." if len(row['texte_original']) > 60 else row['texte_original']
            print(f"   {i}. Score: {row['score_final']:.3f} - Confiance: {row['confiance']:.1%}")
            print(f"      \"{texte}\"")
        
        # Sarcasmes les plus forts
        if sarcasme_count > 0:
            print("\n😏 TOP 3 SARCASMES DÉTECTÉS:")
            sarcasmes = df_final[df_final['sarcasme_detecte'] == True].nlargest(3, 'score_sarcasme')
            for i, (idx, row) in enumerate(sarcasmes.iterrows(), 1):
                texte = row['texte_original'][:60] + "..." if len(row['texte_original']) > 60 else row['texte_original']
                print(f"   {i}. Score sarcasme: {row['score_sarcasme']:.3f}")
                print(f"      \"{texte}\"")
        
        print("\n" + "=" * 70)
        print("✅ ANALYSE COMPLÉTÉE AVEC SUCCÈS")
        print("=" * 70)
        
        return df_final
        
    except Exception as e:
        print(f"\n❌ ERREUR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# ==============================================
# 4. EXÉCUTION PRINCIPALE
# ==============================================

if __name__ == "__main__":
    # Installation requise:
    # pip install pandas numpy matplotlib seaborn scikit-learn
    
    # Chemin du fichier
    chemin_fichier = r"C:\Users\Sy Savane Idriss\project_sentiment_fb\commentaire_election_gov.xlsx"
    
    # Exécuter l'analyse
    print("\n" + "="*70)
    print("🚀 DÉMARRAGE DE L'ANALYSE AVEC VISUALISATION")
    print("="*70)
    
    resultats = analyser_fichier_excel(chemin_fichier)
    
    if resultats is not None:
        # Afficher un mini-rapport
        print("\n📋 RAPPORT FINAL:")
        print(f"   Commentaires analysés: {len(resultats)}")
        print(f"   Score moyen global: {resultats['score_final'].mean():.3f}")
        print(f"   Sentiment dominant: {resultats['sentiment'].mode()[0]}")
        
        # Suggestions d'amélioration
        print("\n💡 SUGGESTIONS D'AMÉLIORATION:")
        print("   1. Vérifier les commentaires à faible confiance")
        print("   2. Analyser les expressions locales détectées")
        print("   3. Comparer avec les étiquettes manuelles")
        print("   4. Examiner les cas de sarcasme détectés")
        
        # Fichiers générés
        print("\n📁 FICHIERS GÉNÉRÉS:")
        print(f"   • {chemin_fichier.replace('.xlsx', '_analyse_detaille.xlsx')}")
        print(f"   • graphiques_sentiment/analyse_complete.png")
        print(f"   • graphiques_sentiment/evolution_sentiments.png")
        print(f"   • graphiques_sentiment/sarcasmes.png")
        
        if 'New_Sentiment' in resultats.columns:
            print(f"   • graphiques_sentiment/comparaison_sentiments.png")
            print(f"   • graphiques_sentiment/matrice_confusion.png")