import pandas as pd
import numpy as np
import re
import string
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os
import warnings
warnings.filterwarnings('ignore')

# ==============================================
# 1. CLASSE DE NETTOYAGE DE TEXTE POUR FRANÇAIS IVOIRIEN
# ==============================================

class NettoyeurTexteIvoirien:
    """Nettoyeur spécialisé pour le français ivoirien des commentaires Facebook"""
    
    def __init__(self):
        # Mots à conserver (noms propres, expressions locales)
        self.mots_speciaux = {
            'ivoirien', 'ivoirienne', 'côte', "côte d'ivoire", 'abidjan', 'yamoussoukro',
            'bouaké', 'daloa', 'korhogo', 'san-pédro', 'gagnoa',
            'ado', 'ouattara', 'alassane', 'rhdp', 'pdp', 'fpi',
            'gbagbo', 'soro', 'bedié', 'konan', 'gnangnan',
            'choco', 'éléphant', 'panthère',
            'ansta', 'anstats', 'anstat', 'institut', 'national', 'statistique'
        }
        
        # Expressions ivoiriennes à conserver
        self.expressions_locales = {
            'tchê', 'walaye', 'saya', 'gôh', 'faforo', 'allô', 'ayo',
            'wah', 'atchê', 'atché', 'atchè', 'atchi', 'atchî',
            'cé ma fa', 'c ma fa', 'ça va aller', 'ça va allé',
            'wéé', 'wé', 'weh', 'aïe', 'aïe aïe', 'aïe aïe aïe',
            'mon frère', 'ma soeur', 'mon cher', 'ma chère',
            'gros', 'grand', 'petit', 'jeune'
        }
        
        # Stop words français
        self.stop_words_fr = {
            'le', 'la', 'les', 'de', 'du', 'des', 'un', 'une', 'et', 'est', 'en', 
            'que', 'qui', 'dans', 'pour', 'par', 'sur', 'avec', 'sans', 'sous', 
            'dont', 'où', 'y', 'à', 'au', 'aux', 'ce', 'cet', 'cette', 'ces',
            'mon', 'ma', 'mes', 'ton', 'ta', 'tes', 'son', 'sa', 'ses', 'notre',
            'nos', 'votre', 'vos', 'leur', 'leurs', 'on', 'nous', 'vous', 'ils',
            'elles', 'eux', 'celui', 'celle', 'ceux', 'celles', 'aucun', 'aucune',
            'certains', 'certaines', 'plusieurs', 'tout', 'tous', 'toute', 'toutes',
            'même', 'comme', 'aussi', 'bien', 'très', 'plus', 'moins', 'peu',
            'beaucoup', 'trop', 'alors', 'donc', 'or', 'ni', 'car', 'mais', 'ou',
            'si', 'que', 'quand', 'comment', 'pourquoi', 'combien'
        }
        
    def nettoyer_texte(self, texte):
        """Nettoie un texte de commentaire Facebook"""
        if not isinstance(texte, str) or pd.isna(texte):
            return ""
        
        # Étape 1: Nettoyage de base
        texte = str(texte).lower()
        
        # Supprimer les URLs
        texte = re.sub(r'http\S+|www\S+|https\S+', '', texte, flags=re.MULTILINE)
        
        # Supprimer les mentions @
        texte = re.sub(r'@\w+', '', texte)
        
        # Supprimer les hashtags
        texte = re.sub(r'#\w+', '', texte)
        
        # Garder les émoticônes de base
        emoticons = re.findall(r'[:;=][\'\"-]?[)DdpP/\\|\[\]{}@*]', texte)
        
        # Remplacer les caractères spéciaux problématiques
        texte = re.sub(r'[âãàáä]', 'a', texte)
        texte = re.sub(r'[êëèé]', 'e', texte)
        texte = re.sub(r'[îïìí]', 'i', texte)
        texte = re.sub(r'[ôöòó]', 'o', texte)
        texte = re.sub(r'[ûüùú]', 'u', texte)
        texte = re.sub(r'[ç]', 'c', texte)
        
        # Supprimer les nombres seuls
        texte = re.sub(r'\b\d+\b', '', texte)
        
        # Étape 2: Gestion des répétitions de caractères
        texte = re.sub(r'(.)\1{2,}', r'\1\1', texte)  # "bonjourrrrr" -> "bonjourr"
        
        # Étape 3: Tokenisation et nettoyage
        tokens = re.findall(r'\b\w+\b', texte)
        
        # Filtrer les tokens
        tokens_filtres = []
        for token in tokens:
            if len(token) <= 1:
                continue
            
            # Garder les mots spéciaux
            if token in self.mots_speciaux:
                tokens_filtres.append(token)
                continue
            
            # Garder les expressions locales
            if token in self.expressions_locales:
                tokens_filtres.append(token)
                continue
            
            # Supprimer les stop words
            if token in self.stop_words_fr:
                continue
            
            # Vérifier si c'est un mot valide (au moins 2 lettres)
            if len(token) >= 2:
                tokens_filtres.append(token)
        
        # Étape 4: Reconstruire le texte
        texte_nettoye = ' '.join(tokens_filtres)
        
        # Ajouter les émoticônes conservées
        if emoticons:
            texte_nettoye += ' ' + ' '.join(emoticons)
        
        # Supprimer les espaces multiples
        texte_nettoye = re.sub(r'\s+', ' ', texte_nettoye).strip()
        
        return texte_nettoye
    
    def nettoyer_dataframe(self, df, colonne_texte):
        """Nettoie une colonne de texte dans un DataFrame"""
        print(f"🧹 Nettoyage de la colonne '{colonne_texte}'...")
        
        # Appliquer le nettoyage
        df['Text_Clean'] = df[colonne_texte].apply(self.nettoyer_texte)
        
        # Statistiques de nettoyage
        total_originaux = df[colonne_texte].dropna().shape[0]
        total_nettoyes = df['Text_Clean'][df['Text_Clean'] != ''].shape[0]
        
        print(f"   ✅ Textes originaux: {total_originaux}")
        print(f"   ✅ Textes nettoyés: {total_nettoyes}")
        print(f"   📉 Réduction moyenne: {((total_originaux - total_nettoyes)/total_originaux*100):.1f}%")
        
        return df

# ==============================================
# 2. ANALYSEUR DE SENTIMENTS POUR FRANÇAIS IVOIRIEN
# ==============================================

class AnalyseurSentimentsIvoirien:
    """Analyseur de sentiments spécialisé pour le contexte ivoirien"""
    
    def __init__(self):
        self.lexique = self._creer_lexique_ivoirien()
        
    def _creer_lexique_ivoirien(self):
        """Crée un lexique de sentiments adapté au contexte ivoirien"""
        lexique = {
            # ========== POSITIF ==========
            'bon': 0.6, 'bonne': 0.6, 'bien': 0.5, 'excellent': 0.9,
            'parfait': 0.8, 'super': 0.7, 'génial': 0.7, 'formidable': 0.7,
            'félicitations': 0.8, 'félicitation': 0.8, 'féliciter': 0.7,
            'bravo': 0.7, 'merci': 0.4, 'remercier': 0.4,
            'content': 0.6, 'heureux': 0.7, 'satisfait': 0.6,
            'utile': 0.5, 'efficace': 0.6, 'pratique': 0.5,
            'clair': 0.4, 'précis': 0.5, 'détaillé': 0.4,
            'progress': 0.5, 'progrès': 0.5, 'amélioration': 0.5,
            'mécanique': 0.3,  # positif dans contexte ivoirien
            'choco': 0.4,  # supporters positifs
            
            # ========== NÉGATIF ==========
            'mauvais': -0.6, 'mal': -0.5, 'nul': -0.7, 'nulle': -0.7,
            'pire': -0.8, 'horrible': -0.8, 'terrible': -0.7,
            'problème': -0.4, 'difficulté': -0.4, 'erreur': -0.5,
            'faux': -0.6, 'incorrect': -0.5, 'inexact': -0.5,
            'incompréhensible': -0.4, 'confus': -0.4, 'compliqué': -0.3,
            'cher': -0.3, 'coûteux': -0.4, 'trop': -0.2,
            'manque': -0.3, 'absent': -0.4, 'insuffisant': -0.4,
            'tchê': -0.5,  # expression négative ivoirienne
            'saya': -0.6,  # très négatif
            
            # ========== CONTEXTE STATISTIQUE ==========
            'statistique': 0.0, 'donnée': 0.0, 'chiffre': 0.0,
            'étude': 0.0, 'recherche': 0.1, 'analyse': 0.0,
            'enquête': 0.0, 'sondage': 0.0, 'résultat': 0.0,
            'rapport': 0.0, 'publication': 0.0, 'information': 0.1,
            'anstat': 0.0, 'institut': 0.0, 'national': 0.0,
            
            # ========== NEUTRE/ADMINISTRATIF ==========
            'question': 0.0, 'demande': 0.0, 'réponse': 0.0,
            'explication': 0.0, 'détail': 0.0, 'exemple': 0.0,
            'ministère': 0.0, 'gouvernement': 0.0, 'administration': 0.0,
            'service': 0.0, 'public': 0.0, 'citoyen': 0.0,
            'population': 0.0, 'habitant': 0.0, 'résident': 0.0,
        }
        return lexique
    
    def analyser_sentiment_texte(self, texte):
        """Analyse le sentiment d'un texte"""
        if not texte or not isinstance(texte, str):
            return {'score': 0, 'sentiment': 'NEUTRE', 'confiance': 0}
        
        # Tokenisation
        mots = re.findall(r'\b\w+\b', texte.lower())
        
        if not mots:
            return {'score': 0, 'sentiment': 'NEUTRE', 'confiance': 0}
        
        # Calcul du score
        scores = []
        negation = False
        
        for mot in mots:
            if mot in ['pas', 'non', 'jamais', 'rien', 'aucun']:
                negation = True
                continue
            
            if mot in self.lexique:
                score = self.lexique[mot]
                if negation:
                    score = -score * 0.7
                    negation = False
                scores.append(score)
            else:
                negation = False
        
        if scores:
            score_moyen = np.mean(scores)
            confiance = min(len(scores) / 10, 1.0)
        else:
            score_moyen = 0
            confiance = 0
        
        # Déterminer la catégorie
        if score_moyen > 0.1:
            sentiment = 'POSITIF'
        elif score_moyen < -0.1:
            sentiment = 'NÉGATIF'
        else:
            sentiment = 'NEUTRE'
        
        return {
            'score': round(score_moyen, 3),
            'sentiment': sentiment,
            'confiance': round(confiance, 2),
            'mots_analysés': len(scores)
        }
    
    def analyser_dataframe(self, df, colonne_texte='Text_Clean'):
        """Analyse les sentiments d'une colonne de DataFrame"""
        print(f"🔍 Analyse des sentiments de '{colonne_texte}'...")
        
        resultats = []
        
        for idx, texte in enumerate(df[colonne_texte]):
            if pd.isna(texte) or texte == '':
                resultats.append({'score': 0, 'sentiment': 'NEUTRE', 'confiance': 0})
            else:
                resultats.append(self.analyser_sentiment_texte(str(texte)))
            
            # Afficher la progression
            if (idx + 1) % 50 == 0:
                print(f"   ✓ {idx + 1}/{len(df)} textes analysés")
        
        # Créer un DataFrame de résultats
        df_resultats = pd.DataFrame(resultats)
        
        # Fusionner avec le DataFrame original
        df['score_sentiment'] = df_resultats['score']
        df['sentiment'] = df_resultats['sentiment']
        df['confiance_sentiment'] = df_resultats['confiance']
        
        print(f"✅ Analyse terminée: {len(df)} commentaires traités")
        
        return df

# ==============================================
# 3. GÉNÉRATEUR DE RAPPORT
# ==============================================

class GenerateurRapportANSTAT:
    """Générateur de rapport d'analyse pour ANSTAT"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.resultats = {}
        
    def analyser_distribution(self):
        """Analyse la distribution des sentiments"""
        if 'sentiment' not in self.df.columns:
            raise ValueError("La colonne 'sentiment' n'existe pas dans le DataFrame")
        
        distribution = self.df['sentiment'].value_counts()
        pourcentages = (distribution / len(self.df) * 100).round(1)
        
        self.resultats['distribution'] = {
            'counts': distribution.to_dict(),
            'percentages': pourcentages.to_dict()
        }
        
        return self
    
    def analyser_tendance_temporelle(self):
        """Analyse l'évolution des sentiments dans le temps"""
        if 'Post Date-Time' not in self.df.columns:
            print("⚠️  Colonne 'Post Date-Time' non trouvée, analyse temporelle ignorée")
            return self
        
        # Convertir en datetime
        self.df['date'] = pd.to_datetime(self.df['Post Date-Time']).dt.date
        
        # Grouper par jour
        daily_stats = self.df.groupby('date').agg({
            'score_sentiment': 'mean',
            'sentiment': lambda x: (x == 'POSITIF').sum() / len(x) * 100
        }).rename(columns={'sentiment': 'pct_positif'})
        
        self.resultats['tendance_temporelle'] = daily_stats
        
        return self
    
    def analyser_mots_cles(self, top_n=20):
        """Analyse les mots-clés les plus fréquents"""
        if 'Text_Clean' not in self.df.columns:
            print("⚠️  Colonne 'Text_Clean' non trouvée")
            return self
        
        # Concaténer tous les textes
        all_text = ' '.join(self.df['Text_Clean'].dropna().astype(str).tolist())
        
        # Extraire les mots
        words = re.findall(r'\b[a-zéèêëàâäôöûüç]{3,}\b', all_text.lower())
        
        # Compter les occurrences
        word_counts = Counter(words)
        
        # Filtrer les mots communs non informatifs
        common_words = {'que', 'est', 'pas', 'pour', 'dans', 'avec', 'mais', 'son', 'ses',
                       'une', 'des', 'les', 'aux', 'du', 'de', 'la', 'le', 'et', 'ou'}
        
        filtered_counts = {word: count for word, count in word_counts.items() 
                          if word not in common_words}
        
        # Top N mots
        top_words = dict(sorted(filtered_counts.items(), 
                               key=lambda x: x[1], reverse=True)[:top_n])
        
        self.resultats['mots_cles'] = top_words
        
        return self
    
    def analyser_longueur_textes(self):
        """Analyse la longueur des textes"""
        if 'Text_Clean' not in self.df.columns:
            return self
        
        self.df['longueur_texte'] = self.df['Text_Clean'].apply(
            lambda x: len(str(x).split()) if pd.notna(x) else 0
        )
        
        stats = {
            'moyenne': self.df['longueur_texte'].mean(),
            'mediane': self.df['longueur_texte'].median(),
            'max': self.df['longueur_texte'].max(),
            'min': self.df['longueur_texte'].min()
        }
        
        self.resultats['longueur_textes'] = stats
        
        return self
    
    def generer_rapport_texte(self, chemin_sortie=None):
        """Génère un rapport texte détaillé"""
        rapport = []
        
        # En-tête
        rapport.append("="*80)
        rapport.append("RAPPORT D'ANALYSE DES COMMENTAIRES - ANSTAT CÔTE D'IVOIRE")
        rapport.append("="*80)
        rapport.append("")
        
        # 1. Synthèse
        rapport.append("1. SYNTHÈSE DE L'ANALYSE")
        rapport.append("-"*40)
        rapport.append(f"Période d'analyse : {self.df['date'].min() if 'date' in self.df.columns else 'N/A'} "
                      f"au {self.df['date'].max() if 'date' in self.df.columns else 'N/A'}")
        rapport.append(f"Nombre total de commentaires : {len(self.df)}")
        rapport.append("")
        
        # 2. Distribution des sentiments
        rapport.append("2. DISTRIBUTION DES SENTIMENTS")
        rapport.append("-"*40)
        
        if 'distribution' in self.resultats:
            for sentiment, count in self.resultats['distribution']['counts'].items():
                pct = self.resultats['distribution']['percentages'].get(sentiment, 0)
                rapport.append(f"  • {sentiment:10} : {count:4d} commentaires ({pct:5.1f}%)")
        rapport.append("")
        
        # 3. Scores moyens
        rapport.append("3. SCORES MOYENS")
        rapport.append("-"*40)
        if 'score_sentiment' in self.df.columns:
            rapport.append(f"  Score moyen global : {self.df['score_sentiment'].mean():.3f}")
            rapport.append(f"  Score médian : {self.df['score_sentiment'].median():.3f}")
            
            # Scores par catégorie
            for sentiment in self.df['sentiment'].unique():
                score_moyen = self.df[self.df['sentiment'] == sentiment]['score_sentiment'].mean()
                rapport.append(f"  • {sentiment} : {score_moyen:.3f}")
        rapport.append("")
        
        # 4. Mots-clés
        rapport.append("4. THÉMATIQUES PRINCIPALES")
        rapport.append("-"*40)
        
        if 'mots_cles' in self.resultats:
            rapport.append("  Mots les plus fréquents :")
            for i, (mot, freq) in enumerate(self.resultats['mots_cles'].items(), 1):
                rapport.append(f"    {i:2d}. {mot:15} : {freq:3d} mentions")
                if i >= 10:  # Limiter à 10 mots
                    break
        rapport.append("")
        
        # 5. Analyse temporelle
        if 'tendance_temporelle' in self.resultats:
            rapport.append("5. ÉVOLUTION TEMPORELLE")
            rapport.append("-"*40)
            
            daily = self.resultats['tendance_temporelle']
            rapport.append(f"  Score moyen quotidien : {daily['score_sentiment'].mean():.3f}")
            rapport.append(f"  % moyen de positivité : {daily['pct_positif'].mean():.1f}%")
            rapport.append("")
        
        # 6. Exemples significatifs
        rapport.append("6. EXEMPLES DE COMMENTAIRES")
        rapport.append("-"*40)
        
        # Exemples positifs
        positifs = self.df[self.df['sentiment'] == 'POSITIF'].sort_values('score_sentiment', ascending=False).head(3)
        if len(positifs) > 0:
            rapport.append("  Commentaires positifs :")
            for idx, row in positifs.iterrows():
                texte = str(row['Text_Clean'])[:80] + "..." if len(str(row['Text_Clean'])) > 80 else str(row['Text_Clean'])
                rapport.append(f"    ✓ Score: {row['score_sentiment']:.3f} - \"{texte}\"")
        
        # Exemples négatifs
        negatifs = self.df[self.df['sentiment'] == 'NÉGATIF'].sort_values('score_sentiment').head(2)
        if len(negatifs) > 0:
            rapport.append("\n  Commentaires négatifs :")
            for idx, row in negatifs.iterrows():
                texte = str(row['Text_Clean'])[:80] + "..." if len(str(row['Text_Clean'])) > 80 else str(row['Text_Clean'])
                rapport.append(f"    ✗ Score: {row['score_sentiment']:.3f} - \"{texte}\"")
        rapport.append("")
        
        # 7. Recommandations
        rapport.append("7. RECOMMANDATIONS")
        rapport.append("-"*40)
        
        if 'distribution' in self.resultats:
            pct_positif = self.resultats['distribution']['percentages'].get('POSITIF', 0)
            pct_negatif = self.resultats['distribution']['percentages'].get('NÉGATIF', 0)
            
            if pct_positif > 30:
                rapport.append("  ✅ Engagement très positif")
                rapport.append("     → Capitaliser sur cette dynamique en mettant en avant")
                rapport.append("       les retours positifs")
            elif pct_negatif > 20:
                rapport.append("  ⚠️  Niveau de critique élevé")
                rapport.append("     → Analyser en détail les préoccupations")
                rapport.append("     → Répondre systématiquement aux commentaires")
            else:
                rapport.append("  ⚖️  Situation équilibrée")
                rapport.append("     → Maintenir le dialogue avec les citoyens")
                rapport.append("     → Renforcer la communication sur les données")
        
        rapport.append("")
        
        # 8. Conclusion
        rapport.append("8. CONCLUSION")
        rapport.append("-"*40)
        rapport.append("  Cette analyse révèle les perceptions des citoyens vis-à-vis")
        rapport.append("  des publications de l'ANSTAT. Les résultats peuvent servir")
        rapport.append("  à améliorer la communication et l'engagement citoyen.")
        rapport.append("")
        
        # Pied de page
        rapport.append("="*80)
        rapport.append(f"Généré le : {datetime.now().strftime('%d/%m/%Y %H:%M')}")
        rapport.append("Méthodologie : Nettoyage de texte + Analyse lexicale adaptée")
        rapport.append("="*80)
        
        # Convertir en texte
        rapport_texte = '\n'.join(rapport)
        
        # Sauvegarder
        if chemin_sortie:
            with open(chemin_sortie, 'w', encoding='utf-8') as f:
                f.write(rapport_texte)
            print(f"✅ Rapport sauvegardé : {chemin_sortie}")
        else:
            # Sauvegarde par défaut
            dossier_rapports = 'rapports_anstat'
            os.makedirs(dossier_rapports, exist_ok=True)
            
            nom_fichier = f"rapport_anstat_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
            chemin_sauvegarde = os.path.join(dossier_rapports, nom_fichier)
            
            with open(chemin_sauvegarde, 'w', encoding='utf-8') as f:
                f.write(rapport_texte)
            print(f"✅ Rapport sauvegardé : {chemin_sauvegarde}")
        
        return rapport_texte
    
    def generer_graphiques(self, dossier_output="graphiques_anstat"):
        """Génère des graphiques d'analyse"""
        os.makedirs(dossier_output, exist_ok=True)
        
        # 1. Camembert des sentiments
        plt.figure(figsize=(10, 8))
        if 'distribution' in self.resultats:
            labels = list(self.resultats['distribution']['counts'].keys())
            sizes = list(self.resultats['distribution']['counts'].values())
            colors = ['#4CAF50', '#F44336', '#FFC107'][:len(labels)]
            
            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            plt.title('Distribution des Sentiments - ANSTAT', fontsize=14, fontweight='bold')
            plt.axis('equal')
            plt.savefig(f'{dossier_output}/distribution_sentiments.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        # 2. Histogramme des scores
        plt.figure(figsize=(12, 6))
        plt.hist(self.df['score_sentiment'], bins=30, color='#2196F3', edgecolor='black', alpha=0.7)
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Neutre')
        plt.xlabel('Score de Sentiment')
        plt.ylabel('Nombre de Commentaires')
        plt.title('Distribution des Scores de Sentiment', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{dossier_output}/histogramme_scores.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. Top mots
        if 'mots_cles' in self.resultats:
            plt.figure(figsize=(12, 6))
            mots = list(self.resultats['mots_cles'].keys())[:15]
            frequences = list(self.resultats['mots_cles'].values())[:15]
            
            plt.barh(range(len(mots)), frequences, color='#FF9800')
            plt.yticks(range(len(mots)), mots)
            plt.xlabel('Fréquence')
            plt.title('Top 15 des Mots les Plus Fréquents', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(f'{dossier_output}/top_mots.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"✅ Graphiques sauvegardés dans '{dossier_output}'")
        
        return dossier_output

# ==============================================
# 4. PIPELINE COMPLÈTE D'ANALYSE
# ==============================================

def analyser_fichier_anstat(chemin_fichier):
    """Pipeline complète d'analyse du fichier ANSTAT"""
    
    print("🚀 DÉMARRAGE DE L'ANALYSE ANSTAT")
    print("="*60)
    
    try:
        # 1. Charger le fichier
        print(f"\n📂 Chargement du fichier: {chemin_fichier}")
        df = pd.read_excel(chemin_fichier)
        print(f"✅ {len(df)} commentaires chargés")
        print(f"   Colonnes disponibles: {list(df.columns)}")
        
        # 2. Nettoyer les textes
        print("\n🧹 PHASE 1: NETTOYAGE DES TEXTES")
        nettoyeur = NettoyeurTexteIvoirien()
        
        # Identifier la colonne de texte
        if 'Comment Text' in df.columns:
            colonne_texte = 'Comment Text'
        elif 'texte_original' in df.columns:
            colonne_texte = 'texte_original'
        else:
            # Chercher une colonne contenant du texte
            text_columns = [col for col in df.columns if 'text' in col.lower() or 'comment' in col.lower()]
            if text_columns:
                colonne_texte = text_columns[0]
            else:
                colonne_texte = df.columns[1]  # Deuxième colonne par défaut
        
        print(f"   Colonne de texte identifiée: '{colonne_texte}'")
        
        # Nettoyer
        df = nettoyeur.nettoyer_dataframe(df, colonne_texte)
        
        # 3. Analyser les sentiments
        print("\n🔍 PHASE 2: ANALYSE DES SENTIMENTS")
        analyseur = AnalyseurSentimentsIvoirien()
        df = analyseur.analyser_dataframe(df, 'Text_Clean')
        
        # 4. Générer le rapport
        print("\n📊 PHASE 3: GÉNÉRATION DU RAPPORT")
        generateur = GenerateurRapportANSTAT(df)
        
        # Exécuter les analyses
        generateur.analyser_distribution()
        generateur.analyser_tendance_temporelle()
        generateur.analyser_mots_cles()
        generateur.analyser_longueur_textes()
        
        # Générer le rapport texte
        rapport = generateur.generer_rapport_texte()
        
        # Générer les graphiques
        generateur.generer_graphiques()
        
        # 5. Sauvegarder les résultats
        print("\n💾 PHASE 4: SAUVEGARDE DES RÉSULTATS")
        
        # Créer le dossier de sortie
        dossier_resultats = 'resultats_anstat'
        os.makedirs(dossier_resultats, exist_ok=True)
        
        # Sauvegarder le DataFrame avec les résultats
        nom_fichier_sortie = os.path.basename(chemin_fichier).replace('.xlsx', '_analyse_complete.xlsx')
        chemin_sortie = os.path.join(dossier_resultats, nom_fichier_sortie)
        
        df.to_excel(chemin_sortie, index=False)
        print(f"✅ Données analysées sauvegardées: {chemin_sortie}")
        
        # Statistiques finales
        print("\n" + "="*60)
        print("📈 STATISTIQUES FINALES")
        print("="*60)
        
        distribution = generateur.resultats.get('distribution', {})
        if 'counts' in distribution:
            for sentiment, count in distribution['counts'].items():
                pct = distribution['percentages'].get(sentiment, 0)
                print(f"   {sentiment:10}: {count:4d} ({pct:5.1f}%)")
        
        if 'score_sentiment' in df.columns:
            print(f"\n   Score moyen global: {df['score_sentiment'].mean():.3f}")
            print(f"   Score médian: {df['score_sentiment'].median():.3f}")
        
        # Afficher un extrait du rapport
        print("\n" + "="*60)
        print("📋 EXTRAIT DU RAPPORT")
        print("="*60)
        lignes_rapport = rapport.split('\n')
        for ligne in lignes_rapport[:20]:  # Afficher les 20 premières lignes
            print(ligne)
        
        print("\n" + "="*60)
        print("✅ ANALYSE COMPLÉTÉE AVEC SUCCÈS")
        print("="*60)
        
        return df
        
    except Exception as e:
        print(f"\n❌ ERREUR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# ==============================================
# 5. EXÉCUTION PRINCIPALE
# ==============================================

if __name__ == "__main__":
    # Chemin du fichier ANSTAT
    chemin_fichier = r"C:\Users\Sy Savane Idriss\project_sentiment_fb\data\Commentaire_ANSTAT_4_semaine.xlsx"
    
    # Vérifier si le fichier existe
    if not os.path.exists(chemin_fichier):
        print(f"❌ Fichier introuvable: {chemin_fichier}")
        print("Vérifiez le chemin et réessayez.")
    else:
        # Exécuter l'analyse complète
        resultats = analyser_fichier_anstat(chemin_fichier)
        
        if resultats is not None:
            print("\n📁 FICHIERS GÉNÉRÉS :")
            print("   • resultats_anstat/ : Données analysées")
            print("   • rapports_anstat/ : Rapport texte")
            print("   • graphiques_anstat/ : Graphiques d'analyse")
            
            print("\n🎯 PROCHAINES ÉTAPES :")
            print("   1. Consulter le rapport texte complet")
            print("   2. Examiner les graphiques générés")
            print("   3. Analyser les mots-clés identifiés")
            print("   4. Adapter la stratégie de communication")