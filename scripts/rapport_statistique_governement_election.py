import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from collections import Counter
import re

class RapportAnalyseSentiments:
    def __init__(self, chemin_fichier):
        """Initialise l'analyseur avec le chemin du fichier"""
        self.chemin_fichier = chemin_fichier
        self.df = None
        self.resultats = {}
        
    def charger_donnees(self):
        """Charge et prépare les données"""
        print("📂 Chargement des données...")
        self.df = pd.read_excel(self.chemin_fichier)
        print(f"✅ {len(self.df)} commentaires chargés")
        
        # Nettoyage des données
        self.df['Post Date-Time'] = pd.to_datetime(self.df['Post Date-Time'], errors='coerce')
        if 'sentiment' in self.df.columns:
            self.df['sentiment'] = self.df['sentiment'].fillna('NEUTRE')
        else:
            self.df['sentiment'] = 'NEUTRE'
            
        # Extraire date et heure séparément
        self.df['Date'] = self.df['Post Date-Time'].dt.date
        self.df['Heure'] = self.df['Post Date-Time'].dt.hour
        
        return self
    
    def analyser_distribution_sentiments(self):
        """Analyse la distribution des sentiments"""
        print("📊 Analyse des sentiments...")
        
        # Distribution simple
        distribution = self.df['sentiment'].value_counts()
        
        # Regrouper les catégories similaires
        sentiment_groups = {
            'POSITIF': ['POSITIF', 'POSITIVE', 'FAVORABLE'],
            'NÉGATIF': ['NÉGATIF', 'NEGATIF', 'NÉGATIVE', 'NEGATIVE'],
            'NEUTRE': ['NEUTRE', 'NEUTRAL'],
            'MIXTE': ['MIXTE', 'MIXED']
        }
        
        grouped_dist = {}
        for group, variants in sentiment_groups.items():
            count = 0
            for variant in variants:
                count += distribution.get(variant, 0)
            if count > 0:
                grouped_dist[group] = count
        
        # Calculer les pourcentages
        total = len(self.df)
        pourcentages = {k: (v/total*100) for k, v in grouped_dist.items()}
        
        self.resultats['distribution'] = grouped_dist
        self.resultats['pourcentages'] = pourcentages
        self.resultats['total'] = total
        
        return self
    
    def analyser_tendance_temporelle(self):
        """Analyse l'évolution des sentiments dans le temps"""
        print("📈 Analyse temporelle...")
        
        if 'Post Date-Time' in self.df.columns:
            # Trier par date
            df_sorted = self.df.sort_values('Post Date-Time')
            
            # Grouper par jour
            df_sorted['Jour'] = df_sorted['Post Date-Time'].dt.date
            daily_stats = df_sorted.groupby('Jour').agg({
                'sentiment': lambda x: (x.str.contains('POSITIF').sum() / len(x) * 100) if len(x) > 0 else 0
            }).rename(columns={'sentiment': '%_Positif'})
            
            # Période d'analyse
            date_debut = df_sorted['Post Date-Time'].min().date()
            date_fin = df_sorted['Post Date-Time'].max().date()
            periode_jours = (date_fin - date_debut).days + 1
            
            self.resultats['periode'] = {
                'debut': date_debut,
                'fin': date_fin,
                'jours': periode_jours
            }
            
            self.resultats['tendance_quotidienne'] = daily_stats
            
        return self
    
    def analyser_mots_cles(self):
        """Analyse les mots-clés les plus fréquents"""
        print("🔍 Analyse des mots-clés...")
        
        # Extraire tous les textes propres
        if 'Text_Clean' in self.df.columns:
            textes = ' '.join(self.df['Text_Clean'].dropna().astype(str).tolist())
        else:
            textes = ' '.join(self.df['texte_original'].dropna().astype(str).tolist())
        
        # Nettoyer et compter les mots
        mots = re.findall(r'\b[a-zéèêëàâäôöûüç]{3,}\b', textes.lower())
        
        # Liste de mots à exclure (stop words français)
        stop_words = {'les', 'des', 'une', 'pour', 'dans', 'avec', 'mais', 'est', 'son', 'ses',
                     'que', 'qui', 'quoi', 'dont', 'où', 'quand', 'comment', 'pourquoi',
                     'sur', 'sous', 'dans', 'avec', 'sans', 'dans', 'aux', 'du', 'de', 'la',
                     'le', 'un', 'une', 'et', 'ou', 'où', 'à', 'au', 'aux', 'en', 'dans',
                     'par', 'pour', 'sur', 'vers', 'avec', 'sans', 'sous', 'chez'}
        
        # Filtrer les mots pertinents
        mots_filtres = [mot for mot in mots if mot not in stop_words and len(mot) > 2]
        
        # Compter les occurrences
        compteur = Counter(mots_filtres)
        top_mots = dict(compteur.most_common(20))
        
        self.resultats['top_mots'] = top_mots
        
        # Analyser par sentiment
        sentiments_analysis = {}
        for sentiment in self.df['sentiment'].unique():
            mask = self.df['sentiment'] == sentiment
            textes_sentiment = ' '.join(self.df[mask]['Text_Clean'].dropna().astype(str).tolist())
            mots_sentiment = re.findall(r'\b[a-zéèêëàâäôöûüç]{3,}\b', textes_sentiment.lower())
            mots_sentiment_filtres = [m for m in mots_sentiment if m not in stop_words]
            sentiments_analysis[sentiment] = dict(Counter(mots_sentiment_filtres).most_common(10))
        
        self.resultats['mots_par_sentiment'] = sentiments_analysis
        
        return self
    
    def analyser_auteurs(self):
        """Analyse des auteurs les plus actifs"""
        print("👥 Analyse des auteurs...")
        
        if 'Author' in self.df.columns:
            # Auteurs les plus actifs
            top_auteurs = self.df['Author'].value_counts().head(10).to_dict()
            
            # Sentiment par auteur
            auteurs_sentiments = {}
            for auteur in list(top_auteurs.keys())[:5]:
                df_auteur = self.df[self.df['Author'] == auteur]
                if len(df_auteur) > 0:
                    sentiments = df_auteur['sentiment'].value_counts().to_dict()
                    auteurs_sentiments[auteur] = {
                        'total_posts': len(df_auteur),
                        'sentiments': sentiments
                    }
            
            self.resultats['top_auteurs'] = top_auteurs
            self.resultats['analyse_auteurs'] = auteurs_sentiments
            
        return self
    
    def analyser_heures_activite(self):
        """Analyse des heures d'activité"""
        print("⏰ Analyse des horaires...")
        
        if 'Heure' in self.df.columns:
            # Distribution par heure
            distribution_heures = self.df['Heure'].value_counts().sort_index().to_dict()
            
            # Heures de pointe
            heures_pointe = sorted(distribution_heures.items(), key=lambda x: x[1], reverse=True)[:3]
            
            self.resultats['distribution_heures'] = distribution_heures
            self.resultats['heures_pointe'] = heures_pointe
            
        return self
    
    def generer_rapport_texte(self, chemin_sortie=None):
        """Génère un rapport texte détaillé"""
        print("\n📄 Génération du rapport...")
        
        rapport = []
        
        # En-tête du rapport
        rapport.append("="*80)
        rapport.append("RAPPORT D'ANALYSE DES SENTIMENTS - PAGE FACEBOOK GOUVERNEMENT IVOIRIEN")
        rapport.append("="*80)
        rapport.append("")
        
        # 1. SYNOPSIS
        rapport.append("1. SYNOPSIS")
        rapport.append("-"*40)
        rapport.append(f"Période analysée : {self.resultats.get('periode', {}).get('debut', 'N/A')} au "
                      f"{self.resultats.get('periode', {}).get('fin', 'N/A')}")
        rapport.append(f"Durée : {self.resultats.get('periode', {}).get('jours', 'N/A')} jours")
        rapport.append(f"Total commentaires analysés : {self.resultats.get('total', 0)}")
        rapport.append("")
        
        # 2. DISTRIBUTION DES SENTIMENTS
        rapport.append("2. DISTRIBUTION DES SENTIMENTS")
        rapport.append("-"*40)
        
        for sentiment, count in self.resultats.get('distribution', {}).items():
            pourcentage = self.resultats.get('pourcentages', {}).get(sentiment, 0)
            rapport.append(f"  • {sentiment} : {count} commentaires ({pourcentage:.1f}%)")
        
        # Interprétation
        rapport.append("\n  Interprétation :")
        if self.resultats.get('pourcentages', {}).get('POSITIF', 0) > 30:
            rapport.append("  → Engagement global très positif")
        elif self.resultats.get('pourcentages', {}).get('NÉGATIF', 0) > 20:
            rapport.append("  → Niveau de critique élevé à surveiller")
        else:
            rapport.append("  → Tonalité globalement équilibrée")
        rapport.append("")
        
        # 3. TENDANCE TEMPORELLE
        rapport.append("3. ÉVOLUTION TEMPORELLE")
        rapport.append("-"*40)
        
        if 'tendance_quotidienne' in self.resultats:
            rapport.append("  Évolution du taux de positivité :")
            for jour, taux in list(self.resultats['tendance_quotidienne']['%_Positif'].items())[:5]:
                rapport.append(f"    {jour} : {taux:.1f}% de positivité")
        rapport.append("")
        
        # 4. MOTS-CLÉS PRINCIPAUX
        rapport.append("4. VOCABULAIRE ET THÉMATIQUES")
        rapport.append("-"*40)
        
        rapport.append("  Mots les plus fréquents :")
        for mot, freq in list(self.resultats.get('top_mots', {}).items())[:10]:
            rapport.append(f"    • {mot} : {freq} occurrences")
        
        # Analyse par sentiment
        rapport.append("\n  Vocabulaire par catégorie de sentiment :")
        for sentiment, mots_dict in self.resultats.get('mots_par_sentiment', {}).items():
            if mots_dict:
                top_mot = list(mots_dict.keys())[0] if mots_dict else "N/A"
                rapport.append(f"    {sentiment} : {top_mot} (et {len(mots_dict)-1} autres)")
        rapport.append("")
        
        # 5. ANALYSE DES AUTEURS
        if 'top_auteurs' in self.resultats:
            rapport.append("5. PROFIL DES CONTRIBUTEURS")
            rapport.append("-"*40)
            
            rapport.append("  Auteurs les plus actifs :")
            for auteur, count in list(self.resultats['top_auteurs'].items())[:5]:
                rapport.append(f"    • {auteur} : {count} commentaires")
            
            if 'analyse_auteurs' in self.resultats:
                rapport.append("\n  Profil de contribution :")
                for auteur, data in self.resultats['analyse_auteurs'].items():
                    sentiments = ', '.join([f"{k}({v})" for k, v in data['sentiments'].items()])
                    rapport.append(f"    {auteur} : {data['total_posts']} posts [{sentiments}]")
            rapport.append("")
        
        # 6. RYTHME D'ACTIVITÉ
        if 'heures_pointe' in self.resultats:
            rapport.append("6. RYTHME D'ACTIVITÉ")
            rapport.append("-"*40)
            
            rapport.append("  Heures de pointe (plus d'engagement) :")
            for heure, count in self.resultats['heures_pointe']:
                rapport.append(f"    • {heure}h00 : {count} commentaires")
            rapport.append("")
        
        # 7. EXEMPLES SIGNIFICATIFS
        rapport.append("7. EXEMPLES SIGNIFICATIFS")
        rapport.append("-"*40)
        
        # Exemples positifs
        if 'POSITIF' in self.resultats.get('distribution', {}):
            df_positifs = self.df[self.df['sentiment'].str.contains('POSITIF')].head(3)
            rapport.append("  Commentaires positifs caractéristiques :")
            for idx, row in df_positifs.iterrows():
                texte = row.get('Text_Clean', row.get('texte_original', ''))
                if len(texte) > 80:
                    texte = texte[:77] + "..."
                rapport.append(f"    ✓ \"{texte}\"")
        
        # Exemples négatifs
        if 'NÉGATIF' in self.resultats.get('distribution', {}):
            df_negatifs = self.df[self.df['sentiment'].str.contains('NÉGATIF')].head(2)
            if len(df_negatifs) > 0:
                rapport.append("\n  Commentaires négatifs caractéristiques :")
                for idx, row in df_negatifs.iterrows():
                    texte = row.get('Text_Clean', row.get('texte_original', ''))
                    if len(texte) > 80:
                        texte = texte[:77] + "..."
                    rapport.append(f"    ✗ \"{texte}\"")
        rapport.append("")
        
        # 8. RECOMMANDATIONS
        rapport.append("8. RECOMMANDATIONS STRATÉGIQUES")
        rapport.append("-"*40)
        
        # Recommandations basées sur l'analyse
        pourcentage_positif = self.resultats.get('pourcentages', {}).get('POSITIF', 0)
        pourcentage_negatif = self.resultats.get('pourcentages', {}).get('NÉGATIF', 0)
        
        rapport.append("  Priorité 1 : Capitaliser sur les points forts")
        if pourcentage_positif > 20:
            rapport.append("    → Amplifier la visibilité des retours positifs")
        else:
            rapport.append("    → Stimuler les interactions positives")
        
        rapport.append("\n  Priorité 2 : Gérer les préoccupations")
        if pourcentage_negatif > 10:
            rapport.append("    → Répondre systématiquement aux critiques")
        else:
            rapport.append("    → Maintenir la qualité du dialogue")
        
        rapport.append("\n  Priorité 3 : Optimiser l'engagement")
        if 'heures_pointe' in self.resultats:
            rapport.append(f"    → Programmer les publications aux heures de pointe ({self.resultats['heures_pointe'][0][0]}h)")
        rapport.append("")
        
        # 9. PERSPECTIVES
        rapport.append("9. PERSPECTIVES D'AMÉLIORATION")
        rapport.append("-"*40)
        rapport.append("  → Analyse comparative avec périodes non-électorales")
        rapport.append("  → Segmentation par type de publication")
        rapport.append("  → Intégration de l'analyse des émojis")
        rapport.append("  → Suivi longitudinal des contributeurs clés")
        rapport.append("")
        
        # Pied de page
        rapport.append("="*80)
        rapport.append("MÉTHODOLOGIE : Analyse automatisée des sentiments avec reconnaissance")
        rapport.append("du contexte ivoirien et des expressions locales.")
        rapport.append(f"Date de génération : {datetime.now().strftime('%d/%m/%Y %H:%M')}")
        rapport.append("="*80)
        
        # Joindre toutes les lignes
        rapport_complet = '\n'.join(rapport)
        
        # Sauvegarder le rapport
        if chemin_sortie:
            with open(chemin_sortie, 'w', encoding='utf-8') as f:
                f.write(rapport_complet)
            print(f"✅ Rapport sauvegardé : {chemin_sortie}")
        else:
            # Sauvegarde par défaut
            dossier_rapports = os.path.join(os.path.dirname(self.chemin_fichier), 'rapports')
            os.makedirs(dossier_rapports, exist_ok=True)
            
            nom_fichier = f"rapport_analyse_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
            chemin_sauvegarde = os.path.join(dossier_rapports, nom_fichier)
            
            with open(chemin_sauvegarde, 'w', encoding='utf-8') as f:
                f.write(rapport_complet)
            print(f"✅ Rapport sauvegardé : {chemin_sauvegarde}")
        
        # Afficher un extrait
        print("\n" + "="*80)
        print("📋 EXTRAIT DU RAPPORT")
        print("="*80)
        print('\n'.join(rapport[:50]))  # Afficher les 50 premières lignes
        
        return rapport_complet
    
    def generer_graphiques(self, dossier_output="graphiques_rapport"):
        """Génère des graphiques d'analyse"""
        print("\n🎨 Génération des graphiques...")
        
        os.makedirs(dossier_output, exist_ok=True)
        
        # 1. Camembert des sentiments
        plt.figure(figsize=(10, 8))
        labels = list(self.resultats['distribution'].keys())
        sizes = list(self.resultats['distribution'].values())
        colors = ['#4CAF50', '#F44336', '#FFC107', '#9E9E9E'][:len(labels)]
        
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Distribution des Sentiments', fontsize=14, fontweight='bold')
        plt.axis('equal')
        plt.savefig(f'{dossier_output}/distribution_sentiments.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Graphique temporel
        if 'tendance_quotidienne' in self.resultats:
            plt.figure(figsize=(12, 6))
            dates = list(self.resultats['tendance_quotidienne'].index)
            valeurs = list(self.resultats['tendance_quotidienne']['%_Positif'])
            
            plt.plot(dates, valeurs, marker='o', linewidth=2, color='#2196F3')
            plt.fill_between(dates, valeurs, alpha=0.2, color='#2196F3')
            plt.axhline(y=np.mean(valeurs), color='red', linestyle='--', label='Moyenne')
            
            plt.xlabel('Date')
            plt.ylabel('% de Positivité')
            plt.title('Évolution de la Positivité dans le Temps', fontsize=14, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'{dossier_output}/evolution_temporelle.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        # 3. Top mots
        if 'top_mots' in self.resultats:
            plt.figure(figsize=(12, 6))
            mots = list(self.resultats['top_mots'].keys())[:15]
            frequences = list(self.resultats['top_mots'].values())[:15]
            
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
    
    def generer_tableau_de_bord(self):
        """Génère un tableau de bord synthétique"""
        print("\n📊 Génération du tableau de bord...")
        
        tableau = []
        tableau.append("="*80)
        tableau.append("TABLEAU DE BORD SYNTHÉTIQUE")
        tableau.append("="*80)
        tableau.append("")
        
        # Indicateurs clés
        tableau.append("INDICATEURS CLÉS")
        tableau.append("-"*40)
        tableau.append(f"🔢 Total commentaires : {self.resultats.get('total', 0)}")
        tableau.append(f"📅 Période : {self.resultats.get('periode', {}).get('jours', 'N/A')} jours")
        tableau.append("")
        
        # Distribution sentiments
        tableau.append("SENTIMENTS")
        tableau.append("-"*40)
        for sentiment, count in self.resultats.get('distribution', {}).items():
            pourcentage = self.resultats.get('pourcentages', {}).get(sentiment, 0)
            tableau.append(f"{sentiment:10} : {count:4d} ({pourcentage:5.1f}%)")
        tableau.append("")
        
        # Top 3 mots
        tableau.append("THÉMATIQUES DOMINANTES")
        tableau.append("-"*40)
        for mot, freq in list(self.resultats.get('top_mots', {}).items())[:5]:
            tableau.append(f"  • {mot:15} : {freq:3d} mentions")
        tableau.append("")
        
        # Recommandation rapide
        tableau.append("RECOMMANDATION PRIORITAIRE")
        tableau.append("-"*40)
        
        pourcentage_positif = self.resultats.get('pourcentages', {}).get('POSITIF', 0)
        pourcentage_negatif = self.resultats.get('pourcentages', {}).get('NÉGATIF', 0)
        
        if pourcentage_positif > 30:
            tableau.append("✅ Excellent engagement positif !")
            tableau.append("→ Capitaliser sur cette dynamique.")
        elif pourcentage_negatif > 15:
            tableau.append("⚠️  Attention aux critiques")
            tableau.append("→ Renforcer la gestion des préoccupations.")
        else:
            tableau.append("⚖️  Situation équilibrée")
            tableau.append("→ Maintenir le dialogue constructif.")
        
        tableau.append("")
        tableau.append("="*80)
        
        # Sauvegarder
        dossier_rapports = os.path.join(os.path.dirname(self.chemin_fichier), 'rapports')
        os.makedirs(dossier_rapports, exist_ok=True)
        
        chemin_tableau = os.path.join(dossier_rapports, f"tableau_de_bord_{datetime.now().strftime('%Y%m%d')}.txt")
        with open(chemin_tableau, 'w', encoding='utf-8') as f:
            f.write('\n'.join(tableau))
        
        print(f"✅ Tableau de bord sauvegardé : {chemin_tableau}")
        print('\n'.join(tableau[:30]))  # Afficher les 30 premières lignes
        
        return '\n'.join(tableau)
    
    def executer_analyse_complete(self):
        """Exécute l'analyse complète et génère tous les rapports"""
        print("🚀 DÉMARRAGE DE L'ANALYSE COMPLÈTE")
        print("="*60)
        
        # Étape 1: Charger les données
        self.charger_donnees()
        
        # Étape 2: Analyses
        self.analyser_distribution_sentiments()
        self.analyser_tendance_temporelle()
        self.analyser_mots_cles()
        self.analyser_auteurs()
        self.analyser_heures_activite()
        
        # Étape 3: Générer les rapports
        rapport = self.generer_rapport_texte()
        self.generer_graphiques()
        self.generer_tableau_de_bord()
        
        print("\n" + "="*60)
        print("✅ ANALYSE COMPLÉTÉE AVEC SUCCÈS")
        print("="*60)
        
        return rapport


# ==============================================
# EXÉCUTION PRINCIPALE
# ==============================================

if __name__ == "__main__":
    # Chemin vers votre fichier
    chemin_fichier = r"C:\Users\Sy Savane Idriss\project_sentiment_fb\data\commentaire_election_gov_analyse_detaille.xlsx"
    
    # Vérifier si le fichier existe
    if not os.path.exists(chemin_fichier):
        print(f"❌ Fichier introuvable: {chemin_fichier}")
        print("Vérifiez le chemin et réessayez.")
    else:
        # Créer et exécuter l'analyseur
        analyseur = RapportAnalyseSentiments(chemin_fichier)
        analyseur.executer_analyse_complete()
        
        # Fichiers générés :
        print("\n📁 FICHIERS GÉNÉRÉS :")
        print(f"   • Rapport détaillé dans : data/rapports/")
        print(f"   • Graphiques dans : graphiques_rapport/")
        print(f"   • Tableau de bord synthétique")
        
        # Exemple d'utilisation avancée
        print("\n💡 UTILISATION AVANCÉE :")
        print("   # Pour un rapport personnalisé :")
        print("   analyseur = RapportAnalyseSentiments(chemin_fichier)")
        print("   analyseur.charger_donnees()")
        print("   analyseur.analyser_distribution_sentiments()")
        print("   rapport = analyseur.generer_rapport_texte('mon_rapport.txt')")