import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Téléchargement des ressources NLTK
import nltk
nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from wordcloud import WordCloud
from textblob import TextBlob
import seaborn as sns
from datetime import datetime

# Initialiser l'analyseur de sentiment
sia = SentimentIntensityAnalyzer()

def clean_text(text):
    """Nettoyer le texte pour l'analyse"""
    if isinstance(text, str):
        # Convertir en minuscules
        text = text.lower()
        # Supprimer les URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Supprimer les mentions
        text = re.sub(r'@\w+', '', text)
        # Supprimer les hashtags
        text = re.sub(r'#\w+', '', text)
        # Supprimer les caractères spéciaux (garder les lettres, chiffres, espaces et accents français)
        text = re.sub(r'[^\w\sàâäéèêëîïôöùûüçÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ]', ' ', text)
        # Supprimer les espaces multiples
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ""

def analyze_sentiment_nltk(text):
    """Analyser le sentiment avec VADER"""
    cleaned_text = clean_text(text)
    scores = sia.polarity_scores(cleaned_text)
    
    # Déterminer la catégorie de sentiment
    if scores['compound'] >= 0.05:
        return 'positive', scores['compound']
    elif scores['compound'] <= -0.05:
        return 'negative', scores['compound']
    else:
        return 'neutral', scores['compound']

def analyze_sentiment_textblob(text):
    """Analyser le sentiment avec TextBlob"""
    cleaned_text = clean_text(text)
    analysis = TextBlob(cleaned_text)
    
    # TextBlob retourne une polarité entre -1 et 1
    polarity = analysis.sentiment.polarity
    
    if polarity > 0.1:  # Seuil un peu plus élevé pour TextBlob
        return 'positive', polarity
    elif polarity < -0.1:  # Seuil un peu plus bas pour TextBlob
        return 'negative', polarity
    else:
        return 'neutral', polarity

def analyze_comments(df):
    """Analyser les sentiments des commentaires"""
    results = []
    
    print(f"Analyse de {len(df)} commentaires...")
    
    for idx, row in df.iterrows():
        comment = row['Comment Text']
        
        # Analyse avec NLTK
        sentiment_nltk, score_nltk = analyze_sentiment_nltk(comment)
        
        # Analyse avec TextBlob
        sentiment_tb, score_tb = analyze_sentiment_textblob(comment)
        
        results.append({
            'Comment': comment,
            'Author': row['Author'] if 'Author' in row else 'Inconnu',
            'Date': row['Post Date-Time'] if 'Post Date-Time' in row else None,
            'Sentiment_NLTK': sentiment_nltk,
            'Score_NLTK': score_nltk,
            'Sentiment_TextBlob': sentiment_tb,
            'Score_TextBlob': score_tb
        })
        
        # Afficher la progression
        if (idx + 1) % 50 == 0:
            print(f"  {idx + 1}/{len(df)} commentaires analysés...")
    
    return pd.DataFrame(results)

def create_visualizations(df_sentiments, sentiment_counts_nltk, percentages_nltk, sentiment_counts_tb, percentages_tb):
    """Créer les visualisations"""
    
    # Palette de couleurs améliorée
    colors = {'positive': '#2ecc71', 'neutral': '#f39c12', 'negative': '#e74c3c'}
    color_list = [colors.get(sent, '#95a5a6') for sent in sentiment_counts_nltk.index]
    
    # Création des graphiques
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    fig.suptitle('Analyse des Sentiments des Commentaires Facebook', fontsize=16, fontweight='bold', y=1.02)
    
    # 1. Diagramme circulaire NLTK
    wedges, texts, autotexts = axes[0, 0].pie(
        sentiment_counts_nltk.values, 
        labels=sentiment_counts_nltk.index, 
        autopct='%1.1f%%', 
        colors=[colors.get(sent, '#95a5a6') for sent in sentiment_counts_nltk.index],
        startangle=90,
        explode=[0.05 if i == sentiment_counts_nltk.index[0] else 0 for i in range(len(sentiment_counts_nltk))]
    )
    axes[0, 0].set_title('Distribution des Sentiments (NLTK VADER)', fontweight='bold')
    
    # 2. Diagramme en barres NLTK (nombres)
    bars1 = axes[0, 1].bar(sentiment_counts_nltk.index, sentiment_counts_nltk.values, 
                          color=[colors.get(sent, '#95a5a6') for sent in sentiment_counts_nltk.index])
    axes[0, 1].set_title('Nombre de Commentaires par Sentiment (NLTK)', fontweight='bold')
    axes[0, 1].set_ylabel('Nombre de Commentaires', fontweight='bold')
    axes[0, 1].set_xlabel('Catégorie de Sentiment', fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Ajouter les valeurs sur les barres
    for bar in bars1:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Diagramme en barres NLTK (pourcentages)
    bars2 = axes[0, 2].bar(percentages_nltk.index, percentages_nltk.values,
                          color=[colors.get(sent, '#95a5a6') for sent in percentages_nltk.index])
    axes[0, 2].set_title('Pourcentage des Sentiments (NLTK)', fontweight='bold')
    axes[0, 2].set_ylabel('Pourcentage (%)', fontweight='bold')
    axes[0, 2].set_xlabel('Catégorie de Sentiment', fontweight='bold')
    axes[0, 2].set_ylim([0, max(percentages_nltk.values) * 1.2])
    axes[0, 2].grid(axis='y', alpha=0.3)
    
    # Ajouter les pourcentages sur les barres
    for bar, perc in zip(bars2, percentages_nltk.values):
        height = bar.get_height()
        axes[0, 2].text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{perc}%', ha='center', va='bottom', fontweight='bold')
    
    # 4. Diagramme circulaire TextBlob
    wedges2, texts2, autotexts2 = axes[1, 0].pie(
        sentiment_counts_tb.values, 
        labels=sentiment_counts_tb.index, 
        autopct='%1.1f%%', 
        colors=[colors.get(sent, '#95a5a6') for sent in sentiment_counts_tb.index],
        startangle=90,
        explode=[0.05 if i == sentiment_counts_tb.index[0] else 0 for i in range(len(sentiment_counts_tb))]
    )
    axes[1, 0].set_title('Distribution des Sentiments (TextBlob)', fontweight='bold')
    
    # 5. Diagramme en barres TextBlob (nombres)
    bars3 = axes[1, 1].bar(sentiment_counts_tb.index, sentiment_counts_tb.values,
                          color=[colors.get(sent, '#95a5a6') for sent in sentiment_counts_tb.index])
    axes[1, 1].set_title('Nombre de Commentaires par Sentiment (TextBlob)', fontweight='bold')
    axes[1, 1].set_ylabel('Nombre de Commentaires', fontweight='bold')
    axes[1, 1].set_xlabel('Catégorie de Sentiment', fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    # Ajouter les valeurs sur les barres
    for bar in bars3:
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # 6. Diagramme en barres TextBlob (pourcentages)
    bars4 = axes[1, 2].bar(percentages_tb.index, percentages_tb.values,
                          color=[colors.get(sent, '#95a5a6') for sent in percentages_tb.index])
    axes[1, 2].set_title('Pourcentage des Sentiments (TextBlob)', fontweight='bold')
    axes[1, 2].set_ylabel('Pourcentage (%)', fontweight='bold')
    axes[1, 2].set_xlabel('Catégorie de Sentiment', fontweight='bold')
    axes[1, 2].set_ylim([0, max(percentages_tb.values) * 1.2])
    axes[1, 2].grid(axis='y', alpha=0.3)
    
    # Ajouter les pourcentages sur les barres
    for bar, perc in zip(bars4, percentages_tb.values):
        height = bar.get_height()
        axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{perc}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Sauvegarder le graphique
    plt.savefig('sentiment_analysis_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 7. Graphique de comparaison côte à côte
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(sentiment_counts_nltk))
    width = 0.35
    
    bars_nltk = ax2.bar(x - width/2, sentiment_counts_nltk.values, width, 
                       label='NLTK VADER', color='#3498db', alpha=0.8)
    bars_tb = ax2.bar(x + width/2, sentiment_counts_tb.values, width, 
                     label='TextBlob', color='#e67e22', alpha=0.8)
    
    ax2.set_xlabel('Catégorie de Sentiment', fontweight='bold')
    ax2.set_ylabel('Nombre de Commentaires', fontweight='bold')
    ax2.set_title('Comparaison NLTK vs TextBlob', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(sentiment_counts_nltk.index)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Ajouter les valeurs sur les barres
    for bars in [bars_nltk, bars_tb]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('sentiment_comparison_bar.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 8. Nuage de mots
    print("Génération des nuages de mots...")
    
    # Préparer le texte pour les nuages de mots
    stop_words_fr = set(stopwords.words('french'))
    custom_stopwords = ['merci', 'monsieur', 'premier', 'ministre', 'mr', 'm', 'le', 'la', 'les', 'de', 'des', 'du', 'et', 'est']
    stop_words_fr.update(custom_stopwords)
    
    # Nuage de mots pour tous les commentaires
    all_comments = ' '.join(df_sentiments['Comment'].astype(str))
    
    # Filtrer les mots vides
    words = [word for word in all_comments.lower().split() if word not in stop_words_fr and len(word) > 2]
    filtered_text = ' '.join(words)
    
    if filtered_text:
        fig3, axes3 = plt.subplots(1, 2, figsize=(18, 8))
        
        # Nuage de mots pour tous les commentaires
        wordcloud_all = WordCloud(width=800, height=400, 
                                 background_color='white',
                                 max_words=150,
                                 colormap='viridis',
                                 contour_width=1,
                                 contour_color='steelblue').generate(filtered_text)
        
        axes3[0].imshow(wordcloud_all, interpolation='bilinear')
        axes3[0].set_title('Mots les plus fréquents - Tous les Commentaires', fontsize=14, fontweight='bold')
        axes3[0].axis('off')
        
        # Nuage de mots pour les commentaires positifs seulement
        positive_df = df_sentiments[df_sentiments['Sentiment_NLTK'] == 'positive']
        if len(positive_df) > 0:
            positive_comments = ' '.join(positive_df['Comment'].astype(str))
            positive_words = [word for word in positive_comments.lower().split() 
                            if word not in stop_words_fr and len(word) > 2]
            positive_filtered = ' '.join(positive_words)
            
            if positive_filtered:
                wordcloud_pos = WordCloud(width=800, height=400, 
                                         background_color='white',
                                         max_words=100,
                                         colormap='summer',
                                         contour_width=1,
                                         contour_color='green').generate(positive_filtered)
                
                axes3[1].imshow(wordcloud_pos, interpolation='bilinear')
                axes3[1].set_title('Mots les plus fréquents - Commentaires Positifs', fontsize=14, fontweight='bold')
                axes3[1].axis('off')
        
        plt.tight_layout()
        plt.savefig('wordclouds.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return fig

def print_detailed_statistics(df_sentiments, sentiment_counts_nltk, percentages_nltk, sentiment_counts_tb, percentages_tb):
    """Afficher les statistiques détaillées"""
    
    print("=" * 100)
    print(" " * 30 + "ANALYSE DES SENTIMENTS DES COMMENTAIRES FACEBOOK")
    print("=" * 100)
    
    print(f"\n📊 STATISTIQUES GÉNÉRALES:")
    print(f"   {'─' * 40}")
    print(f"   • Total des commentaires analysés: {len(df_sentiments):,}")
    
    if 'Date' in df_sentiments.columns and df_sentiments['Date'].notna().any():
        dates = pd.to_datetime(df_sentiments['Date'].dropna())
        if len(dates) > 0:
            print(f"   • Période: {dates.min().strftime('%d/%m/%Y')} au {dates.max().strftime('%d/%m/%Y')}")
            print(f"   • Durée: {(dates.max() - dates.min()).days + 1} jours")
    
    print(f"   • Nombre d'auteurs uniques: {df_sentiments['Author'].nunique()}")
    
    print(f"\n📈 RÉSULTATS DÉTAILLÉS:")
    print(f"   {'─' * 40}")
    
    # Tableau des résultats
    print("\n   " + "=" * 60)
    print("   | {:<15} | {:<10} | {:<10} | {:<10} |".format(
        "SENTIMENT", "NLTK (n)", "NLTK (%)", "TextBlob (n)"
    ))
    print("   " + "=" * 60)
    
    sentiments = ['positive', 'neutral', 'negative']
    for sentiment in sentiments:
        nltk_count = sentiment_counts_nltk.get(sentiment, 0)
        nltk_perc = percentages_nltk.get(sentiment, 0)
        tb_count = sentiment_counts_tb.get(sentiment, 0)
        
        print("   | {:<15} | {:<10} | {:<10.1f} | {:<10} |".format(
            sentiment.capitalize(), nltk_count, nltk_perc, tb_count
        ))
    
    print("   " + "=" * 60)
    
    # Scores moyens
    print(f"\n📊 SCORES MOYENS:")
    print(f"   {'─' * 40}")
    for sentiment in sentiments:
        if sentiment in df_sentiments['Sentiment_NLTK'].unique():
            nltk_mean = df_sentiments[df_sentiments['Sentiment_NLTK'] == sentiment]['Score_NLTK'].mean()
            tb_mean = df_sentiments[df_sentiments['Sentiment_TextBlob'] == sentiment]['Score_TextBlob'].mean()
            print(f"   • {sentiment.capitalize()}:")
            print(f"     - Score moyen NLTK: {nltk_mean:.3f}")
            print(f"     - Score moyen TextBlob: {tb_mean:.3f}")
    
    # Concordance entre les méthodes
    concordance = (df_sentiments['Sentiment_NLTK'] == df_sentiments['Sentiment_TextBlob']).mean() * 100
    
    print(f"\n🔄 CONCORDANCE ENTRE LES MÉTHODES:")
    print(f"   {'─' * 40}")
    print(f"   • Pourcentage de concordance: {concordance:.1f}%")
    print(f"   • Nombre de divergences: {len(df_sentiments) - (df_sentiments['Sentiment_NLTK'] == df_sentiments['Sentiment_TextBlob']).sum()}")
    
    # Matrice de concordance
    concordance_matrix = pd.crosstab(
        df_sentiments['Sentiment_NLTK'], 
        df_sentiments['Sentiment_TextBlob'],
        rownames=['NLTK →'],
        colnames=['TextBlob ↓']
    )
    
    print(f"\n📋 MATRICE DE CONCORDANCE:")
    print(f"   {'─' * 40}")
    print(concordance_matrix.to_string())
    
    # Top 10 des auteurs les plus actifs
    if 'Author' in df_sentiments.columns:
        author_counts = df_sentiments['Author'].value_counts().head(10)
        print(f"\n👥 TOP 10 DES AUTEURS LES PLUS ACTIFS:")
        print(f"   {'─' * 40}")
        for i, (author, count) in enumerate(author_counts.items(), 1):
            print(f"   {i:2}. {author:<30} : {count:>3} commentaires")
    
    print(f"\n💾 FICHIERS GÉNÉRÉS:")
    print(f"   {'─' * 40}")
    print("   1. sentiment_analysis_comparison.png - Graphiques de comparaison")
    print("   2. sentiment_comparison_bar.png - Graphique barre comparatif")
    print("   3. wordclouds.png - Nuages de mots")
    print("   4. sentiment_results.csv - Résultats détaillés")
    print("   5. sentiment_summary.csv - Résumé statistique")

def main():
    """Fonction principale"""
    
    # Chemin vers votre fichier CSV
    csv_path = r"C:\Users\Sy Savane Idriss\project_sentiment_fb\data\fb_comment.csv"
    
    print("=" * 80)
    print("DÉMARRAGE DE L'ANALYSE DES SENTIMENTS")
    print("=" * 80)
    print(f"📂 Lecture du fichier: {csv_path}")
    
    try:
        # Lecture du fichier CSV
        df = pd.read_csv(csv_path, encoding='utf-8')
        
        print(f"✅ Fichier chargé avec succès")
        print(f"   • Nombre de lignes: {len(df)}")
        print(f"   • Colonnes disponibles: {list(df.columns)}")
        
        # Vérifier les colonnes nécessaires
        if 'Comment Text' not in df.columns:
            raise ValueError("La colonne 'Comment Text' est introuvable dans le fichier CSV")
        
        # Nettoyage des données
        print("\n🧹 Nettoyage des données...")
        
        # Conversion des dates si la colonne existe
        if 'Post Date-Time' in df.columns:
            df['Post Date-Time'] = pd.to_datetime(df['Post Date-Time'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
        
        # Analyse des sentiments
        print("\n🔍 Analyse des sentiments en cours...")
        df_sentiments = analyze_comments(df)
        
        # Calcul des statistiques
        sentiment_counts_nltk = df_sentiments['Sentiment_NLTK'].value_counts()
        sentiment_counts_tb = df_sentiments['Sentiment_TextBlob'].value_counts()
        
        percentages_nltk = (sentiment_counts_nltk / len(df_sentiments) * 100).round(1)
        percentages_tb = (sentiment_counts_tb / len(df_sentiments) * 100).round(1)
        
        # Création des visualisations
        print("\n🎨 Création des visualisations...")
        create_visualizations(df_sentiments, sentiment_counts_nltk, percentages_nltk, sentiment_counts_tb, percentages_tb)
        
        # Affichage des statistiques
        print("\n📊 Génération des statistiques...")
        print_detailed_statistics(df_sentiments, sentiment_counts_nltk, percentages_nltk, sentiment_counts_tb, percentages_tb)
        
        # Sauvegarde des résultats
        print("\n💾 Sauvegarde des résultats...")
        
        # Fichier complet
        output_file = 'sentiment_results.csv'
        df_sentiments.to_csv(output_file, index=False, encoding='utf-8')
        print(f"   ✅ {output_file} - Résultats détaillés sauvegardés")
        
        # Fichier résumé
        summary_df = pd.DataFrame({
            'Sentiment': sentiment_counts_nltk.index,
            'Count_NLTK': sentiment_counts_nltk.values,
            'Percentage_NLTK': percentages_nltk.values,
            'Count_TextBlob': [sentiment_counts_tb.get(sent, 0) for sent in sentiment_counts_nltk.index],
            'Percentage_TextBlob': [percentages_tb.get(sent, 0) for sent in sentiment_counts_nltk.index]
        })
        summary_df.to_csv('sentiment_summary.csv', index=False, encoding='utf-8')
        print(f"   ✅ sentiment_summary.csv - Résumé statistique sauvegardé")
        
        # Rapport final
        print("\n" + "=" * 80)
        print("ANALYSE TERMINÉE AVEC SUCCÈS!")
        print("=" * 80)
        print(f"\n🎯 RÉSULTATS PRINCIPAUX (NLTK VADER):")
        print(f"   • Positif: {percentages_nltk.get('positive', 0):.1f}%")
        print(f"   • Neutre: {percentages_nltk.get('neutral', 0):.1f}%")
        print(f"   • Négatif: {percentages_nltk.get('negative', 0):.1f}%")
        
        print(f"\n📈 TENDANCE GÉNÉRALE:")
        if percentages_nltk.get('positive', 0) > percentages_nltk.get('negative', 0):
            print("   ✅ Sentiment globalement POSITIF")
        elif percentages_nltk.get('negative', 0) > percentages_nltk.get('positive', 0):
            print("   ❌ Sentiment globalement NÉGATIF")
        else:
            print("   ⚖️ Sentiment globalement NEUTRE")
        
    except FileNotFoundError:
        print(f"❌ ERREUR: Fichier non trouvé à l'emplacement: {csv_path}")
        print("   Vérifiez le chemin du fichier.")
    except Exception as e:
        print(f"❌ ERREUR: {str(e)}")
        print("   Assurez-vous que le fichier CSV est au bon format.")

if __name__ == "__main__":
    main()