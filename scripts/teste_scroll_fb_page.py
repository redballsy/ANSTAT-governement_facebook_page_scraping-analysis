import pyautogui
import time
import numpy as np
import pytesseract
import cv2
import hashlib

# --- CONFIGURATION ---
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Sy Savane Idriss\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

IMG_EXIT = r"C:\Users\Sy Savane Idriss\project_sentiment_fb\image\exit.png"
IMG_VIEW_MORE = r"C:\Users\Sy Savane Idriss\project_sentiment_fb\image\view_more.png"
CONFIDENCE = 0.75

def calculer_hash_unique(region):
    """
    Calcule un hash MD5 unique d'une région d'écran
    Plus fiable qu'une simple moyenne de pixels
    """
    try:
        screenshot = pyautogui.screenshot(region=region)
        img_array = np.array(screenshot)
        
        # Réduire la taille pour une comparaison plus rapide
        img_small = cv2.resize(img_array, (40, 40))
        
        # Convertir en niveaux de gris
        img_gray = cv2.cvtColor(img_small, cv2.COLOR_RGB2GRAY)
        
        # Appliquer un léger flou pour ignorer les petites variations
        img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
        
        # Calculer le hash MD5 de l'image
        img_bytes = img_blur.tobytes()
        hash_md5 = hashlib.md5(img_bytes).hexdigest()
        
        return hash_md5
    except:
        return None

def detecter_animation_continue(region):
    """
    Détection rapide d'animation (GIF/vidéo)
    """
    try:
        frames = []
        for _ in range(2):
            frame = np.array(pyautogui.screenshot(region=region))
            frames.append(frame)
            time.sleep(0.15)
        
        if len(frames) < 2:
            return False
            
        gray1 = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frames[1], cv2.COLOR_RGB2GRAY)
        diff = cv2.absdiff(gray1, gray2)
        pixels_changes = np.count_nonzero(diff > 25)
        
        return pixels_changes > 80
    except:
        return False

def verifier_date_limite():
    try:
        w, h = pyautogui.size()
        screenshot = pyautogui.screenshot(region=(0, 0, w//3, h))
        text = pytesseract.image_to_string(screenshot, lang='fra+eng').lower()
        limites = ['2 sem', '2w', '3 sem', '3w', 'décembre', 'janvier']
        return any(mot in text for mot in limites)
    except: 
        return False

def verifier_bouton_deja_clique(position_actuelle, positions_recentes, seuil_distance=30):
    """
    Vérifie si cette position est trop proche d'une position récemment cliquée
    """
    temps_actuel = time.time()
    
    for pos, heure in positions_recentes[:10]:  # Vérifier les 10 plus récentes
        # Si moins de 15 secondes et position proche
        if temps_actuel - heure < 15:
            distance = ((pos[0] - position_actuelle[0])**2 + 
                       (pos[1] - position_actuelle[1])**2)**0.5
            
            if distance < seuil_distance:
                return True, distance
    
    return False, 0

def chercher_un_bouton():
    """
    Cherche UN seul bouton View More à la fois
    Retourne la position OU None si aucun trouvé
    """
    w, h = pyautogui.size()
    chasse_region = (0, h//3, w, h//2)
    
    try:
        # Chercher UN bouton View More
        view_more_pos = pyautogui.locateCenterOnScreen(
            IMG_VIEW_MORE, 
            confidence=CONFIDENCE, 
            region=chasse_region
        )
        
        if view_more_pos:
            return (view_more_pos.x, view_more_pos.y)
        else:
            return None
    except:
        return None

def traiter_bouton_unique(view_more_pos, hashes_posts_traites, positions_recentes, limite_memoire):
    """
    Traite UN seul bouton View More avec toutes les vérifications
    """
    position_actuelle = view_more_pos
    
    # VÉRIFICATION 1: Position déjà cliquée récemment ?
    deja_clique, distance = verifier_bouton_deja_clique(position_actuelle, positions_recentes)
    
    if deja_clique:
        print(f"⚠️ Position déjà cliquée il y a peu (distance: {distance:.0f}px)")
        return False, None  # Bouton ignoré
    
    # VÉRIFICATION 2: Hash unique du post
    w, h = pyautogui.size()
    post_region = (
        max(0, position_actuelle[0] - 250),  # Zone plus large
        max(0, position_actuelle[1] - 150),
        500,
        300
    )
    
    post_hash = calculer_hash_unique(post_region)
    
    if post_hash and post_hash in hashes_posts_traites:
        print(f"⚠️ Post déjà traité (hash identique)")
        return False, None  # Bouton ignoré
    
    # NOUVEAU POST CONFIRMÉ
    print(f"--- Nouveau Post détecté ---")
    print(f"Position: {position_actuelle}")
    
    return True, post_hash

def process_post(view_more_pos, hashes_posts_traites, positions_recentes, limite_memoire, posts_processed):
    """
    Traite un post complet (clic, scroll commentaires, sortie)
    """
    w, h = pyautogui.size()
    
    # Mémoriser AVANT de cliquer
    positions_recentes.insert(0, (view_more_pos, time.time()))
    
    # Limiter la taille des mémoires
    if len(positions_recentes) > limite_memoire:
        positions_recentes = positions_recentes[:limite_memoire]
    
    # Cliquer sur le bouton "View More"
    pyautogui.click(view_more_pos)
    time.sleep(1.5)
    
    # DÉLAI 1: Attendre 5 secondes avant de scroller
    print("⏳ Attente de 5 secondes avant de scroller...")
    for i in range(5, 0, -1):
        print(f"  {i}...")
        time.sleep(1)
    
    # Définir la région de scan pour les commentaires
    region_scan = (w//4, h//4, w//2, h//2)
    
    # --- SCROLL INITIAL 2 FOIS ---
    print("Scroll initial 2 fois...")
    for i in range(2):
        pyautogui.scroll(-600)  # RALENTI: -600 au lieu de -1000
        time.sleep(0.6)  # RALENTI: 0.6s au lieu de 0.3s
        print(f"  Scroll {i+1}/2 effectué")
    
    # --- VÉRIFICATION ANIMATION ---
    if detecter_animation_continue(region_scan):
        print("⚠️ Animation détectée, attente 4s...")
        time.sleep(4)
        
        if detecter_animation_continue(region_scan):
            print("❌ Animation persiste, sortie immédiate...")
            pyautogui.press('esc')
            time.sleep(0.5)
            pyautogui.scroll(-600)  # RALENTI
            return posts_processed
    
    # --- VALIDATION DÉMARRAGE SCROLL ---
    print("Validation du démarrage du scroll...")
    scroll_demarre = False
    
    for _ in range(5):
        pyautogui.scroll(-600)  # RALENTI
        time.sleep(0.7)  # RALENTI
        
        # Vérifier si le bouton original a disparu
        try:
            if not pyautogui.locateOnScreen(IMG_VIEW_MORE, confidence=CONFIDENCE-0.1):
                scroll_demarre = True
                print("✅ Bouton disparu, scroll activé!")
                break
        except:
            scroll_demarre = True
            break
    
    # --- SCROLL PROFOND DES COMMENTAIRES ---
    if scroll_demarre:
        print("Scroll des commentaires en cours...")
        scroll_count = 0
        memoire_screenshots = []
        
        while scroll_count < 80:
            avant = np.array(pyautogui.screenshot(region=region_scan))
            
            # Garder les 2 derniers screenshots
            memoire_screenshots.append(avant)
            if len(memoire_screenshots) > 2:
                memoire_screenshots.pop(0)
            
            # Vérifier si on est coincé (même image 2 fois)
            if (len(memoire_screenshots) == 2 and 
                np.array_equal(memoire_screenshots[0], memoire_screenshots[1])):
                print("🔄 Même image détectée 2 fois, probablement fin")
                break
            
            pyautogui.scroll(-1000)  # RALENTI: -1000 au lieu de -1500
            time.sleep(1.0)  # RALENTI: 1.0s au lieu de 0.7s
            apres = np.array(pyautogui.screenshot(region=region_scan))
            
            if np.array_equal(avant, apres):
                print("✅ Fin des commentaires (plus de mouvement).")
                break
            
            scroll_count += 1
    else:
        print("❌ Le bouton n'a jamais disparu. Post vide ou bloqué.")
    
    # DÉLAI 2: Attendre 5 secondes avant de sortir
    print("⏳ Attente de 5 secondes avant de sortir...")
    for i in range(5, 0, -1):
        print(f"  {i}...")
        time.sleep(1)
    
    # --- SORTIE DE L'INTERFACE ---
    try:
        exit_pos = pyautogui.locateCenterOnScreen(IMG_EXIT, confidence=CONFIDENCE)
        if exit_pos:
            pyautogui.click(exit_pos)
            print("✅ Sortie via bouton EXIT")
        else:
            pyautogui.press('esc')
            print("✅ Sortie via ESC")
    except:
        pyautogui.press('esc')
        print("✅ Sortie via ESC (exception)")
    
    posts_processed += 1
    time.sleep(0.5)
    
    return posts_processed

def automate():
    print("!!! BOT FACEBOOK - SYSTÈME 'UN BOUTON À LA FOIS' !!!")
    print("=== NOUVELLE STRATÉGIE ===")
    print("1. Prend une photo de l'écran")
    print("2. Cherche UN bouton View More")
    print("3. Traite le bouton trouvé COMPLÈTEMENT")
    print("4. Scrolle SEULEMENT quand aucun bouton n'est visible")
    print("5. Scroll ralenti pour une meilleure détection")
    print("===================\n")
    
    time.sleep(3)
    
    posts_processed = 0
    w, h = pyautogui.size()
    
    # Système de mémoire amélioré
    hashes_posts_traites = set()  # Set pour vérifications rapides
    positions_recentes = []  # Liste des positions récentes (position, heure)
    
    # Limite de mémoire
    limite_memoire = 50
    
    # Compteur d'écrans vides consécutifs
    ecran_vide_consecutifs = 0
    max_ecrans_vides_avant_scroll_fort = 5

    while True:
        try:
            # Vérifier la limite de date
            if verifier_date_limite():
                print(">>> LIMITE DE DATE ATTEINTE.")
                break
            
            # STRATÉGIE: Chercher UN bouton à la fois
            print("\n📸 Analyse de l'écran en cours...")
            bouton_trouve = chercher_un_bouton()
            
            if bouton_trouve:
                print(f"🎯 Bouton trouvé à la position: {bouton_trouve}")
                ecran_vide_consecutifs = 0  # Réinitialiser le compteur
                
                # Vérifier si ce bouton est valide à traiter
                valide, post_hash = traiter_bouton_unique(
                    bouton_trouve, 
                    hashes_posts_traites, 
                    positions_recentes, 
                    limite_memoire
                )
                
                if valide:
                    # Mémoriser le hash si disponible
                    if post_hash:
                        hashes_posts_traites.add(post_hash)
                    
                    # Limiter la taille de la mémoire des hashs
                    if len(hashes_posts_traites) > limite_memoire:
                        hashes_posts_traites = set(list(hashes_posts_traites)[:limite_memoire])
                    
                    # Traiter le post complet
                    posts_processed = process_post(
                        bouton_trouve,
                        hashes_posts_traites,
                        positions_recentes,
                        limite_memoire,
                        posts_processed
                    )
                    
                    # Après avoir traité un bouton, faire un PETIT scroll pour vérifier s'il reste d'autres boutons
                    print("\nPetit scroll pour vérifier s'il reste d'autres boutons...")
                    pyautogui.scroll(-300)  # Très petit scroll
                    time.sleep(1.0)  # Attente pour laisser charger
                    
                else:
                    # Bouton invalide (déjà traité)
                    print("Bouton ignoré (déjà traité)")
                    
                    # Petit scroll pour passer ce bouton déjà traité
                    pyautogui.scroll(-400)
                    time.sleep(0.8)
                
            else:
                # Aucun bouton trouvé sur cet écran
                ecran_vide_consecutifs += 1
                print(f"📭 Aucun bouton détecté (écran vide #{ecran_vide_consecutifs})")
                
                # Scroll adaptatif selon le nombre d'écrans vides consécutifs
                if ecran_vide_consecutifs == 1:
                    # Premier écran vide: scroll TRÈS léger
                    print("Premier écran vide - scroll très léger...")
                    pyautogui.scroll(-250)
                    time.sleep(1.2)
                    
                elif ecran_vide_consecutifs == 2:
                    # Deuxième écran vide: scroll léger
                    print("Deuxième écran vide - scroll léger...")
                    pyautogui.scroll(-350)
                    time.sleep(1.0)
                    
                elif ecran_vide_consecutifs == 3:
                    # Troisième écran vide: scroll moyen
                    print("Troisième écran vide - scroll moyen...")
                    pyautogui.scroll(-450)
                    time.sleep(0.9)
                    
                elif ecran_vide_consecutifs == 4:
                    # Quatrième écran vide: scroll normal
                    print("Quatrième écran vide - scroll normal...")
                    pyautogui.scroll(-550)
                    time.sleep(0.8)
                    
                else:
                    # Après 4 écrans vides: scroll plus important
                    scroll_distance = -650
                    if ecran_vide_consecutifs > 10:
                        scroll_distance = -800
                    
                    print(f"Écran vide #{ecran_vide_consecutifs} - scroll de {abs(scroll_distance)} pixels...")
                    pyautogui.scroll(scroll_distance)
                    time.sleep(0.7)
                
                # Réinitialiser après un certain nombre d'écrans vides
                if ecran_vide_consecutifs > 15:
                    print("⚠️ Beaucoup d'écrans vides consécutifs - vérification...")
                    pyautogui.scroll(-1000)  # Grand scroll pour sortir d'une potentielle boucle
                    time.sleep(1.5)
                    ecran_vide_consecutifs = 5  # Réinitialiser partiellement

        except KeyboardInterrupt:
            print("\n🛑 Arrêt manuel par l'utilisateur.")
            break
            
        except Exception as e:
            print(f"❌ Erreur: {e}")
            pyautogui.press('esc')
            time.sleep(1.5)
            continue

    print(f"\n✅ TERMINÉ. {posts_processed} posts traités avec succès.")
    print(f"📊 Mémoire: {len(hashes_posts_traites)} hashes, {len(positions_recentes)} positions")

if __name__ == "__main__":
    pyautogui.FAILSAFE = True
    print("Démarrage dans 3 secondes...")
    automate()