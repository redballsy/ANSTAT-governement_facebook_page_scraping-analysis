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

def automate():
    print("!!! BOT FACEBOOK - SYSTÈME ANTI-DOUBLONS AMÉLIORÉ !!!")
    print("=== INSTRUCTIONS ===")
    print("1. Double vérification: hash + position")
    print("2. Attente 5s après clic sur 'View More'")
    print("3. Attente 5s avant clic sur 'Exit'")
    print("4. Scroll adaptatif pour éviter les doublons")
    print("===================\n")
    
    time.sleep(3)
    
    posts_processed = 0
    w, h = pyautogui.size()
    
    # Région où chercher les boutons "View More"
    chasse_region = (0, h//3, w, h//2)
    
    # Système de mémoire amélioré
    hashes_posts_traites = set()  # Set pour vérifications rapides
    positions_recentes = []  # Liste des positions récentes (position, heure)
    
    # Limite de mémoire
    limite_memoire = 50
    
    # Compteur pour debug
    debug_counter = 0

    while True:
        try:
            debug_counter += 1
            
            # Vérifier la limite de date
            if verifier_date_limite():
                print(">>> LIMITE DE DATE ATTEINTE.")
                break
            
            # Chercher le bouton "View More"
            view_more_pos = None
            try:
                view_more_pos = pyautogui.locateCenterOnScreen(
                    IMG_VIEW_MORE, 
                    confidence=CONFIDENCE, 
                    region=chasse_region
                )
            except:
                pass
            
            if view_more_pos:
                position_actuelle = (view_more_pos.x, view_more_pos.y)
                
                # VÉRIFICATION 1: Position déjà cliquée récemment ?
                deja_clique, distance = verifier_bouton_deja_clique(position_actuelle, positions_recentes)
                
                if deja_clique:
                    print(f"⚠️ Position déjà cliquée il y a peu (distance: {distance:.0f}px)")
                    print("Scroll adaptatif pour éviter doublon...")
                    
                    # Scroll plus important pour bien passer
                    pyautogui.scroll(-1000)
                    time.sleep(0.6)
                    pyautogui.scroll(-600)
                    time.sleep(0.4)
                    continue
                
                # VÉRIFICATION 2: Hash unique du post
                post_region = (
                    max(0, view_more_pos.x - 250),  # Zone plus large
                    max(0, view_more_pos.y - 150),
                    500,
                    300
                )
                
                post_hash = calculer_hash_unique(post_region)
                
                if post_hash and post_hash in hashes_posts_traites:
                    print(f"⚠️ Post déjà traité (hash identique)")
                    print("Scroll pour passer au suivant...")
                    
                    # Scroll adaptatif
                    pyautogui.scroll(-900)
                    time.sleep(0.5)
                    continue
                
                # NOUVEAU POST CONFIRMÉ
                print(f"--- Nouveau Post {posts_processed + 1} ---")
                print(f"Position: {position_actuelle}")
                
                # Mémoriser AVANT de cliquer
                positions_recentes.insert(0, (position_actuelle, time.time()))
                if post_hash:
                    hashes_posts_traites.add(post_hash)
                
                # Limiter la taille des mémoires
                if len(positions_recentes) > limite_memoire:
                    positions_recentes = positions_recentes[:limite_memoire]
                
                if len(hashes_posts_traites) > limite_memoire:
                    # Garder seulement les plus récents
                    hashes_posts_traites = set(list(hashes_posts_traites)[:limite_memoire])
                
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
                    pyautogui.scroll(-1000)
                    time.sleep(0.3)
                    print(f"  Scroll {i+1}/2 effectué")
                
                # --- VÉRIFICATION ANIMATION ---
                if detecter_animation_continue(region_scan):
                    print("⚠️ Animation détectée, attente 4s...")
                    time.sleep(4)
                    
                    if detecter_animation_continue(region_scan):
                        print("❌ Animation persiste, sortie immédiate...")
                        pyautogui.press('esc')
                        time.sleep(0.5)
                        pyautogui.scroll(-1000)  # Scroll important après sortie
                        continue
                
                # --- VALIDATION DÉMARRAGE SCROLL ---
                print("Validation du démarrage du scroll...")
                scroll_demarre = False
                
                for _ in range(5):
                    pyautogui.scroll(-1000)
                    time.sleep(0.4)
                    
                    # Vérifier si le bouton original a disparu
                    try:
                        if not pyautogui.locateOnScreen(IMG_VIEW_MORE, confidence=CONFIDENCE-0.1, region=chasse_region):
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
                        
                        pyautogui.scroll(-1500)
                        time.sleep(0.7)
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
                
                # --- SCROLL POUR PASSER AU POST SUIVANT ---
                print("Scroll pour post suivant...")
                
                # Scroll adaptatif selon le nombre de posts traités
                if posts_processed < 20:
                    scroll_force = -800
                elif posts_processed < 50:
                    scroll_force = -1000
                else:
                    scroll_force = -1200  # Plus fort après 50 posts
                
                pyautogui.scroll(scroll_force)
                time.sleep(0.7)
                    
            else:
                # Pas de bouton trouvé, scroll normal
                pyautogui.scroll(-400)
                time.sleep(0.2)

        except KeyboardInterrupt:
            print("\n🛑 Arrêt manuel par l'utilisateur.")
            break
            
        except Exception as e:
            print(f"❌ Erreur: {e}")
            pyautogui.press('esc')
            time.sleep(1)
            continue

    print(f"\n✅ TERMINÉ. {posts_processed} posts traités avec succès.")
    print(f"📊 Mémoire: {len(hashes_posts_traites)} hashes, {len(positions_recentes)} positions")

if __name__ == "__main__":
    pyautogui.FAILSAFE = True
    print("Démarrage dans 3 secondes...")
    automate()