import pyautogui
import time
import random
import pytesseract

# --- CONFIGURATION ---
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Sy Savane Idriss\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

IMG_EXIT = r"C:\Users\Sy Savane Idriss\project_sentiment_fb\image\exit.png"
IMG_VIEW_MORE = r"C:\Users\Sy Savane Idriss\project_sentiment_fb\image\view_more.png"
IMG_VOIR_PLUS = r"C:\Users\Sy Savane Idriss\project_sentiment_fb\image\voir_plus.png"  # Bouton "Voir plus de réponses"
IMG_PAUSE = r"C:\Users\Sy Savane Idriss\project_sentiment_fb\image\pause_button.png"  # Bouton pause vidéo
CONFIDENCE = 0.80  # Légèrement augmenté pour éviter les confusions

def verifier_date_limite():
    """Analyse la zone de date sur la gauche."""
    try:
        w, h = pyautogui.size()
        screenshot = pyautogui.screenshot(region=(0, 0, w//2, h))
        text = pytesseract.image_to_string(screenshot, lang='fra+eng').lower()
        limites = ['décembre', 'decembre', '4 sem', '4w', '5 sem', '5w', '6 sem', '6w']
        return any(mot in text for mot in limites)
    except:
        return False

def cliquer_voir_plus_si_present():
    """Clique sur le bouton 'Voir plus de réponses' s'il est présent."""
    try:
        w, h = pyautogui.size()
        region_popup = (w//4, h//4, w//2, h//2)
        
        # Rechercher le bouton 'voir_plus' dans la région de la popup
        bouton = pyautogui.locateOnScreen(
            IMG_VOIR_PLUS, 
            confidence=CONFIDENCE,
            region=region_popup
        )
        
        if bouton:
            center = pyautogui.center(bouton)
            print(f"  -> Bouton 'Voir plus de réponses' détecté, clic à Y: {center.y}")
            pyautogui.click(center)
            time.sleep(1.5)  # Attendre chargement des réponses
            return True
        return False
                    
    except:
        return False

def gerer_video_si_presente():
    """Vérifie et gère la vidéo si le bouton pause est présent."""
    try:
        w, h = pyautogui.size()
        region_popup = (w//4, h//4, w//2, h//2)
        
        # Rechercher le bouton pause dans la popup
        bouton_pause = pyautogui.locateOnScreen(
            IMG_PAUSE,
            confidence=CONFIDENCE,
            region=region_popup
        )
        
        if bouton_pause:
            center = pyautogui.center(bouton_pause)
            print(f"  ⏸️  Bouton PAUSE détecté, clic pour mettre en pause à Y: {center.y}")
            pyautogui.click(center)
            time.sleep(0.5)  # Court délai après la pause
            return True
        return False
        
    except:
        return False

def automate():
    print("!!! DÉMARRAGE MODE SCAN MULTI-POSTS AVEC GESTION VIDÉO !!!")
    print("Le bot va mettre en pause les vidéos automatiquement")
    time.sleep(5)
    posts_processed = 0

    while True:
        try:
            # 1. Vérifier la date avant de scanner l'écran
            if verifier_date_limite():
                print(">>> LIMITE 4 SEMAINES ATTEINTE. ARRÊT.")
                break

            # 2. Capturer TOUS les boutons "View more" actuellement visibles sur l'écran
            boutons = list(pyautogui.locateAllOnScreen(IMG_VIEW_MORE, confidence=CONFIDENCE))

            if boutons:
                print(f"--- {len(boutons)} bouton(s) détecté(s) sur cet écran ---")
                
                # On trie les boutons du haut vers le bas pour ne pas les sauter
                boutons.sort(key=lambda b: b.top)

                for b in boutons:
                    center = pyautogui.center(b)
                    
                    # Petite vérification : on clique seulement si le bouton est bien dans la zone utile
                    print(f"Traitement du bouton à la position Y: {center.y}")
                    pyautogui.click(center)
                    time.sleep(2.5) # Attente ouverture popup

                    # 3. Scroll INTERNE (Popup)
                    w, h = pyautogui.size()
                    pyautogui.moveTo(w / 2, h / 2)
                    last_shot = pyautogui.screenshot(region=(w//4, h//4, w//2, h//2))
                    
                    while True:
                        # Vérifier et gérer la vidéo AVANT toute action
                        gerer_video_si_presente()
                        
                        # Vérifier et cliquer sur "Voir plus" si présent
                        cliquer_voir_plus_si_present()
                        
                        # Scroller dans la popup
                        pyautogui.scroll(-1000)
                        time.sleep(1.0) # Un peu plus lent pour laisser charger
                        
                        current_shot = pyautogui.screenshot(region=(w//4, h//4, w//2, h//2))
                        # Si l'image ne bouge plus, on sort
                        if list(last_shot.getdata())[::500] == list(current_shot.getdata())[::500]:
                            # Dernière vérification avant de sortir
                            gerer_video_si_presente()
                            break
                        last_shot = current_shot

                    # 4. Sortie de la Popup
                    try:
                        exit_pos = pyautogui.locateCenterOnScreen(IMG_EXIT, confidence=CONFIDENCE)
                        if exit_pos: pyautogui.click(exit_pos)
                        else: pyautogui.press('esc')
                    except:
                        pyautogui.press('esc')
                    
                    posts_processed += 1
                    print(f"Post {posts_processed} terminé.")
                    time.sleep(1.5) # Pause pour laisser la popup se fermer proprement

                # 5. Une fois TOUS les boutons de l'écran traités, on descend pour chercher la suite
                print("Plus de boutons sur cet écran. Scroll vers la suite...")
                for _ in range(18): # On descend suffisamment pour renouveler l'écran
                    pyautogui.press('down')
                    time.sleep(0.05)

            else:
                # Si aucun bouton n'est trouvé, on descend par petits coups
                print("Recherche de nouveaux posts...")
                for _ in range(5):
                    pyautogui.press('down')
                    time.sleep(0.1)

        except Exception as e:
            print(f"Erreur rencontrée : {e}")
            pyautogui.press('esc')
            time.sleep(1)
            for _ in range(10): pyautogui.press('down')
            continue

if __name__ == "__main__":
    pyautogui.FAILSAFE = True
    automate()