import pyautogui
import time
import random
import pytesseract

# --- CONFIGURATION ---
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Sy Savane Idriss\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

IMG_EXIT = r"C:\Users\Sy Savane Idriss\project_sentiment_fb\image\exit.png"
IMG_VIEW_MORE = r"C:\Users\Sy Savane Idriss\project_sentiment_fb\image\view_more.png"
IMG_VOIR_PLUS = r"C:\Users\Sy Savane Idriss\project_sentiment_fb\image\voir_plus.png"
CONFIDENCE = 0.80 

def cliquer_voir_plus_si_present():
    """Clique sur le bouton 'Voir plus de réponses' s'il est présent."""
    try:
        w, h = pyautogui.size()
        region_popup = (w//4, h//4, w//2, h//2)
        
        bouton = pyautogui.locateOnScreen(
            IMG_VOIR_PLUS, 
            confidence=CONFIDENCE,
            region=region_popup
        )
        
        if bouton:
            center = pyautogui.center(bouton)
            print(f"  -> Bouton 'Voir plus de réponses' détecté, clic à Y: {center.y}")
            pyautogui.click(center)
            
            # RETOUR AU CENTRE APRÈS LE CLIC
            pyautogui.moveTo(w / 2, h / 2)
            print(f"  ↺ Retour au centre de l'écran")
            
            time.sleep(1.5)
            return True
        return False
    except:
        return False

def detecter_view_more_dans_popup():
    """Détecte si l'image view_more.png apparaît dans la popup."""
    try:
        w, h = pyautogui.size()
        region_popup = (w//4, h//4, w//2, h//2)
        
        bouton = pyautogui.locateOnScreen(
            IMG_VIEW_MORE, 
            confidence=CONFIDENCE,
            region=region_popup
        )
        
        if bouton:
            return True
        return False
    except:
        return False

def automate():
    print("!!! DÉMARRAGE MODE SCAN ILLIMITÉ (SANS LIMITE DE DATE) !!!")
    time.sleep(5)
    posts_processed = 0

    while True:
        try:
            # 1. Capturer TOUS les boutons "View more" actuellement visibles
            boutons = list(pyautogui.locateAllOnScreen(IMG_VIEW_MORE, confidence=CONFIDENCE))

            if boutons:
                print(f"--- {len(boutons)} bouton(s) détecté(s) sur cet écran ---")
                
                # Tri du haut vers le bas
                boutons.sort(key=lambda b: b.top)
                
                # Liste pour garder les boutons déjà traités sur cet écran
                boutons_traites = []

                for b in boutons:
                    # Vérifier si ce bouton a déjà été traité (position similaire)
                    deja_traite = False
                    for bt in boutons_traites:
                        # Si la position Y est similaire (à 10 pixels près), considérer comme déjà traité
                        if abs(b.top - bt) < 10:
                            deja_traite = True
                            break
                    
                    if deja_traite:
                        print(f"  ⏭️  Bouton à Y:{b.top} déjà traité, passage au suivant")
                        continue
                        
                    center = pyautogui.center(b)
                    print(f"Traitement du bouton à la position Y: {center.y}")
                    pyautogui.click(center)
                    time.sleep(2.5)

                    # 2. Scroll INTERNE (Popup)
                    w, h = pyautogui.size()
                    region_popup = (w//4, h//4, w//2, h//2)
                    pyautogui.moveTo(w / 2, h / 2)  # Souris au centre initial
                    
                    # Liste pour garder les 10 dernières signatures de screenshots
                    signatures = []
                    # Timer pour détecter view_more pendant 5 secondes
                    view_more_detection_start = None
                    view_more_detected = False
                    
                    while True:
                        # Vérifier et cliquer sur "Voir plus" si présent
                        cliquer_voir_plus_si_present()
                        
                        # Vérifier si view_more apparaît dans la popup
                        if detecter_view_more_dans_popup():
                            if view_more_detection_start is None:
                                # Premier détection, on démarre le timer
                                view_more_detection_start = time.time()
                                print("  ⚠️  View_more détecté dans popup, vérification pendant 5s...")
                            elif time.time() - view_more_detection_start >= 5:
                                # View_more présent pendant 5 secondes, on sort
                                print("  ⚠️  View_more présent depuis 5s, sortie de la popup...")
                                view_more_detected = True
                                break
                        else:
                            # Reset le timer si view_more n'est plus détecté
                            view_more_detection_start = None
                        
                        # Scroller vers le bas
                        pyautogui.scroll(-1000)
                        time.sleep(1.0)
                        
                        # Prendre screenshot et créer sa signature
                        current_shot = pyautogui.screenshot(region=region_popup)
                        current_signature = list(current_shot.getdata())[::50]  # 1 pixel sur 50
                        
                        # Vérifier si cette signature existe déjà dans les 10 dernières
                        if current_signature in signatures:
                            print("  🔄 Screenshot répété détecté - fin du scroll")
                            break
                        
                        # Ajouter la nouvelle signature
                        signatures.append(current_signature)
                        
                        # Garder seulement les 10 dernières signatures
                        if len(signatures) > 10:
                            signatures.pop(0)  # Retire la plus ancienne

                    # 3. Sortie de la Popup
                    try:
                        # Si on a détecté view_more pendant 5s, on force exit
                        if view_more_detected:
                            print("  ⚠️  Forcer sortie via bouton EXIT (view_more détecté)")
                            exit_pos = pyautogui.locateCenterOnScreen(IMG_EXIT, confidence=CONFIDENCE)
                            if exit_pos: 
                                pyautogui.click(exit_pos)
                                print("  ✅ Sortie via bouton EXIT")
                            else: 
                                pyautogui.press('esc')
                                print("  ✅ Sortie via ESC")
                        else:
                            # Sortie normale
                            exit_pos = pyautogui.locateCenterOnScreen(IMG_EXIT, confidence=CONFIDENCE)
                            if exit_pos: 
                                pyautogui.click(exit_pos)
                                print("  ✅ Sortie via bouton EXIT")
                            else: 
                                pyautogui.press('esc')
                                print("  ✅ Sortie via ESC")
                    except:
                        pyautogui.press('esc')
                        print("  ✅ Sortie via ESC (exception)")
                    
                    posts_processed += 1
                    print(f"  📊 Post {posts_processed} terminé.")
                    
                    # Marquer ce bouton comme traité
                    boutons_traites.append(b.top)
                    
                    # 4. SCROLL VERS LE COMMENTAIRE SUIVANT APRÈS CHAQUE POPUP
                    print("  ⬇️  Scroll vers le commentaire suivant...")
                    for _ in range(5):  # Moins de scrolls pour aller au commentaire suivant
                        pyautogui.press('down')
                        time.sleep(0.1)
                    
                    # Attendre un peu pour laisser la page se stabiliser
                    time.sleep(1.0)

                # 5. Une fois TOUS les boutons de l'écran traités, on descend pour chercher la suite
                print("Plus de boutons sur cet écran. Scroll vers la suite...")
                for _ in range(15):  # Scroll plus important pour changer de section
                    pyautogui.press('down')
                    time.sleep(0.05)
                    
                # Attendre que de nouveaux posts se chargent
                time.sleep(2.0)

            else:
                # Si aucun bouton trouvé, on descend par petits coups
                print("Recherche de nouveaux posts...")
                for _ in range(5):
                    pyautogui.press('down')
                    time.sleep(0.1)
                    
                # Attendre un peu entre les recherches
                time.sleep(1.5)

        except Exception as e:
            print(f"❌ Erreur rencontrée : {e}")
            pyautogui.press('esc')
            time.sleep(1)
            for _ in range(10): 
                pyautogui.press('down')
                time.sleep(0.05)
            continue

if __name__ == "__main__":
    pyautogui.FAILSAFE = True
    automate()