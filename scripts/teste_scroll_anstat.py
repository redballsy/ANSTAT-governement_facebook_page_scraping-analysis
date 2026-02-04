import pyautogui
import time
import random
import pytesseract

# --- CONFIGURATION ---
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Sy Savane Idriss\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

IMG_EXIT = r"C:\Users\Sy Savane Idriss\project_sentiment_fb\image\exit.png"
IMG_VIEW_MORE = r"C:\Users\Sy Savane Idriss\project_sentiment_fb\image\view_more.png"
CONFIDENCE = 0.75

def verifier_date_limite():
    """Analyse la zone de date sur la gauche."""
    try:
        w, h = pyautogui.size()
        screenshot = pyautogui.screenshot(region=(0, 0, w//2, h))
        text = pytesseract.image_to_string(screenshot, lang='fra+eng').lower()
        
        # --- MODIFIÉ POUR 4 SEMAINES ---
        # On ajoute '4 sem' et '4w'. On garde décembre car 4 semaines avant fin janvier = fin décembre.
        limites = ['décembre', 'decembre', '4 sem', '4w', '5 sem', '5w', '6 sem', '6w']
        return any(mot in text for mot in limites)
    except:
        return False

def automate():
    print("!!! DÉMARRAGE MODE CLAVIER (LENT & FURTIF) !!!")
    time.sleep(5)
    posts_processed = 0

    while True:
        try:
            # 1. Vérifier la date (Toutes les 10 lignes de scroll environ pour la performance)
            if verifier_date_limite():
                print(">>> LIMITE 4 SEMAINES ATTEINTE. ARRÊT.")
                break

            # 2. Chercher le bouton 'View more' sur l'écran actuel
            view_more_pos = pyautogui.locateCenterOnScreen(IMG_VIEW_MORE, confidence=CONFIDENCE)
            
            if view_more_pos:
                print(f"--- Post {posts_processed + 1} détecté ---")
                pyautogui.click(view_more_pos)
                time.sleep(2) # Attente ouverture

                # 3. Scroll INTERNE (Popup) - On garde le scroll souris ici car c'est une liste à part
                w, h = pyautogui.size()
                pyautogui.moveTo(w / 2, h / 2)
                last_shot = pyautogui.screenshot(region=(w//4, h//4, w//2, h//2))
                
                while True:
                    pyautogui.scroll(-1000)
                    time.sleep(0.8)
                    current_shot = pyautogui.screenshot(region=(w//4, h//4, w//2, h//2))
                    if list(last_shot.getdata())[::500] == list(current_shot.getdata())[::500]:
                        break
                    last_shot = current_shot

                # 4. Sortie (Image ou Echap)
                try:
                    exit_pos = pyautogui.locateCenterOnScreen(IMG_EXIT, confidence=CONFIDENCE)
                    if exit_pos: pyautogui.click(exit_pos)
                    else: pyautogui.press('esc')
                except: pyautogui.press('esc')
                
                posts_processed += 1
                print(f"Post {posts_processed} fini. Reprise du défilement...")
                
                # Petit délai pour laisser la fenêtre se fermer
                time.sleep(1)
                
                # On force un petit scroll pour dépasser le bouton qu'on vient de cliquer
                for _ in range(15):
                    pyautogui.press('down')

            else:
                # 5. DÉFILEMENT DOUX (Simule la touche Bas)
                # On descend de 3 lignes puis on regarde à nouveau
                for _ in range(3):
                    pyautogui.press('down')
                    time.sleep(0.1) # Très léger délai pour la fluidité

        except Exception as e:
            pyautogui.press('esc')
            for _ in range(10): pyautogui.press('down')
            continue

if __name__ == "__main__":
    pyautogui.FAILSAFE = True
    automate()