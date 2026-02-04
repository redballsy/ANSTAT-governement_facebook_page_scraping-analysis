import pyautogui
import time

# --- Configuration ---
# Une valeur positive scrolle vers le haut, négative vers le bas.
# L'intensité dépend de votre système (ex: -100 ou -1000).
VITESSE_SCROLL = -500 
DELAI_ENTRE_SCROLLS = 0.01 # Pour un mouvement "liquide"

print("Lancement dans 3 secondes... Placez votre souris sur la fenêtre cible.")
time.sleep(3)

try:
    print("Scroll en cours... Appuyez sur Ctrl+C dans le terminal pour arrêter.")
    while True:
        # La fonction scroll simule le coup de molette
        pyautogui.scroll(VITESSE_SCROLL)
        
        # Un court délai évite de saturer le processeur ou de faire planter l'application cible
        time.sleep(DELAI_ENTRE_SCROLLS)

except KeyboardInterrupt:
    print("\nScript arrêté proprement.")