import pyautogui
import time
import os
from datetime import datetime

# --- Image Paths ---
IMG_EXIT = r"C:\Users\Sy Savane Idriss\project_sentiment_fb\image\exit.png"
IMG_VIEW_MORE = r"C:\Users\Sy Savane Idriss\project_sentiment_fb\image\view_more.png"
LOG_DIR = r"C:\Users\Sy Savane Idriss\project_sentiment_fb\logs"

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# --- Settings ---
CONFIDENCE = 0.75
MAX_SEARCH_SCROLLS = 15 
FIXED_DELAY = 2  # Your requested 2-second speed

def take_error_screenshot(reason):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(LOG_DIR, f"error_{reason}_{timestamp}.png")
    try:
        pyautogui.screenshot(filename)
    except:
        pass

def automate():
    print("!!! STARTING IN 5 SECONDS !!!")
    time.sleep(5)
    
    posts_processed = 0
    consecutive_empty_scrolls = 0

    while True:
        try:
            # 1. Search for 'View more'
            view_more_pos = pyautogui.locateCenterOnScreen(IMG_VIEW_MORE, confidence=CONFIDENCE)
            
            if view_more_pos:
                print(f"--- Post {posts_processed + 1} ---")
                pyautogui.click(view_more_pos)
                consecutive_empty_scrolls = 0 
                time.sleep(FIXED_DELAY) # Wait 2s for popup
                
                # 2. Position mouse for focus
                w, h = pyautogui.size()
                pyautogui.moveTo(w / 2, h / 2) 
                
                # 3. FAST SMART SCROLL
                last_shot = pyautogui.screenshot()
                while True:
                    pyautogui.scroll(-1200) 
                    time.sleep(FIXED_DELAY) # Wait 2s to check freeze
                    current_shot = pyautogui.screenshot()
                    
                    # Quick pixel check for freeze
                    if list(last_shot.getdata()) == list(current_shot.getdata()):
                        print("Bottom reached.")
                        break
                    last_shot = current_shot
                    print("Scrolling...")

                # 4. Fast Exit
                try:
                    exit_pos = pyautogui.locateCenterOnScreen(IMG_EXIT, confidence=CONFIDENCE)
                    if exit_pos:
                        pyautogui.click(exit_pos)
                    else:
                        pyautogui.press('esc')
                except:
                    pyautogui.press('esc')

                posts_processed += 1
                time.sleep(FIXED_DELAY) # Wait 2s before moving to next
                pyautogui.scroll(-2000) 
                time.sleep(FIXED_DELAY) # Wait 2s to settle feed

            else:
                # 5. SEARCH LOGIC
                consecutive_empty_scrolls += 1
                
                if consecutive_empty_scrolls >= MAX_SEARCH_SCROLLS:
                    print("Limit reached. Checking 2x deeper...")
                    found_extra = False
                    for i in range(2): # Your 2 extra checks
                        pyautogui.scroll(-3500)
                        time.sleep(FIXED_DELAY)
                        if pyautogui.locateOnScreen(IMG_VIEW_MORE, confidence=CONFIDENCE):
                            print("Found post in deep search!")
                            consecutive_empty_scrolls = 0
                            found_extra = True
                            break
                    
                    if not found_extra:
                        print("No more posts found.")
                        break
                else:
                    print(f"Searching... ({consecutive_empty_scrolls}/{MAX_SEARCH_SCROLLS})")
                    pyautogui.scroll(-1500)
                    time.sleep(FIXED_DELAY)

        except Exception as e:
            print(f"Error: {e}")
            take_error_screenshot("loop_error")
            pyautogui.scroll(-2000)
            time.sleep(FIXED_DELAY)
            continue

    print(f"Task Complete. Total: {posts_processed}")

if __name__ == "__main__":
    pyautogui.FAILSAFE = True 
    automate()