# ui_display.py

import tkinter as tk
import threading
import queue
import keyboard
from PIL import ImageTk, Image


message_queue = queue.Queue()
running = True
current_window = None
current_label = None  # 用于跟踪当前显示的图片标签

def start_ui():
    def ui_thread():
        global current_window, current_label
        
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口
        root.attributes("-topmost", True)
        
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        # 创建初始窗口
        current_window = tk.Toplevel(root)
        current_window.attributes("-fullscreen", True)
        current_window.attributes("-topmost", True)
        current_window.configure(bg='black')
        current_window.wm_attributes("-toolwindow", True)
        
        # 创建初始标签（空内容）
        current_label = tk.Label(current_window, bg='black')
        current_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        def update_loop():
            global current_label
            current_image = ""
            
            while running:
                try:
                    new_image = message_queue.get(timeout=0.5)
                    if new_image != current_image:
                        current_image = new_image
                        
                        try:
                            image_path = r'C:\Users\vivi\Downloads/API.py源码/MI_images/' + str(current_image) + '.png'
                            img = Image.open(image_path)
                            
                            # 调整图片大小
                            img_width, img_height = img.size
                            ratio = min(screen_width / img_width, screen_height / img_height)
                            new_size = (int(img_width * ratio), int(img_height * ratio))
                            img = img.resize(new_size, Image.Resampling.LANCZOS)
                            
                            # 更新图片而不重建窗口
                            photo = ImageTk.PhotoImage(img)
                            current_label.configure(image=photo)
                            current_label.image = photo  # 保持引用
                            
                        except Exception as e:
                            print(f"Error loading image: {e}")
                            
                except queue.Empty:
                    pass

        threading.Thread(target=update_loop, daemon=True).start()
        threading.Thread(target=listen_for_quit, daemon=True).start()
        root.mainloop()

    threading.Thread(target=ui_thread, daemon=True).start()


def update_text_from_outside(new_text: str):
    """供外部调用的函数，用于更新UI显示内容"""
    message_queue.put(new_text)


def listen_for_quit():
    global running
    keyboard.wait("q")
    running = False