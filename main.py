import tkinter as tk
import customtkinter as ctk
from ctypes import windll
from utility.imageLoader import ImageLoader
from window.appWindow import AppWindow
windll.shcore.SetProcessDpiAwareness(1)
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

if __name__ == '__main__':
    ImageLoader.loadAll()
    root = AppWindow()
    root.mainloop()


