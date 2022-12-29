import tkinter as tk
from meta.appMeta import AppMeta
from utility.imageLoader import ImageLoader
import customtkinter as ctk

class HomeFrame(ctk.CTkFrame):
    def __init__(self, master: any, *args, **kwargs):
        super().__init__(master,*args, **kwargs)
        label = ctk.CTkLabel(self, text="Home" ,font=ctk.CTkFont(size=15, weight="bold"))
        label.pack()