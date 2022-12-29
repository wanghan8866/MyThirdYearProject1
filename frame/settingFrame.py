import tkinter as tk
from meta.appMeta import AppMeta
from utility.imageLoader import ImageLoader
from widget.imageNode import ImageNode
import customtkinter as ctk


class SettingFrame(ctk.CTkFrame):
    def __init__(self, master: any, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        label = ctk.CTkLabel(self, text="Settings", font=ctk.CTkFont(size=15, weight="bold"))
        label.pack()

        imageLabel = ImageNode(self, name="setting")
        imageLabel.pack()
