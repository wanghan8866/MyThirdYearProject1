import tkinter as tk
from tkinter import ttk
from ctypes import windll
import customtkinter as ctk
from utility.imageLoader import ImageLoader
from meta.appMeta import AppMeta
from utility.nameSplitor import splitByUpper


class ImageNode(ctk.CTkFrame):
    def __init__(self, container: ctk.CTkFrame, name: str, mode="m", linker=None, callback=None, *arg, **kwargs):
        super().__init__(container, *arg, **kwargs)

        self.mode = mode
        self.selected = False
        self.linker = linker
        self.name = name
        label = ctk.CTkLabel(self, text="", image=ImageLoader.All_Images["bird"])
        label.pack(pady=10)
        label = ctk.CTkLabel(self, text=splitByUpper(name))
        label.pack()
        self.callback=callback

        self.configure(width=100, height=100, border_width=2, border_color="gray")
        self.pack_propagate(False)
        self.events_binding()


    # def onEnter(self, *args):
    #     self.configure(border_color="red")
    # def onLeave(self, *args):
    #     self.configure(border_color="black")

    def onClick(self,*args):
        if self.selected:
            self.configure(border_color="gray")
            self.selected = False
        else:
            self.configure(border_color="red")
            self.selected = True
        if self.mode == "s":
            for child in self.master.winfo_children():
                if child == self or type(child) != ImageNode:
                    continue
                if child.selected:
                    child.configure(border_color="gray")
                    child.selected = False
        if self.linker:
            # print("connected")
            self.linker.content = AppMeta.GAMES.get(self.name, [])
            self.linker.clearAll()
            self.linker.populate()
        if self.callback is not None:
            self.callback()


    def events_binding(self):
        # self.bind("<Enter>", self.onEnter)
        # self.bind("<Leave>", self.onLeave)
        self.bind("<Button-1>", self.onClick)
        for child in self.winfo_children():
            child.bind("<Button-1>", self.onClick)

    def events_unbinding(self):
        self.unbind("<Button-1>")
        for child in self.winfo_children():
            child.unbind("<Button-1>")

    def clear(self):
        self.events_unbinding()
        self.destroy()
