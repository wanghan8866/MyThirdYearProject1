import tkinter as tk
from widget.imageNode import ImageNode
from utility.imageLoader import ImageLoader
import customtkinter as ctk
from ctypes import windll


class SelectSection(ctk.CTkFrame):
    def __init__(self, parent, name, mode, content, linker=None, callback=None, **kwargs):
        ctk.CTkFrame.__init__(self, parent, **kwargs)
        self.mode = mode
        self.canvas = ctk.CTkCanvas(self, borderwidth=0,
                                    background="#ffffff",
                                    height=100)
        self.frame = ctk.CTkFrame(self.canvas, bg_color="#ffffff")
        self.vsb = ctk.CTkScrollbar(self, orientation="horizontal", command=self.canvas.xview,
                                    button_color="#0fc7e4")
        self.canvas.configure(xscrollcommand=self.vsb.set)
        self.name = name
        self.label = ctk.CTkLabel(self, text=self.name + ": ")
        self.label.pack(side="top")
        self.vsb.pack(side="bottom", fill="x")
        self.canvas.pack(side="left", fill="x", expand=True)
        self.canvas.create_window((0, 0), window=self.frame, anchor="nw",
                                  tags="self.frame")

        self.frame.bind("<Configure>", self.onFrameConfigure)
        self.content = content
        self.linker = linker
        self.callback=callback

        self.populate()

    def populate(self):
        '''Put in some fake data'''
        for row, name in enumerate(self.content):
            ImageNode(self.frame, name, mode=self.mode, linker=self.linker, callback=self.callback).grid(row=0, column=row, padx=5, pady=0)
            t = "this is the second column for row %s" % row
            # ImageNode(self.frame).grid(row=1, column=row)

    def clearAll(self):
        for child in self.frame.winfo_children():
            if type(child) == ImageNode:
                child.clear()

    def onFrameConfigure(self, event):
        '''Reset the scroll region to encompass the inner frame'''
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def getSelected(self):
        result = []
        for child in self.frame.winfo_children():
            if type(child) == ImageNode:
                if child.selected:
                    result.append(child.name)
        return result
