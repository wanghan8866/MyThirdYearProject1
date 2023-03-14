import tkinter as tk
from meta.appMeta import AppMeta
from utility.imageLoader import ImageLoader
import customtkinter as ctk
import math
from meta.trainingMeta import SelectionType, ListType, NumberType
from utility.nameSplitor import splitByUnder


def nextPerfectSquare(N):
    nextN = math.floor(math.sqrt(N)) + 1

    return nextN * nextN


def getCoords(i, size: int):
    # size = int(math.sqrt(total))
    c = i // size
    r = i % size
    return r, c


def optionmenu_callback(choice):
    print("optionmenu dropdown clicked:", choice)


class Component(ctk.CTkFrame):
    def __init__(self, master: any, name: str, value, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        label = ctk.CTkLabel(self, text=splitByUnder(name) + ":", font=ctk.CTkFont(size=15, weight="bold"), anchor='w')
        label.pack(fill="both", expand=True)
        # print(type(value))
        if isinstance(value, SelectionType):
            self.entry = ctk.CTkOptionMenu(master=self,
                                           values=[str(v) for v in value.options])
            self.entry.pack(side="bottom", padx=10, pady=5)
            self.entry.set(str(value.default))  # set initial value
        if isinstance(value, NumberType):
            self.entry = ctk.CTkEntry(master=self, placeholder_text=str(value.default))
            self.entry.pack(side="bottom", padx=10, pady=5)  # set initial value
            self.entry.insert(0, str(value.default))
        if isinstance(value, ListType):
            self.entry = ctk.CTkEntry(master=self, placeholder_text=str(value.default))
            self.entry.pack(side="bottom", padx=10, pady=5)  # set initial value
            self.entry.insert(0, str(value.default))
        # print("component", self["height"])

    def get(self):
        return self.entry.get()


class ModelTestFrame(ctk.CTkFrame):
    def __init__(self, master: any, name: str, settings: dict, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        print(settings)
        label = ctk.CTkLabel(self, text=name, font=ctk.CTkFont(size=15, weight="bold"))
        self.settings = settings
        self.components = []

        heights = 0
        max_height = int(self["height"])
        current_row = 0
        current_col = 0
        for i, (key, value) in enumerate(self.settings.items()):
            # print(i, key, value)
            component = Component(self, key, value)
            component.grid(row=current_row, column=current_col, sticky="nsew")
            component.update()
            heights += int(component.winfo_height())
            self.components.append(component)

            if heights >= max_height:
                current_row = 0
                current_col += 1
                heights = 0
            else:
                current_row += 1
            # print(heights, max_height)
        # label.pack()

    def getAllInputs(self):
        settings = {}
        for i, (key, value) in enumerate(self.settings.items()):
            # print(i, key, value)

            settings[key] = self.components[i].get()

        return settings
