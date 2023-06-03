import math

import customtkinter as ctk

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
        self.key = name
        self.label = ctk.CTkLabel(self, text=splitByUnder(name) + ":", font=ctk.CTkFont(size=15, weight="bold"),
                                  anchor='w')
        self.label.pack(fill="both", expand=True)
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
        self.canvas = ctk.CTkCanvas(self, borderwidth=0,
                                    background="#373737",
                                    height=int(self["height"]), width=int(self["width"]))
        self.frame = ctk.CTkFrame(self.canvas, bg_color="#373737", height=int(self["height"]), width=int(self["width"]))
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

        heights = 0
        max_height = int(self["height"])
        # print("max h",max_height)
        self.current_row = 0
        self.current_col = 0
        for i, (key, value) in enumerate(self.settings.items()):
            # print(i, key, value)
            component = Component(self.frame, key, value)
            component.grid(row=self.current_row, column=self.current_col, sticky="nsew")
            component.update()
            heights += int(component.winfo_height())
            self.components.append(component)

            if heights >= max_height:
                self.current_row = 0
                self.current_col += 1
                heights = 0
            else:
                self.current_row += 1
            # print(heights, max_height)
        # label.pack()

    def onFrameConfigure(self, event):
        '''Reset the scroll region to encompass the inner frame'''
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def addComponent(self, component: Component):
        self.components.append(component)
        component.grid(row=self.current_row, column=self.current_col, sticky="nsew")
        component.update()
        self.current_row += 1

    def getAllInputs(self):
        settings = {}
        for i, (key, value) in enumerate(self.settings.items()):
            # print(i, key, value)

            settings[key] = self.components[i].get()

        return settings

    def get_components(self, key: str):
        for com in self.components:
            if com.key == key:
                return com
        return None
