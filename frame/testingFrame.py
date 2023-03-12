import tkinter as tk
from meta.appMeta import AppMeta
from utility.imageLoader import ImageLoader
import customtkinter as ctk
from widget.selectSection import SelectSection
import numpy as np
from typing import Tuple
from window.gameWindow import GameWindow


class TestingFrame(ctk.CTkFrame):
    def __init__(self, master: any, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        label = ctk.CTkLabel(self, text="Testing Arena", font=ctk.CTkFont(size=15, weight="bold"))
        label.pack()
        self.game = "Snake-gen"
        # self.section1 = SelectSection(self, "Agent", mode="m", content=[])
        # print(AppMeta.GAMES.get("Snake-gen",[]))
        self.section1 = SelectSection(self, "Agent", mode="m", content=AppMeta.GAMES.get("Snake-gen", []))
        # self.section = SelectSection(self, "Game", mode="s", content=list(AppMeta.GAMES.keys()), linker=self.section1)
        # self.section.pack(fill="x", padx=20)
        self.section1.pack(fill="x", padx=20, pady=10)
        # self.section1.pack(fill="x", padx=20, pady=20)

        button = ctk.CTkButton(self, text="Launch", command=self.on_click)
        button.pack(side="bottom", pady=20)
        self.canvas = ctk.CTkCanvas(self)
        self.canvas.pack(expand=1, fill=tk.BOTH, pady=10, padx=20)
        self.gridmap = np.zeros(shape=(10, 10), dtype=np.uint8)
        self.grid_length = 30
        self.color_map = ["white", "pink", "green", "blue", "red"]
        # print()
        self.hoffset = int(self.canvas["width"]) // 2 - self.grid_length * 2
        self.voffset = 10
        for i, row in enumerate(self.gridmap):
            for j, col in enumerate(row):
                self.canvas.create_rectangle(i * self.grid_length + self.hoffset, j * self.grid_length + self.voffset,
                                             (i + 1) * self.grid_length + self.hoffset,
                                             (j + 1) * self.grid_length + self.voffset, width=2, fill="white")
        # oval_element = self.canvas.create_oval(20, 20, 100, 100, width=2, fill="white")
        self.canvas.bind('<Button-1>', self.object_click_event)
        self.canvas.bind('<Button-3>', self.object_de_click_event)
        self.canvas.pack()

    def fromItemIndexToCoords(self, index: int) -> Tuple[int, int]:
        pass

    def fromCoordsToItemIndex(self, x: int, y: int) -> Tuple:
        x = (x - self.hoffset) // (self.grid_length)
        y = (y - self.voffset) // (self.grid_length)
        item = y + x * 10 + 1
        # print(x, y, item)
        return x, y, item

    def object_click_event(self, event):
        # item = self.canvas.find_closest(event.x, event.y)
        x, y, new_item = self.fromCoordsToItemIndex(event.x, event.y)
        if x >= 10 or x < 0 or y >= 10 or y < 0:
            return
        self.gridmap[y, x] = (self.gridmap[y, x] + 1) % 5
        # print(self.gridmap[y, x] )
        self.canvas.itemconfigure(new_item, fill=self.color_map[self.gridmap[y, x]])
        # print('Clicked object at: ', event.x, event.y, new_item)

    def object_de_click_event(self, event):
        # item = self.canvas.find_closest(event.x, event.y)
        x, y, new_item = self.fromCoordsToItemIndex(event.x, event.y)
        if x >= 10 or x < 0 or y >= 10 or y < 0:
            return
        self.gridmap[y, x] = 0
        self.canvas.itemconfigure(new_item, fill="white")
        # print('Clicked object at: ', event.x, event.y, new_item)

    def on_click(self, *args):
        # print(self.gridmap)
        # print(self.section.getSelected())
        # print(self.section1.getSelected())
        GameWindow(self, [self.game], self.section1.getSelected(), pattern=self.gridmap.T)
