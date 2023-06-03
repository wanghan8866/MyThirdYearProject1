import os
import tkinter as tk
from typing import Tuple

import customtkinter as ctk
import numpy as np
from customtkinter import filedialog

from agents.snake_game_gen.patterns import Pattern
from agents.snake_game_gen.snake_env3 import load_snake
from meta.appMeta import AppMeta
from widget.selectSection import SelectSection
from window.errorbox import ErrorBox
from window.gameWindow import GameWindow


def load_snake_testing(pattern, directory, name):
    snake = load_snake(directory, name)
    snake.use_pattern(Pattern(pattern))
    return snake


class TestingFrame(ctk.CTkFrame):
    def __init__(self, master: any, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        label = ctk.CTkLabel(self, text="Testing Arena", font=ctk.CTkFont(size=15, weight="bold"))
        label.pack()
        self.game = "Snake-gen"
        # self.section1 = SelectSection(self, "Agent", mode="m", content=[])
        # print(AppMeta.GAMES.get("Snake-gen",[]))
        self.section1 = SelectSection(self, "Agent", mode="m", content=AppMeta.GAMES.get("Snake-gen", []),
                                      callback=self.on_select)
        # self.section = SelectSection(self, "Game", mode="s", content=list(AppMeta.GAMES.keys()), linker=self.section1)
        # self.section.pack(fill="x", padx=20)
        self.section1.pack(fill="x", padx=20, pady=10)
        # self.section1.pack(fill="x", padx=20, pady=20)

        self.grid_length = 30
        self.voffset = 5
        self.canvas = ctk.CTkCanvas(self, height=10 * self.grid_length + 2 * self.voffset)
        self.canvas.pack(expand=1, fill=tk.BOTH, pady=10, padx=20)
        self.gridmap = np.zeros(shape=(10, 10), dtype=np.uint8)

        self.color_map = ["white", "pink", "green", "blue", "red"]
        # print()
        self.hoffset = int(self.canvas["width"]) // 2 - self.grid_length * 2
        self.snake_directory = None

        for i, row in enumerate(self.gridmap):
            for j, col in enumerate(row):
                self.canvas.create_rectangle(i * self.grid_length + self.hoffset, j * self.grid_length + self.voffset,
                                             (i + 1) * self.grid_length + self.hoffset,
                                             (j + 1) * self.grid_length + self.voffset, width=2, fill="white")

        self.bottom_section_frame = ctk.CTkFrame(self)
        self.bottom_section_frame.pack(fill="both", side="bottom", expand=True)

        self.general_snake_agent_frame = ctk.CTkFrame(self.bottom_section_frame)
        button = ctk.CTkButton(self.general_snake_agent_frame, text="Launch", command=self.on_click)
        button.pack(side="bottom", pady=10)

        self.genetic_snake_agent_frame = ctk.CTkFrame(self.bottom_section_frame)
        button = ctk.CTkButton(self.genetic_snake_agent_frame, text="load existing genetic agent",
                               command=self.on_btn_for_load)
        button.grid(row=0, column=0, pady=10, padx=15)
        self.file_path_label = ctk.CTkLabel(self.genetic_snake_agent_frame,
                                            text="Genetic snake agent:", anchor="w")
        self.file_path_label.grid(row=0, column=1, pady=10, padx=5)
        self.agent_selection = ctk.CTkOptionMenu(master=self.genetic_snake_agent_frame,
                                                 values=["None"])
        self.agent_selection.grid(row=0, column=2, pady=10, padx=5)
        button = ctk.CTkButton(self.genetic_snake_agent_frame, text="Launch", command=self.on_click)
        button.grid(row=0, column=3, pady=10, padx=15)

        # oval_element = self.canvas.create_oval(20, 20, 100, 100, width=2, fill="white")
        self.canvas.bind('<Button-1>', self.object_click_event)
        self.canvas.bind('<Button-3>', self.object_de_click_event)
        self.canvas.pack()

    def on_select(self):
        # print("selected", self.section1.getSelected())
        if len(self.section1.getSelected()) == 0:
            self.general_snake_agent_frame.pack_forget()
            self.genetic_snake_agent_frame.pack_forget()
            return
        if 'genetic Agent' in self.section1.getSelected():
            self.genetic_snake_agent_frame.pack(side="bottom", pady=10)
            self.general_snake_agent_frame.pack_forget()
        else:
            self.genetic_snake_agent_frame.pack_forget()
            self.general_snake_agent_frame.pack(side="bottom", pady=10)

    def on_btn_for_load(self):
        frame = self.section1.getSelected()
        if len(frame) == 0:
            ErrorBox("Snake Loading", "No snake selected!")
            return
        directory = filedialog.askdirectory()

        if directory == "":
            ErrorBox("Snake Loading", "Wrong directory format!")
            return
        print(directory)
        self.snake_directory = directory

        print(sorted([snake_file for snake_file in os.listdir(directory)
                      if snake_file.startswith("snake") and snake_file[-1].isdigit()],
                     key=lambda name: int(name.split("_")[1])))

        self.agent_selection.configure(
            values=sorted([snake_file for snake_file in os.listdir(directory)
                           if snake_file.startswith("snake") and snake_file[-1].isdigit()],
                          key=lambda name: int(name.split("_")[1])), )
        self.agent_selection.update()

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
        creating_snake = None
        if self.snake_directory is not None:
            print(self.snake_directory)
            print(self.agent_selection.get())
            creating_snake = lambda pattern: load_snake_testing(pattern, self.snake_directory,
                                                                self.agent_selection.get())

        GameWindow(self, [self.game],
                   self.section1.getSelected(),
                   pattern=self.gridmap.T,
                   create_snake=creating_snake)
