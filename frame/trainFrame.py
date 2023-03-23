import tkinter as tk
from meta.appMeta import AppMeta
from utility.imageLoader import ImageLoader
import customtkinter as ctk
from customtkinter import filedialog
from frame.modelTestFrame import ModelTestFrame, Component, NumberType
from window.trainingWindow import TrainingWindow
from widget.selectSection import SelectSection
from meta.trainingMeta import TrainingMeta
from frame.modelTestFrame import ModelTestFrame
from utility.nameSplitor import splitByUpper
from training.genetic_snake_training_env import GeneticSnakeTrainingEnv
from snake_game_gen.snake_env3 import load_snake
import os
import json
from window.errorbox import ErrorBox
from training.dqn_snake_training_env import DQNTrainingEnv

class TrainingFrame(ctk.CTkFrame):
    def __init__(self, master: any, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        label = ctk.CTkLabel(self, text="Training Agents for Snake Game", font=ctk.CTkFont(size=15, weight="bold"))
        label.pack()

        self.section1 = SelectSection(self, "Model", mode="s", content=list(TrainingMeta.MODELS.keys()),
                                      callback=self.on_selected)
        # self.section = SelectSection(self, "Game", mode="s", content=list(AppMeta.GAMES.keys()), linker=self.section1)
        # self.section.pack(fill="x", padx=20)
        self.section1.pack(fill="x", padx=20, pady=10)
        self.content_frame = ctk.CTkFrame(self, border_color="gray75")
        open_button = ctk.CTkButton(self, text="Open a directory", command=self.open_dir)
        open_button.pack()

        self.content_frame.pack(fill="both", padx=20, pady=5)
        self.env = None
        self.settings = None
        self.snake_file_names = {}

        self.frame_map = {}
        for key, values in TrainingMeta.MODELS.items():
            self.frame_map[key] = ModelTestFrame(self.content_frame, name=splitByUpper(key), settings=values,
                                                 corner_radius=0, fg_color="transparent", border_color="gray35",
                                                 height=260)

        button = ctk.CTkButton(self, text="Launch", command=self.on_click)
        button.pack(side="bottom", pady=5)

    def open_dir(self):
        frame = self.section1.getSelected()
        if len(frame) == 0:
            ErrorBox("Snake Loading", "No snake selected!")
            return
        directory = filedialog.askdirectory()

        if directory == "":
            ErrorBox("Snake Loading", "Wrong directory format!")
            return
        print(directory)
        print(os.listdir(directory))

        max_name = ""
        max_iteration = 0
        generations = []
        for file in os.listdir(directory):
            if "_" in file and "." not in file:
                gens = int(file.split("_")[1])
                # print(gens)
                self.snake_file_names[gens] = file
                generations.append(gens)
                if gens > max_iteration:
                    max_name = file
                    max_iteration = gens
        print(max_name)
        with open(f"{directory}/settings.json", 'r', encoding='utf-8') as fp:
            settings = json.load(fp)

        print("dir: selected", self.section1.getSelected())
        # settings_iter =
        for com, (key, value) in zip(self.frame_map[frame[0]].components, settings.items()):
            value = settings[com.key]
            if isinstance(com.entry, ctk.CTkOptionMenu):
                com.entry.set(value)
            elif isinstance(com.entry, ctk.CTkEntry):
                com.entry.delete(0, len(com.entry.get()))
                com.entry.insert(0, value)

            # com.configure(text="")
        print(self.frame_map[frame[0]].get_components("generation"))
        print(self.frame_map[frame[0]].get_components("working_directory"))
        entry = self.frame_map[frame[0]].get_components("generation").entry
        # for i, gens in enumerate(generations):
        entry.configure(values=[f"{gens}" for gens in sorted(generations)], )
        entry.set(f"{max_iteration}")

        entry.update()
        entry = self.frame_map[frame[0]].get_components("working_directory").entry
        entry.delete(0, len(entry.get()))
        entry.insert(0, directory)
        # print("gens",generations)
        # print("keys", settings)
        # self.frame_map[frame[0]].get_components("working_directory")

        # snake = load_snake(directory, max_name)
        # self.setting = settings
        self.settings = settings
        self.settings["working_directory"] = directory
        self.settings["generation"] = max_iteration
        self.env = GeneticSnakeTrainingEnv(self.settings, times=50, path=directory,
                                           current_gen=max_iteration + 1,
                                           create_snake=lambda: load_snake(directory, max_name))

        # print(settings)

    def on_click(self, *args):
        # print(self.gridmap)
        # print(self.section.getSelected())
        print(self.section1.getSelected())
        if len(self.section1.getSelected()) == 0:
            ErrorBox("Snake training", "No snake selected!")
            return
        self.settings = self.frame_map[self.section1.getSelected()[0]].getAllInputs()
        # print(settings)
        if self.section1.getSelected()[0]=="DoubleDeepQLearning":
            self.env=DQNTrainingEnv(self.settings)
            TrainingWindow(self, "Snake-DQN", "", settings=self.settings, training_env=self.env, speed=1)

        else:
            if self.env is None:
                directory = filedialog.askdirectory()

                if directory == "":
                    ErrorBox("Snake Loading", "Wrong directory format!")
                    return
                print(directory)
                print(os.listdir(directory))
                self.env = GeneticSnakeTrainingEnv(self.settings, times=50, path="models/gen/test2")
            if self.settings["generation"] != "0":
                print("generation env")
                self.env = GeneticSnakeTrainingEnv(self.settings, times=50, path=self.settings["working_directory"],
                                                   current_gen=eval(self.settings["generation"]) + 1,
                                                   create_snake=lambda: load_snake(self.settings["working_directory"],
                                                                                   self.snake_file_names[
                                                                                       eval(self.settings["generation"])]))

            TrainingWindow(self, "Snake-gen", "", settings=self.settings, training_env=self.env, speed=1)

    def on_selected(self):

        print("selected", self.section1.getSelected())
        for frame in self.frame_map:
            if len(self.section1.getSelected()) == 0:
                self.frame_map[frame].pack_forget()
                continue
            if self.section1.getSelected()[0] == frame:
                print("selected")
                self.frame_map[frame].pack(fill="both")
            else:
                self.frame_map[frame].pack_forget()
