import tkinter as tk
from meta.appMeta import AppMeta
from utility.imageLoader import ImageLoader
import customtkinter as ctk
from window.trainingWindow import TrainingWindow
from widget.selectSection import SelectSection
from meta.trainingMeta import TrainingMeta
from frame.modelTestFrame import ModelTestFrame
from utility.nameSplitor import splitByUpper
from training.genetic_snake_training_env import GeneticSnakeTrainingEnv


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
        self.content_frame.pack(fill="both", padx=20, pady=5)

        self.frame_map = {}
        for key, values in TrainingMeta.MODELS.items():
            self.frame_map[key] = ModelTestFrame(self.content_frame, name=splitByUpper(key), settings=values,
                                                 corner_radius=0, fg_color="transparent", border_color="gray35",
                                                 height=260)

        button = ctk.CTkButton(self, text="Launch", command=self.on_click)
        button.pack(side="bottom", pady=20)

    def on_click(self, *args):
        # print(self.gridmap)
        # print(self.section.getSelected())
        print(self.section1.getSelected())
        if len(self.section1.getSelected()) == 0:
            return
        settings = self.frame_map[self.section1.getSelected()[0]].getAllInputs()
        # print(settings)
        env = GeneticSnakeTrainingEnv(settings,times=50,path="models/gen/test1")

        TrainingWindow(self, "Snake-gen", "", settings=settings, training_env=env, speed=1)

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
