import customtkinter as ctk

from frame.homeFrame import HomeFrame
from frame.settingFrame import SettingFrame
from frame.singleFrame import SinglePlayerFrame
from frame.testingFrame import TestingFrame
from frame.trainFrame import TrainingFrame
from meta.appMeta import AppMeta
from utility.imageLoader import ImageLoader


class AppWindow(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title(AppMeta.APP_TITLE)
        self.geometry(AppMeta.SIZE)
        self.resizable(False, False)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self.navigation_frame = ctk.CTkFrame(self, corner_radius=0)
        self.navigation_frame.grid(row=0, column=0, sticky="nsew")
        self.navigation_frame.grid_rowconfigure(5, weight=1)

        self.navigation_frame_label = ctk.CTkLabel(self.navigation_frame, text="  Menu ",
                                                   image=ImageLoader.getTkImage("logo"),
                                                   compound="left",
                                                   font=ctk.CTkFont(size=20, weight="bold"))

        self.navigation_frame_label.grid(row=0, column=0, padx=20, pady=20)

        self.home_button = ctk.CTkButton(self.navigation_frame, corner_radius=0, height=40,
                                         border_spacing=10, text="Home", fg_color="transparent",
                                         text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                                         image=ImageLoader.getTkImage("home"), anchor="w",
                                         command=lambda: self.select_frame_by_name("home"))

        self.home_button.grid(row=1, column=0, sticky="ew")

        self.single_button = ctk.CTkButton(self.navigation_frame, corner_radius=0, height=40,
                                           border_spacing=10, text="Single-Player Mode", fg_color="transparent",
                                           text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                                           image=ImageLoader.getTkImage("user"), anchor="w",
                                           command=lambda: self.select_frame_by_name("single"))

        self.single_button.grid(row=2, column=0, sticky="ew")

        self.multi_button = ctk.CTkButton(self.navigation_frame, corner_radius=0, height=40,
                                          border_spacing=10, text="Training agents", fg_color="transparent",
                                          text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                                          image=ImageLoader.getTkImage("chat"), anchor="w",
                                          command=lambda: self.select_frame_by_name("train"))

        self.multi_button.grid(row=3, column=0, sticky="ew")

        self.test_button = ctk.CTkButton(self.navigation_frame, corner_radius=0, height=40,
                                         border_spacing=10, text="Experimenting", fg_color="transparent",
                                         text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                                         image=ImageLoader.getTkImage("user"), anchor="w",
                                         command=lambda: self.select_frame_by_name("test"))

        self.test_button.grid(row=4, column=0, sticky="ew")

        self.setting_button = ctk.CTkButton(self.navigation_frame, corner_radius=0, height=40,
                                            border_spacing=10, text="Settings", fg_color="transparent",
                                            text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                                            image=ImageLoader.getTkImage("user"), anchor="w",
                                            command=lambda: self.select_frame_by_name("setting"))

        self.setting_button.grid(row=5, column=0, sticky="ew")

        self.home_frame = HomeFrame(self, corner_radius=0, fg_color="transparent")
        self.single_frame = SinglePlayerFrame(self, corner_radius=0, fg_color="transparent")
        self.train_frame = TrainingFrame(self, corner_radius=0, fg_color="transparent")
        self.setting_frame = SettingFrame(self, corner_radius=0, fg_color="transparent")
        self.testing_frame = TestingFrame(self, corner_radius=0, fg_color="transparent")

        self.select_frame_by_name("home")

    def select_frame_by_name(self, name):
        self.home_button.configure(fg_color=("gray75", "gray25") if name == "home" else "transparent")
        self.single_button.configure(fg_color=("gray75", "gray25") if name == "single" else "transparent")
        self.multi_button.configure(fg_color=("gray75", "gray25") if name == "train" else "transparent")
        self.test_button.configure(fg_color=("gray75", "gray25") if name == "test" else "transparent")
        self.setting_button.configure(fg_color=("gray75", "gray25") if name == "setting" else "transparent")

        if name == "home":
            self.home_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.home_frame.grid_forget()
        if name == "single":
            self.single_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.single_frame.grid_forget()
        if name == "train":
            self.train_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.train_frame.grid_forget()
        if name == "test":
            self.testing_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.testing_frame.grid_forget()

        if name == "setting":
            self.setting_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.setting_frame.grid_forget()
