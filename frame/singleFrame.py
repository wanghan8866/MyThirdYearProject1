import customtkinter as ctk

from meta.appMeta import AppMeta
from widget.selectSection import SelectSection
from window.gameWindow import GameWindow


class SinglePlayerFrame(ctk.CTkFrame):

    def on_click(self, *args):
        GameWindow(self, self.section.getSelected(), self.section1.getSelected())

    def __init__(self, master: any, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        label = ctk.CTkLabel(self, text="Single Player", font=ctk.CTkFont(size=15, weight="bold"))
        label.pack()

        self.section1 = SelectSection(self, "Agent", mode="m", content=[])
        self.section = SelectSection(self, "Game", mode="s", content=list(AppMeta.GAMES.keys()), linker=self.section1)
        self.section.pack(fill="x", padx=20)

        self.section1.pack(fill="x", padx=20, pady=20)

        button = ctk.CTkButton(self, text="Launch", command=self.on_click)
        button.pack(side="bottom", pady=20)
