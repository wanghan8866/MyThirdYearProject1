import customtkinter as ctk

from utility.nameSplitor import splitByUpper
from widget.gameCanvas import GameCanvas


class GameWindow(ctk.CTkToplevel):
    layouts = [(0, 0), (0, 1), (1, 0), (1, 1)]

    def __init__(self, master, games: list, agents: list, pattern=None, create_snake=None, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.title(games[0])

        label = ctk.CTkLabel(self, text=games[0])
        label.pack()
        self.game_frame = ctk.CTkFrame(self)
        self.game_frame.pack()
        self.human_agents = []
        for i, agent in enumerate(agents):
            frame = ctk.CTkFrame(self.game_frame, border_color="black", border_width=3, corner_radius=0)
            if agent == "human":
                frame.configure(border_color="red", border_width=3)

            name = ctk.CTkLabel(frame, text=splitByUpper(agent))
            name.pack()
            if agent == "genetic Agent":
                print("creating existing snake")
                game = GameCanvas(frame, game_name=games[0], agent_name=agent, background="black", pattern=pattern,
                                  create_env=create_snake)
            else:
                game = GameCanvas(frame, game_name=games[0], agent_name=agent, background="black", pattern=pattern)
            if agent == "human":
                self.human_agents.append(game)
            game.pack(padx=5, pady=5, fill="both", expand=True)
            frame.grid(row=GameWindow.layouts[i][0], column=GameWindow.layouts[i][1], padx=5, pady=5, sticky="news")

        self.events_binding()

    def onKey(self, e):
        for agent in self.human_agents:
            agent.on_key_press(e)

    def events_binding(self):
        self.bind("<Key>", self.onKey)
        self.bind("<Button-1>", self.onKey)
