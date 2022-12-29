import customtkinter as ctk
from widget.gameCanvas import GameCanvas
from utility.nameSplitor import splitByUpper


class GameWindow(ctk.CTkToplevel):
    layouts = [(0, 0), (0, 1), (1, 0), (1, 1)]

    def __init__(self, master, games: list, agents: list, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.title("Game")
        # self.geometry("400x400")
        # self.resizable(False, False)
        label = ctk.CTkLabel(self, text="window")
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
            game = GameCanvas(frame, game_name=games[0], agent_name=agent, background="black")
            if agent == "human":
                self.human_agents.append(game)
            game.pack(padx=5, pady=5, fill="both", expand=True)
            frame.grid(row=GameWindow.layouts[i][0], column=GameWindow.layouts[i][1], padx=5, pady=5, sticky="news")
        # self.game_frame.grid_propagate(False)
        # self.grid_propagate(False)
        # self.pack_propagate(False)
        self.events_binding()
        print(games)
        print(agents)

    def onKey(self, e):
        for agent in self.human_agents:
            agent.on_key_press(e)
        # print("key", args[0])

    def events_binding(self):
        self.bind("<Key>", self.onKey)
        self.bind("<Button-1>", self.onKey)
