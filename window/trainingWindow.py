import random

import customtkinter as ctk
from widget.gameCanvas import GameCanvas
from utility.nameSplitor import splitByUpper
from training.base_training_env import BaseTrainingEnv
from training.genetic_snake_training_env import load_stats
from PIL import Image, ImageTk
from utility.nameSplitor import splitByUnder
from time import time
from snake_game_gen.Win_counter import WinCounter
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)


class TrainingWindow(ctk.CTkToplevel):
    layouts = [(0, 0), (0, 1), (1, 0), (1, 1)]

    def __init__(self, master, game: str, agent: str, settings: dict, training_env: BaseTrainingEnv, speed: int,
                 display_nn: bool = False, *args,
                 **kwargs):
        super().__init__(master, *args, **kwargs)
        self.title("Test")
        print("settings in train window", settings)
        # self.geometry("400x400")
        # self.resizable(False, False)
        # label = ctk.CTkLabel(self, text="window")
        # label.pack()
        self.game_canvas = ctk.CTkCanvas(self, width=400, height=400)
        self.game_canvas.grid(row=0, column=0)

        self.stats_frame = ctk.CTkFrame(self)
        self.stats_frame.grid(row=0, column=1)
        heights = 0
        max_height = 250
        current_row = 0
        current_col = 0
        max_row = 0
        self.training_env = training_env
        for i, (key, value) in enumerate(settings.items()):
            # print(i, key, value)
            component = ctk.CTkLabel(self.stats_frame, text=f"{splitByUnder(key)}: {value}", anchor="w",
                                     font=ctk.CTkFont(size=15, weight="bold"))
            component.grid(row=current_row, column=current_col, sticky="nsew")
            component.update()
            heights += int(component.winfo_height())

            if heights >= max_height:
                current_row = 0
                current_col += 1
                heights = 0

            else:
                current_row += 1
                if current_row > max_row:
                    max_row = current_row
            # print(heights, max_height,max_row)
        current_row = max_row
        # print(((time() - self.training_env.t1)/3600.))
        self.generation_label = ctk.CTkLabel(
            self.stats_frame,

            text=f"======================= Gneration {self.training_env.current_generation} at time {((time() - self.training_env.t1) / 3600.):f} in Hours =======================",
            anchor="w",
            font=ctk.CTkFont(size=15, weight="bold"))
        self.generation_label.grid(row=current_row + 1, column=0, columnspan=2, sticky="nsew")

        self.fitness_label = ctk.CTkLabel(
            self.stats_frame,
            text=f"Max fitness: {self.training_env.population.fittest_individual.fitness}", anchor="w",
            font=ctk.CTkFont(size=15, weight="bold"))
        self.fitness_label.grid(row=current_row + 2, column=0, sticky="nsew")

        self.score_label = ctk.CTkLabel(
            self.stats_frame,
            text=f"Best Score: {self.training_env.population.fittest_individual.score}", anchor="w",
            font=ctk.CTkFont(size=15, weight="bold"))
        self.score_label.grid(row=current_row + 3, column=0, sticky="nsew")

        self.avg_fitness_label = ctk.CTkLabel(
            self.stats_frame,
            text=f"Average fitness: {self.training_env.population.average_fitness}", anchor="w",
            font=ctk.CTkFont(size=15, weight="bold"))
        self.avg_fitness_label.grid(row=current_row + 4, column=0, sticky="nsew")

        self.win_label = ctk.CTkLabel(
            self.stats_frame,
            text=f"Wins: {(WinCounter.counter / self.training_env._current_individual) if self.training_env._current_individual != 0 else 0.0}",
            anchor="w",
            font=ctk.CTkFont(size=15, weight="bold"))
        self.win_label.grid(row=current_row + 5, column=0, sticky="nsew")

        self.which_one = ctk.CTkLabel(
            self.stats_frame,
            text=f"Iteration: {self.training_env._current_individual}",
            anchor="w",
            font=ctk.CTkFont(size=15, weight="bold"))
        self.which_one.grid(row=current_row + 6, column=0, sticky="nsew")

        self.speed_label = ctk.CTkLabel(
            self.stats_frame,
            text=f"Game Speed: in ms per frame",
            anchor="w",
            font=ctk.CTkFont(size=15, weight="bold"))
        self.speed_label.grid(row=3, column=1, sticky="nsew")
        self.speed_entry = ctk.CTkEntry(
            self.stats_frame,
            placeholder_text=f"{speed}",
            font=ctk.CTkFont(size=15, weight="bold"))
        self.speed_entry.grid(row=4, column=1, sticky="nsew")

        self.display_label = ctk.CTkLabel(
            self.stats_frame,
            text=f"Game Speed: in ms per frame",
            anchor="w",
            font=ctk.CTkFont(size=15, weight="bold"))
        self.display_label.grid(row=5, column=1, sticky="nsew")
        self.display_entry = ctk.CTkOptionMenu(
            self.stats_frame,
            values=["True", "False"],
            font=ctk.CTkFont(size=15, weight="bold"),
            command=self.select_display
        )
        self.display_entry.grid(row=6, column=1, sticky="nsew")
        self.display_entry.set(str(display_nn))

        self.graph_frame = ctk.CTkFrame(self)
        self.graph_frame.grid(row=1, column=0, columnspan=2)

        fig = Figure(figsize=(10, 5),
                     dpi=100)

        # creating the Tkinter canvas
        # containing the Matplotlib figure
        self.graph_canvas = FigureCanvasTkAgg(fig,
                                              master=self.graph_frame)
        self.graph_canvas.draw()

        # placing the canvas on the Tkinter window
        self.graph_canvas.get_tk_widget().pack()

        # placing the toolbar on the T

        # print(games)
        # print(agents)
        self.speed = speed

        self.images = [None]
        self.display_nn = display_nn
        self.old_generation = 0

        self.after(self.speed, self.training)

    def select_display(self, choice):
        self.display_nn = eval(self.display_entry.get())

    def training(self):
        self.training_env.update(self.display_nn)
        img = self.training_env.snake.render(mode="2d_array")

        self.images[0] = ImageTk.PhotoImage(image=Image.fromarray(img))
        # self.create_image((self.layout[i][0] * obs.shape[1], self.layout[i][1] * obs.shape[0]), anchor="nw",
        #                   image=self.images[0])
        self.game_canvas.create_image((0, 0), anchor="nw",
                                      image=self.images[0])
        # if self.display_nn:
        #     self.training_env.myCanvas.update_network(self.training_env.snake.observation)

        self.generation_label.configure(

            text=f"======================= Gneration {self.training_env.current_generation} at time {((time() - self.training_env.t1) / 3600.):f} in Hours =======================",
        )
        if self.training_env.current_generation != self.old_generation:
            self.generation_update()
            self.old_generation = self.training_env.current_generation

        self.fitness_label.configure(
            text=f"Max fitness: {self.training_env.population.fittest_individual.fitness}", anchor="w",
        )

        self.score_label.configure(
            text=f"Best Score: {self.training_env.population.fittest_individual.score}", anchor="w",
        )

        self.avg_fitness_label.configure(
            text=f"Average fitness: {self.training_env.population.average_fitness}", anchor="w",
        )

        self.win_label.configure(
            text=f"Wins: {(WinCounter.counter / self.training_env._current_individual) if self.training_env._current_individual != 0 else 0.0}",
        )
        self.which_one.configure(
            text=f"Iteration: {self.training_env._current_individual}",
        )
        # print(img.shape)
        try:
            self.speed = int(self.speed_entry.get())
        except Exception:
            self.speed_entry.delete(0, len(self.speed_entry.get()))
            self.speed_entry.insert(0, self.speed)

        # self.training()
        self.after(self.speed, self.training)

    def generation_update(self):
        data = load_stats(self.training_env.model_path + "/snake_stats.csv", normalize=False)
        # print(data["apples"]["max"])
        fig = Figure(figsize=(10, 5),
                     dpi=100)

        # list of squares
        # order = random.randint(0, 10)
        # y = [i ** order for i in range(101)]

        # adding the subplot
        plot1 = fig.add_subplot(111)

        # plotting the graph
        plot1.plot(data["apples"]["max"])
        plot1.set_title("Max apples number in each generation")
        plot1.set_xlabel("Generations", color="C0")
        plot1.set_ylabel("Apples", color="C0")

        self.graph_canvas.figure = fig
        self.graph_canvas.draw()
