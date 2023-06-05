import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk
from gym import Env

from meta.appMeta import AppMeta
from utility.agentCreator import AgentCreator
from utility.envCreator import EnvCreator

print()
dones = False
MOVE_INCREMENT = 20
MOVES_PER_SECOND = 10
GAME_SPEED = 1000 // MOVES_PER_SECOND


class GameCanvas(ctk.CTkFrame):
    layouts = [[(0, 0)], [(0, 0), (1, 0)], [(0, 0), (1, 0), (0.5, 1)], [(0, 0), (1, 0), (0, 1), (1, 1)]]

    def __init__(self, container, game_name: str, agent_name: str, pattern=None, background="black", create_env=None,
                 *args, **kwargs):
        super().__init__(container, *args, **kwargs)

        if create_env is not None:
            self.env = create_env(pattern=pattern)
        else:
            self.env: Env = EnvCreator.createEnv(game_name, agent_name=agent_name, pattern=pattern)

        self.obs = self.env.reset()
        array = self.env.render(mode="rgb_array")

        self.configure(width=array.shape[0], height=array.shape[1])
        self.flip = False
        self.game_name = game_name
        self.agent_name = agent_name

        if game_name == "FlappyBird":
            self.flip = True
            array = np.transpose(array, axes=(1, 0, 2))
        print(array.shape)
        print(self.env.action_space)
        self.speed = GAME_SPEED
        self.speed_label = ctk.CTkLabel(
            self,
            text=f"Game Speed: in ms per frame",
            anchor="w",
            font=ctk.CTkFont(size=15, weight="bold"))
        self.speed_label.pack(side="top")

        self.speed_entry = ctk.CTkEntry(
            self,
            placeholder_text=f"{self.speed}",
            font=ctk.CTkFont(size=15, weight="bold"))
        self.speed_entry.pack(side="top")

        self.canvas = ctk.CTkCanvas(self, background=background, width=array.shape[1], height=array.shape[0])
        self.canvas.pack(side="bottom", fill="both", expand=True)

        if len(array.shape) > 2:
            self.configure(height=array.shape[0], width=array.shape[1])
        self.container = container

        self.agent = AgentCreator.createAgent(agent_name, game_name, self.env)

        self.images = [None]

        self.current_action = None
        self.total_reward = 0

        self.canvas.after(self.speed, self.move)

    def move(self):

        action = self.agent.predict(self.obs, self.env.action_space, self.current_action)

        self.obs, rewards, done, info = self.env.step(action)
        self.current_action = None
        self.total_reward = rewards
        array = self.agent.render()
        if self.flip:
            array = np.transpose(array, axes=(1, 0, 2))
        self.images[0] = ImageTk.PhotoImage(image=Image.fromarray(array))

        self.canvas.create_image((0, 0), anchor="nw",
                                 image=self.images[0])

        if not done:

            try:
                self.speed = int(self.speed_entry.get())
            except Exception:
                self.speed_entry.delete(0, len(self.speed_entry.get()))
                self.speed_entry.insert(0, self.speed)

            self.canvas.after(self.speed, self.move)
        else:

            self.canvas.delete("all")
            self.canvas.create_text(int(self.cget("width")) // 2, int(self.cget("height")) // 2,
                                    text=f"Game Over with score {self.total_reward}",
                                    fill="white")
            self.env.close()

            print("done")

    def on_key_press(self, e):
        key = e.keysym
        if key in AppMeta.KEY_MAP:
            self.current_action = EnvCreator.returnKeyAction(self.game_name, key)


if __name__ == '__main__':
    root = ctk.CTk()

    canvas1 = GameCanvas(root, game_name="Skiing", agent_name="randomAgent", width=300, height=300, background="black")
    canvas1.grid(row=0, column=0)
    canvas1 = GameCanvas(root, game_name="Skiing", agent_name="randomAgent", width=300, height=300, background="black")
    canvas1.grid(row=0, column=1)
    canvas = GameCanvas(root, game_name="BattleZone", agent_name="randomAgent", width=300, height=300,
                        background="black")
    canvas.grid(row=1, column=0)

    root.mainloop()
