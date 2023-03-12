# import tkinter
import customtkinter as tk
import numpy as np
import colorsys
from snake_game_gen.snake_env import Snake, load_snake
from snake_game_gen.settings import settings
import time
import cv2

tk.set_default_color_theme("dark-blue")


class NN_canvas:
    def __init__(self, parent, snake: Snake, bg, width, height, *args, **kwargs):
        # super().__init__(parent, *args, **kwargs)
        self.bg = bg
        self.snake = snake
        self.horizontal_distance_between_layers = 50
        self.vertical_distance_between_nodes = 10
        self.num_neurons_in_largest_layer = max(self.snake.network.layer_nodes)
        self.neuron_locations = {}
        self.img = np.zeros((1000, 10000, 3), dtype='uint8')
        self.height = height
        self.width = width

    def update_network(self):
        self.img = np.zeros((1000, 1000, 3), dtype='uint8')
        vertical_space = 8
        radius = 8
        # height = int(self["height"])
        height = self.height
        width = self.width
        # width = int(self["width"])
        layer_nodes = self.snake.network.layer_nodes
        default_offset = 30
        h_offset = default_offset
        inputs = self.snake.vision_as_array
        out = self.snake.network.feed_forward(inputs)
        max_out = np.argmax(out)
        # print(layer_nodes)
        for layer, num_nodes in enumerate(layer_nodes):
            # print(num_nodes)
            v_offset = (height - ((2 * radius + vertical_space) * num_nodes)) / 2
            activation = None
            if layer > 0:
                activation = self.snake.network.params["A" + str(layer)]

            for node in range(num_nodes):
                x_loc = int(h_offset)
                y_loc = int(node * (radius * 2 + vertical_space) + v_offset)
                t = (layer, node)
                colour = "white"
                saturation = 0.
                if t not in self.neuron_locations:
                    self.neuron_locations[t] = (x_loc, y_loc + radius)
                # print(t, inputs[node])
                if layer == 0:
                    if inputs[node] > 0:
                        # green

                        colour = colorsys.hsv_to_rgb(120 / 360., inputs[node], 1.)
                        colour = [int(c * 255) for c in colour]
                        colour = [colour[-1], colour[1], colour[0]]
                        # colour = (0, 255, 0)
                    else:
                        colour = (125, 125, 125)
                    text = f"{self.snake.vision_as_array[node, 0]:.2f}"
                    cv2.putText(self.img, text,
                                (int(h_offset - 25),
                                 int(node * (radius * 2 + vertical_space) + v_offset + radius)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, colour, lineType=cv2.LINE_AA)
                elif layer > 0 and layer < len(layer_nodes) - 1:
                    # print(activation[node, 0])
                    if activation[node, 0] > 0:
                        saturation = max(min(activation[node, 0], 1.), 0.)
                        colour = colorsys.hsv_to_rgb(1., saturation, 1.)
                        colour = [int(c * 255) for c in colour]
                        colour = (colour[-1], colour[1], colour[0])
                    else:
                        colour = (125, 125, 125)
                    # print(colour)
                    # print(colour)
                    # colour = '#%02x%02x%02x' % (colour[0], colour[1], colour[2])
                    # print(colour)
                elif layer == len(layer_nodes) - 1:
                    text = self.snake.possible_directions[node].upper()
                    if node == max_out:
                        colour = (0, 255, 255)

                    else:
                        colour = (125, 125, 125)
                    # self.create_text(h_offset + 30, node * (radius * 2 + vertical_space) + v_offset + 1.5 * radius,
                    #                  text=text, fill=colour)

                    cv2.putText(self.img, text,
                                (int(h_offset + 30),
                                 int(node * (radius * 2.5 + vertical_space) + v_offset + 1.5 * radius)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, colour, lineType=cv2.LINE_AA)
                # print(x_loc,y_loc)
                cv2.circle(self.img, (x_loc + radius, y_loc + radius), radius, colour, -1, lineType=cv2.LINE_AA)
                # self.create_oval(x_loc, y_loc, x_loc + radius * 2, y_loc + radius * 2, fill=colour)
            h_offset += 150

        for l in range(1, len(layer_nodes)):
            weights = self.snake.network.params["W" + str(l)]
            # print(weights.shape)
            prev_nodes = weights.shape[1]
            curr_nodes = weights.shape[0]
            for prev_node in range(prev_nodes):
                for curr_node in range(curr_nodes):
                    if weights[curr_node, prev_node] > 0:
                        # colour = "red"
                        colour = (0, 150, 0)

                    else:
                        # colour = "gray"
                        colour = (125, 125, 125)
                    start = self.neuron_locations[(l - 1, prev_node)]
                    end = self.neuron_locations[(l, curr_node)]

                    # self.create_line(start[0] + radius * 2, start[1], end[0], end[1], fill=colour)
                    cv2.line(self.img,
                             (start[0] + radius, start[1]),
                             end,
                             colour, 1, lineType=cv2.LINE_AA)
        # self.update()
        cv2.imshow('n', self.img)


if __name__ == '__main__':
    # init tk
    root = tk.CTk()
    displaying = True
    env = load_snake("models/test_64", f"snake_1699", settings)
    # env = Snake([10,10], hidden_layer_architecture=settings['hidden_network_architecture'],
    #                   hidden_activation=settings['hidden_layer_activation'],
    #                   output_activation=settings['output_layer_activation'],
    #                   lifespan=settings['lifespan'],
    #                   apple_and_self_vision=settings['apple_and_self_vision'])
    # env.look()
    # env.step(action=-1)
    # create canvas
    myCanvas=None
    if displaying:
        myCanvas = NN_canvas(root, snake=env, bg="white", height=1000, width=1000)
    # env = load_snake("models/tests", f"snake_1400", settings)
    # It will check your custom environment and output additional warnings if needed
    # check_env(env)
    episodes = 100
    rewards_history = []
    avg_reward_history = []
    print(env.observation_space.shape)
    # myCanvas.pack()
    for episode in range(episodes):
        # if episode == 41:
        #     myCanvas = NN_canvas(root, snake=env, bg="white", height=1000, width=1000)
        #     displaying=True
        done = False
        obs = env.reset()
        rewards = 0
        while not done:
            if displaying:
                action = env.possible_directions[env.action_space.sample()]
                t_end = time.time() + 0.1
                k =  -1
                # action = -1
                while time.time() < t_end:
                    if k == -1:
                        # pass
                        k = cv2.waitKey(1)
                        # print(k)

                        if k == 97:
                            action = "l"
                        elif k == 100:
                            action = "r"
                        elif k == 119:
                            action = "u"
                        elif k == 115:
                            action = "d"
                        # print(k)
                    else:
                        continue
                # print(f"action: {action}")
                env.render()
                myCanvas.update_network()
            action = -1

            obs, reward, done, info = env.step(action)
            # print("obs", obs)
            #
            # root.update()

            # A=env.render(mode="rgb_array")
            # print(A.shape)
            # print("reward", reward)

            # print(obs.shape)
            rewards += reward
        # print(env.steps_history)
        # print(len(env.steps_history))

        # rewards_history.append(rewards)
        avg_reward = (len(env.snake_array))
        avg_reward_history.append(avg_reward)
        env.calculate_fitness()
        print(f"games: {episode + 1}, avg_score: {avg_reward} fit: {env.fitness}")
        print(f"games: {episode + 1}, avg_score: {env.steps_history}")
        print(f"games: {episode + 1}, avg_score: {env.steps_history}")
        if len(env.steps_history)>0:
            print(f"games: {episode + 1}, avg_score: {np.mean(env.steps_history)}\n")
        else:
            print()

    # draw arcs
    print()
    print(f"games: average reward over {episodes} games: {np.mean(avg_reward_history)}")
    print(f"games: std reward over {episodes} games: {np.std(avg_reward_history)}")
    # add to window and show

    # root.mainloop()
