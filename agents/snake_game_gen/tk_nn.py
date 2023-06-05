
import customtkinter as tk
import numpy as np
import colorsys
from agents.snake_game_gen.snake_env import Snake, load_snake
from agents.snake_game_gen.settings import settings
import time
import cv2

tk.set_default_color_theme("dark-blue")


class NN_canvas:
    def __init__(self, parent, snake: Snake, bg, width, height, *args, **kwargs):
        
        self.bg = bg
        self.snake = snake
        self.horizontal_distance_between_layers = 50
        self.vertical_distance_between_nodes = 10
        self.num_neurons_in_largest_layer = max(self.snake.network.layer_nodes)
        self.neuron_locations = {}
        self.img = np.zeros((1000, 10000, 3), dtype='uint8')
        self.height = height
        self.width = width

    def update_network(self,obs=None):
        self.img = np.zeros((1000, 1000, 3), dtype='uint8')
        vertical_space = 8
        radius = 8
        
        height = self.height
        width = self.width
        
        layer_nodes = self.snake.network.layer_nodes
        default_offset = 30
        h_offset = default_offset
        inputs = self.snake.vision_as_array
        out = self.snake.network.feed_forward(inputs)
        max_out = np.argmax(out)
        
        for layer, num_nodes in enumerate(layer_nodes):
            
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
                
                if layer == 0:
                    if inputs[node] > 0:
                        

                        colour = colorsys.hsv_to_rgb(120 / 360., inputs[node], 1.)
                        colour = [int(c * 255) for c in colour]
                        colour = [colour[-1], colour[1], colour[0]]
                        
                    else:
                        colour = (125, 125, 125)
                    text = f"{self.snake.vision_as_array[node, 0]:.2f}"
                    cv2.putText(self.img, text,
                                (int(h_offset - 25),
                                 int(node * (radius * 2 + vertical_space) + v_offset + radius)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, colour, lineType=cv2.LINE_AA)
                elif layer > 0 and layer < len(layer_nodes) - 1:
                    
                    if activation[node, 0] > 0:
                        saturation = max(min(activation[node, 0], 1.), 0.)
                        colour = colorsys.hsv_to_rgb(1., saturation, 1.)
                        colour = [int(c * 255) for c in colour]
                        colour = (colour[-1], colour[1], colour[0])
                    else:
                        colour = (125, 125, 125)
                    
                    
                    
                    
                elif layer == len(layer_nodes) - 1:
                    text = self.snake.possible_directions[node].upper()
                    if node == max_out:
                        colour = (0, 255, 255)

                    else:
                        colour = (125, 125, 125)
                    
                    

                    cv2.putText(self.img, text,
                                (int(h_offset + 30),
                                 int(node * (radius * 2.5 + vertical_space) + v_offset + 1.5 * radius)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, colour, lineType=cv2.LINE_AA)
                
                cv2.circle(self.img, (x_loc + radius, y_loc + radius), radius, colour, -1, lineType=cv2.LINE_AA)
                
            h_offset += 150

        for l in range(1, len(layer_nodes)):
            weights = self.snake.network.params["W" + str(l)]
            
            prev_nodes = weights.shape[1]
            curr_nodes = weights.shape[0]
            for prev_node in range(prev_nodes):
                for curr_node in range(curr_nodes):
                    if weights[curr_node, prev_node] > 0:
                        
                        colour = (0, 150, 0)

                    else:
                        
                        colour = (125, 125, 125)
                    start = self.neuron_locations[(l - 1, prev_node)]
                    end = self.neuron_locations[(l, curr_node)]

                    
                    cv2.line(self.img,
                             (start[0] + radius, start[1]),
                             end,
                             colour, 1, lineType=cv2.LINE_AA)
        
        cv2.imshow('n', self.img)


if __name__ == '__main__':
    
    root = tk.CTk()
    displaying = True
    env = load_snake("models/test_64", f"snake_1699", settings)
    
    
    
    
    
    
    
    
    myCanvas=None
    if displaying:
        myCanvas = NN_canvas(root, snake=env, bg="white", height=1000, width=1000)
    
    
    
    episodes = 100
    rewards_history = []
    avg_reward_history = []
    print(env.observation_space.shape)
    
    for episode in range(episodes):
        
        
        
        done = False
        obs = env.reset()
        rewards = 0
        while not done:
            if displaying:
                action = env.possible_directions[env.action_space.sample()]
                t_end = time.time() + 0.1
                k =  -1
                
                while time.time() < t_end:
                    if k == -1:
                        
                        k = cv2.waitKey(1)
                        

                        if k == 97:
                            action = "l"
                        elif k == 100:
                            action = "r"
                        elif k == 119:
                            action = "u"
                        elif k == 115:
                            action = "d"
                        
                    else:
                        continue
                
                env.render()
                myCanvas.update_network()
            action = -1

            obs, reward, done, info = env.step(action)
            
            
            

            
            
            

            
            rewards += reward
        
        

        
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

    
    print()
    print(f"games: average reward over {episodes} games: {np.mean(avg_reward_history)}")
    print(f"games: std reward over {episodes} games: {np.std(avg_reward_history)}")
    

    
