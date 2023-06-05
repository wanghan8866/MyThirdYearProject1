import customtkinter as tk
import numpy as np
import colorsys
from agents.PER.ranked.network import LinearDeepQNetwork, softmax, relu
import cv2
import torch as T

tk.set_default_color_theme("dark-blue")


class Q_NN_canvas:
    def __init__(self, parent, network: LinearDeepQNetwork, bg, width, height, *args, **kwargs):

        self.bg = bg
        self.network = network
        self.horizontal_distance_between_layers = 50
        self.vertical_distance_between_nodes = 10

        self.layers = []
        for t in list(network.parameters())[::2]:
            self.layers.append(t.cpu().detach().numpy().shape[0])

        self.weights = list(self.network.parameters())

        self.num_neurons_in_largest_layer = max(self.layers)
        self.neuron_locations = {}
        self.img = np.zeros((1000, 10000, 3), dtype='uint8')
        self.height = height
        self.width = width

    def update_network(self, obs):
        self.img = np.zeros((1000, 1000, 3), dtype='uint8')
        vertical_space = 8
        radius = 8

        height = self.height
        width = self.width

        layer_nodes = self.layers
        default_offset = 30
        h_offset = default_offset
        inputs = obs
        state = T.tensor(np.array([inputs]), dtype=T.float).to(self.network.device)
        out = self.network.forward(state)
        max_out = T.argmax(out).item()

        A_prev = obs
        params = {}
        x = 0
        for i in range(0, len(self.weights), 2):
            W = self.weights[i].cpu().detach().numpy()
            b = np.array([[w] for w in self.weights[i + 1].cpu().detach().numpy()])

            z = np.dot(W, A_prev) + b
            if i == len(self.weights) - 2:

                A_prev = softmax(z)
            else:
                A_prev = relu(z)

            params["W" + str(x)] = W
            params["b" + str(x)] = b
            params['A' + str(x)] = A_prev
            x += 1

        for layer, num_nodes in enumerate(layer_nodes):

            v_offset = (height - ((2 * radius + vertical_space) * num_nodes)) / 2
            activation = None
            if layer > 0:
                activation = params["A" + str(layer)]

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
                    text = f"{obs[node]:.2f}"
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

                    text = ["U", "D", "L", "R"][node]

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
            weights = params["W" + str(l)]

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

        cv2.imshow('Q_n', self.img)
