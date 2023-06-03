import math
import random
from typing import Tuple

import cv2
import gym
import numpy as np
from gym import spaces
from matplotlib import pyplot as plt

angle_16 = math.tan(22.5 / 180 * math.pi)


def plotPixel(x1, y1, x2, y2, dx, dy, decide):
    # pk is initial decision making parameter
    # Note:x1&y1,x2&y2, dx&dy values are interchanged
    # and passed in plotPixel function so
    # it can handle both cases when m>1 & m<1
    pk = 2 * dy - dx
    points = []

    # for (int i = 0; i <= dx; i++) {

    # if decide:
    #     print(dx, y1, x1, y2, x2)
    # else:
    #     print(dx, x1, y1, x2, y2)
    for i in range(0, dx + 1):
        if decide:
            # print(f"({y1},{x1})", end=" ")
            points.append((y1, x1))

        else:
            # print(f"({x1},{y1})", end=" ")
            points.append((x1, y1))

        # checking either to decrement or increment the
        # value if we have to plot from (0,100) to (100,0)
        if (x1 < x2):
            x1 = x1 + 1
        else:
            x1 = x1 - 1
        if (pk < 0):

            # decision value will decide to plot
            # either  x1 or y1 in x's position
            if (decide == 0):

                # putpixel(x1, y1, RED);
                pk = pk + 2 * dy
            else:

                # (y1,x1) is passed in xt
                # putpixel(y1, x1, YELLOW);
                pk = pk + 2 * dy
        else:
            if (y1 < y2):
                y1 = y1 + 1
            else:
                y1 = y1 - 1

            # if (decide == 0):
            #   # putpixel(x1, y1, RED)
            # else:
            #   #  putpixel(y1, x1, YELLOW);
            pk = pk + 2 * dy - 2 * dx
    return points


# Driver code

def findPoints(x1, y1, x2, y2):
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    # print(dx,dy)
    # If slope is less than one
    if (dx > dy):
        # passing argument as 0 to plot(x,y)
        points = plotPixel(x1, y1, x2, y2, dx, dy, 0)

    # if slope is greater than or equal to 1
    else:
        # passing argument as 1 to plot (y,x)
        points = plotPixel(y1, x1, y2, x2, dy, dx, 1)
    # print(points)
    return points


class Direction:
    UP = 0
    RIGHT = 3
    DOWN = 1
    LEFT = 2

    @staticmethod
    def toString(dir):
        if dir == 0:
            return "UP"
        elif dir == 3:
            return "RIGHT"
        elif dir == 1:
            return "DOWN"
        elif dir == 2:
            return "LEFT"


def collision_with_apple(score, size, current_state):
    apple_position = [random.randint(1, size - 1), random.randint(1, size - 1)]
    score += 1
    try:
        while current_state[apple_position[0], apple_position[1]]:
            apple_position = [random.randint(1, size - 1), random.randint(1, size - 1)]
    except Exception as e:
        print(apple_position)
        raise e
    return apple_position, score


def collision_with_boundaries(snake_head, size):
    if snake_head[0] >= size or snake_head[0] < 0 or snake_head[1] >= size or snake_head[1] < 0:
        return 1
    else:
        return 0


def collision_with_self(snake_position):
    snake_head = snake_position[0]

    if snake_head in snake_position[1:]:
        # print(snake_position)
        return 1
    else:
        return 0


class SnakeEnv3(gym.Env):

    def __init__(self, size):
        super(SnakeEnv3, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(4)
        self.direction = 3
        self.size = size
        self.max_snake_length = int(pow(self.size, 2))
        # Example for using image as input (channel-first; channel-last also works):
        # self.observation_space = spaces.Box(low=-500, high=500,
        #                                     shape=(5 + SNAKE_LEN_GOAL,), dtype=np.float64)
        self.state = np.zeros(shape=(size, size), dtype='uint8')
        # self.observation_space = spaces.Box(low=0, high=255,
        #                                     shape=(self.size * 10, self.size * 10, 3), dtype=np.uint8)
        self.observation_space = spaces.Box(low=-500, high=500,
                                            shape=(56,), dtype=np.float64)

        self.apple_num = 0
        self.max_index = self.size - 1

    def step(self, action):
        # self.prev_actions.append(action)
        # cv2.imshow('a', self.img)
        # cv2.waitKey(1)
        self.state = np.zeros(shape=(self.size, self.size), dtype='uint8')
        self.img = np.zeros((self.size * 10, self.size * 10, 3), dtype='uint8')
        # Display Apple
        cv2.rectangle(self.img, (self.apple_position[0] * 10, self.apple_position[1] * 10),
                      (self.apple_position[0] * 10 + 10, self.apple_position[1] * 10 + 10), (0, 0, 255), 3)
        self.state[self.apple_position[0], self.apple_position[1]] = 2
        # Display Snake
        for position in self.snake_position:
            cv2.rectangle(self.img, (position[0] * 10, position[1] * 10),
                          (position[0] * 10 + 10, position[1] * 10 + 10), (0, 255, 0), 3)
            self.state[position[0], position[1]] = 1

        # Takes step after fixed time
        # t_end = time.time() + 0.2
        # k = -1
        # action = -1
        # while time.time() < t_end:
        #     if k == -1:
        #         # pass
        #         k = cv2.waitKey(1)
        #         # print(k)
        #
        #         if k == 97:
        #             action = Direction.LEFT
        #         elif k == 100:
        #             action = Direction.RIGHT
        #         elif k == 119:
        #             action = Direction.UP
        #         elif k == 115:
        #             action = Direction.DOWN
        #         # print(k)
        #     else:
        #         continue

        if action == -1:
            action = self.direction

        if action == Direction.RIGHT:
            self.snake_head[0] += 1
            self.direction = Direction.RIGHT
        elif action == Direction.LEFT:
            self.snake_head[0] -= 1
            self.direction = Direction.LEFT
        elif action == Direction.DOWN:
            self.snake_head[1] += 1
            self.direction = Direction.DOWN
        elif action == Direction.UP:
            self.snake_head[1] -= 1
            self.direction = Direction.UP

        apple_reward = 0
        death_reward = 0
        # print(self.snake_head, self.apple_position)
        # Increase Snake length on eating apple
        if self.snake_head == self.apple_position:
            self.apple_position, self.score = collision_with_apple(self.score, self.size, current_state=self.state)
            self.snake_position.insert(0, list(self.snake_head))
            apple_reward = 20
            # apple_reward = 250
            self.apple_num += 1

        else:
            self.snake_position.insert(0, list(self.snake_head))
            self.snake_position.pop()

        # On collision kill the snake and print the score
        if collision_with_boundaries(self.snake_head, self.size) == 1 or collision_with_self(self.snake_position) == 1:
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # self.img = np.zeros((500, 500, 3), dtype='uint8')
            # cv2.putText(self.img, 'Your Score is {}'.format(self.score), (140, 250), font, 1, (255, 255, 255), 2,
            #             cv2.LINE_AA)
            # cv2.imshow('a', self.img)
            if len(self.snake_position) >= self.max_snake_length - 5:
                death_reward = 150
            else:
                # death_reward = -100. / math.pow(len(self.snake_position), 0.8)
                death_reward = -100.
            # print("collide", collision_with_boundaries(self.snake_head, self.size),
            #       collision_with_self(self.snake_position))
            # print(self.snake_position)
            self.done = True

        # euclidean_dist_to_apple = np.linalg.norm(np.array(self.snake_head) - np.array(self.apple_position), ord=1)
        # print(euclidean_dist_to_apple)

        # self.total_reward = ((75 - euclidean_dist_to_apple) + apple_reward + death_reward) / 100
        # print(euclidean_dist_to_apple)
        self.total_reward = apple_reward + death_reward

        # print(self.total_reward)

        # self.reward = self.total_reward - self.prev_reward
        # self.prev_reward = self.total_reward

        # if self.done:
        #     self.reward = -10
        info = {"apple score": self.apple_num}

        # head_x = self.snake_head[0]
        # head_y = self.snake_head[1]
        #
        # snake_length = len(self.snake_position)
        # apple_delta_x = self.apple_position[0] - head_x
        # apple_delta_y = self.apple_position[1] - head_y
        #
        # # create observation:
        #
        # observation = [head_x, head_y, apple_delta_x, apple_delta_y, snake_length] + list(self.prev_actions)
        # observation = np.array(observation)
        observation = self.img
        # print(observation)
        # print(self.total_reward)
        # print(self.state.T)
        if self.snake_position[-2][0] + 1 == self.snake_position[-1][0]:
            self.tail_direction = Direction.LEFT
        elif self.snake_position[-2][0] - 1 == self.snake_position[-1][0]:
            self.tail_direction = Direction.RIGHT
        elif self.snake_position[-2][1] + 1 == self.snake_position[-1][1]:
            self.tail_direction = Direction.UP
        else:
            self.tail_direction = Direction.DOWN
        # print("head:", Direction.toString(self.direction), Direction.toString(self.tail_direction))
        self.observation = self.look()
        # print("obs", observation)
        return self.observation, self.total_reward, self.done, info

    def look_in_direction(self, slope: Slope) -> Tuple[Vision, DrawableVision]:
        dist_to_wall = None
        dist_to_apple = np.inf
        dist_to_self = np.inf

        wall_location = None
        apple_location = None
        self_location = None

        position = self.snake_array[0].copy()
        distance = 1.0
        total_distance = 0.0

        # Can't start by looking at yourself
        position.x += slope.run
        position.y += slope.rise
        total_distance += distance
        body_found = False  # Only need to find the first occurance since it's the closest
        food_found = False  # Although there is only one food, stop looking once you find it

        # Keep going until the position is out of bounds
        while self._within_wall(position):
            if not body_found and self._is_body_location(position):
                dist_to_self = total_distance
                self_location = position.copy()
                body_found = True
                self.intersections.append(self_location)
            if not food_found and self._is_apple_location(position):
                dist_to_apple = total_distance
                apple_location = position.copy()
                self.intersections.append(apple_location)
                food_found = True

            wall_location = position
            position.x += slope.run
            position.y += slope.rise
            total_distance += distance
        assert (total_distance != 0.0)

        # @TODO: May need to adjust numerator in case of VISION_16 since step size isn't always going to be on a tile
        dist_to_wall = 1.0 / total_distance

        if self.apple_and_self_vision == 'binary':
            dist_to_apple = 1.0 if dist_to_apple != np.inf else 0.0
            dist_to_self = 1.0 if dist_to_self != np.inf else 0.0

        elif self.apple_and_self_vision == 'distance':
            dist_to_apple = 1.0 / dist_to_apple
            dist_to_self = 1.0 / dist_to_self

        vision = Vision(dist_to_wall, dist_to_apple, dist_to_self)
        drawable_vision = DrawableVision(wall_location, apple_location, self_location)
        # drawable_vision = None
        self.intersections.append(wall_location)
        return (vision, drawable_vision)

    def look(self):
        x = self.snake_position[0][0]
        y = self.snake_position[0][1]
        dy1 = y * angle_16
        dy1 = 0 if dy1 < 1e-3 else dy1
        dy2 = abs(self.max_index - y) * angle_16
        dy2 = 0 if dy2 < 1e-3 else dy2
        dx1 = x * angle_16
        dx1 = 0 if dx1 < 1e-3 else dx1
        dx2 = abs(self.max_index - x) * angle_16
        dx2 = 0 if dx2 < 1e-3 else dx2
        # print(dy1, dy2, dx1, dx2)

        obs = np.array([
            *self.look_at_direction((x, y), (0, -1)),
            *self.look_at_direction_16((x, y), (x + dy1, 0)),
            *self.look_at_direction((x, y), (1, -1)),
            *self.look_at_direction_16((x, y), (self.max_index, y - dx2)),
            *self.look_at_direction((x, y), (1, 0)),
            *self.look_at_direction_16((x, y), (self.max_index, y + dx2)),
            *self.look_at_direction((x, y), (1, 1)),
            *self.look_at_direction_16((x, y), (x + dy2, self.max_index)),

            *self.look_at_direction((x, y), (0, 1)),
            *self.look_at_direction_16((x, y), (x - dy2, self.max_index)),
            *self.look_at_direction((x, y), (-1, 1)),
            *self.look_at_direction_16((x, y), (0, y + dx1)),

            *self.look_at_direction((x, y), (-1, 0)),
            *self.look_at_direction_16((x, y), (0, y - dx1)),
            *self.look_at_direction((x, y), (-1, -1)),
            *self.look_at_direction_16((x, y), (x - dy1, 0)),
            0., 0., 0., 0., 0., 0., 0., 0.
        ], dtype=np.float64)
        obs[48 + self.direction] = 1
        obs[52 + self.tail_direction] = 1
        # up, upright,right,rightdown, down, downleft,left,leftup
        return obs

    def look_at_direction(self, current_pos, direction: Tuple[int, int]):
        print("d", direction)
        x = current_pos[0] + direction[0]
        y = current_pos[1] + direction[1]
        distance = abs(direction[0]) + abs(direction[1])
        found_food = 0
        found_body = 0
        while not collision_with_boundaries((x, y), self.size):
            if found_body == 0 and self.state[x, y] == 1:
                found_body = 1.
            if found_food == 0 and self.state[x, y] == 2:
                found_food = 1.
            x += direction[0]
            y += direction[1]
            distance += abs(direction[0]) + abs(direction[1])
        # print((x, y), found_body, found_food, distance)
        cv2.line(self.img, (current_pos[0] * 10 + 5, current_pos[1] * 10 + 5), (x * 10 + 5, y * 10 + 5),
                 (255, 100, 100), thickness=1)
        return distance / 2. / self.size, found_body, found_food

    def look_at_direction_16(self, current_pos, new_pos: Tuple[float, float]):
        print(new_pos)
        x = current_pos[0]
        y = current_pos[1]
        x2 = new_pos[0]
        y2 = new_pos[1]
        points = findPoints(x, y, x2, y2)
        found_food = 0
        found_body = 0
        # print(x,y)
        for point in points[1:]:
            if collision_with_boundaries(point, self.size):
                x2 = point[0]
                y2 = point[1]
                break
            x2 = point[0]
            y2 = point[1]
            # print(x2,y2)
            if found_body == 0 and self.state[x2, y2] == 1:
                found_body = 1.
            if found_food == 0 and self.state[x2, y2] == 2:
                found_food = 1.

        distance = abs(x2 - x) + abs(y2 - y)
        # print((x, y), found_body, found_food, distance)
        # print((x * 10 + 5, y * 10 + 5),(x2 * 10 + 5, y2 * 10 + 5))
        cv2.line(self.img, (x * 10 + 5, y * 10 + 5), (x2 * 10 + 5, y2 * 10 + 5),
                 (255, 100, 100), thickness=1)
        return (distance + 1) / 2. / self.size, found_body, found_food

    def reset(self):
        self.img = np.zeros((self.size * 10, self.size * 10, 3), dtype='uint8')
        # Initial Snake and Apple position
        half = int(self.size // 2)
        self.snake_position = [[half, half], [half - 1, half], [half - 2, half]]
        self.apple_position = [random.randint(1, self.size - 1), random.randint(1, self.size - 1)]
        self.score = 0
        self.prev_button_direction = 1
        self.button_direction = 1
        self.snake_head = [half, half]
        # print(self.snake_position)
        self.apple_num = 0
        self.direction = Direction.RIGHT
        self.state = np.zeros(shape=(self.size, self.size), dtype='uint8')
        self.prev_reward = 0

        self.done = False

        # head_x = self.snake_head[0]
        # head_y = self.snake_head[1]
        #
        # snake_length = len(self.snake_position)
        # apple_delta_x = self.apple_position[0] - head_x
        # apple_delta_y = self.apple_position[1] - head_y
        #
        # self.prev_actions = deque(maxlen=SNAKE_LEN_GOAL)  # however long we aspire the snake to be
        #
        # for i in range(SNAKE_LEN_GOAL):
        #     self.prev_actions.append(-1)  # to create history

        # create observation:
        # observation = [head_x, head_y, apple_delta_x, apple_delta_y, snake_length] + list(self.prev_actions)
        # observation = np.array(observation)
        # observation = self.img
        observation = np.zeros((56,), dtype=np.float64)
        return observation

    def render(self, mode='human'):
        if mode == "human":
            # cv2.waitKey(1)
            cv2.imshow('a', self.img)


        else:
            return self.img


from stable_baselines3.common.env_checker import check_env

PRINT_NUM = 10
if __name__ == '__main__':
    env = SnakeEnv3(size=15)
    # It will check your custom environment and output additional warnings if needed
    check_env(env)
    episodes = 20
    rewards_history = []
    avg_reward_history = []
    print(env.observation_space.shape)
    for episode in range(episodes):
        done = False
        obs = env.reset()
        rewards = 0
        while not done:
            random_action = env.action_space.sample()
            # print(f"action: {random_action}")
            obs, reward, done, info = env.step(random_action)
            env.render()
            # A=env.render(mode="rgb_array")
            # print(A.shape)
            print("reward", reward)
            for i in range(0, 16):
                # print(obs[0:3])
                # print()
                print(i, ", {:.4f}, {}, {}".format(*obs[3 * (i):3 * (i + 1)]))
            print(obs[-4:])
            print(obs[-8:-4])
            print()

            # print(obs.shape)
            rewards += reward
        rewards_history.append(rewards)
        avg_reward = np.mean(rewards_history[-100:])
        avg_reward_history.append(avg_reward)
        if episode % PRINT_NUM == 0:
            print(f"games: {episode + 1}, avg_score: {avg_reward}")

    # plt.plot(rewards_history)
    # plt.plot(avg_reward_history)
    plt.show()
