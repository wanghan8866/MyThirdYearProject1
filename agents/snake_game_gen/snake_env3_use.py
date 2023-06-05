from path_finding import Mixed
from settings import settings
from snake_env3 import *

PRINT_NUM = 1
if __name__ == '__main__':
    displaying = True
    using_path_finding = False
    path_correctness = []
    p = Pattern(np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 2, 2, 2, 0, 0, 0, 0],
        [0, 0, 3, 0, 0, 2, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
        [2, 1, 0, 0, 4, 2, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 2, 0, 0, 0, 0],
        [2, 2, 2, 2, 2, 2, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
    
    
    
    
    
    
    
    X = np.array(
        [200, 400, 600, 800, 1000, 1200, 1400, 1450, 1500, 1550, 1575, 1600, 1625, 1650, 1675, 1690, 1695, 1699])
    X = [400]
    y_scores = []
    y_steps = []
    for name in X:
        env = load_snake("models/test_64", f"snake_{name}", settings)
        
        
        
        episodes = 100
        rewards_history = []
        avg_reward_history = []
        path_history = []
        

        for episode in range(episodes):
            path_correctness.append([])
            done = False
            obs = env.reset()
            rewards = 0
            while not done:
                action = env.possible_directions[env.action_space.sample()]
                action = -1
                path = None

                path = Mixed(env, env.apple_location).run_mixed()
                
                if path is None:
                    action = -1
                    
                    
                elif path[1] is None:
                    action = -1
                else:
                    
                    if using_path_finding:
                        result = path[1] - env.snake_array[0]
                        old_action = action
                        if result == Point(0, 1):
                            action = "d"
                        elif result == Point(0, -1):
                            action = "u"
                        elif result == Point(1, 0):
                            action = "r"
                        else:
                            action = "l"
                        if old_action == action:
                            path_correctness[episode].append(1)
                        else:
                            path_correctness[episode].append(0)
                    else:
                        action = -1
                    

                if displaying:
                    t_end = time.time() + 0.1
                    k = -1
                    
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

                    env.render(drawing_vision=False, path=path)

                

                
                obs, reward, done, info = env.step(action)
                

                
                
                
                
                
            
            avg_reward = len(env.snake_array)
            avg_reward_history.append(avg_reward)
            if len(env.steps_history) == 0:
                path_history.append(0)
            else:
                path_history.append(np.mean(env.steps_history))
            
            
        print()
        print(f"snake: snake_{name}")
        print(f"games: average reward over {episodes} games: {np.mean(avg_reward_history)}")
        print(f"games: std reward over {episodes} games: {np.std(avg_reward_history)}")
        print(f"games: average steps over {episodes} games: {np.mean(path_history)}")
        print(f"games: std steps over {episodes} games: {np.std(path_history)}")
        y_scores.append(np.mean(avg_reward_history))
        y_steps.append(np.mean(path_history))
    plt.plot(X, y_scores)
    plt.show()
    plt.plot(X, y_steps)
    plt.show()
