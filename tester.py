import numpy as np
import math
import cv2

class Tester(object):

    def __init__(self, agent, env, config, model_path, num_episodes=100000, max_ep_steps=400, test_ep_steps=100000):
        self.num_episodes = num_episodes
        self.max_ep_steps = max_ep_steps
        self.test_ep_steps = test_ep_steps
        self.agent = agent
        self.env = env
        self.config = config
        
        self.agent.is_training = False
        self.agent.load_weights(model_path)
        # self.agent.load_checkpoint(model_path)
        self.policy = lambda x: agent.act(x)

    def test(self, debug=False, visualize=True):
        avg_reward = 0

        while True: # for episode in range(self.num_episodes):

            state = self.env.reset()
            current_, v, a = self.env.get_info()

            # Gray
            state = np.reshape([cv2.resize(state, (self.config.image_size, self.config.image_size))], (1, self.config.image_size, self.config.image_size)) # (1, 96, 96)        
            history = np.stack((state, state, state, state, state, state), axis=1) # (1, 6, 96, 96)

            npv = np.ones([1, 1, self.config.image_size, self.config.image_size])*int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))    # (1, 6, 96, 96)
            npa = np.ones([1, 1, self.config.image_size, self.config.image_size])*int(math.sqrt(a.x**2 + a.y**2 + a.z**2))          # (1, 6, 96, 96)
            history_value = np.append(history, npv, axis=1)         # (1, 7, 96, 96)
            history_value = np.append(history_value, npa, axis=1)   # (1, 8, 96, 96)

            episode_steps = 0
            episode_reward = 0.

            done = False
            while not done:

                action = self.policy(history_value)
                next_state, reward, done, info = self.env.step(action)
                next_, v, a = self.env.get_info()

                ll = int(math.sqrt((next_.x - current_.x)**2 + (next_.y - current_.y)**2 + (next_.z - current_.z)**2))
                vv = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
                aa = int(math.sqrt(a.x**2 + a.y**2 + a.z**2))
                if not done: reward += 2 * (ll - 2) + 3 if vv > 40 else -10

                # self.env.draw_waypoint_union(self.env.world.debug, current_, next_)
                # self.env.draw_string(self.env.world.debug, current_, str('%15.0f km/h' % vv) )
                # print(ll, vv, aa)

                next_history = np.reshape([cv2.resize(next_state, (self.config.image_size, self.config.image_size))], (1, 1, self.config.image_size, self.config.image_size)) # (1, 1, 96, 96)
                next_history = np.append(next_history, history[:, :5, :, :], axis=1) # (1, 6, 96, 96)

                npv = np.ones([1, 1, self.config.image_size, self.config.image_size])*vv    # (1, 6, 96, 96)
                npa = np.ones([1, 1, self.config.image_size, self.config.image_size])*aa    # (1, 6, 96, 96)
                next_history_value = np.append(next_history, npv, axis=1)                   # (1, 7, 96, 96)
                next_history_value = np.append(next_history_value, npa, axis=1)             # (1, 8, 96, 96)        
                
                current_ = next_
                state = next_state
                history = next_history
                history_value = next_history_value
                episode_reward += reward

                episode_steps += 1
                if done:
                    for actor in self.env.actor_list:
                        actor.destroy()
                        # carla.command.DestroyActor(actor)
                    self.env.vehicle.destroy()

            if debug:
                print('[Test] episode: %3d, episode_reward: %5f' % (episode, episode_reward))

            avg_reward += episode_reward
        avg_reward /= self.num_episodes
        print("avg reward: %5f" % (avg_reward))




