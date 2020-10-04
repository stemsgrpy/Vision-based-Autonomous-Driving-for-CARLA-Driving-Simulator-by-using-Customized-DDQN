import os
import math
import scipy
import numpy as np
from config import Config
from core.logger import TensorBoardLogger
from core.util import get_output_folder
import cv2

class Trainer:
    def __init__(self, agent, env, config: Config):
        self.agent = agent
        self.env = env
        self.config = config
        self.SaveImage = True

        if not os.path.exists('./history'):
            os.mkdir('./history')

        # non-Linear epsilon decay
        epsilon_final = self.config.epsilon_min
        epsilon_start = self.config.epsilon
        epsilon_decay = self.config.eps_decay

        self.epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
            -1. * frame_idx / epsilon_decay)

        self.outputdir = get_output_folder(self.config.output, self.config.env)
        self.agent.save_config(self.outputdir)
        self.board_logger = TensorBoardLogger(self.outputdir)

    def train(self, pre_fr=0):
        losses = []
        all_rewards = []
        episode_reward = 0
        ep_num = 0
        is_win = False

        state = self.env.reset() # (360, 480)
        
        current_, v, a = self.env.get_info()

        '''
        # RGB
        state = np.reshape([cv2.resize(state, (self.config.image_size, self.config.image_size)).transpose(2,0,1)], (1, 3, self.config.image_size, self.config.image_size)) #(1, 3, 96, 96)
        history = np.stack((state, state, state, state, state, state), axis=1) # (1, 3*6, 96, 96)
        history = np.reshape([np.concatenate(history)], (1, 18, self.config.image_size, self.config.image_size))    #(1, 18, 96, 96)
        '''

        # Gray
        state = np.reshape([cv2.resize(state, (self.config.image_size, self.config.image_size))], (1, self.config.image_size, self.config.image_size)) # (1, 96, 96)        
        history = np.stack((state, state, state, state, state, state), axis=1) # (1, 6, 96, 96)

        npv = np.ones([1, 1, self.config.image_size, self.config.image_size])*int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))    # (1, 6, 96, 96)
        npa = np.ones([1, 1, self.config.image_size, self.config.image_size])*int(math.sqrt(a.x**2 + a.y**2 + a.z**2))          # (1, 6, 96, 96)
        history_value = np.append(history, npv, axis=1)         # (1, 7, 96, 96)
        history_value = np.append(history_value, npa, axis=1)   # (1, 8, 96, 96)

        if self.SaveImage:
            img = history_value.transpose(0,2,3,1)
            scipy.misc.imsave('history/history0.jpg',img[0][:,:,0])
            scipy.misc.imsave('history/history1.jpg',img[0][:,:,1])
            scipy.misc.imsave('history/history2.jpg',img[0][:,:,2])
            scipy.misc.imsave('history/history3.jpg',img[0][:,:,3])
            scipy.misc.imsave('history/history4.jpg',img[0][:,:,4])
            scipy.misc.imsave('history/history5.jpg',img[0][:,:,5])
            scipy.misc.imsave('history/history6.jpg',img[0][:,:,6]) # npv
            scipy.misc.imsave('history/history7.jpg',img[0][:,:,7]) # npa
        
        for fr in range(pre_fr + 1, self.config.frames + 1):
            # self.env.render()
            epsilon = self.epsilon_by_frame(fr)

            # action = self.agent.act(state, epsilon)
            action = self.agent.act(history_value, epsilon)

            next_state, reward, done, _ = self.env.step(action)
            next_, v, a = self.env.get_info()

            # Recalculate reward
            ll = int(math.sqrt((next_.x - current_.x)**2 + (next_.y - current_.y)**2 + (next_.z - current_.z)**2))
            vv = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
            aa = int(math.sqrt(a.x**2 + a.y**2 + a.z**2))
            if not done: reward += 2 * (ll - 2) + 3 if vv > 40 else -10

            '''
            # Draw path trajectory
            # self.env.draw_waypoint_union(self.env.world.debug, current_, next_)
            # self.env.draw_string(self.env.world.debug, current_, str('%15.0f km/h' % vv) )
            '''

            next_history = np.reshape([cv2.resize(next_state, (self.config.image_size, self.config.image_size))], (1, 1, self.config.image_size, self.config.image_size)) # (1, 1, 96, 96)
            next_history = np.append(next_history, history[:, :5, :, :], axis=1) # (1, 6, 96, 96)
            
            npv = np.ones([1, 1, self.config.image_size, self.config.image_size])*vv    # (1, 6, 96, 96)
            npa = np.ones([1, 1, self.config.image_size, self.config.image_size])*aa    # (1, 6, 96, 96)
            next_history_value = np.append(next_history, npv, axis=1)                   # (1, 7, 96, 96)
            next_history_value = np.append(next_history_value, npa, axis=1)             # (1, 8, 96, 96)

            if self.SaveImage:
                img = next_history_value.transpose(0,2,3,1)
                scipy.misc.imsave('history/history'+str(fr)+'0.jpg',img[0][:,:,0])
                scipy.misc.imsave('history/history'+str(fr)+'1.jpg',img[0][:,:,1])
                scipy.misc.imsave('history/history'+str(fr)+'2.jpg',img[0][:,:,2])
                scipy.misc.imsave('history/history'+str(fr)+'3.jpg',img[0][:,:,3])
                scipy.misc.imsave('history/history'+str(fr)+'4.jpg',img[0][:,:,4])
                scipy.misc.imsave('history/history'+str(fr)+'5.jpg',img[0][:,:,5])
                scipy.misc.imsave('history/history'+str(fr)+'6.jpg',img[0][:,:,6]) # npv
                scipy.misc.imsave('history/history'+str(fr)+'7.jpg',img[0][:,:,7]) # npa          

            # self.agent.buffer.add(state, action, reward, next_state, done)
            self.agent.buffer.add(history_value, action, reward, next_history_value, done)

            current_ = next_
            state = next_state
            history = next_history
            history_value = next_history_value
            episode_reward += reward
            
            loss = 0
            # if self.agent.buffer.size() > self.config.batch_size:
            if self.agent.buffer.size() > self.config.min_buff:
                loss = self.agent.learning(fr)
                losses.append(loss)
                self.board_logger.scalar_summary('Loss per frame', fr, loss)

            if fr % self.config.print_interval == 0:
                print("frames: %5d, reward: %5f, loss: %4f episode: %4d, epsilon: %4f" % (fr, np.mean(all_rewards[-10:]), loss, ep_num, self.epsilon_by_frame(fr)))

            if fr % self.config.log_interval == 0:
                self.board_logger.scalar_summary('Reward per episode', ep_num, all_rewards[-1])

            if self.config.checkpoint and fr % self.config.checkpoint_interval == 0:
                self.agent.save_checkpoint(fr, self.outputdir)

            if done:

                for actor in self.env.actor_list:
                    actor.destroy()
                    # carla.command.DestroyActor(actor)
                self.env.vehicle.destroy()
                # print("All cleaned up!")

                state = self.env.reset()
                current_, v, a = self.env.get_info()

                # Gray
                state = np.reshape([cv2.resize(state, (self.config.image_size, self.config.image_size))], (1, self.config.image_size, self.config.image_size)) # (1, 96, 96)        
                history = np.stack((state, state, state, state, state, state), axis=1) # (1, 6, 96, 96)

                npv = np.ones([1, 1, self.config.image_size, self.config.image_size])*int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))    # (1, 6, 96, 96)
                npa = np.ones([1, 1, self.config.image_size, self.config.image_size])*int(math.sqrt(a.x**2 + a.y**2 + a.z**2))          # (1, 6, 96, 96)
                history_value = np.append(history, npv, axis=1)         # (1, 7, 96, 96)
                history_value = np.append(history_value, npa, axis=1)   # (1, 8, 96, 96)

                all_rewards.append(episode_reward)
                episode_reward = 0
                ep_num += 1
                avg_reward = float(np.mean(all_rewards[-100:]))
                self.board_logger.scalar_summary('Best 100-episodes average reward', ep_num, avg_reward)

                if len(all_rewards) >= 100 and avg_reward >= self.config.win_reward and all_rewards[-1] > self.config.win_reward:
                    is_win = True
                    self.agent.save_model(self.outputdir, 'best')
                    print('Ran %d episodes best 100-episodes average reward is %3f. Solved after %d trials ✔' % (ep_num, avg_reward, ep_num - 100))
                    if self.config.win_break:
                        break

        if not is_win:
            print('Did not solve after %d episodes' % ep_num)
            self.agent.save_model(self.outputdir, 'last')
