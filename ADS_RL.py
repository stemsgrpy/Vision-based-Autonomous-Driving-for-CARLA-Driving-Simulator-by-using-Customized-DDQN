import argparse
import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math
import torch
from torch.optim import Adam
from core.util import get_class_attr_val
from config import Config
from buffer import ReplayBuffer
from model import Create_DQN, Create_ResDQN
from trainer import Trainer
from tester import Tester
from torchsummary import summary

try:
    sys.path.append(glob.glob('../../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

SHOW_PREVIEW = False
IM_WIDTH = 480
IM_HEIGHT = 360

SECONDS_PER_EPISODE = 100000

'''
red = carla.Color(255, 0, 0)
green = carla.Color(0, 255, 0)
blue = carla.Color(47, 210, 231)
cyan = carla.Color(0, 255, 255)
yellow = carla.Color(255, 255, 0)
orange = carla.Color(255, 162, 0)
white = carla.Color(255, 255, 255)
'''

class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0

    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    actor_list = []

    front_camera = None
    collision_hist = []

    def __init__(self):
        self.client  = carla.Client("localhost", 2000)
        self.client .set_timeout(2.0)
        # self.world = self.client .get_world()
        self.world = self.client.load_world('Town05')
        self.blueprint_library = self.world.get_blueprint_library()

        self.bp = self.blueprint_library.filter("model3")[0]
        print(self.bp)

    def reset(self):
        self.collision_hist = []
        self.actor_list = []

        self.spawn_point = random.choice(self.world.get_map().get_spawn_points())
        print(self.spawn_point)

        self.vehicle = self.world.spawn_actor(self.bp, self.spawn_point)
        self.old_l = self.vehicle.get_location()
        v = self.vehicle.get_velocity()

        # self.cam_bp = self.blueprint_library.find("sensor.camera.rgb")
        self.cam_bp = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        self.cam_bp.set_attribute("image_size_x", f"{self.im_width}")
        self.cam_bp.set_attribute("image_size_y", f"{self.im_height}")
        self.cam_bp.set_attribute("fov", "110")

        spawn_point = carla.Transform(carla.Location(x=2.5, z=2.5), carla.Rotation(pitch=-30))
        self.sensor = self.world.spawn_actor(self.cam_bp, spawn_point, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))
        # self.sensor.listen(lambda image: image.save_to_disk('output/%06d.png' % image.frame_number))

        # self.vehicle.set_autopilot(True)
        # self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0.0))

        time.sleep(4)

        colsensor = self.world.get_blueprint_library().find('sensor.other.collision')
        self.colsensor = self.world.spawn_actor(colsensor, spawn_point, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()

        return self.front_camera

    '''
    def draw_transform(self, debug, trans, col=carla.Color(255, 0, 0), lt=1):
        debug.draw_arrow(
        trans.location, trans.location + trans.get_forward_vector(),
        thickness=0.05, arrow_size=0.1, color=col, life_time=lt)

    def draw_waypoint_union(self, debug, w0, w1, lt=1):
        debug.draw_line(
        w0 + carla.Location(z=0.25),
        w1 + carla.Location(z=0.25),
        thickness=0.1, color=green, life_time=lt, persistent_lines=False)
        debug.draw_point(w1 + carla.Location(z=0.25), 0.1, green, lt, False)

    def draw_string(self, debug, w0, text, lt=1):
        debug.draw_string(w0, text, False, orange, lt)
        time.sleep(0.5)
    '''

    def process_img(self, image):

        image.convert(carla.ColorConverter.CityScapesPalette)

        i = np.array(image.raw_data)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]

        if self.SHOW_CAM:
            cv2.imshow("camera.semantic_segmentation",i3)
            cv2.waitKey(1)

        i3 = cv2.cvtColor(i3, cv2.COLOR_BGR2GRAY)
        self.front_camera = i3

    def collision_data(self, event):
        self.collision_hist.append(event)

    def get_info(self):
        return self.vehicle.get_location(), self.vehicle.get_velocity(), self.vehicle.get_acceleration()

    def step(self, action):

        '''
        # Simple Action (action number: 3)
        action_test = action -1
        self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=action_test*self.STEER_AMT))
        '''

        # Complex Action (action number: 9)
        if action == 0:     # forward
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0, brake=0.0))
        elif action == 1:   # left
            self.vehicle.apply_control(carla.VehicleControl(throttle=0, steer=-1, brake=0.0))
        elif action == 2:   # right
            self.vehicle.apply_control(carla.VehicleControl(throttle=0, steer=1, brake=0.0))
        elif action == 3:   # forward_left
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1.0, brake=0.0))
        elif action == 4:   # forward_right
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1.0, brake=0.0))
        elif action == 5:   # brake
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=-0.0, brake=1.0))
        elif action == 6:   # brake_left
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=-1.0, brake=1.0))
        elif action == 7:   # brake_right
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=1.0, brake=1.0))
        elif action == 8:   # none
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=-0.0, brake=0.0))
        
        '''
        if len(self.collision_hist) != 0:
            done = True
            reward = -200 + (time.time()-self.episode_start)*15/SECONDS_PER_EPISODE
        else:
            done = False
            reward = 1
        '''

        done = False
        if len(self.collision_hist) != 0:
            done = True
            reward = -200
        else:
            reward = 1

        if time.time() > self.episode_start + SECONDS_PER_EPISODE:
            done = True
        
        return self.front_camera, reward, done, None

class ResDDQNAgent:
    def __init__(self, config: Config):
        self.config = config
        self.is_training = True
        # self.buffer = deque(maxlen=self.config.max_buff)
        self.buffer = ReplayBuffer(self.config.max_buff)

        self.model = Create_ResDQN(self.config.state_shape, self.config.action_dim)
        self.target_model = Create_ResDQN(self.config.state_shape, self.config.action_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        self.model_optim = Adam(self.model.parameters(), lr=self.config.learning_rate)

        if self.config.use_cuda:
            self.cuda()

    def act(self, state, epsilon=None):
        if epsilon is None: epsilon = self.config.epsilon_min
        if random.random() > epsilon or not self.is_training: # np.random.random()
            # state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            state = torch.tensor(state, dtype=torch.float) # (1, 6, 96, 96)
            if self.config.use_cuda:
                state = state.cuda()
            q_value = self.model(state)
            action = q_value.max(1)[1].item() #action = np.argmax(qs[0])
        else:
            # action = np.random.randint(0, env.action_space_size)
            action = random.randrange(self.config.action_dim)
        return action

    def learning(self, fr):
        s0, a, r, s1, done = self.buffer.sample(self.config.batch_size)

        s0 = torch.tensor(s0, dtype=torch.float)
        s1 = torch.tensor(s1, dtype=torch.float)
        a = torch.tensor(a, dtype=torch.long)
        r = torch.tensor(r, dtype=torch.float)
        done = torch.tensor(done, dtype=torch.float)

        if self.config.use_cuda:
            s0 = s0.cuda()
            s1 = s1.cuda()
            a = a.cuda()
            r = r.cuda()
            done = done.cuda()

        #print(s0.shape, s0.is_contiguous())
        #exit()

        q_values = self.model(s0).cuda()
        next_q_values = self.model(s1).cuda()
        next_q_state_values = self.target_model(s1).cuda()

        q_value = q_values.gather(1, a.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_state_values.gather(1, next_q_values.max(1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = r + self.config.gamma * next_q_value * (1 - done)
        # Notice that detach the expected_q_value
        loss = (q_value - expected_q_value.detach()).pow(2).mean()

        self.model_optim.zero_grad()
        loss.backward()
        self.model_optim.step()

        if fr % self.config.update_tar_interval == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        return loss.item()

    def cuda(self):
        self.model.cuda()
        self.target_model.cuda()

    def load_weights(self, model_path):
        model = torch.load(model_path)
        if 'model' in model:
            self.model.load_state_dict(model['model'])
        else:
            self.model.load_state_dict(model)

    def save_model(self, output, name=''):
        torch.save(self.model.state_dict(), '%s/model_%s.pkl' % (output, name))

    def save_config(self, output):
        with open(output + '/config.txt', 'w') as f:
            attr_val = get_class_attr_val(self.config)
            for k, v in attr_val.items():
                f.write(str(k) + " = " + str(v) + "\n")

    def save_checkpoint(self, fr, output):
        checkpath = output + '/checkpoint_model'
        os.makedirs(checkpath, exist_ok=True)
        torch.save({
            'frames': fr,
            'model': self.model.state_dict()
        }, '%s/checkpoint_fr_%d.tar'% (checkpath, fr))

    def load_checkpoint(self, model_path):
        checkpoint = torch.load(model_path)
        fr = checkpoint['frames']
        self.model.load_state_dict(checkpoint['model'])
        self.target_model.load_state_dict(checkpoint['model'])
        return fr

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train', dest='train', action='store_true', help='train model')
    parser.add_argument('--test', dest='test', action='store_true', help='test model')
    parser.add_argument('--retrain', dest='retrain', action='store_true', help='retrain model')
    parser.add_argument('--model_path', type=str, help='if test or retrain, import the model')
    args = parser.parse_args()

    '''
    dqn = Create_ResDQN(6, 9).cuda()
    summary(dqn, (8, 96, 96))
    exit()
    '''

    config = Config()
    config.env = 'Autonomous_Driving'

    config.gamma = 0.99
    config.epsilon = 1
    config.epsilon_min = 0.1
    config.eps_decay = 600000
    config.frames = 4000000
    config.use_cuda = True
    config.learning_rate = 0.001
    config.max_buff = 20000
    config.min_buff = 2000
    config.update_tar_interval = 100
    config.batch_size = 32
    config.print_interval = 5000
    config.log_interval = 5000
    config.checkpoint = True
    config.checkpoint_interval = 5000
    config.win_reward = 10
    config.win_break = False

    config.state_shape = 6  # 3*6 -> RGB
    config.action_dim = 9   # # forward, left, right, forward_left, forward_right, brake, brake_left, brake_right, none
    config.image_size = 96

    # if not os.path.isdir('models'):
    #     os.makedirs('models')

    env = CarEnv()
    agent = ResDDQNAgent(config)

    if args.train:
        trainer = Trainer(agent, env, config)
        trainer.train()

    elif args.test:
        if args.model_path is None:
            print('please add the model path:', '--model_path xxxx')
            exit(0)
        tester = Tester(agent, env, config, args.model_path)
        tester.test()

    elif args.retrain:
        fr = agent.load_checkpoint(args.model_path)
        trainer = Trainer(agent, env, config)
        trainer.train(fr)

    '''
    for _ in range(1,2):
        print(_)
        
        env.reset()

        for actor in env.actor_list:
            actor.destroy()
            #carla.command.DestroyActor(actor)
        env.vehicle.destroy()
        print("All cleaned up!")
    '''


