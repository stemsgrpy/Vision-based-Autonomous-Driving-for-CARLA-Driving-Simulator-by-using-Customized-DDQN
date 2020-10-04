# Vision-based-Autonomous-Driving-for-CARLA-Driving-Simulator-by-using-Customized-DDQN

## CARLA Simulator
- The simulation platform provides open digital assets (urban layouts, buildings, vehicles), as shown in Fig1.
- Environment (**Uphill**, **Downhill**, **Street_view**, **Shadow**, **Crossroads**)
- Download [CARLA](http://carla.org/) (CARLA_0.9.5 version)
- Running CARLA
```
  ./CarlaUE4.sh (Linux)
  CarlaUE4.exe (Windows)
```

<p align="center">
  <img width="500" src="/README/carla.jpg">
</p>
<p align="center">
  Figure 1: Urban Layout
</p>

<p align="center">
  <img src="/README/Uphill.JPG" alt="Description" width="320" height="180" border="0" />
  <img src="/README/Downhill.JPG" alt="Description" width="320" height="180" border="0" />
  <img src="/README/Crossroads.JPG" alt="Description" width="320" height="180" border="0" />
  <img src="/README/Shadow.JPG" alt="Description" width="320" height="180" border="0" />
  <img src="/README/Street_view.JPG" alt="Description" width="320" height="180" border="0" />
</p>
<p align="center">
  Figure 2: CARLA Simulation Environment 
</p>

## Image Data
- CARLA Simulator contains different urban layouts and can also generate objects.
  - Urban layout **Town05** is used as experimental site
  - Consist Consecutive Samples
  - Grayscale Image Segmentation
    - **RGB**, **Segmentation**, **Grayscale Segmentation**

<p align="center">
  <img src="/README/RGB0.png" alt="Description" width="120" height="120" border="0" />
  <img src="/README/RGB1.png" alt="Description" width="120" README="120" border="0" />
  <img src="/README/RGB2.png" alt="Description" width="120" README="120" border="0" />
  <img src="/README/RGB3.png" alt="Description" width="120" height="120" border="0" />
  <img src="/README/RGB4.png" alt="Description" width="120" README="120" border="0" />
  <img src="/README/RGB5.png" alt="Description" width="120" height="120" border="0" />
</p>
<p align="center">
  Figure 3: Consecutive Samples of RGB 
</p>

<p align="center">
  <img src="/README/Segmentation0.png" alt="Description" width="120" height="120" border="0" />
  <img src="/README/Segmentation1.png" alt="Description" width="120" README="120" border="0" />
  <img src="/README/Segmentation2.png" alt="Description" width="120" README="120" border="0" />
  <img src="/README/Segmentation3.png" alt="Description" width="120" height="120" border="0" />
  <img src="/README/Segmentation4.png" alt="Description" width="120" README="120" border="0" />
  <img src="/README/Segmentation5.png" alt="Description" width="120" height="120" border="0" />
</p>
<p align="center">
  Figure 4: Consecutive Samples of Segmentation 
</p>

<p align="center">
  <img src="/README/GrayscaleSegmentation0.png" alt="Description" width="120" height="120" border="0" />
  <img src="/README/GrayscaleSegmentation1.png" alt="Description" width="120" README="120" border="0" />
  <img src="/README/GrayscaleSegmentation2.png" alt="Description" width="120" README="120" border="0" />
  <img src="/README/GrayscaleSegmentation3.png" alt="Description" width="120" height="120" border="0" />
  <img src="/README/GrayscaleSegmentation4.png" alt="Description" width="120" README="120" border="0" />
  <img src="/README/GrayscaleSegmentation5.png" alt="Description" width="120" height="120" border="0" />
</p>
<p align="center">
  Figure 5: Consecutive Samples of Grayscale Segmentation 
</p>

## End-to-end (Input to Output)
- State (Input)  
  - Consist Consecutive Samples  
    - **Grayscale Segmentation Images**  
    - **Vehicle Velocity**  
    - **Vehicle Acceleration**  

<p align="center">
  <img src="/README/GrayscaleSegmentation0.png" alt="Description" width="100" height="100" border="0" />
  <img src="/README/GrayscaleSegmentation1.png" alt="Description" width="100" README="100" border="0" />
  <img src="/README/GrayscaleSegmentation2.png" alt="Description" width="100" README="100" border="0" />
  <img src="/README/GrayscaleSegmentation3.png" alt="Description" width="100" height="100" border="0" />
  <img src="/README/GrayscaleSegmentation4.png" alt="Description" width="100" README="100" border="0" />
  <img src="/README/GrayscaleSegmentation5.png" alt="Description" width="100" height="100" border="0" />
  <img src="/README/Image_npv.jpg" alt="Description" width="100" height="100" border="0" />
  <img src="/README/Image_npa.jpg" alt="Description" width="100" height="100" border="0" />
</p>
<p align="center">
  Figure 6: Consecutive Samples of CARLA Simulator 
</p>

- Action (Output)  
  - **Discrete** (Select one action)  
    - Simple Action
    - **Complex Action**  
```
    # Simple Action (action number: 3)
    action_test = action -1
    self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=action_test*self.STEER_AMT))
```   

```
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
```

- Agent
  - ResDDQNAgent  
    - **ResNet** (Linear-24 add 2 inputs, Velocity and Acceleration)  
    - **DDQN**  

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 96, 96]           1,728
       BatchNorm2d-2           [-1, 32, 96, 96]              64
              ReLU-3           [-1, 32, 96, 96]               0
            Conv2d-4           [-1, 64, 48, 48]          18,432
       BatchNorm2d-5           [-1, 64, 48, 48]             128
            Conv2d-6           [-1, 64, 48, 48]           2,048
       BatchNorm2d-7           [-1, 64, 48, 48]             128
     ResidualBlock-8           [-1, 64, 48, 48]               0
            Conv2d-9          [-1, 128, 24, 24]          73,728
      BatchNorm2d-10          [-1, 128, 24, 24]             256
           Conv2d-11          [-1, 128, 24, 24]           8,192
      BatchNorm2d-12          [-1, 128, 24, 24]             256
    ResidualBlock-13          [-1, 128, 24, 24]               0
           Conv2d-14          [-1, 256, 12, 12]         294,912
      BatchNorm2d-15          [-1, 256, 12, 12]             512
           Conv2d-16          [-1, 256, 12, 12]          32,768
      BatchNorm2d-17          [-1, 256, 12, 12]             512
    ResidualBlock-18          [-1, 256, 12, 12]               0
           Conv2d-19            [-1, 512, 6, 6]       1,179,648
      BatchNorm2d-20            [-1, 512, 6, 6]           1,024
           Conv2d-21            [-1, 512, 6, 6]         131,072
      BatchNorm2d-22            [-1, 512, 6, 6]           1,024
    ResidualBlock-23            [-1, 512, 6, 6]               0
           Linear-24                  [-1, 512]         263,680
             ReLU-25                  [-1, 512]               0
           Linear-26                    [-1, 9]           4,617
================================================================
Total params: 2,014,729
Trainable params: 2,014,729
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.28
Forward/backward pass size (MB): 17.30
Params size (MB): 7.69
Estimated Total Size (MB): 25.27
----------------------------------------------------------------
```

- Reward
  - Each Reward  
    - **Vehicle Live**  
    - **Vehicle Collision**  
  - Recalculate Reward  
    - **Moving Length**  
    - **Moving Velocity**  

```
    # self.env.step(action)
    done = False
    if len(self.collision_hist) != 0:
        done = True
        reward = -200
    else:
        reward = 1

    # Recalculate reward
    ll = int(math.sqrt((next_.x - current_.x)**2 + (next_.y - current_.y)**2 + (next_.z - current_.z)**2))
    vv = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
    aa = int(math.sqrt(a.x**2 + a.y**2 + a.z**2))
    if not done: reward += 2 * (ll - 2) + 3 if vv > 40 else -10
```

## Reinforcement Learning Customized DDQN (ADS_RL)
### Train
```
python ADS_RL.py --train
```

### Test
```
python ADS_RL.py --test --model_path out/Autonomous_Driving-runx/model_xxxx.pkl
```

### Retrain
```
python ADS_RL.py --retrain --model_path out/Autonomous_Driving-runx/checkpoint_model/checkpoint_fr_xxxxx.tar
```

## Result

<p align="center">
  <img src="/README/1_600.gif" alt="Description" width="320" height="213" border="0" />
  <img src="/README/2_600.gif" alt="Description" width="320" height="213" border="0" />
  <img src="/README/3_600.gif" alt="Description" width="320" height="213" border="0" />
  <img src="/README/4_600.gif" alt="Description" width="320" height="213" border="0" />
  <img src="/README/5_600.gif" alt="Description" width="320" height="213" border="0" />
  <img src="/README/6_600.gif" alt="Description" width="320" height="213" border="0" />
</p>
<p align="center">
  Figure 7: Reinforcement Learning Customized DDQN on CARLA Simulator
</p>

## Reference
https://github.com/blackredscarf/pytorch-DQN  
[Introduction-Self-driving cars with Carla and Python](https://pythonprogramming.net/introduction-self-driving-autonomous-cars-carla-python/)  
https://github.com/stemsgrpy/Discrete-Control-for-Atari-Game-by-using-DDQN  