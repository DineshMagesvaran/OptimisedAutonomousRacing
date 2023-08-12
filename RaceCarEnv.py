import gym
import math
from Graphics import Graphics
import numpy as np
import math

STATE_H = 96
STATE_W = 96

class RaceCarEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        super(RaceCarEnv, self).__init__() # Initialising RaceCarEnv as a child class of Gym
        self.action_space = gym.spaces.Box(np.array([-1, 0, 0]).astype(np.float32), np.array([1, 1, 1]).astype(np.float32)) # steer, gas, brake
        self.action_space.n = 5
        self.observation_space = gym.spaces.Box(low =0, high = 255, shape = (STATE_H, STATE_W, 3), dtype = np.uint8) # x coord, y coord, heading 
        self.initialize_variables()
        self.icr = 0
        self.graphics = Graphics()
        self.hf_thickness = self.graphics.hf_thickness # half of the track thickness
        self.episode_counter = 0
        self.validation = False

    # Attributes in this function are reset everytime reset() is called
    def initialize_variables(self):
        self.car_heading = 0 #heading in world coordinates
        self.car_speed = 0 
        self.pos_x , self.pos_y = 0, 0
        self.acceleration = 0
        self.steering = 0
        self.line_reached = False
        self.dt = 0.1
        self.virtual_wheel_heading = 0 # Virtual wheel heading is the average angle of the left and right front steering wheels
        self.previous_xy = [0, 0]
        self.acceleration_gain = 1 # Gain which adjusts acceleration
        self.steering_gain = 0.1 # Gain which adjusts steering
        self.checkpoint_passed = [False, False, False] # Stores checkpoints that have been passed
        self.cp_reward_collected = [False, False, False] # Stores rewards that have been collected
        self.time_elapsed = 0

    # Updates if checkpoint 1 has been passed
    def update_checkpoint1(self):
        if -self.hf_thickness < self.previous_xy[0] < self.hf_thickness and self.previous_xy[1] < 50:
            if  -self.hf_thickness < self.pos_x < self.hf_thickness and self.pos_y > 50:
                self.checkpoint_passed[0] = True
                print ("Checkpoint 1 Reached!")

    # Updates if checkpoint 2 has been passed
    def update_checkpoint2(self):
        if self.checkpoint_passed[0] == False:
            return None
        if (-40 - self.hf_thickness) < self.previous_xy[0] < (-40 + self.hf_thickness) and self.previous_xy[1] > 50:
            if  (-40 - self.hf_thickness) < self.pos_x < (-40 + self.hf_thickness) and self.pos_y < 50:
                self.checkpoint_passed[1] = True
                print("Checkpoint 2 Reached!")

    # Updates if checkpoint 3 has been passed
    def update_checkpoint3(self):
        if self.checkpoint_passed[1] == False:
            return None
        if (-40 - self.hf_thickness) < self.previous_xy[0] < (-40 + self.hf_thickness) and self.previous_xy[1] > 0:
            if  (-40 - self.hf_thickness) < self.pos_x < (-40 + self.hf_thickness) and self.pos_y < 0:
                self.checkpoint_passed[2] = True
                print("Checkpoint 3 Reached!")

    # Resets environment
    def reset(self):
        self.initialize_variables()
        state_image = self.graphics.reset_graphics()
        self.episode_counter += 1
        if self.validation:
            self.episode_counter = -1
        return state_image

    # Moving the car 1 time step based on given action
    def step(self, action):
        self.time_elapsed += self.dt
        
        # Acceleration is computed from both the brake and acceleration values
        self.acceleration = (action[1] - action[2]) * self.acceleration_gain
        self.steering = action[0] * self.steering_gain
        
        state = self.getNextState()
        reward = self.getReward()
        done = self.isDone()
        info = "Hello World"
        
        car_heading_angle = self.car_heading * 180 / math.pi
        state_image = self.graphics.updateGraphics(self.pos_x, self.pos_y, car_heading_angle, self.episode_counter, 
        self.car_speed, self.time_elapsed)
        return state_image, reward, done, info

    def get_acceleration(self):
        return self.acceleration

    def get_steering(self):
        return self.steering

    # Computes the next state of the car
    def getNextState(self):
        self.previous_xy[0], self.previous_xy[1] = self.pos_x, self.pos_y
        
        # The code block below makes use of the Ackermann steering princple to calculate speed and position
        self.virtual_wheel_heading = self.steering # self.steering values are mapped to virtual wheel heading angle
        self.car_speed += self.get_acceleration() * self.dt
        speed_y = self.car_speed * math.cos(self.car_heading)
        speed_x = -self.car_speed * math.sin(self.car_heading)
        self.pos_y += speed_y * self.dt
        self.pos_x += speed_x * self.dt    
        if not self.virtual_wheel_heading == 0: 
            self.icr = 2 / math.tan(self.virtual_wheel_heading)
            angular_vel = self.car_speed / self.icr
        else: 
            angular_vel = 0
        self.car_heading += angular_vel * self.dt

        return self.pos_x, self.pos_y, self.car_heading, self.car_speed

    # Using information from previous state and current state, reward is calculated
    def getReward(self):
        reward = -0.1
        self.update_checkpoint1()
        self.update_checkpoint2()
        self.update_checkpoint3()
        reward += self.get_progress_as_reward()
        for i in range(len(self.cp_reward_collected)):
            if self.cp_reward_collected[i] == False:
                if self.checkpoint_passed[i] == True:
                    reward += 1000
                    self.cp_reward_collected[i] = True  

        if self.crossed_finish_line():
            reward += 10000
        if self.is_out_of_track():
            reward -= 10000
        if self.stops_moving_forward():
            reward -= 10000
        
        # If the car does not complete the track in 500 seconds, a negative reward is given
        if self.time_elapsed > 500:
            reward -= 10000
        return reward

    # Rewards are given the further it travels along the track
    def get_progress_as_reward(self):
        reward = 0
        straight_reward_scale = 1 # Reward of 1 for every metre advanced along straight segment
        curve_reward_scale = 100 # Reward of 100 for every radian advanced along curved segment
        if  -self.hf_thickness < self.pos_x < self.hf_thickness and 0 <= self.pos_y <= 50: # First straight segment
            reward += (self.pos_y - self.previous_xy[1]) * straight_reward_scale
        elif (-40 - self.hf_thickness) < self.pos_x < (-40 + self.hf_thickness) and 0 <= self.pos_y <= 50: # Second straight segment
            reward += (self.previous_xy[1] - self.pos_y) * straight_reward_scale
        elif self.pos_y > 50: # First curved segment
            previous_angle = math.atan2(self.previous_xy[1] - 50, self.previous_xy[0] - (-20))
            curr_angle = math.atan2(self.pos_y - 50, self.pos_x - (-20))
            reward += abs(curr_angle - previous_angle) * curve_reward_scale
        elif self.pos_y < 0: # Second curved segment
            previous_angle = math.atan2(self.previous_xy[1] - 0, self.previous_xy[0] - (-20))
            curr_angle = math.atan2(self.pos_y - 0, self.pos_x - (-20))
            reward += abs(curr_angle - previous_angle) * curve_reward_scale
        return reward

    def stops_moving_forward(self):
        return self.car_speed < 0

    # Cross check if the finish line is crossed
    def crossed_finish_line(self):
        if self.checkpoint_passed[2] == False:
            return self.line_reached
        if self.previous_xy[0] > -self.hf_thickness and self.previous_xy[0] < self.hf_thickness: 
            if self.pos_x > -self.hf_thickness  and self.pos_x < self.hf_thickness:
                if self.previous_xy[1] < 0 and self.pos_y > 0:
                    self.line_reached = True
        return self.line_reached

    # Check if the car is out of track
    def is_out_of_track(self):
        is_in_first_seg = (-self.hf_thickness < self.pos_x < self.hf_thickness) and 0 <= self.pos_y <= 50 
        is_in_circle_seg = self.checkCircleSeg()
        is_in_third_seg = ((-40 - self.hf_thickness) < self.pos_x < (-40 + self.hf_thickness)) and 0 <= self.pos_y <= 50 
        return not (is_in_first_seg or is_in_circle_seg or is_in_third_seg)

    # Helper function to check if the car is in the curved segments
    def checkCircleSeg(self):
        isWithinRange = False
        if self.pos_y > 50:
            distance = math.sqrt(math.pow(self.pos_x - (-20), 2) + math.pow(self.pos_y - 50, 2))
            isWithinRange = (20 - self.hf_thickness) < distance < (20 + self.hf_thickness)
        
        if self.pos_y < 0:
            distance = math.sqrt(math.pow(self.pos_x - (-20), 2) + math.pow(self.pos_y - 0, 2))
            isWithinRange = (20 - self.hf_thickness) < distance < (20 + self.hf_thickness)
        
        return isWithinRange

    # Check if the episode is terminated
    def isDone(self):
        time_limit_exceeded = self.time_elapsed > 500
        if time_limit_exceeded:
            print ("Episode done: ", "Time limit exceeded")
        if self.is_out_of_track():
            print ("Episode done: ", "Car out of track")
        if self.crossed_finish_line():
            print ("Episode done: ", "CROSSED FINISH LINE!!!")
        if self.stops_moving_forward():
            print("Episode done: ", "Stops moving forward")
        return time_limit_exceeded or self.is_out_of_track() or self.crossed_finish_line() or self.stops_moving_forward()
    
    # Render graphics
    def render(self, mode='human', close=False):
        car_heading_angle = self.car_heading * 180 / math.pi
        imagedata = self.graphics.updateGraphics(self.pos_x, self.pos_y, car_heading_angle, self.episode_counter, 
        self.car_speed, self.time_elapsed)
        return imagedata

    def update_validation(self, validation):
        self.validation = validation