# Import Dependencies
import os
import gymnasium as gym
import pygame
from pygame import gfxdraw
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# Load Model
envName = 'CartPole-v1'
env = gym.make(envName, render_mode='human')

# Function to run the model
def runModel(episodes=5):
    for episode in range(1, episodes+1):
        state = env.reset()
        done = False
        truncated = False
        score = 0 
        
        while not done:
            env.render()
            action = env.action_space.sample()
            n_state, reward, done, truncated, info = env.step(action)
            score+=reward
        print('Episode:{} Score:{}'.format(episode, score))
    env.close()

# Run the Model
#runModel(7)

# Set model path
PPO_path = os.path.join('Saved_Models', 'PPO_model1K')

# Train Model
def trainAndSaveModel():
    # Train Model
    env = DummyVecEnv([lambda: env])
    model = PPO('MlpPolicy', env, verbose = 1)
    model.learn(total_timesteps=1000)
    
    # Save Model
    print(PPO_path)
    model.save(PPO_path)

model = PPO.load(PPO_path, env=env)
runModel(7)