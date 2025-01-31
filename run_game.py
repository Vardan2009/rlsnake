from env import Snake, SCREEN_W, SCREEN_H
import pygame

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import logger
import imageio
import numpy as np

import json

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

import optuna

import subprocess, os, platform

from constants import *

pygame.init()

screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
clock = pygame.time.Clock()

font = pygame.font.Font(REG_FONT_PATH, 12)
title_font = pygame.font.Font(TITLE_FONT_PATH, 48)

def human_game():
    env = Snake(screen, clock, font)
    obs, _ = env.reset()

    close = False

    action = 2

    while True:
        env.render();        
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT: close = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w: action = 0
                if event.key == pygame.K_d: action = 1
                if event.key == pygame.K_s: action = 2
                if event.key == pygame.K_a: action = 3
        
        _, reward, done, _, _ = env.step(action)
        
        if done or close: break
        
model = None

def optimize_hyperparameters(trial):
    global timesteps
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-7, 1e-4)
    n_steps = trial.suggest_int('n_steps', 256, 2048) 
    ent_coef = trial.suggest_uniform('ent_coef', 0.005, 0.05)

    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    clock = pygame.time.Clock()
    font = pygame.font.Font(REG_FONT_PATH, 12)
    env = Snake(screen, clock, font)

    model = PPO("MlpPolicy", env, verbose=1, n_steps=n_steps, learning_rate=learning_rate, ent_coef=ent_coef)

    
    model.learn(total_timesteps=100000)

    obs, _ = env.reset()
    total_reward = 0
    done = False
    for i in range(100000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        if done: break
    return total_reward

def start_optimization(timesteps):
    study = optuna.create_study(direction='maximize')
    study.optimize(optimize_hyperparameters, n_trials=20) 
    
    best_trial = study.best_trial
    print("Best trial:", best_trial)
    print(f"Best hyperparameters: {best_trial.params}")
    
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    clock = pygame.time.Clock()
    font = pygame.font.Font(REG_FONT_PATH, 12)
    
    env = Snake(screen, clock, font)
    
    best_model = PPO("MlpPolicy", env, verbose=1, 
                     n_steps=best_trial.params['n_steps'], 
                     learning_rate=best_trial.params['learning_rate'], 
                     ent_coef=best_trial.params['ent_coef'])
    
    with open("hyperparams.json", "w+") as outfile: 
        json.dump(best_trial.params, outfile)
    
    log = logger.configure("logging", ["csv", "stdout"])
    best_model.set_logger(log)
    
        
    checkpoint_callback = CheckpointCallback(save_freq=1000000, save_path='./final_model_logs/',
                                         name_prefix='rlsnake_model_')
    
    best_model.learn(total_timesteps=timesteps, callback=checkpoint_callback)
    
    print("Training done! Recording 1000 frame gif...")
    frames = []
    obs, _ = env.reset()
    
    for _ in range(1000):
        action, _ = best_model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
        screen = env.render()
        frame = pygame.surfarray.array3d(screen)
        frame = frame.swapaxes(0, 1)
        frames.append(frame)
        if done: break
        

    imageio.mimsave("best_game.gif", frames, duration=1/30)
    print("Saving model...")
    best_model.save("last_model")
    print("Completed")

def run_trained_rl():
    if not model:
        show_error("No model loaded/trained!")
        return
    
    env = Snake(screen, clock, font)
    obs, _ = env.reset()

    close = False

    while True:
        env.render();
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT: close = True
            
        action, _ = model.predict(obs, deterministic = True)
        
        obs, reward, done, _, _ = env.step(action)
        
        if done or close: break


error_msg = "test"
error_remain = -1

def show_error(err):
    global error_msg, error_remain
    error_msg = err
    error_remain = 2

def main_menu():
    global model, error_msg, error_remain
    choice = 0
    screen_width, screen_height = 800, 600
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("RLSnake Menu")

    timesteps = 50000000 # 50 000 000

    while True:
        screen.fill(COLOR_GROUND)
        
        # Title
        title_img = title_font.render("RLSnake", True, COLOR_SNAKE)
        screen.blit(title_img, (screen_width // 2 - title_img.get_width() // 2, 20))

        # Menu options with better spacing and highlighting
        options = [
            ("Human", 0),
            ("RL Agent", 1),
            (f"Start RL Training (for {timesteps} timesteps)", 2),
            ("Load last trained model", 3),
            ("Open last trained model playback", 4),
            ("Exit", 5)
        ]

        for i, (text, option) in enumerate(options):
            y_position = (screen_height - (len(options) * 40)) // 2 + i * 40
            if choice == option:
                text_img = font.render(f"→ {text} ←", True, COLOR_SNAKE)
            else:
                text_img = font.render(text, True, COLOR_GROUND_BORDER)
            screen.blit(text_img, (screen_width // 2 - text_img.get_width() // 2, y_position))

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s: 
                    choice = min(choice + 1, len(options) - 1)
                elif event.key == pygame.K_w: 
                    choice = max(choice - 1, 0)
                elif event.key == pygame.K_d:
                    if choice == 0: 
                        human_game()
                    elif choice == 1: 
                        run_trained_rl()
                    elif choice == 2: 
                        start_optimization(timesteps)
                    elif choice == 3: 
                        try:
                            model = PPO.load("last_model")
                        except FileNotFoundError:
                            show_error("No trained model found!")
                    elif choice == 4: 
                        try:
                            os.startfile("best_game.gif")
                        except FileNotFoundError:
                            show_error("No last game record found!")
                    elif choice == 5: 
                        return
                elif event.key == pygame.K_ESCAPE:
                    return

            if event.type == pygame.MOUSEWHEEL and choice == 2:
                timesteps += event.y * 5000

        clock.tick(60)
        
        if error_remain > 0:
            error_img = font.render(f"{error_msg}", True, (255, 0, 0))
            screen.blit(error_img, (20, 20))
            error_remain -= 1 / 60

        pygame.display.flip()

if __name__ == '__main__':
    main_menu()