import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import random
import os

import util

from constants import *

# Define new gymnasium environment
class Snake(gym.Env):
    def __init__(self, screen, clock, font):
        super(Snake, self).__init__()
        
        self.board = None
        self.reward = 0

        # Action Space: 4 (Up, Down, Left, Right)        
        self.action_space = spaces.Discrete(4)

        # FOR make_state_exp
        # self.observation_space = spaces.Box(
        #     low=0,
        #     high=max(BOARD_W, BOARD_H),
        #     dtype=np.uint8,
        #     # BOARD_W*BOARD_H + 1
        #     shape=(BOARD_W*BOARD_H + 1, )
        # )
        
        self.observation_space = spaces.Box(
            low=0,
            high=BOARD_W * BOARD_H,
            dtype=np.uint8,
            shape=(9, )
        )
        
        # self.max_dist = util.dist((0, 0), (BOARD_W, BOARD_H))
                
        self.screen = screen
        self.clock = clock
        self.font = font
        
        self.agent_view_area = None
        
        self.coin_idx = 0
        
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.reward = None
        self.board = np.zeros(shape=(BOARD_W, BOARD_H))
        
        self.snake = [(1, 1)]  # Snake's body will be a list of coordinates
        self.snake_length = 1
        
        # Place the snake's head
        self.board[self.snake[0]] = CELL_HEAD
        
        # Initialize coin position property
        self.coin_position = COINS_POS[0]
        
        # Place a coin at a random position
        self.place_coin()
        
        self.coin_idx = 0
        
        self.last_action = (0, 1)
        
        return self.make_state(), {}
    
    def make_state_array(self):
        result = np.array([-1 for _ in range(BOARD_W*BOARD_H + 1)])
        result[0], result[1] = self.last_action
        result[2], result[3] = self.coin_position
        
        idxcounter = 4
        for segment in self.snake:
            result[idxcounter], result[idxcounter + 1] = segment
            idxcounter += 2
        
        return result
    
    def make_state_exp(self):
        # coin pos, head pos, length, collision flag
        result = np.array([-1 for _ in range(9)])
        result[0], result[1] = self.coin_position
        result[2], result[3] = self.snake[0]
        result[4] = len(self.snake)
        self.agent_view_area = (
            self.is_on_collision_course((1, 0)),
            self.is_on_collision_course((0, 1)),
            self.is_on_collision_course((-1, 0)),
            self.is_on_collision_course((0, -1))
        )
        result[5], result[6], result[7], result[8] = self.agent_view_area
        return result
    
    def make_state_simple(self):
        return np.array(self.board)
        
    def make_state(self): return self.make_state_exp()
    
    def is_on_collision_course(self,dir):
        rows, cols = self.board.shape
        x, y = self.snake[0]
        
        dx, dy = dir
        
        nx, ny = x + dx, y + dy
            
        if nx < 0 or nx >= rows or ny < 0 or ny >= cols or self.board[nx, ny] == CELL_SNAKE:
            return True
    
        return False
    
    def place_coin(self):
        self.coin_idx += 1
        # coin_position =  COINS_POS[self.coin_idx % len(COINS_POS)]
        coin_position = (random.randint(0, BOARD_W - 1), random.randint(0, BOARD_H - 1))
        while coin_position in self.snake:  # Ensure the coin doesn't appear on the snake
            self.coin_idx += 1
            # coin_position =  COINS_POS[self.coin_idx % len(COINS_POS)]
            coin_position = (random.randint(0, BOARD_W - 1), random.randint(0, BOARD_H - 1))
        
        # Save the coin position
        self.coin_position = coin_position
        
        self.board[coin_position] = CELL_COIN
        
    def step(self, action):
        delta_pos = (0, 0)
        
        if action == 0:
            delta_pos = (0, -1)  # Up
        elif action == 1:
            delta_pos = (1, 0)   # Right
        elif action == 2:
            delta_pos = (0, 1)   # Down
        elif action == 3:
            delta_pos = (-1, 0)  # Left
        
        collision = False
        
        # Compute new head position
        head_x, head_y = self.snake[0]
        new_head = (head_x + delta_pos[0], head_y + delta_pos[1])
        
        
        # Check for wall collision
        if (new_head[0] < 0 or new_head[1] < 0 or 
            new_head[0] >= BOARD_W or new_head[1] >= BOARD_H):
            collision = True
        
        reward = 0
        
        # Check if the snake eats a coin
        if not collision:
            if self.board[new_head] == CELL_COIN:
                self.board[new_head] = CELL_HEAD  # New head
                self.snake.insert(0, new_head)  # Add to the front of the snake
                self.snake_length += 1
                self.place_coin()  # Place a new coin
                reward = 100
            else:
                # Move the snake
                self.board[self.snake[-1]] = CELL_EMPTY  # Remove the tail
                self.snake.pop()  # Remove the tail from the snake body
                
                if self.board[new_head] == CELL_SNAKE:  # Check if the head collides with the body
                    collision = True
                else:
                    self.board[new_head] = CELL_HEAD  # Place new head
                    self.snake.insert(0, new_head)  # Add the new head at the front
            # reward = (self.max_dist - util.dist(new_head, self.coin_position)) + (self.snake_length - 1) * 15
        else:
            reward = -100
        
        # Update the board state
        for segment in self.snake[1:]:
            self.board[segment] = CELL_SNAKE
            
        # Save the value in the class instance (used for rendering)
        self.reward = reward
        
        if action != 0:
            self.last_action = delta_pos
        
        return self.make_state(), reward, collision, False, {}
    
    def render(self, mode="human"):
        pygame.display.set_caption("RLSnake")
        # Clear the entire screen with white
        self.screen.fill(COLOR_GROUND_BORDER)
        
        # Compute the cell width and height (shape of each cell)
        cell_w = CELL_SZ
        cell_h = CELL_SZ
        
        # Iterate over the board
        for y in range(BOARD_H):
            for x in range(BOARD_W):
                color = COLOR_GROUND
                
                # Determine color from the value of the cell
                if self.board[x, y] == CELL_SNAKE or self.board[x, y] == CELL_HEAD: 
                    color = COLOR_SNAKE
                elif self.board[x, y] == CELL_COIN: 
                    color = COLOR_COIN
                
                # Draw the cell
                pygame.draw.rect(self.screen, color, ((SCREEN_W - cell_w * BOARD_W) // 2 + x * cell_w - 1, (SCREEN_H - cell_h * BOARD_H) // 2 + y * cell_h - 1, cell_w + 1, cell_h + 1))
                if self.board[x, y] != CELL_SNAKE and self.board[x, y] != CELL_HEAD and self.board[x, y] != CELL_COIN:
                    pygame.draw.rect(self.screen, COLOR_GROUND_BORDER, ((SCREEN_W - cell_w * BOARD_W) // 2 + x * cell_w - 1, (SCREEN_H - cell_h * BOARD_H) // 2 + y * cell_h - 1, cell_w + 1, cell_h + 1), 1)

        rew_img = self.font.render('''Step reward: {reward}'''.format(reward=self.reward), True, (255, 255, 255))
        score_img = self.font.render('''Score: {len}'''.format(len=len(self.snake)), True, (255, 255, 255))
        self.screen.blit(score_img, (20, 20))
        self.screen.blit(rew_img, (20, 40))
        
        # right, up, left, down
        
        pygame.draw.rect(self.screen, (self.agent_view_area[3] * 255, 0, 0), (20 + 25, 70, 25, 25))
        pygame.draw.rect(self.screen, (self.agent_view_area[2] * 255, 0, 0), (20     , 70 + 25, 25, 25))
        pygame.draw.rect(self.screen, (self.agent_view_area[0] * 255, 0, 0), (20 + 50, 70 + 25, 25, 25))
        pygame.draw.rect(self.screen, (self.agent_view_area[1] * 255, 0, 0), (20 + 25, 70 + 50, 25, 25))

        # Display using pygame
        pygame.display.flip()
        
        # Set FPS
        self.clock.tick(7)
        
        return self.screen
    
    def close(self):
        pygame.quit()
