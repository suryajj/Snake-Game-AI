import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000;
BATCH_SIZE = 1000;
LR = 0.001;

class Agent:
  def __init__(self):
    self.n_games = 0;
    self.epsilon = 0;
    self.gamma = 0.9; #discount rate which determines how much the agent cares about rewards in the future
    self.memory = deque(maxlen=MAX_MEMORY)
    self.model = Linear_QNet(11, 256, 3) #11 values for states, training, and then 3 values for action 
    self.trainer = QTrainer(self.model, lr=LR, gamma = self.gamma)
  def get_state(self, game):
    head = game.snake[0];
    point_r = Point(head.x + 20, head.y)
    point_l = Point(head.x - 20, head.y)
    point_u = Point(head.x, head.y - 20)
    point_d = Point(head.x, head.y + 20)

    dir_l = game.direction == Direction.LEFT
    dir_r = game.direction == Direction.RIGHT
    dir_u = game.direction == Direction.UP
    dir_d = game.direction == Direction.DOWN

    state = [
       (dir_r and game.is_collision(point_r)) or
       (dir_l and game.is_collision(point_l)) or
       (dir_u and game.is_collision(point_u)) or
       (dir_d and game.is_collision(point_d)), 

       (dir_r and game.is_collision(point_d)) or
       (dir_d and game.is_collision(point_l)) or
       (dir_l and game.is_collision(point_u)) or
       (dir_u and game.is_collision(point_r)),

       (dir_r and game.is_collision(point_u)) or
       (dir_u and game.is_collision(point_l)) or
       (dir_l and game.is_collision(point_d)) or
       (dir_d and game.is_collision(point_r)),

       dir_l,
       dir_r,
       dir_u,
       dir_d,

       game.food.x < head.x, #MAYBE CHANGE IT TO game.head.x 
       game.food.x > head.x,
       game.food.y < head.y, #food up
       game.food.y > head.y #food down
    ]
    return np.array(state, dtype=int)

  def remember(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))

  def train_long_memory(self):
    if len(self.memory) > BATCH_SIZE:
      mini_sample = random.sample(self.memory, BATCH_SIZE)
    else:
      mini_sample = self.memory
    
    states, actions, rewards, next_states, dones = zip(*mini_sample);

    self.trainer.train_step(states, actions, rewards, next_states, dones)

    
  def train_short_memory(self, state, action, reward, next_state, done):
    self.trainer.train_step(state, action, reward, next_state, done)

  def get_action(self, state):
    self.epsilon = 80 - self.n_games
    final_move = [0,0,0]
    if random.randint(0, 200) < self.epsilon:
      move = random.randint(0, 2)
      final_move[move] = 1;
    else:
      state0 = torch.tensor(state, dtype=torch.float)
      prediction = self.model(state0)
      move = torch.argmax(prediction).item() # returns the index of the max value
      final_move[move] = 1;

    return final_move;

def train():
    plot_scores = [];
    plot_mean_scores = [];
    total_score = 0
    record = 0;
    agent = Agent() #creates an agent object
    game = SnakeGameAI() #creates a snakegame object
    while True:
      state_old = agent.get_state(game) #gets the old state

      final_move = agent.get_action(state_old) #gets the move by giving 
                                               #'get_action' the current state

      reward, done, score = game.play_step(final_move) #outputs after playing the move

      state_new = agent.get_state(game) #new state after the move

      agent.train_short_memory(state_old, final_move, reward, state_new, done) #train after each move

      agent.remember(state_old, final_move, reward, state_new, done) #adds the parameters to the memory

      if done:
        game.reset()
        agent.n_games += 1;
        agent.train_long_memory() #train long term memory after a game is done

        if score > record:
          record = score; #measure the highest record
          agent.model.save() #saves the model if high score reached
        print('Game:', agent.n_games, 'Score:', score, 'Record:', record)

        plot_scores.append(score)
        total_score += score
        mean_score = total_score/agent.n_games
        plot_mean_scores.append(mean_score)
        plot(plot_scores, plot_mean_scores)
if __name__ == '__main__':
    train();