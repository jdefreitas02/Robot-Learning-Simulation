###########################
# DO NOT MODIFY THIS FILE #
###########################

# Imports from external libraries
import time
import numpy as np
import pygame
import torch

# Imports from this project
import config
import constants
from environment import Environment
from robot import Robot
from graphics import Graphics
from demonstrator import Demonstrator


# Set the random seed
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)
# Initialize Pygame
pygame.init()
# Create an environment (the "physical" world)
environment = Environment()
# Create a robot (the robot's "brain" making the decisions) and reset the episode
robot = Robot()
# Create a graphics object (this will create a window and draw on the window)
graphics = Graphics(environment)
# Create a demonstrator object (this is the "human" that provides demonstrations)
demonstrator = Demonstrator(environment)

# If we are in development mode, then we give the robot access to the environment.
if config.MODE == 'development':
    robot.environment = environment


# MAIN TRAINING LOOP
print('STARTING TRAINING')
# Initialise the costs
time_cost = 0
step_cost = 0
reset_cost = 0
demo_cost = 0
# Initialise the money
money = constants.INIT_MONEY
# Reset the environment and get the observation
observation = environment.random_reset()
# Set the start time
start_time = time.time()
# Keep running the training until all the money has run out
training_running = True
while training_running:
    # Check for any user input
    for event in pygame.event.get():
        # Closing the window
        if event.type == pygame.QUIT:
            training_running = False
    # Get the action
    action_type, action_value = robot.training_action(observation, money)
    # If the action is a 'regular' action, take a step in the environment with the requested action (defined by action_value)
    if action_type == 1:
        cost = constants.COST_PER_STEP
        if money >= cost:
            action = action_value
            next_observation, distance_to_goal = environment.step(action)
            robot.receive_transition(observation, action, next_observation, distance_to_goal)
            observation = next_observation
            step_cost += cost
        else:
            print('You cannot execute an action, there is not enough money.')
    # If the action is a 'reset' action, then reset the environment to a random initial state
    elif action_type == 2:
        cost = constants.COST_PER_RESET
        if money >= cost:
            observation = environment.random_reset()
            reset_cost += cost
        else:
            print('You cannot reset the environment, there is not enough money.')
    # If the action is a 'demo' action, then get a demonstration from where the robot currently is, and for the requested length (defined within action_value)
    elif action_type == 3:
        demo_state = environment.state
        demo_length = action_value
        cost = constants.COST_PER_DEMO + demo_length * constants.COST_PER_DEMO_STEP
        if money >= cost:
            demo = demonstrator.generate_demo(demo_state, demo_length)
            robot.receive_demo(demo)
            demo_cost += cost
        else:
            print('You cannot get a demonstration, there is not enough money.')
    # If the action is a 'finish' action, then finish training
    elif action_type == 4:
        print('ENDING TRAINING')
        training_running = False
        break
    # If the action is any other action, do nothing
    else:
        print(f'Action type {action_type} is invalid.')
    # Draw the environment, and any visualisations, on the window, and step the timer
    graphics.draw(environment, robot.visualisation_lines)
    # Calculate the money spent on time
    time_cost = constants.COST_PER_SECOND * (time.time() - start_time)
    # Calculate the total cost
    total_cost = time_cost + step_cost + reset_cost + demo_cost
    # Calculate the money left
    money = constants.INIT_MONEY - total_cost
    # Check if the money is negative
    if money < 0:
        print('Money is negative, you will receive a penalty.')
        print('ENDING TRAINING')
        training_running = False
        break
#print(f"{time_cost=}, {step_cost=}, {reset_cost=}, {demo_cost=}, {total_cost=}, {money=}")
# MAIN TESTING LOOP
print('STARTING TESTING')
testing_running = True
init_time = time.time()
# Reset the environment to a random initial state and get the observation
observation = environment.random_reset()
while testing_running:
    # Check for any user input
    for event in pygame.event.get():
        # Closing the window
        if event.type == pygame.QUIT:
            testing_running = False
    # Get the action
    action = robot.testing_action(observation)
    # Execute the action
    next_observation, distance_to_goal = environment.step(action)
    # Calculate how long has elapsed so far
    time_elapsed = time.time() - init_time
    # Check to see if the robot has reached the finish
    if environment.state[0] >= constants.GOAL_LINE_X:
        print(f'The robot reached the goal! Time elapsed = {time_elapsed}')
        print('ENDING TESTING')
        testing_running = False
        break
    # Check to see if the time has run out
    elif time_elapsed >= constants.MAX_TEST_TIME:
        print(f'The robot run out of time! Distance to goal = {distance_to_goal}')
        print('ENDING TESTING')
        testing_running = False
        break
    # Draw the environment, and any visualisations, on the window, and step the timer
    graphics.draw(environment, robot.visualisation_lines)
    # Update the observation
    observation = next_observation

# If we have reached this point, quit pygame and end the program. Goodbye!
pygame.quit()
