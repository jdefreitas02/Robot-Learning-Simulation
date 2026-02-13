####################################
#      YOU MAY EDIT THIS FILE      #
# ALL OF YOUR CODE SHOULD GO HERE #
####################################

# Imports from external libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

# Imports from this project
# You should not import any other modules, including config.py
# If you want to create some configuration parameters for your algorithm, keep them within this robot.py file
import config
import constants
from graphics import VisualisationLine

# Configure matplotlib for interactive mode
plt.ion()

# CONFIGURATION PARAMETERS HERE. Add whatever configuration parameters you like here.
# Remember, you will only be submitting this robot.py file, no other files.


# The Robot class (which could be called "Agent") is the "brain" of the robot, and is used to decide what action to execute in the environment
class Robot:

    # Initialise a new robot
    def __init__(self):
        # The environment (only available during development mode)
        self.environment = None
        # A list of visualisations which will be displayed on the bottom half of the window
        self.visualisation_lines = []

    # Get the next training action
    def training_action(self, obs, money):
        # Random action
        action_type = 1
        action_value = np.random.uniform(-constants.MAX_ACTION_MAGNITUDE, constants.MAX_ACTION_MAGNITUDE, 2)
        return action_type, action_value

    # Get the next testing action
    def testing_action(self, obs):
        # Random action
        action = np.random.uniform(-constants.MAX_ACTION_MAGNITUDE, constants.MAX_ACTION_MAGNITUDE, 2)
        return action

    # Receive a transition
    def receive_transition(self, obs, action, next_obs, distance_to_goal):
        pass

    # Receive a new demonstration
    def receive_demo(self, demo):
        pass
