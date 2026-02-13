###########################
# DO NOT MODIFY THIS FILE #
###########################

# Imports from external libraries
import numpy as np

# Imports from this project
import constants


# The Demonstrator class is the "human" which provides the demonstrations to the robot
class Demonstrator:

    # Initialise a new demonstrator
    def __init__(self, environment):
        # The demonstrator has access to the true environment
        self.environment = environment

    # Generate a demonstration using the cross-entropy method
    def generate_demo(self, init_state, demo_length):
        # Create some placeholders for the data
        sampled_actions = np.zeros([constants.DEMO_CEM_NUM_ITER, constants.DEMO_CEM_NUM_PATHS, demo_length, constants.ACTION_DIMENSION], dtype=np.float32)
        sampled_paths = np.zeros([constants.DEMO_CEM_NUM_ITER, constants.DEMO_CEM_NUM_PATHS, demo_length + 1, 2], dtype=np.float32)
        path_distances = np.zeros([constants.DEMO_CEM_NUM_ITER, constants.DEMO_CEM_NUM_PATHS], dtype=np.float32)
        action_mean = np.zeros([constants.DEMO_CEM_NUM_ITER, demo_length, 2], dtype=np.float32)
        action_std = np.zeros([constants.DEMO_CEM_NUM_ITER, demo_length, 2], dtype=np.float32)
        # Loop over the CEM iterations
        for iter_num in range(constants.DEMO_CEM_NUM_ITER):
            # Loop over all the paths that will be sampled
            for path_num in range(constants.DEMO_CEM_NUM_PATHS):
                # The start of each path is the robot's current state
                sampled_paths[iter_num, path_num, 0] = init_state
                curr_state = init_state
                # Sample actions for each step of the episode
                for step in range(demo_length):
                    # If this is the first iteration, then sample a uniformly random action
                    if iter_num == 0:
                        # If this is the first action in the path, sample a random action
                        if step == 0:
                            # First, compute the angle
                            angle = np.random.uniform(0, 2 * np.pi)
                            # Then, convert this into x and y actions
                            action_x = constants.MAX_ACTION_MAGNITUDE * np.cos(angle)
                            action_y = constants.MAX_ACTION_MAGNITUDE * np.sin(angle)
                            action = np.array([action_x, action_y])
                        # Otherwise, perturb the previous action slightly, to maintain smoothness
                        else:
                            action = action + np.random.uniform(low=-0.4*constants.MAX_ACTION_MAGNITUDE, high=0.4*constants.MAX_ACTION_MAGNITUDE, size=2)
                            action = np.clip(action, -constants.MAX_ACTION_MAGNITUDE, constants.MAX_ACTION_MAGNITUDE)
                    # If this is not the first iteration, then sample an action from the distribution calculated in the previous iteration
                    else:
                        action = np.random.normal(loc=action_mean[iter_num-1, step], scale=action_std[iter_num-1, step])
                        # We need to clip this action because the normal distribution is unbounded
                        action = np.clip(action, a_min=-constants.MAX_ACTION_MAGNITUDE, a_max=constants.MAX_ACTION_MAGNITUDE)
                    # Calculate the next state using the environment dynamics
                    next_state = self.environment.dynamics(curr_state, action)
                    # Populate the placeholders with the action and next state
                    sampled_actions[iter_num, path_num, step] = action
                    sampled_paths[iter_num, path_num, step + 1] = next_state
                    # Update the state in this planning path
                    curr_state = next_state
                # Calculate the distance between the final state and the goal, for this path
                distance = np.abs(2.0 - next_state[0])
                path_distances[iter_num, path_num] = distance
            # Find the elite paths, which we do here by getting the paths with the minimum distance to the goal
            elites = np.argsort(path_distances[iter_num])[:constants.DEMO_CEM_NUM_ELITES]
            # Use the elite paths to update the action distribution
            elite_actions = sampled_actions[iter_num, elites]
            action_mean[iter_num] = np.mean(elite_actions, axis=0)
            action_std[iter_num] = np.std(elite_actions, axis=0)
        # The demonstration is the best path after the iterations
        best_path = np.argmin(path_distances[-1])
        # Now create the observation-action pairs
        demonstration = []
        for step in range(demo_length):
            state = sampled_paths[-1, best_path, step]
            observation = self.environment.observation_function(state)
            action = sampled_actions[-1, best_path, step]
            demonstration.append((observation, action))
        return demonstration
