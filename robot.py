####################################
#      YOU MAY EDIT THIS FILE      #
# ALL OF YOUR CODE SHOULD GO HERE #
####################################
##
## add CEM visualisations back in
## check if model learned is optimal
## potentially get second demo path - make it move a certain distance from the start of the other demo ?
## change stuck dynamics during testing - should move until no longer stuck (maybe until predicted velocity is high)
##
# Imports from external libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import collections

# Imports from this project
import config
import constants
from graphics import VisualisationLine

# Define the Neural Networks
class DynamicsModel(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(DynamicsModel, self).__init__()
        # Input: Observation + Action
        # Output: Delta Observation (Next Obs - Current Obs)
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, obs_dim)
        )

    def forward(self, obs, act):
        # We scale action up slightly to match magnitude of observations for better training
        x = torch.cat([obs, act * 10.0], dim=-1)
        delta = self.net(x)
        return delta

class DistanceModel(nn.Module):
    def __init__(self, obs_dim):
        super(DistanceModel, self).__init__()
        # Input: Observation
        # Output: Predicted Distance to Goal
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, obs):
        return self.net(obs)

# The Robot class
class Robot:

    def __init__(self):
        # --- INIT PARAMS ---
        self.environment = None # Only available in dev mode
        self.visualisation_lines = []
        
        # --- DEVELOPMENT FLAGS ---
        # Set LOAD_MODEL = True to skip training and load 'robot_model.pth'
        # Set SAVE_MODEL = True to save the model after training finishes
        # Set FORCE_START_POS = True to teleport robot to 'start_pos_coords' (Dev mode only)
        # UPDATED: Set LOAD=False so we can train with the new Tube Exploration logic
        self.LOAD_MODEL = False
        self.SAVE_MODEL = True
        self.FORCE_START_POS = True
        self.start_pos_coords = [0.1, 0.7] # X, Y
        self.model_path = 'robot_model.pth'

        # --- HYPERPARAMETERS ---
        self.demo_length = 30          # Length of each demo segment
        self.target_demos = 2          # We want 2 chained demos
        self.num_demos_collected = 0
        
        # Phase 1: Random Exploration (Learn basics)
        self.random_steps = 3000
        
        # Phase 2: Refinement Exploration (Use trained model to generate better data)
        self.refinement_steps = 3000
        self.refinement_direction = 1 # 1 = Forward, -1 = Backward
        
        self.training_epochs = 100      
        self.batch_size = 64
        self.lr = 0.001
        
        # MPC / CEM Parameters
        self.planning_horizon = 20
        self.cem_iterations = 5
        self.cem_num_samples = 200
        self.cem_num_elites = 20
        self.plan_duration = 5 # Receding horizon step count
        
        # --- STATE MANAGEMENT ---
        self.demo_buffer = []
        self.replay_index = 0
        self.steps_explored = 0
        self.state_machine = 'START'   
        # Sequence: START -> REPLAY -> GET_DEMO_2 -> REPLAY -> EXPLORE_RANDOM -> TRAIN_1 -> RESET -> EXPLORE_REFINEMENT -> TRAIN_2 -> DONE
        
        # Exploration helper
        self.current_explore_action = None
        self.explore_action_duration = 0
        
        # Stuck Detection
        self.recent_obs_buffer = collections.deque(maxlen=20)
        
        # Plan Buffer for Testing
        self.planned_actions = []
        self.cached_trajectory_lines = []
        
        # Testing state
        self.has_tested_start = False

        # --- DATA STORAGE ---
        self.memory = collections.deque(maxlen=20000)
        
        # --- VISUALISATION BOUNDARIES ---
        self.obs_min = np.full(constants.OBSERVATION_DIMENSION, np.inf)
        self.obs_max = np.full(constants.OBSERVATION_DIMENSION, -np.inf)
        self.demo_observations = []
        self.demo_actions = [] # Stores ALL expert actions from all demos
        self.dynamics_points = []
        
        self.inv_W = None 
        
        # --- MODELS ---
        self.device = torch.device("cpu") # Rules say NO GPU
        self.dynamics_model = DynamicsModel(constants.OBSERVATION_DIMENSION, constants.ACTION_DIMENSION).to(self.device)
        self.distance_model = DistanceModel(constants.OBSERVATION_DIMENSION).to(self.device)
        
        self.dynamics_opt = optim.Adam(self.dynamics_model.parameters(), lr=self.lr)
        self.distance_opt = optim.Adam(self.distance_model.parameters(), lr=self.lr)

    # -------------------------------------------------------------------------
    # SAVE / LOAD UTILS
    # -------------------------------------------------------------------------
    def save_model(self):
        print(f"Saving model to {self.model_path}...")
        data = {
            'dynamics_state': self.dynamics_model.state_dict(),
            'distance_state': self.distance_model.state_dict(),
            'demo_observations': self.demo_observations,
            'demo_actions': self.demo_actions,
            'obs_min': self.obs_min,
            'obs_max': self.obs_max,
        }
            
        torch.save(data, self.model_path)
        print("Model saved successfully.")

    def load_model(self):
        print(f"Loading model from {self.model_path}...")
        try:
            checkpoint = torch.load(self.model_path, weights_only=False)
            self.dynamics_model.load_state_dict(checkpoint['dynamics_state'])
            self.distance_model.load_state_dict(checkpoint['distance_state'])
            self.demo_observations = checkpoint['demo_observations']
            self.demo_actions = checkpoint['demo_actions']
            self.obs_min = checkpoint['obs_min']
            self.obs_max = checkpoint['obs_max']
            
            print("Model loaded successfully.")
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False

    # -------------------------------------------------------------------------
    # VISUALISATION HELPERS
    # -------------------------------------------------------------------------
    def update_obs_bounds(self, obs):
        """Dynamically track the bounds of the abstract observation space."""
        self.obs_min = np.minimum(self.obs_min, obs)
        self.obs_max = np.maximum(self.obs_max, obs)

    def _fit_inverse_obs_mapping(self):
        """Fits a linear mapping from observation to true state using Least Squares."""
        if self.environment is None:
            return
        
        states = []
        obses = []
        # Sample random states
        for _ in range(200):
            x = np.random.uniform(0, 2)
            y = np.random.uniform(0, 1)
            s = np.array([x, y])
            o = self.environment.observation_function(s)
            states.append(s)
            obses.append(np.append(o, 1.0)) # Add bias term
            
        states = np.array(states)
        obses = np.array(obses)
        
        # Solve for W in (obses @ W = states)
        self.inv_W, _, _, _ = np.linalg.lstsq(obses, states, rcond=None)

    def get_state_from_obs(self, obs):
        """Converts observation to state. Uses exact inverse if env available, else adaptive projection."""
        if self.inv_W is not None:
            obs_with_bias = np.append(obs, 1.0)
            return obs_with_bias @ self.inv_W
        else:
            # Fallback adaptive projection
            range_obs = self.obs_max - self.obs_min + 1e-6
            x = 2.0 * (obs[0] - self.obs_min[0]) / range_obs[0]
            y = 1.0 * (obs[1] - self.obs_min[1]) / range_obs[1]
            return np.array([x, y])

    def draw_background_visualisations(self):
        """Draw the grid, demonstration path, and learned vector field."""
        self.visualisation_lines = []
        
        # Lazy init exact inverse mapping
        if self.inv_W is None and self.environment is not None:
            self._fit_inverse_obs_mapping()
        
        # 1. Gridlines
        colour = (50, 50, 50) 
        width = 0.002
        for row in range(11):
            y = row * 0.1
            self.visualisation_lines.append(VisualisationLine(0.0, y, 2.0, y, colour, width))
        for col in range(21):
            x = col * 0.1
            self.visualisation_lines.append(VisualisationLine(x, 0.0, x, 1.0, colour, width))

        # 2. Demonstration Path (Teal)
        if len(self.demo_observations) > 1:
            for i in range(len(self.demo_observations) - 1):
                s1 = self.get_state_from_obs(self.demo_observations[i])
                s2 = self.get_state_from_obs(self.demo_observations[i+1])
                self.visualisation_lines.append(VisualisationLine(s1[0], s1[1], s2[0], s2[1], (0, 255, 255), 0.005))

        # 3. Dynamics vectors (Only draw if model is trained)
        if self.state_machine == 'DONE':
            self.dynamics_model.eval()
            
            actions = [
                (np.array([0.0, 0.04]), (255, 0, 0)),    # Up -> Red
                (np.array([0.04, 0.0]), (0, 200, 0)),    # Right -> Green
                (np.array([0.0, -0.04]), (0, 0, 255)),   # Down -> Blue
                (np.array([-0.04, 0.0]), (200, 200, 0))  # Left -> Yellow
            ]
            
            # If we have access to the environment, plot a beautiful uniform grid
            if self.environment is not None:
                for row in range(10):
                    for col in range(20):
                        x = col * 0.1 + 0.05
                        y = row * 0.1 + 0.05
                        state = np.array([x, y])
                        obs = self.environment.observation_function(state)
                        
                        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                        
                        for act, col_rgb in actions:
                            act_t = torch.tensor(act, dtype=torch.float32).unsqueeze(0).to(self.device)
                            with torch.no_grad():
                                delta = self.dynamics_model(obs_t, act_t).squeeze(0).cpu().numpy()
                            next_obs = obs + delta
                            next_state = self.get_state_from_obs(next_obs)
                            self.visualisation_lines.append(VisualisationLine(x, y, next_state[0], next_state[1], col_rgb, 0.003))

    # -------------------------------------------------------------------------
    # HELPERS FOR STUCK DETECTION
    # -------------------------------------------------------------------------
    def _update_stuck_buffer(self, obs):
        self.recent_obs_buffer.append(obs)

    def _is_stuck(self):
        if len(self.recent_obs_buffer) < 20: return False
        diffs = [np.linalg.norm(self.recent_obs_buffer[i] - self.recent_obs_buffer[i-1]) for i in range(1, len(self.recent_obs_buffer))]
        return np.mean(diffs) < 0.0015

    # -------------------------------------------------------------------------
    # SHARED PLANNER (USED FOR REFINEMENT & TESTING)
    # -------------------------------------------------------------------------
    def _run_cem(self, obs, direction=1):
        self.dynamics_model.eval()
        self.distance_model.eval()
        
        curr_obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 1. Setup Target Tracking
        best_dist = float('inf')
        closest_idx = 0
        if len(self.demo_observations) > 0:
            # Find closest demo point
            for i, demo_obs in enumerate(self.demo_observations):
                d = np.linalg.norm(obs - demo_obs)
                if d < best_dist: best_dist, closest_idx = d, i
            
            target_seq = []
            for t in range(self.planning_horizon):
                if direction == 1:
                    # Looking forward
                    idx = min(closest_idx + t, len(self.demo_observations) - 1)
                else:
                    # Looking backward
                    idx = max(closest_idx - t, 0)
                target_seq.append(self.demo_observations[idx])
            target_tensor = torch.tensor(np.array(target_seq), dtype=torch.float32).to(self.device)
        else:
            target_tensor = None

        # 2. Initialize Mean (Warm Start)
        action_mean = torch.zeros(self.planning_horizon, constants.ACTION_DIMENSION).to(self.device)
        if len(self.demo_actions) > 0:
            for t in range(self.planning_horizon):
                if direction == 1:
                    idx = min(closest_idx + t, len(self.demo_actions) - 1)
                    # Forward action
                    act = self.demo_actions[idx]
                else:
                    idx = max(closest_idx - t, 0)
                    # Reverse action: -1.0 * action
                    act = -1.0 * self.demo_actions[idx]
                
                action_mean[t] = torch.tensor(act, dtype=torch.float32).to(self.device)
        
        action_std = torch.ones(self.planning_horizon, constants.ACTION_DIMENSION).to(self.device) * 0.5 * constants.MAX_ACTION_MAGNITUDE
        best_action_seq = None
        mean_trajectories = []

        # 3. CEM Loop
        for _ in range(self.cem_iterations):
            noise = torch.randn(self.cem_num_samples, self.planning_horizon, constants.ACTION_DIMENSION).to(self.device)
            samples = torch.clamp(action_mean.unsqueeze(0) + action_std.unsqueeze(0) * noise, 
                                  -constants.MAX_ACTION_MAGNITUDE, constants.MAX_ACTION_MAGNITUDE)
            
            sim_obs = curr_obs.repeat(self.cem_num_samples, 1)
            costs = torch.zeros(self.cem_num_samples).to(self.device)
            
            for t in range(self.planning_horizon):
                actions_t = samples[:, t, :]
                delta = self.dynamics_model(sim_obs, actions_t)
                sim_obs = sim_obs + delta
                
                dist_cost = self.distance_model(sim_obs).squeeze(-1)
                vel_rew = -10.0 * torch.norm(delta, dim=1)
                
                track_cost = 0
                if target_tensor is not None:
                    track_cost = 10000.0 * torch.sum((sim_obs - target_tensor[t].unsqueeze(0))**2, dim=1)
                
                # If backward, disable distance cost (or flip it) so we don't fight to get to the goal
                final_dist_weight = 0.05 if direction == 1 else 0.0
                
                costs += (final_dist_weight * dist_cost) + track_cost + vel_rew
            
            elites = samples[torch.topk(costs, self.cem_num_elites, largest=False)[1]]
            action_mean = elites.mean(dim=0)
            action_std = elites.std(dim=0) + 1e-5
            best_action_seq = action_mean

            # Vis trace
            sim_obs_mean = curr_obs.clone()
            path = [sim_obs_mean.squeeze(0).cpu().numpy()]
            for t in range(self.planning_horizon):
                with torch.no_grad():
                    d = self.dynamics_model(sim_obs_mean, action_mean[t].unsqueeze(0))
                sim_obs_mean = sim_obs_mean + d
                path.append(sim_obs_mean.squeeze(0).cpu().numpy())
            mean_trajectories.append(path)

        # 4. Cache Lines for Viz
        self.cached_trajectory_lines = []
        for i, path in enumerate(mean_trajectories):
            brightness = int(50 + 205 * (i+1)/self.cem_iterations)
            for t in range(len(path)-1):
                s1, s2 = self.get_state_from_obs(path[t]), self.get_state_from_obs(path[t+1])
                self.cached_trajectory_lines.append(VisualisationLine(s1[0], s1[1], s2[0], s2[1], (brightness, brightness, brightness), 0.002))
        
        return best_action_seq.detach().cpu().numpy()

    # -------------------------------------------------------------------------
    # DATA COLLECTION & TRAINING
    # -------------------------------------------------------------------------
    def training_action(self, obs, money):
        # --- LOAD MODEL SHORTCUT ---
        # If enabled, this loads the model/env and immediately ends training
        if self.LOAD_MODEL and self.state_machine == 'START':
             if self.load_model():
                 self.state_machine = 'DONE'
                 return 4, 0 # End training immediately
             else:
                 print("Load failed, falling back to training...")
        
        # Update stuck buffer
        self._update_stuck_buffer(obs)

        action_type, action_value = 4, 0
        
        # --- PHASE 1: START -> REQUEST DEMO 1 ---
        if self.state_machine == 'START':
            print("STATE: START -> REPLAY (Requesting Demo 1)")
            self.num_demos_collected = 1
            self.replay_index = 0
            self.demo_buffer = [] 
            self.state_machine = 'REPLAY' 
            self.draw_background_visualisations()
            return 3, self.demo_length 

        # --- PHASE 2: REPLAYING DEMO 1 or 2 ---
        if self.state_machine == 'REPLAY':
            if self.replay_index < len(self.demo_buffer):
                action = self.demo_buffer[self.replay_index]
                self.replay_index += 1
                action_value = np.clip(action, -constants.MAX_ACTION_MAGNITUDE, constants.MAX_ACTION_MAGNITUDE)
                action_type = 1
            else:
                # Replay Finished
                if self.num_demos_collected < self.target_demos:
                    print("STATE: REPLAY -> GET_DEMO_2")
                    # Immediate transition to Request Demo 2
                    self.num_demos_collected += 1
                    self.replay_index = 0
                    self.demo_buffer = []
                    self.state_machine = 'REPLAY' # Will be in REPLAY mode next step
                    return 3, self.demo_length
                else:
                    print("STATE: REPLAY -> EXPLORE_RANDOM")
                    self.state_machine = 'EXPLORE_RANDOM'
                    self.steps_explored = 0
            
        # --- 2. RANDOM EXPLORATION (Learn Basics) ---
        if self.state_machine == 'EXPLORE_RANDOM':
            if self.steps_explored >= self.random_steps:
                print("STATE: EXPLORE_RANDOM -> TRAIN_1")
                self.state_machine = 'TRAIN_1'
            else:
                self.steps_explored += 1
                
                # --- HEURISTIC: STUCK DETECTION ---
                is_stuck = self._is_stuck()
                
                # If stuck, override behavior: Big steps, long duration
                if is_stuck:
                    if self.explore_action_duration <= 0:
                        print("STUCK")
                        # Pick new random direction
                        angle = np.random.uniform(0, 2 * np.pi)
                        self.current_explore_action = np.array([constants.MAX_ACTION_MAGNITUDE * np.cos(angle), constants.MAX_ACTION_MAGNITUDE * np.sin(angle)])
                        self.explore_action_duration = 200 
                    self.explore_action_duration -= 1
                    action_value = self.current_explore_action
                
                else:
                    # Normal Exploration (Mix of Fine and Coarse)
                    # Mix in fine-grained exploration (20% chance)
                    if np.random.random() < 0.20:
                         angle = np.random.uniform(0, 2 * np.pi)
                         mag = np.random.uniform(0.1, 0.3) * constants.MAX_ACTION_MAGNITUDE
                         action_value = np.array([mag * np.cos(angle), mag * np.sin(angle)])
                    else:
                        # Correlated Random Walk
                        if self.explore_action_duration <= 0:
                            angle = np.random.uniform(0, 2 * np.pi)
                            self.current_explore_action = np.array([constants.MAX_ACTION_MAGNITUDE * np.cos(angle), constants.MAX_ACTION_MAGNITUDE * np.sin(angle)])
                            self.explore_action_duration = 10 
                        
                        self.explore_action_duration -= 1
                        action_value = self.current_explore_action
                action_type = 1

        # --- 3. FIRST TRAINING ---
        if self.state_machine == 'TRAIN_1':
            print(f"Training Model 1 on {len(self.memory)} data points...")
            self.train_models()
            print("STATE: TRAIN_1 -> RESET")
            self.state_machine = 'RESET'
            return 2, 0 # Reset for refinement phase

        # --- 4. PREPARE REFINEMENT ---
        if self.state_machine == 'RESET':
            print("STATE: RESET -> EXPLORE_REFINEMENT")
            self.state_machine = 'EXPLORE_REFINEMENT'
            self.steps_explored = 0
            self.planned_actions = [] # Clear planner buffer
            self.recent_obs_buffer.clear() # Clear stuck buffer
            self.refinement_direction = 1 # Start Forward
            return 2, 0 # Ensure we are reset

        # --- 5. REFINEMENT EXPLORATION (DAgger-lite) ---
        if self.state_machine == 'EXPLORE_REFINEMENT':
            if self.steps_explored >= self.refinement_steps:
                print("STATE: EXPLORE_REFINEMENT -> TRAIN_2")
                self.state_machine = 'TRAIN_2'
            else:
                self.steps_explored += 1
                
                # Check where we are on the demo path
                current_idx = 0
                if len(self.demo_observations) > 0:
                    dists = [np.linalg.norm(obs - d) for d in self.demo_observations]
                    current_idx = np.argmin(dists)
                
                # AUTO-REVERSE LOGIC
                if self.refinement_direction == 1:
                    # Going forward: if near end, flip to backward
                    if current_idx >= len(self.demo_observations) - 10:
                        print("Switching to BACKWARD refinement")
                        self.refinement_direction = -1
                        self.planned_actions = [] # Clear old plan
                else:
                    # Going backward: if near start, flip to forward
                    if current_idx <= 5:
                        print("Switching to FORWARD refinement")
                        self.refinement_direction = 1
                        self.planned_actions = []

                if self._is_stuck():
                    self.planned_actions = []
                    # Simple escape heuristic if stuck during refinement
                    if self.explore_action_duration <= 0:
                        print("STUCK during refinement")
                        angle = np.random.uniform(0, 2*np.pi)
                        self.current_explore_action = np.array([constants.MAX_ACTION_MAGNITUDE*np.cos(angle), constants.MAX_ACTION_MAGNITUDE*np.sin(angle)])
                        self.explore_action_duration = 200
                    self.explore_action_duration -= 1
                    action_value = self.current_explore_action
                elif len(self.planned_actions) > 0:
                    action_value = self.planned_actions.pop(0)
                else:
                    # Run CEM with direction
                    best_seq = self._run_cem(obs, direction=self.refinement_direction)
                    self.planned_actions = list(best_seq[:self.plan_duration])
                    action_value = self.planned_actions.pop(0)
                
                action_type = 1

        # --- 6. FINAL TRAINING ---
        if self.state_machine == 'TRAIN_2':
            print(f"Training Model 2 on {len(self.memory)} data points (Refinement Added)...")
            self.train_models()
            print("STATE: TRAIN_2 -> DONE")
            if self.SAVE_MODEL: self.save_model()
            self.state_machine = 'DONE'
            self.recent_obs_buffer.clear()
            action_type, action_value = 4, 0

        self.draw_background_visualisations()
        return action_type, action_value

    def receive_transition(self, obs, action, next_obs, distance_to_goal):
        self.update_obs_bounds(obs)
        self.update_obs_bounds(next_obs)
        self.memory.append((obs, action, next_obs, distance_to_goal))

    def receive_demo(self, demo):
        # demo is list of (obs, action)
        
        # 1. Store for Warm Start Planning AND Biased Exploration
        for obs, act in demo:
            self.update_obs_bounds(obs)
            self.demo_observations.append(obs)
            self.demo_actions.append(act)
            
        # 2. Extract actions for immediate Replay
        self.demo_buffer = [d[1] for d in demo]

    def train_models(self):
        if len(self.memory) < self.batch_size:
            return

        obs_batch, act_batch, next_obs_batch, dist_batch = zip(*self.memory)
        
        obs_t = torch.tensor(np.array(obs_batch), dtype=torch.float32).to(self.device)
        act_t = torch.tensor(np.array(act_batch), dtype=torch.float32).to(self.device)
        next_obs_t = torch.tensor(np.array(next_obs_batch), dtype=torch.float32).to(self.device)
        dist_t = torch.tensor(np.array(dist_batch), dtype=torch.float32).unsqueeze(1).to(self.device)
        
        delta_target = next_obs_t - obs_t
        dataset = torch.utils.data.TensorDataset(obs_t, act_t, delta_target, dist_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.dynamics_model.train()
        self.distance_model.train()

        for epoch in range(self.training_epochs):
            total_dyn_loss = 0
            total_dist_loss = 0
            
            for o, a, d_target, dist_target in loader:
                # Train Dynamics
                pred_delta = self.dynamics_model(o, a)
                dyn_loss = nn.MSELoss()(pred_delta, d_target)
                self.dynamics_opt.zero_grad()
                dyn_loss.backward()
                self.dynamics_opt.step()
                
                # Train Distance
                pred_dist = self.distance_model(o)
                dist_loss = nn.MSELoss()(pred_dist, dist_target)
                self.distance_opt.zero_grad()
                dist_loss.backward()
                self.distance_opt.step()
                
                total_dyn_loss += dyn_loss.item()
                total_dist_loss += dist_loss.item()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: DynLoss={total_dyn_loss:.4f}, DistLoss={total_dist_loss:.4f}")
                
        # Setup static points for vector field visualization
        num_samples = min(40, len(self.memory))
        idxs = np.random.choice(len(self.memory), num_samples, replace=False)
        self.dynamics_points = [self.memory[i][0] for i in idxs]

    # -------------------------------------------------------------------------
    # TESTING PHASE (MPC / CEM)
    # -------------------------------------------------------------------------
    def testing_action(self, obs):
        
        # --- START POS OVERRIDE (Dev Only) ---
        if not self.has_tested_start:
             self.has_tested_start = True
             # If we are forcing start pos, we do it ONCE at the beginning of test
             if self.FORCE_START_POS and self.environment is not None:
                  print(f"Forcing start position to {self.start_pos_coords}")
                  self.environment.state = np.array(self.start_pos_coords)
                  # IMPORTANT: Get fresh obs from new state so planner isn't confused
                  obs = self.environment.observation_function(self.environment.state)

        self.update_obs_bounds(obs)
        self._update_stuck_buffer(obs)
        
        # 1. EMERGENCY STUCK RECOVERY
        if self._is_stuck():
            self.planned_actions = [] # Clear buffer
            pass # fall through to CEM to find best escape direction
        
        # 2. EXECUTE BUFFERED PLAN
        elif len(self.planned_actions) > 0:
            action = self.planned_actions.pop(0)
            self.draw_background_visualisations()
            self.visualisation_lines.extend(self.cached_trajectory_lines)
            return action

        # Run CEM
        best_seq = self._run_cem(obs)
        self.planned_actions = list(best_seq[:self.plan_duration])
        first_action = self.planned_actions.pop(0)
        
        self.draw_background_visualisations()
        self.visualisation_lines.extend(self.cached_trajectory_lines)
        
        # HEURISTIC: Check if stuck (Final Override)
        if self._is_stuck():
            print("STUCK (During Testing)")
            norm = np.linalg.norm(first_action)
            if norm > 1e-6:
                # Normalize and scale to Max
                first_action = (first_action / norm) * constants.MAX_ACTION_MAGNITUDE
                
        return first_action