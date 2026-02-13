####################################
#      YOU MAY EDIT THIS FILE      #
# ALL OF YOUR CODE SHOULD GO HERE #
####################################

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
            nn.Linear(obs_dim + act_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, obs_dim)
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
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, obs):
        return self.net(obs)

# The Robot class
class Robot:

    def __init__(self):
        # --- INIT PARAMS ---
        self.environment = None # Only available in dev mode
        self.visualisation_lines = []
        
        # --- HYPERPARAMETERS ---
        self.demo_length = 30          # Length of each demo segment
        self.target_demos = 2          # We want 2 chained demos
        self.num_demos_collected = 0
        
        # Increased random exploration since we removed the biased steps and reset cost
        self.random_steps = 10000      
        
        self.training_epochs = 50      
        self.batch_size = 64
        self.lr = 0.001
        
        # MPC / CEM Parameters
        self.planning_horizon = 20
        self.cem_iterations = 5
        self.cem_num_samples = 200
        self.cem_num_elites = 20
        self.plan_duration = 10         # Number of steps to execute before replanning
        
        # --- STATE MANAGEMENT ---
        self.demo_buffer = []
        self.replay_index = 0
        self.steps_explored = 0
        self.state_machine = 'START'   
        # Sequence: START -> REPLAY -> GET_DEMO_2 -> REPLAY -> EXPLORE -> TRAIN
        
        # Exploration helper
        self.current_explore_action = None
        self.explore_action_duration = 0
        
        # Stuck Detection
        self.recent_obs_buffer = collections.deque(maxlen=20)
        
        # Plan Buffer for Testing
        self.planned_actions = []
        self.cached_trajectory_lines = []
        
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
            else:
                # Fallback if no environment: draw vectors at random sampled points
                if hasattr(self, 'dynamics_points') and len(self.dynamics_points) > 0:
                    for obs in self.dynamics_points:
                        s1 = self.get_state_from_obs(obs)
                        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                        
                        for act, col_rgb in actions:
                            act_t = torch.tensor(act, dtype=torch.float32).unsqueeze(0).to(self.device)
                            with torch.no_grad():
                                delta = self.dynamics_model(obs_t, act_t).squeeze(0).cpu().numpy()
                            next_obs = obs + delta
                            s2 = self.get_state_from_obs(next_obs)
                            self.visualisation_lines.append(VisualisationLine(s1[0], s1[1], s2[0], s2[1], col_rgb, 0.004))

    # -------------------------------------------------------------------------
    # HELPERS FOR STUCK DETECTION
    # -------------------------------------------------------------------------
    def _update_stuck_buffer(self, obs):
        self.recent_obs_buffer.append(obs)

    def _is_stuck(self):
        """Returns True if the robot hasn't moved much in the last 20 steps."""
        if len(self.recent_obs_buffer) < 20:
            return False
        
        # Calculate average Euclidean distance between consecutive observations in buffer
        diffs = []
        for i in range(1, len(self.recent_obs_buffer)):
            dist = np.linalg.norm(self.recent_obs_buffer[i] - self.recent_obs_buffer[i-1])
            diffs.append(dist)
        
        avg_movement = np.mean(diffs)
        # Threshold: 0.001 is very small movement in obs space
        return avg_movement < 0.0015

    # -------------------------------------------------------------------------
    # DATA COLLECTION & TRAINING
    # -------------------------------------------------------------------------
    def training_action(self, obs, money):
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
                expert_action = self.demo_buffer[self.replay_index]
                self.replay_index += 1
                noise = 0.0
                action = expert_action + noise
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
                    print("STATE: REPLAY -> EXPLORE")
                    # Immediate transition to Random Exploration
                    self.state_machine = 'EXPLORE'
                    self.steps_explored = 0
                    # Fall through to the EXPLORE logic immediately below
            
        # --- PHASE 3: RANDOM EXPLORATION (Correlated Random Walk) ---
        # Note: We use 'if' here (not elif) so we can fall through from the state change above
        if self.state_machine == 'EXPLORE':
            if self.steps_explored >= self.random_steps:
                print("STATE: EXPLORE -> TRAIN")
                self.state_machine = 'TRAIN'
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
                        mag = constants.MAX_ACTION_MAGNITUDE
                        x = mag * np.cos(angle)
                        y = mag * np.sin(angle)
                        self.current_explore_action = np.array([x, y])
                        self.explore_action_duration = 200 # FORCE LONG DURATION
                    
                    self.explore_action_duration -= 1
                    action_value = self.current_explore_action
                
                else:
                    # Normal Exploration (Mix of Fine and Coarse)
                    # Mix in fine-grained exploration (20% chance)
                    if np.random.random() < 0.20:
                         small_mag = np.random.uniform(0.1, 0.3) * constants.MAX_ACTION_MAGNITUDE
                         angle = np.random.uniform(0, 2 * np.pi)
                         x = small_mag * np.cos(angle)
                         y = small_mag * np.sin(angle)
                         action_value = np.array([x, y])
                    else:
                        # Correlated Random Walk
                        if self.explore_action_duration <= 0:
                            angle = np.random.uniform(0, 2 * np.pi)
                            mag = constants.MAX_ACTION_MAGNITUDE 
                            x = mag * np.cos(angle)
                            y = mag * np.sin(angle)
                            self.current_explore_action = np.array([x, y])
                            self.explore_action_duration = 10 
                        
                        self.explore_action_duration -= 1
                        action_value = self.current_explore_action
                
                action_type = 1

        # --- PHASE 4: TRAIN ---
        if self.state_machine == 'TRAIN':
            print(f"Training on {len(self.memory)} data points...")
            self.train_models()
            print("STATE: TRAIN -> DONE")
            self.state_machine = 'DONE'
            # Clear stuck buffer before testing
            self.recent_obs_buffer.clear()
            action_type = 4
            action_value = 0

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
        self.update_obs_bounds(obs)
        self._update_stuck_buffer(obs)
        
        # 1. EMERGENCY STUCK RECOVERY
        # If we are stuck, we discard any cached plan and force the "Max Magnitude" heuristic immediately
        if self._is_stuck():
            self.planned_actions = [] # Clear buffer
            # Run a single CEM pass to find the best direction, then floor it
            # We reuse the logic below but bypass the plan buffer check
            pass # fall through to CEM
        
        # 2. EXECUTE BUFFERED PLAN (Receding Horizon Control)
        # If we are NOT stuck and have actions in the buffer, execute the next one
        elif len(self.planned_actions) > 0:
            action = self.planned_actions.pop(0)
            
            # Maintain visualisations (since we aren't re-running CEM)
            self.draw_background_visualisations()
            self.visualisation_lines.extend(self.cached_trajectory_lines)
            return action

        # 3. RUN CEM (Replanning)
        self.dynamics_model.eval()
        self.distance_model.eval()
        
        curr_obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device) 
        
        # --- PATH TRACKING COST SETUP ---
        # 1. Find the closest observation in the demo
        best_dist = float('inf')
        closest_idx = 0
        
        for i, demo_obs in enumerate(self.demo_observations):
            dist = np.linalg.norm(obs - demo_obs)
            if dist < best_dist:
                best_dist = dist
                closest_idx = i
                
        # 2. Extract the target sequence of OBSERVATIONS (not actions)
        target_obs_seq = []
        for t in range(self.planning_horizon):
            idx = closest_idx + t
            if idx < len(self.demo_observations):
                target_obs_seq.append(self.demo_observations[idx])
            else:
                target_obs_seq.append(self.demo_observations[-1])
        
        target_obs_tensor = torch.tensor(np.array(target_obs_seq), dtype=torch.float32).to(self.device) # [Horizon, ObsDim]

        # 3. Initialize CEM with Zero Mean (Unbiased)
        action_mean = torch.zeros(self.planning_horizon, constants.ACTION_DIMENSION).to(self.device)
        action_std = torch.ones(self.planning_horizon, constants.ACTION_DIMENSION).to(self.device) * 0.5 * constants.MAX_ACTION_MAGNITUDE

        best_action_seq = None
        mean_trajectories = []

        # Cross Entropy Method Loop
        for i in range(self.cem_iterations):
            noise = torch.randn(self.cem_num_samples, self.planning_horizon, constants.ACTION_DIMENSION).to(self.device)
            samples = action_mean.unsqueeze(0) + action_std.unsqueeze(0) * noise
            samples = torch.clamp(samples, -constants.MAX_ACTION_MAGNITUDE, constants.MAX_ACTION_MAGNITUDE)
            
            sim_obs = curr_obs.repeat(self.cem_num_samples, 1)
            cumulative_cost = torch.zeros(self.cem_num_samples).to(self.device)
            
            for t in range(self.planning_horizon):
                actions_t = samples[:, t, :] 
                delta_pred = self.dynamics_model(sim_obs, actions_t)
                sim_obs = sim_obs + delta_pred
                
                # Cost 1: Distance to Goal (Global Heuristic)
                dist_pred = self.distance_model(sim_obs).squeeze(-1) 
                
                # Cost 2: Path Tracking (Stay on the demo tube)
                target_step = target_obs_tensor[t].unsqueeze(0) # [1, ObsDim]
                tracking_cost = torch.sum((sim_obs - target_step)**2, dim=1) # Sq Euclidean Distance
                
                # Combine costs (High weight on tracking to fix offset drift)
                cumulative_cost += dist_pred + (20.0 * tracking_cost)
                
            _, elite_idxs = torch.topk(cumulative_cost, self.cem_num_elites, largest=False)
            elites = samples[elite_idxs] 
            
            action_mean = elites.mean(dim=0)
            action_std = elites.std(dim=0) + 1e-5
            best_action_seq = action_mean

            # Simulate the mean path for visualisation
            sim_obs_mean = curr_obs.clone()
            path = [sim_obs_mean.squeeze(0).cpu().numpy()]
            for t in range(self.planning_horizon):
                with torch.no_grad():
                    delta = self.dynamics_model(sim_obs_mean, action_mean[t].unsqueeze(0))
                sim_obs_mean = sim_obs_mean + delta
                path.append(sim_obs_mean.squeeze(0).cpu().numpy())
            mean_trajectories.append(path)

        # --- UPDATE VISUALISATIONS (and cache them) ---
        self.cached_trajectory_lines = [] 
        
        for iter_num, path in enumerate(mean_trajectories):
            intensity = (iter_num + 1.0) / self.cem_iterations
            brightness = int(50 + 205 * intensity)
            colour = (brightness, brightness, brightness)
            width = 0.002 + 0.003 * intensity
            
            for t in range(len(path) - 1):
                s1 = self.get_state_from_obs(path[t])
                s2 = self.get_state_from_obs(path[t+1])
                line = VisualisationLine(s1[0], s1[1], s2[0], s2[1], colour, width)
                self.cached_trajectory_lines.append(line)
        
        self.draw_background_visualisations()
        self.visualisation_lines.extend(self.cached_trajectory_lines)

        # --- PROCESS RESULT ---
        # Convert tensor actions to numpy list
        best_actions_np = best_action_seq.detach().cpu().numpy()
        
        # Store the first 'plan_duration' steps into the buffer
        self.planned_actions = list(best_actions_np[:self.plan_duration])

        first_action = self.planned_actions.pop(0)
        
        # HEURISTIC: Check if stuck (Final Override)
        if self._is_stuck():
            print("STUCK (During Testing)")
            norm = np.linalg.norm(first_action)
            if norm > 1e-6:
                # Normalize and scale to Max
                first_action = (first_action / norm) * constants.MAX_ACTION_MAGNITUDE
                
        return first_action