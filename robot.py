import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import collections

import config
import constants

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
        # Scale action up slightly to match magnitude of observations for better training
        x = torch.cat([obs, act * 10.0], dim=-1)
        delta = self.net(x) 
        return delta

class Robot:
    def __init__(self):
        # Hyperparameters
        self.demo_length = 30          
        self.max_demos = 6          # Changed: Maximum allowed demos to prevent blowing the budget
        self.goal_threshold = 0.1  # Lowered: We only stop when we are right on the goal line
        self.current_dist = float('inf') # Tracks the latest distance to goal
        self.num_demos_collected = 0
        
        # Phase 1: Random Exploration (Learn basics)
        self.random_steps = 100
        
        # Phase 2: Refinement Exploration (Use trained model to generate better data)
        self.refinement_rounds = 10          
        self.refinement_steps = 500
        self.refinement_direction = 1 # 1 = Forward, -1 = Backward
        
        self.training_epochs = 10      
        self.batch_size = 64
        self.lr = 0.001
        
        # MPC / CEM Parameters
        self.planning_horizon = 2    
        self.cem_iterations = 6
        self.cem_num_samples = 100
        self.cem_num_elites = 10     
        self.plan_duration = 2        
        
        # State Management
        self.demo_buffer = []
        self.replay_index = 0
        self.steps_explored = 0
        self.state_machine = 'START'   
        
        self.current_explore_action = None
        self.explore_action_duration = 0
        
        self.recent_obs_buffer = collections.deque(maxlen=20)
        self.planned_actions = []
        
        # Data Storage
        self.memory = collections.deque(maxlen=20000)
        self.demo_observations = []
        self.demo_actions = [] 
        
        # Models
        self.device = torch.device("cpu") # GPU prohibited
        self.dynamics_model = DynamicsModel(constants.OBSERVATION_DIMENSION, constants.ACTION_DIMENSION).to(self.device)
        self.dynamics_opt = optim.Adam(self.dynamics_model.parameters(), lr=self.lr)

        self.visualisation_lines = []
    # -------------------------------------------------------------------------
    # STUCK DETECTION
    # -------------------------------------------------------------------------
    def _update_stuck_buffer(self, obs):
        self.recent_obs_buffer.append(obs)

    def _is_stuck(self):
        if len(self.recent_obs_buffer) < 20: return False
        diffs = [np.linalg.norm(self.recent_obs_buffer[i] - self.recent_obs_buffer[i-1]) for i in range(1, len(self.recent_obs_buffer))]
        return np.mean(diffs) < 0.0015
            
    def _is_jittering(self):
        current_obs = self.recent_obs_buffer[-1]
        max_spread = max([np.linalg.norm(current_obs - obs) for obs in self.recent_obs_buffer])
        return max_spread < 0.05

    # -------------------------------------------------------------------------
    # SHARED PLANNER (USED FOR REFINEMENT & TESTING)
    # -------------------------------------------------------------------------
    def _run_cem(self, obs, direction=1, recovery=False, test=False):
        self.dynamics_model.eval()
        
        curr_obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 1. Setup Target Tracking
        best_dist = float('inf')
        closest_idx = 0
        if len(self.demo_observations) > 0:
            for i, demo_obs in enumerate(self.demo_observations):
                d = np.linalg.norm(obs - demo_obs)
                if d < best_dist: 
                    best_dist, closest_idx = d, i
            
            target_obs_seq = []
            target_act_seq = []
            
            for t in range(self.planning_horizon):
                if direction == 1:
                    # Looking forward
                    obs_idx = min(closest_idx + t + 1, len(self.demo_observations) - 1)
                    act_idx = min(closest_idx + t, len(self.demo_actions) - 1)
                    target_obs_seq.append(self.demo_observations[obs_idx])
                    target_act_seq.append(self.demo_actions[act_idx])
                else:
                    # Looking backward
                    obs_idx = max(closest_idx - t - 1, 0)
                    act_idx = max(closest_idx - t - 1, 0)
                    target_obs_seq.append(self.demo_observations[obs_idx])
                    target_act_seq.append(-1.0 * self.demo_actions[act_idx])

            target_tensor = torch.tensor(np.array(target_obs_seq), dtype=torch.float32).to(self.device)
            target_act_tensor = torch.tensor(np.array(target_act_seq), dtype=torch.float32).to(self.device)
        else:
            target_tensor = None
            target_act_tensor = None

        # 2. Initialize Mean
        action_mean = torch.zeros(self.planning_horizon, constants.ACTION_DIMENSION).to(self.device)
        if target_act_tensor is not None and not recovery:
             action_mean = target_act_tensor.clone()
        
        std_mag = 0.5 if not recovery else 0.8
        action_std = torch.ones(self.planning_horizon, constants.ACTION_DIMENSION).to(self.device) * std_mag * constants.MAX_ACTION_MAGNITUDE
        
        best_action_seq = None

        # 3. CEM Loop
        for iter_i in range(self.cem_iterations):
            noise = torch.randn(self.cem_num_samples, self.planning_horizon, constants.ACTION_DIMENSION).to(self.device)
            samples = torch.clamp(action_mean.unsqueeze(0) + action_std.unsqueeze(0) * noise, 
                                  -constants.MAX_ACTION_MAGNITUDE, constants.MAX_ACTION_MAGNITUDE)
            
            sim_obs = curr_obs.repeat(self.cem_num_samples, 1)
            costs = torch.zeros(self.cem_num_samples).to(self.device)
            
            for t in range(self.planning_horizon):
                actions_t = samples[:, t, :]
                delta = self.dynamics_model(sim_obs, actions_t)
                sim_obs = sim_obs + delta
                
                track_cost = 0
                act_cost = 0
                stuck_cost = 0 
                
                # Sticky Area Punishment Calculation
                speed = torch.norm(delta, dim=1)
                effort = torch.norm(actions_t, dim=1)
                efficiency = speed / (effort + 1e-6)
                
                if target_tensor is not None:
                    track_cost = 7000.0 * torch.sum((sim_obs - target_tensor[t].unsqueeze(0))**2, dim=1)                        
                    if not recovery:
                        if direction == 1:
                            act_cost = 3000.0 * torch.sum((actions_t - target_act_tensor[t].unsqueeze(0))**2, dim=1)
                        if test:
                            stuck_cost = 300.0 * torch.relu(0.5 - efficiency)
                
                costs += track_cost + act_cost + stuck_cost
            
            elites = samples[torch.topk(costs, self.cem_num_elites, largest=False)[1]]
            action_mean = elites.mean(dim=0)
            action_std = elites.std(dim=0) + 1e-5
            best_action_seq = action_mean

        return best_action_seq.detach().cpu().numpy()

    # -------------------------------------------------------------------------
    # TRAINING MAIN LOOP
    # -------------------------------------------------------------------------
    def training_action(self, obs, money):
        # SAFETY NET: If budget is running critically low, abort training to avoid penalties.
        # You may need to tune '100' depending on how much a step costs in your constants.py
        if money < 7:
            self.state_machine = 'DONE'
            return 4, 0

        self._update_stuck_buffer(obs)
        action_type, action_value = 4, 0
        
        # --- 1. COLLECT DEMOS ---
        if self.state_machine == 'START':
            self.num_demos_collected, self.replay_index, self.demo_buffer = 1, 0, []
            self.state_machine = 'REPLAY'
            # print('COLLECTING DEMONSTRATION 1')
            return 3, self.demo_length 

        if self.state_machine == 'REPLAY':
            if self.replay_index < len(self.demo_buffer):
                action = self.demo_buffer[self.replay_index]
                self.replay_index += 1
                action_value = np.clip(action, -constants.MAX_ACTION_MAGNITUDE, constants.MAX_ACTION_MAGNITUDE)
                action_type = 1
            else:
                # DYNAMIC DEMO LOGIC: 
                # If we are basically at the goal, or hit our budget cap, stop.
                if self.current_dist < self.goal_threshold or self.num_demos_collected >= self.max_demos:
                    self.state_machine = 'EXPLORE_RANDOM'
                    self.steps_explored = 0
                else:
                    self.num_demos_collected += 1
                    self.replay_index, self.demo_buffer = 0, []
                    self.state_machine = 'REPLAY'
                    
                    # CALCULATE DYNAMIC LENGTH:
                    # Estimate steps needed based on a conservative average speed (e.g. 0.025 units/step).
                    # We add a small buffer (+3 steps) to ensure it definitively crosses the line.
                    estimated_steps = int(self.current_dist / 0.025) + 3
                    
                    # Cap it at self.demo_length so we don't ask for a massive demo in one go
                    dynamic_length = min(self.demo_length, max(5, estimated_steps))
                    # print(f'COLLECTING DEMONSTRATION {self.num_demos_collected} with length {dynamic_length}')
                    return 3, dynamic_length
            
        # --- 2. RANDOM EXPLORATION ---
        if self.state_machine == 'EXPLORE_RANDOM':
            if self.steps_explored >= self.random_steps:
                self.state_machine = 'TRAIN_1'
            else:
                self.steps_explored += 1
                is_stuck = self._is_stuck()
                
                if is_stuck:
                    if self.explore_action_duration <= 0:
                        angle = np.random.uniform(0, 2 * np.pi)
                        self.current_explore_action = np.array([
                            constants.MAX_ACTION_MAGNITUDE * np.cos(angle),
                            constants.MAX_ACTION_MAGNITUDE * np.sin(angle)
                        ])
                        self.explore_action_duration = 200 
                    self.explore_action_duration -= 1
                    action_value = self.current_explore_action
                else:
                    if np.random.random() < 0.20:
                        angle = np.random.uniform(0, 2 * np.pi)
                        mag = np.random.uniform(0.1, 0.3) * constants.MAX_ACTION_MAGNITUDE
                        action_value = np.array([mag * np.cos(angle), mag * np.sin(angle)])
                    else:
                        if self.explore_action_duration <= 0:
                            angle = np.random.uniform(0, 2 * np.pi)
                            self.current_explore_action = np.array([
                                constants.MAX_ACTION_MAGNITUDE * np.cos(angle),
                                constants.MAX_ACTION_MAGNITUDE * np.sin(angle)
                            ])
                            self.explore_action_duration = 5 
                        self.explore_action_duration -= 1
                        action_value = self.current_explore_action
                action_type = 1

        # --- 3. FIRST TRAINING ---
        if self.state_machine == 'TRAIN_1':
            self.train_models()

            self.current_refinement_round = 0
            self.refinement_resets_used = 0
            self.max_refinement_resets = 6
            self.frames_stuck_count = 0

            self.state_machine = 'EXPLORE_REFINEMENT'
            self.steps_explored = 0
            self.planned_actions = [] 
            self.recent_obs_buffer.clear()
            self.refinement_direction = 1

            self.stuck_frames_counter = 0
            self.target_bump_offset = 0
            
            # Keep loop alive without teleporting robot
            return 1, np.array([0.0, 0.0])

        # --- 4. REFINEMENT EXPLORATION ---
        if self.state_machine == 'EXPLORE_REFINEMENT':
            if self.steps_explored >= self.refinement_steps:
                self.state_machine = 'TRAIN_2'
            else:
                self.steps_explored += 1
                
                closest_idx = 0
                if len(self.demo_observations) > 0:
                    dists = [np.linalg.norm(obs - d) for d in self.demo_observations]
                    closest_idx = np.argmin(dists)
                
                is_jittering = self._is_jittering()
                
                if is_jittering and not self._is_stuck():
                    self.stuck_frames_counter += 1
                    if self.stuck_frames_counter > 20:
                        self.target_bump_offset += 2 
                        self.stuck_frames_counter = 0 
                else:
                    self.stuck_frames_counter = 0
                    if self.target_bump_offset > 0:
                        self.target_bump_offset -= 1
                
                # Apply offset based on direction
                if self.refinement_direction == 1:
                    current_idx = min(closest_idx + self.target_bump_offset, len(self.demo_observations) - 1)
                else:
                    current_idx = max(closest_idx - self.target_bump_offset, 0)
                
                # Auto-Reverse
                if self.refinement_direction == 1:
                    if current_idx >= len(self.demo_observations) - 10:
                        self.refinement_direction = -1
                        self.planned_actions = [] 
                else:
                    if current_idx <= 2:
                        self.refinement_direction = 1
                        self.planned_actions = []

                # Logic & Stuck Handling
                is_stuck = self._is_stuck()

                if is_stuck:
                    self.frames_stuck_count += 1
                else:
                    self.frames_stuck_count = 0
                    if len(self.planned_actions) > self.plan_duration:
                        self.planned_actions = []

                if self.frames_stuck_count > 200:
                    if self.refinement_resets_used < self.max_refinement_resets:
                        self.refinement_resets_used += 1
                        self.frames_stuck_count = 0
                        self.planned_actions = []
                        self.recent_obs_buffer.clear()
                        return 2, 0 
                    else:
                        self.frames_stuck_count = 0
                        self.planned_actions = [] 

                # If stuck and not currently executing a recovery plan
                if is_stuck and len(self.planned_actions) <= self.plan_duration:
                    self.planned_actions = []
                    
                    best_seq = self._run_cem(obs, direction=self.refinement_direction, recovery=True)
                    first_action = best_seq[0]
                    
                    norm = np.linalg.norm(first_action)
                    if norm > 1e-6:
                        recovery_action = (first_action / norm) * constants.MAX_ACTION_MAGNITUDE
                    else:
                        recovery_action = np.array([constants.MAX_ACTION_MAGNITUDE, 0.0])
                        
                    recovery_action = recovery_action.flatten()
                    self.planned_actions = [recovery_action for _ in range(500)]
                    action_value = self.planned_actions.pop(0)

                # Execute buffered plan
                elif len(self.planned_actions) > 0:
                    action_value = self.planned_actions.pop(0).flatten()
                    self.last_cem_action = action_value 
                    
                # Normal Planning
                else:
                    best_seq = self._run_cem(obs, direction=self.refinement_direction, recovery=False)
                    self.planned_actions = list(best_seq[:self.plan_duration])
                    action_value = self.planned_actions.pop(0).flatten()
                    self.last_cem_action = action_value
                
                action_type = 1

        # --- 5. ITERATIVE REFINEMENT TRAINING ---
        if self.state_machine == 'TRAIN_2':
            self.current_refinement_round += 1
            self.train_models()

            if self.current_refinement_round < self.refinement_rounds:
                self.state_machine = 'EXPLORE_REFINEMENT'
                
                self.steps_explored = 0
                self.planned_actions = []
                self.recent_obs_buffer.clear()
                
                return 1, np.array([0.0, 0.0]) 
            else:
                self.state_machine = 'DONE'
                self.recent_obs_buffer.clear()
                return 4, 0

        return action_type, action_value

    def receive_transition(self, obs, action, next_obs, distance_to_goal):
        self.memory.append((obs, action, next_obs, distance_to_goal))
        self.current_dist = distance_to_goal  # Track the most recent distance!

    def receive_demo(self, demo):
        for obs, act in demo:
            self.demo_observations.append(obs)
            self.demo_actions.append(act)
            
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

        for epoch in range(self.training_epochs):
            total_dyn_loss = 0
            for o, a, d_target, dist_target in loader:
                pred_delta = self.dynamics_model(o, a)
                dyn_loss = nn.MSELoss()(pred_delta, d_target)
                self.dynamics_opt.zero_grad()
                dyn_loss.backward()
                self.dynamics_opt.step()
                total_dyn_loss += dyn_loss.item()

    # -------------------------------------------------------------------------
    # TESTING MAIN LOOP
    # -------------------------------------------------------------------------
    def testing_action(self, obs):
        self._update_stuck_buffer(obs)
        is_stuck = self._is_stuck()

        # --- 1. EXIT RECOVERY ---
        if not is_stuck and len(self.planned_actions) > self.plan_duration:
            self.planned_actions = []

        # --- 2. ENTER RECOVERY ---
        if is_stuck:
            if len(self.planned_actions) <= self.plan_duration:
                self.planned_actions = []
                
                best_seq = self._run_cem(obs, direction=1, recovery=True)

                first_action = best_seq[0]
                norm = np.linalg.norm(first_action)
                if norm > 1e-6:
                    recovery_action = (first_action / norm) * constants.MAX_ACTION_MAGNITUDE
                else:
                    recovery_action = np.array([constants.MAX_ACTION_MAGNITUDE, 0.0])
                
                self.planned_actions = [recovery_action for _ in range(200)]
                return self.planned_actions.pop(0)

        # --- 3. EXECUTE BUFFERED PLAN ---
        if len(self.planned_actions) > 0:
            action = self.planned_actions.pop(0)
            return action

        # --- 4. STANDARD REPLAN (CEM) ---
        best_seq = self._run_cem(obs, direction=1, test=True)
        self.planned_actions = list(best_seq[:self.plan_duration])
        first_action = self.planned_actions.pop(0)
        
        return first_action