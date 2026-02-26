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
            nn.Linear(obs_dim + act_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, obs_dim)
        )

    def forward(self, obs, act):
        # Scale action up slightly to match magnitude of observations for better training
        x = torch.cat([obs, act * 10.0], dim=-1)
        delta = self.net(x) 
        return delta

class BCModel(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(BCModel, self).__init__()
        # Input: Observation
        # Output: Predicted Expert Action
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim)
        )

    def forward(self, obs):
        return self.net(obs)

class Robot:
    def __init__(self):
        # Hyperparameters
        self.demo_length = 30          
        self.max_demos = 6          
        self.current_dist = float('inf') 
        self.min_dist = float('inf')      # NEW: Tracks the closest we've ever been
        self.crossed_line = False         # NEW: Flag to definitively know we passed the goal
        self.num_demos_collected = 0
        self.last_dist = None   
        
        # Phase 1: Random Exploration (Learn basics)
        self.random_steps = 100
        
        # Phase 2: Refinement Exploration (Use trained model to generate better data)
        self.refinement_rounds = 30          
        self.refinement_steps = 200
        self.refinement_direction = 1 
        
        self.training_epochs = 10      
        self.batch_size = 64
        self.lr = 0.001
        
        # MPC / CEM Parameters
        self.planning_horizon = 2    
        self.cem_iterations = 4
        self.cem_num_samples = 40
        self.cem_num_elites = 5     
        self.plan_duration = 1        
        
        # State Management
        self.demo_buffer = []
        self.replay_index = 0
        self.steps_explored = 0
        self.state_machine = 'START'   
        
        self.current_explore_action = None
        self.explore_action_duration = 0
        
        # Fallback tracking
        self.recent_obs_buffer = collections.deque(maxlen=20)
        self.planned_actions = []
        self.recovery_steps = 0
        self.recovery_action = np.zeros(2)
        
        # Data Storage
        self.memory = collections.deque(maxlen=20000)
        self.demo_observations = []
        self.demo_actions = [] 
        
        # Synthetic BC Data Storage
        self.synthetic_dataset = []
        self.current_trajectory = []
        self.last_recorded_obs = None  # Tracks progress for filtering stuck frames
        
        # Models
        self.device = torch.device("cpu") # GPU prohibited
        self.dynamics_model = DynamicsModel(constants.OBSERVATION_DIMENSION, constants.ACTION_DIMENSION).to(self.device)
        self.dynamics_opt = optim.Adam(self.dynamics_model.parameters(), lr=self.lr)

        self.bc_model = BCModel(constants.OBSERVATION_DIMENSION, constants.ACTION_DIMENSION).to(self.device)
        self.bc_opt = optim.Adam(self.bc_model.parameters(), lr=0.001)
        self.visualisation_lines = []

    # -------------------------------------------------------------------------
    # STUCK DETECTION
    # -------------------------------------------------------------------------
    def _update_stuck_buffer(self, obs):
        self.recent_obs_buffer.append(obs)

    def _is_stuck(self):
        if len(self.recent_obs_buffer) < 20: return False
        diffs = [np.linalg.norm(self.recent_obs_buffer[i] - self.recent_obs_buffer[i-1]) for i in range(1, len(self.recent_obs_buffer))]
        return np.mean(diffs) < 0.002
    
    def _is_stuck_test(self):
        if len(self.recent_obs_buffer) < 20: return False
        diffs = [np.linalg.norm(self.recent_obs_buffer[i] - self.recent_obs_buffer[i-1]) for i in range(1, len(self.recent_obs_buffer))]
        return np.mean(diffs) < 0.001
            
    def _is_jittering(self):
        current_obs = self.recent_obs_buffer[-1]
        max_spread = max([np.linalg.norm(current_obs - obs) for obs in self.recent_obs_buffer])
        return max_spread < 0.10

    # -------------------------------------------------------------------------
    # SHARED PLANNER (USED FOR REFINEMENT)
    # -------------------------------------------------------------------------
    def _run_cem(self, obs, direction=1, recovery=False, bump_offset=0):
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
            lookahead = 1 + bump_offset

            for t in range(self.planning_horizon):
                if direction == 1:
                    obs_idx = min(closest_idx + t + lookahead, len(self.demo_observations) - 1)
                    target_obs_seq.append(self.demo_observations[obs_idx])
                else:
                    obs_idx = max(closest_idx - t - lookahead, 0)
                    target_obs_seq.append(self.demo_observations[obs_idx])

            target_tensor = torch.tensor(np.array(target_obs_seq), dtype=torch.float32).to(self.device)
        else:
            target_tensor = None

        # 2. Initialize Mean
        action_mean = torch.zeros(self.planning_horizon, constants.ACTION_DIMENSION).to(self.device)
        
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
                if target_tensor is not None:
                    track_cost = 10000.0 * torch.sum((sim_obs - target_tensor[t].unsqueeze(0))**2, dim=1)                        
                costs += track_cost
            
            elites = samples[torch.topk(costs, self.cem_num_elites, largest=False)[1]]
            action_mean = elites.mean(dim=0)
            action_std = elites.std(dim=0) + 1e-5
            best_action_seq = action_mean

        return best_action_seq.detach().cpu().numpy()

    # -------------------------------------------------------------------------
    # TRAINING MAIN LOOP
    # -------------------------------------------------------------------------
    def training_action(self, obs, money):
        if money < 7:
            if self.state_machine != 'DONE':
                print("Budget critically low! Forcing final model training and exit.")
                self.train_models() 
            self.state_machine = 'DONE'
            return 4, 0

        self._update_stuck_buffer(obs)
        action_type, action_value = 4, 0
        
        # --- 1. COLLECT DEMOS ---
        if self.state_machine == 'START':
            print("STATE: START -> Requesting initial demo.")
            self.num_demos_collected, self.replay_index, self.demo_buffer = 1, 0, []
            self.state_machine = 'REPLAY'
            return 3, self.demo_length 

        if self.state_machine == 'REPLAY':
            if self.replay_index < len(self.demo_buffer):
                action = self.demo_buffer[self.replay_index]
                self.replay_index += 1
                action_value = np.clip(action, -constants.MAX_ACTION_MAGNITUDE, constants.MAX_ACTION_MAGNITUDE)
                action_type = 1
            else:
                if self.crossed_line or self.num_demos_collected >= self.max_demos:
                    print(f"Goal crossed or max demos hit. Moving to EXPLORE_RANDOM. Collected {self.num_demos_collected} demos.")
                    self.state_machine = 'EXPLORE_RANDOM'
                    self.steps_explored = 0
                else:
                    self.num_demos_collected += 1
                    self.replay_index, self.demo_buffer = 0, []
                    self.state_machine = 'REPLAY'
                    
                    estimated_steps = int(self.current_dist / 0.025) + 10
                    dynamic_length = min(self.demo_length, max(5, estimated_steps))
                    print(f"Requesting continuing demo (Number {self.num_demos_collected}), length {dynamic_length}...")
                    return 3, dynamic_length
            
        # --- 2. RANDOM EXPLORATION ---
        if self.state_machine == 'EXPLORE_RANDOM':
            if self.steps_explored >= self.random_steps:
                print("STATE: EXPLORE_RANDOM -> TRAIN_1")
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
            print("Starting Initial Phase 1 Training...")
            self.train_models()

            self.current_refinement_round = 0
            self.refinement_resets_used = 0
            self.max_refinement_resets = 7 - self.num_demos_collected
            self.frames_stuck_count = 0

            print("STATE: TRAIN_1 -> EXPLORE_REFINEMENT")
            self.state_machine = 'EXPLORE_REFINEMENT'
            self.steps_explored = 0
            self.planned_actions = [] 
            self.recent_obs_buffer.clear()
            self.refinement_direction = 1

            self.stuck_frames_counter = 0
            self.target_bump_offset = 0
            
            return 1, np.array([0.0, 0.0])

        # --- 4. REFINEMENT EXPLORATION ---
        if self.state_machine == 'EXPLORE_REFINEMENT':
            if self.steps_explored >= self.refinement_steps:
                print(f"Completed {self.refinement_steps} exploration steps. Transitioning to TRAIN_2.")
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
                        if self.refinement_direction == 1 and closest_idx >= len(self.demo_observations) - 5:
                            print("Jittering near END of demo. Forcing early turnaround (BACKWARD).")
                            self.refinement_direction = -1
                            self.target_bump_offset = 0
                            self.planned_actions = []
                        elif self.refinement_direction == -1 and closest_idx <= 5:
                            print("Jittering near START of demo. Forcing early turnaround (FORWARD).")
                            self.refinement_direction = 1
                            self.target_bump_offset = 0
                            self.planned_actions = []
                        else:
                            # Cap the bump offset to 8 to prevent massive unphysical target jumps
                            self.target_bump_offset = min(self.target_bump_offset + 2, 8) 
                            print(f"ref direction: {self.refinement_direction}, Jittering detected near frame {closest_idx}/{len(self.demo_observations) - 1}. Bump offset: {self.target_bump_offset}")
                        self.stuck_frames_counter = 0 
                else:
                    self.stuck_frames_counter = 0
                    if self.target_bump_offset > 0:
                        self.target_bump_offset -= 1
                
                if self.refinement_direction == 1:
                    current_idx = min(closest_idx + self.target_bump_offset, len(self.demo_observations) - 1)
                else:
                    current_idx = max(closest_idx - self.target_bump_offset, 0)
                
                # Reverse directions
                if self.refinement_direction == 1:
                    if current_idx >= len(self.demo_observations) - 1:
                        self.refinement_direction = -1
                        self.planned_actions = [] 
                        
                        # Only keep this synthetic demo if we successfully logged enough frames
                        if len(self.current_trajectory) > 10:
                            self.synthetic_dataset.extend(self.current_trajectory)
                            print(f"Added synthetic trajectory of length {len(self.current_trajectory)}. Total synthetic dataset size: {len(self.synthetic_dataset)}")
                            
                        self.current_trajectory = [] # Clear ready for next forward pass
                        self.last_recorded_obs = None
                else:
                    if current_idx <= 0:
                        self.refinement_direction = 1
                        self.planned_actions = []
                        self.current_trajectory = [] # Ensure clean slate
                        self.last_recorded_obs = None

                is_stuck = self._is_stuck()

                if is_stuck:
                    self.frames_stuck_count += 1
                else:
                    self.frames_stuck_count = 0
                    if len(self.planned_actions) > self.plan_duration:
                        self.planned_actions = []

                if self.frames_stuck_count > 100:
                    if self.refinement_resets_used < self.max_refinement_resets:
                        print(f"Stuck for >100 frames! Hard reset used: {self.refinement_resets_used + 1}/{self.max_refinement_resets}")
                        self.refinement_resets_used += 1
                        self.frames_stuck_count = 0
                        self.planned_actions = []
                        self.recent_obs_buffer.clear()
                        self.current_trajectory = []
                        self.last_recorded_obs = None
                        return 2, 0 
                    else:
                        self.frames_stuck_count = 0
                        self.planned_actions = [] 
                        self.current_trajectory = []
                        self.last_recorded_obs = None

                # --- ACTION SELECTION ---
                if is_stuck and len(self.planned_actions) <= self.plan_duration:
                    self.planned_actions = []
                    best_seq = self._run_cem(obs, direction=self.refinement_direction, recovery=True, bump_offset=self.target_bump_offset)
                    first_action = best_seq[0]
                    norm = np.linalg.norm(first_action)
                    if norm > 1e-6:
                        recovery_action = (first_action / norm) * constants.MAX_ACTION_MAGNITUDE
                    else:
                        recovery_action = np.array([constants.MAX_ACTION_MAGNITUDE, 0.0])
                    
                    clean_action = recovery_action.flatten()
                    self.planned_actions = [clean_action.copy() for _ in range(4)] # buffer rest

                elif len(self.planned_actions) > 0:
                    clean_action = self.planned_actions.pop(0).flatten()
                else:
                    best_seq = self._run_cem(obs, direction=self.refinement_direction, recovery=False, bump_offset=self.target_bump_offset)
                    self.planned_actions = list(best_seq[:self.plan_duration])
                    clean_action = self.planned_actions.pop(0).flatten()
                
                # --- DAGGER NOISE & THE "CLEAN LABEL" TRICK ---
                action_value = clean_action.copy()
                
                if self.refinement_direction == 1:
                    # PROGRESS FILTER: Check if we have actually moved enough to justify recording this frame.
                    moved_enough = True
                    if self.last_recorded_obs is not None:
                        if np.linalg.norm(obs - self.last_recorded_obs) < 0.005:
                            moved_enough = False

                    if moved_enough and not is_stuck:
                        # 1. Add noise for actual physical execution
                        action_value += np.random.normal(0, 0.005, size=constants.ACTION_DIMENSION)
                        action_value = np.clip(action_value, -constants.MAX_ACTION_MAGNITUDE, constants.MAX_ACTION_MAGNITUDE)
                        
                        # 2. Save the perfect, un-noised label to our BC dataset
                        self.current_trajectory.append((obs, clean_action.copy()))
                        self.last_recorded_obs = obs.copy()

                action_type = 1

        # --- 5. ITERATIVE REFINEMENT TRAINING ---
        if self.state_machine == 'TRAIN_2':
            print(f"Starting Refinement Training Round {self.current_refinement_round + 1}/{self.refinement_rounds}...")
            self.current_refinement_round += 1
            self.train_models()

            if self.current_refinement_round < self.refinement_rounds:
                print(f"Round {self.current_refinement_round} complete. Back to EXPLORE_REFINEMENT.")
                self.state_machine = 'EXPLORE_REFINEMENT'
                
                self.steps_explored = 0
                self.planned_actions = []
                self.recent_obs_buffer.clear()
                self.current_trajectory = [] # Reset for safety
                self.last_recorded_obs = None
                
                return 1, np.array([0.0, 0.0]) 
            else:
                print("All refinement rounds complete. STATE -> DONE.")
                self.state_machine = 'DONE'
                self.recent_obs_buffer.clear()
                return 4, 0

        return action_type, action_value

    def receive_transition(self, obs, action, next_obs, distance_to_goal):
        self.memory.append((obs, action, next_obs, distance_to_goal))
        self.current_dist = distance_to_goal  
        
        # INFLECTION POINT DETECTION
        if distance_to_goal < self.min_dist:
            self.min_dist = distance_to_goal

        if self.state_machine == 'REPLAY':
            if self.last_dist is not None:
                # 1. Distance definitively increased after reaching a minimum
                if distance_to_goal > self.min_dist + 0.015 and self.min_dist < 0.1:
                    self.crossed_line = True
                    print("Detected goal crossing via distance increase after minimum. Marking goal as crossed to avoid demo corruption.")
                
                # 2. Distance is artificially locked at exactly 0.05 
                elif abs(distance_to_goal - 0.05) < 1e-7 and abs(self.last_dist - 0.05) < 1e-7:
                    self.crossed_line = True
                    print("Detected potential wall collision (distance locked at 0.05). Marking goal as crossed to avoid demo corruption.")
                    
            self.last_dist = distance_to_goal
    def receive_demo(self, demo):
        for obs, act in demo:
            self.demo_observations.append(obs)
            self.demo_actions.append(act)
            
        self.demo_buffer = [d[1] for d in demo]

    def train_models(self):
        # 1. Train Dynamics Model (for the CEM Expert)
        if len(self.memory) >= self.batch_size:
            print(f"Training Dynamics Model on {len(self.memory)} transitions...")
            obs_batch, act_batch, next_obs_batch, dist_batch = zip(*self.memory)
            
            obs_t = torch.tensor(np.array(obs_batch), dtype=torch.float32).to(self.device)
            act_t = torch.tensor(np.array(act_batch), dtype=torch.float32).to(self.device)
            next_obs_t = torch.tensor(np.array(next_obs_batch), dtype=torch.float32).to(self.device)
            
            delta_target = next_obs_t - obs_t
            dataset = torch.utils.data.TensorDataset(obs_t, act_t, delta_target)
            loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            self.dynamics_model.train()
            for epoch in range(self.training_epochs):
                total_dyn_loss = 0.0
                for o, a, d_target in loader:
                    pred_delta = self.dynamics_model(o, a)
                    dyn_loss = nn.MSELoss()(pred_delta, d_target)
                    self.dynamics_opt.zero_grad()
                    dyn_loss.backward()
                    self.dynamics_opt.step()
                    total_dyn_loss += dyn_loss.item()
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    print(f"  Dynamics Epoch {epoch+1}/{self.training_epochs} - Avg Loss: {total_dyn_loss / len(loader):.6f}")

        # 2. Train Behavioural Cloning Model (The Fast Student)
        bc_obs = []
        bc_act = []
        
        # EXPERT WEIGHTING: Add the original, perfectly clean expert demos 10 times over
        if len(self.demo_observations) > 0:
            for _ in range(10):
                bc_obs.extend(self.demo_observations)
                bc_act.extend(self.demo_actions)
                
        # Include the auto-generated, noise-recovering synthetic dataset
        if len(self.synthetic_dataset) > 0:
            synth_o, synth_a = zip(*self.synthetic_dataset)
            bc_obs.extend(synth_o)
            bc_act.extend(synth_a)

        if len(bc_obs) >= self.batch_size:
            print(f"Training BC Model on {len(bc_obs)} transitions (Expert + Synthetic)...")
            bc_o_t = torch.tensor(np.array(bc_obs), dtype=torch.float32).to(self.device)
            bc_a_t = torch.tensor(np.array(bc_act), dtype=torch.float32).to(self.device)
            
            bc_dataset = torch.utils.data.TensorDataset(bc_o_t, bc_a_t)
            bc_loader = torch.utils.data.DataLoader(bc_dataset, batch_size=self.batch_size, shuffle=True)
            
            self.bc_model.train()
            for epoch in range(self.training_epochs): 
                total_bc_loss = 0.0
                for o, a in bc_loader:
                    pred_a = self.bc_model(o)
                    loss = nn.MSELoss()(pred_a, a)
                    self.bc_opt.zero_grad()
                    loss.backward()
                    self.bc_opt.step()
                    total_bc_loss += loss.item()
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    print(f"  BC Epoch {epoch+1}/{self.training_epochs} - Avg Loss: {total_bc_loss / len(bc_loader):.6f}")

    # -------------------------------------------------------------------------
    # TESTING MAIN LOOP
    # -------------------------------------------------------------------------
    def testing_action(self, obs):
        self._update_stuck_buffer(obs)
        is_stuck = self._is_stuck_test()
        
        # 1. Clear recovery buffer if unstuck
        if not is_stuck and len(self.planned_actions) > 0:
            print("Robot unstuck! Resuming Behavioural Cloning.")
            self.planned_actions = []
            
        # 2. CEM Fallback Recovery
        if is_stuck:
            if len(self.planned_actions) == 0:
                print("Robot stuck during testing! Reverting to CEM for recovery.")
                
                # Use the Dynamics Model and CEM to mathematically plan an escape route
                best_seq = self._run_cem(obs, direction=1, recovery=True)
                first_action = best_seq[0]
                
                norm = np.linalg.norm(first_action)
                if norm > 1e-6:
                    recovery_action = (first_action / norm) * constants.MAX_ACTION_MAGNITUDE
                else:
                    recovery_action = np.array([constants.MAX_ACTION_MAGNITUDE, 0.0])
                
                # Buffer the CEM's chosen recovery action for 5 frames to push out of the mud
                self.planned_actions = [recovery_action.flatten() for _ in range(5)]
                
            return self.planned_actions.pop(0)
        
        # 3. Pure Behavioural Cloning Execution
        self.bc_model.eval()
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            action = self.bc_model(obs_t).squeeze(0).numpy()
            action = np.clip(action, -constants.MAX_ACTION_MAGNITUDE, constants.MAX_ACTION_MAGNITUDE)
        
        return action