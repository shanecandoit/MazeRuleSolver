import random

class Rule:
    def __init__(self, condition1_pos, condition1_state, condition2_pos, condition2_state, action):
        self.condition1_pos = condition1_pos
        self.condition1_state = condition1_state
        self.condition2_pos = condition2_pos
        self.condition2_state = condition2_state
        self.action = action

    def __repr__(self):
        return f"Rule(cond1_pos={self.condition1_pos}, cond1_state='{self.condition1_state}', " \
               f"cond2_pos={self.condition2_pos}, cond2_state='{self.condition2_state}', action='{self.action}')"

    def matches(self, agent_row, agent_col, maze):
        relative_positions = {
            'NORTH': (-1, 0),
            'SOUTH': (1, 0),
            'EAST': (0, 1),
            'WEST': (0, -1),
            'ANY': (0, 0)  # 'ANY' is a special value
        }
        # check that these positions are valid
        assert self.condition1_pos in relative_positions
        assert self.condition2_pos in relative_positions

        # Check condition 1
        if self.condition1_pos == 'ANY':
            cond1_match = True
        else:
            pos1_dr, pos1_dc = relative_positions[self.condition1_pos]
            check_row1, check_col1 = agent_row + pos1_dr, agent_col + pos1_dc
            cond1_match = (0 <= check_row1 < maze.height and 0 <= check_col1 < maze.width and maze.grid[check_row1][check_col1] == self.condition1_state)
        # Check condition 2
        if self.condition2_pos == 'ANY':
            cond2_match = True
        else:
            pos2_dr, pos2_dc = relative_positions[self.condition2_pos]
            check_row2, check_col2 = agent_row + pos2_dr, agent_col + pos2_dc
            cond2_match = (0 <= check_row2 < maze.height and 0 <= check_col2 < maze.width and maze.grid[check_row2][check_col2] == self.condition2_state)

        return cond1_match and cond2_match

    def get_action(self):
        return self.action

    @staticmethod
    def create_k_rules(k):
        """Create k random rules for the agent with better coverage."""
        rules = []
        relative_positions = ['NORTH', 'SOUTH', 'EAST', 'WEST', 'ANY']
        # Maze states should be consistent with Maze class constants
        maze_states = [Maze.WALL, Maze.PATH, Maze.START, Maze.END, 'ANY'] 
        actions = ['NORTH', 'SOUTH', 'EAST', 'WEST']

        # Create a mix of rule types for better coverage
        
        # 1. Simple single-condition rules (easier to match) - 40% of rules
        simple_rules_count = int(k * 0.4)
        for _ in range(simple_rules_count):
            pos = random.choice(relative_positions)
            state = random.choice(maze_states)
            action = random.choice(actions)
            # Use the same condition twice to create single-condition rules
            rules.append(Rule(pos, state, pos, state, action))
        
        # 2. Wall-avoidance rules - 30% of rules
        avoidance_rules_count = int(k * 0.3)
        for _ in range(avoidance_rules_count):
            # Rules that avoid walls
            wall_pos = random.choice(['NORTH', 'SOUTH', 'EAST', 'WEST'])
            safe_actions = [action for action in actions if action != wall_pos]
            if safe_actions:
                action = random.choice(safe_actions)
                # If wall is in direction X, don't go in direction X
                rules.append(Rule(wall_pos, Maze.WALL, 'ANY', 'ANY', action))
        
        # 3. Path-following rules - 20% of rules  
        path_rules_count = int(k * 0.2)
        for _ in range(path_rules_count):
            # Rules that move toward paths
            path_pos = random.choice(['NORTH', 'SOUTH', 'EAST', 'WEST'])
            action = path_pos  # Move toward the path
            rules.append(Rule(path_pos, Maze.PATH, 'ANY', 'ANY', action))
        
        # 4. Random complex rules - 10% of rules
        remaining_count = k - len(rules)
        for _ in range(remaining_count):
            cond1_pos = random.choice(relative_positions)
            cond1_state = random.choice(maze_states)
            cond2_pos = random.choice(relative_positions)
            cond2_state = random.choice(maze_states)
            action = random.choice(actions)
            rules.append(Rule(cond1_pos, cond1_state, cond2_pos, cond2_state, action))
        
        return rules

class Maze:
    WALL = '#'
    PATH = '.'
    START = 'S'
    END = 'E'

    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height
        self.grid = []
        self.start_pos = (0, 0) # Default start position
        self.end_pos = (0, 0)   # Default end position
        self.generate_maze()

    def generate_maze(self):
        # Initialize grid with all walls
        self.grid = [[Maze.WALL for _ in range(self.width)] for _ in range(self.height)]

        # Choose a random starting point for carving, allowing edges
        # Ensure start_row and start_col are within bounds and are even indices for carving
        start_row = random.randrange(0, self.height, 2)
        start_col = random.randrange(0, self.width, 2)
        
        # Ensure the starting point is within the grid and is a valid carving start
        # If width/height are odd, the last index might not be reachable by randrange(..., 2)
        # For simplicity, we'll ensure it's within bounds.
        start_row = min(start_row, self.height - 1)
        start_col = min(start_col, self.width - 1)

        self.grid[start_row][start_col] = Maze.PATH

        stack = [(start_row, start_col)]

        while stack:
            curr_row, curr_col = stack[-1]
            
            neighbors = []
            # Define possible directions (row_offset, col_offset) for carving 2 cells away
            directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]

            for dr, dc in directions:
                n_row, n_col = curr_row + dr, curr_col + dc
                # Check if the neighbor cell is within bounds and is still a wall
                if 0 <= n_row < self.height and 0 <= n_col < self.width and self.grid[n_row][n_col] == Maze.WALL:
                    # Calculate the wall cell between current and neighbor
                    wall_row, wall_col = curr_row + dr // 2, curr_col + dc // 2
                    neighbors.append(((n_row, n_col), (wall_row, wall_col)))
            
            if neighbors:
                (next_row, next_col), (wall_row, wall_col) = random.choice(neighbors)
                self.grid[wall_row][wall_col] = Maze.PATH # Carve the wall
                self.grid[next_row][next_col] = Maze.PATH # Carve the next cell
                stack.append((next_row, next_col))
            else:
                stack.pop()

        # Set start position
        self.start_pos = (1, 1)
        self.grid[self.start_pos[0]][self.start_pos[1]] = Maze.START

        # Find the furthest point for the end position using BFS
        queue = [(self.start_pos[0], self.start_pos[1], 0)]  # (row, col, distance)
        visited = {(self.start_pos[0], self.start_pos[1])}
        max_dist = 0
        furthest_pos = self.start_pos

        head = 0
        while head < len(queue):
            r, c, dist = queue[head]
            head += 1

            if dist > max_dist:
                max_dist = dist
                furthest_pos = (r, c)

            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 < nr < self.height - 1 and 0 < nc < self.width - 1 and self.grid[nr][nc] != Maze.WALL and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc, dist + 1))

        self.end_pos = furthest_pos
        self.grid[self.end_pos[0]][self.end_pos[1]] = Maze.END

    def print_maze(self, agent=None):
        for r_idx, row in enumerate(self.grid):
            display_row = []
            for c_idx, cell in enumerate(row):
                if agent and agent.row == r_idx and agent.col == c_idx:
                    # Show agent position with special notation based on what's underneath
                    if (r_idx, c_idx) == agent.maze.start_pos:
                        display_row.append('@')  # Agent at start
                    elif (r_idx, c_idx) == agent.maze.end_pos:
                        display_row.append('&')  # Agent at end
                    else:
                        display_row.append('A')  # Agent elsewhere
                else:
                    display_row.append(cell)
            print(" ".join(display_row))

    def is_valid_move(self, row, col):
        # Check if the move is within bounds and not into a wall
        return 0 <= row < self.height and 0 <= col < self.width and self.grid[row][col] != self.WALL

class Agent:
    def __init__(self, maze, start_row, start_col):
        self.maze = maze
        self.row = start_row
        self.col = start_col
        self.ensemble = []  # This will be populated with Rule objects
        
        # Tracking attributes for scoring
        self.position_history = []
        self.unique_positions = set()
        self.score = 0

        # Create more rules with better coverage - 501
        self.ensemble = Rule.create_k_rules(501)
        
        # Add some guaranteed basic rules for better coverage
        self._add_basic_survival_rules()
        
    def _add_basic_survival_rules(self):
        """Add basic rules that should almost always have some match."""
        # Always try to move away from walls if possible
        basic_rules = [
            # If north is wall, try south
            Rule('NORTH', Maze.WALL, 'ANY', 'ANY', 'SOUTH'),
            # If south is wall, try north
            Rule('SOUTH', Maze.WALL, 'ANY', 'ANY', 'NORTH'),
            # If east is wall, try west
            Rule('EAST', Maze.WALL, 'ANY', 'ANY', 'WEST'),
            # If west is wall, try east
            Rule('WEST', Maze.WALL, 'ANY', 'ANY', 'EAST'),
            # If on path, try to stay moving (ANY position, PATH state)
            Rule('ANY', Maze.PATH, 'ANY', 'ANY', 'NORTH'),
            Rule('ANY', Maze.PATH, 'ANY', 'ANY', 'SOUTH'),
            Rule('ANY', Maze.PATH, 'ANY', 'ANY', 'EAST'),
            Rule('ANY', Maze.PATH, 'ANY', 'ANY', 'WEST'),
        ]
        self.ensemble.extend(basic_rules)

    def move(self, action):
        new_row, new_col = self.row, self.col
        if action == 'NORTH':
            new_row -= 1
        elif action == 'SOUTH':
            new_row += 1
        elif action == 'EAST':
            new_col += 1
        elif action == 'WEST':
            new_col -= 1

        if self.maze.is_valid_move(new_row, new_col):
            self.row, self.col = new_row, new_col
            # Track position for scoring
            self.position_history.append((self.row, self.col))
            self.unique_positions.add((self.row, self.col))
            return True
        return False

    def decide_action(self):
        action_votes = {'NORTH': 0, 'SOUTH': 0, 'EAST': 0, 'WEST': 0}
        matching_rules_count = 0

        for rule in self.ensemble:
            if rule.matches(self.row, self.col, self.maze):
                action_votes[rule.get_action()] += 1
                matching_rules_count += 1

        if matching_rules_count == 0:
            # Fallback 1: Try simple heuristic - move away from walls
            return self._heuristic_action()

        # Find the action with the most votes
        best_action = None
        max_votes = -1
        
        # Collect all actions with max votes to handle ties
        top_actions = []
        for action, votes in action_votes.items():
            if votes > max_votes:
                max_votes = votes
                top_actions = [action] # Start a new list of top actions
            elif votes == max_votes and votes > 0: # If votes are equal to max_votes and not zero
                top_actions.append(action)

        if not top_actions: # Should not happen if matching_rules_count > 0, but as a safeguard
            return self._heuristic_action()

        # Tie-breaking: randomly choose from the top actions
        best_action = random.choice(top_actions)

        # return the action with the most votes
        # maybe also return the votes
        return best_action, action_votes
    
    def _heuristic_action(self):
        """Simple heuristic for when no rules match - avoid walls and prefer valid moves.
        Make a rule up on the spot that matches the current situation and suggests a valid action."""
        possible_actions = []
        action_scores = {}
        best_action = None
        best_score = -float('inf')
        best_pos = None
        best_state = None
        
        for action in ['NORTH', 'SOUTH', 'EAST', 'WEST']:
            new_row, new_col = self.row, self.col
            if action == 'NORTH':
                new_row -= 1
            elif action == 'SOUTH':
                new_row += 1
            elif action == 'EAST':
                new_col += 1
            elif action == 'WEST':
                new_col -= 1

            if self.maze.is_valid_move(new_row, new_col):
                possible_actions.append(action)
                # Score actions: prefer unexplored areas
                score = 1
                if (new_row, new_col) not in self.unique_positions:
                    score += 2  # Bonus for new positions
                # Bonus for moving toward end (simple Manhattan distance)
                old_dist = abs(self.row - self.maze.end_pos[0]) + abs(self.col - self.maze.end_pos[1])
                new_dist = abs(new_row - self.maze.end_pos[0]) + abs(new_col - self.maze.end_pos[1])
                if new_dist < old_dist:
                    score += 1  # Bonus for getting closer to end
                action_scores[action] = score
                if score > best_score:
                    best_score = score
                    best_action = action
                    best_pos = action  # Use the direction as the relative position
                    # Get the state of the cell we're moving into
                    if 0 <= new_row < self.maze.height and 0 <= new_col < self.maze.width:
                        best_state = self.maze.grid[new_row][new_col]
                    else:
                        best_state = 'ANY'
        if not possible_actions:
            return None  # Truly stuck
        # Create a new rule that matches the current situation and suggests the best action
        if best_action is not None:
            new_rule = Rule(best_pos, best_state, 'ANY', 'ANY', best_action)
            self.ensemble.append(new_rule)
        # Return in same format as normal decision
        votes = {action: action_scores.get(action, 0) for action in ['NORTH', 'SOUTH', 'EAST', 'WEST']}
        return best_action, votes

    def run(self, n_steps=10):
        """Run the agent for a specified number of steps, tracking positions."""
        print(f"Starting run for {n_steps} steps...")
        stuck_count = 0
        
        for step in range(n_steps):
            print(f"\nStep {step + 1}:")
            print(f"  Current position: ({self.row}, {self.col})")

            # Decide action
            decision_result = self.decide_action()
            
            if decision_result is None:
                print("  No valid moves available - truly stuck!")
                stuck_count += 1
            else:
                chosen_action, votes = decision_result
                # Check if this was a heuristic decision (low vote counts indicate heuristic)
                total_votes = sum(votes.values())
                if total_votes <= 4:  # Likely a heuristic decision
                    print(f"  Heuristic action: {chosen_action} (scores: {votes})")
                else:
                    print(f"  Decided action: {chosen_action} (votes: {votes})")
            
            # Try to move
            if decision_result and chosen_action:
                if self.move(chosen_action):
                    print(f"  Moved successfully to ({self.row}, {self.col})")
                    stuck_count = 0  # Reset stuck counter
                else:
                    print("  Could not move - blocked by wall")
                    stuck_count += 1
            else:
                print("  No action decided - staying in place")
                stuck_count += 1
                
                # If stuck for 3 consecutive steps, try a random valid move
                if stuck_count >= 3:
                    print("  Agent stuck - trying random valid action")
                    possible_actions = []
                    for action in ['NORTH', 'SOUTH', 'EAST', 'WEST']:
                        new_row, new_col = self.row, self.col
                        if action == 'NORTH':
                            new_row -= 1
                        elif action == 'SOUTH':
                            new_row += 1
                        elif action == 'EAST':
                            new_col += 1
                        elif action == 'WEST':
                            new_col -= 1
                        
                        if self.maze.is_valid_move(new_row, new_col):
                            possible_actions.append(action)
                    
                    if possible_actions:
                        random_action = random.choice(possible_actions)
                        if self.move(random_action):
                            print(f"  Random move succeeded: {random_action} to ({self.row}, {self.col})")
                            stuck_count = 0
        
        print("\nRun completed!")
        print(f"Total positions visited: {len(self.position_history)}")
        print(f"Unique positions visited: {len(self.unique_positions)}")
        
    def reset_position_tracking(self):
        """Clear the position history and unique positions for a fresh run."""
        self.position_history.clear()
        self.unique_positions.clear()
        
    def get_score(self):
        """Return the number of unique positions visited."""
        return len(self.unique_positions)

    def print_agent_state(self):
        print(f"Agent State: Position=({self.row}, {self.col}), Rules={len(self.ensemble)}")

    def get_state(self):
        return {'row': self.row, 'col': self.col, 'direction': self.direction}


if __name__ == "__main__":
    # 1. Instantiate Maze and Agent
    my_maze = Maze(width=10, height=10)
    my_agent = Agent(my_maze, my_maze.start_pos[0], my_maze.start_pos[1])

    # 2. Show initial state
    print("\nInitial Maze:")
    my_maze.print_maze(my_agent)
    my_agent.print_agent_state()

    # 3. Run agent for 10 steps
    print("\n=== Agent Run: 10 Steps ===")
    my_agent.position_history.append((my_agent.row, my_agent.col))
    my_agent.unique_positions.add((my_agent.row, my_agent.col))
    my_agent.run(n_steps=10)

    # 4. Show final state and metrics
    print("\nFinal Maze with Agent:")
    my_maze.print_maze(my_agent)
    print("\nPerformance Metrics:")
    print(f"  Final score (unique positions): {my_agent.get_score()}")
    print(f"  Position history: {my_agent.position_history}")
    print(f"  Unique positions: {sorted(list(my_agent.unique_positions))}")

    # 5. Heuristic fallback test: agent with no rules
    print("\n=== Heuristic Fallback Test ===")
    test_agent = Agent(my_maze, my_maze.start_pos[0], my_maze.start_pos[1])
    test_agent.ensemble = []  # Remove all rules
    test_agent.position_history.append((test_agent.row, test_agent.col))
    test_agent.unique_positions.add((test_agent.row, test_agent.col))
    test_agent.run(n_steps=3)
    print(f"  Position history: {test_agent.position_history}")
    print(f"  Unique positions: {sorted(list(test_agent.unique_positions))}")
    print("   Agent length of ensemble:", len(test_agent.ensemble))  # Should be >0

    # 6. Test reset functionality
    print("\nTesting reset_position_tracking()...")
    test_agent.reset_position_tracking()
    print(f"  Score after reset: {test_agent.get_score()}")
    print(f"  Position history after reset: {test_agent.position_history}")
