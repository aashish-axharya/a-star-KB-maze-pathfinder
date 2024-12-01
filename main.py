from queue import PriorityQueue
import math

DIRECTIONS = {
    'UP': (-1, 0),
    'DOWN': (1, 0),
    'LEFT': (0, -1),
    'RIGHT': (0, 1)
}

def euclidean_distance(p1, p2):
    """Calculate the Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_neighbors(grid, current):
    """Get valid neighboring cells using only cardinal directions."""
    neighbors = []
    rows, cols = len(grid), len(grid[0])
    
    for direction in DIRECTIONS.values():
        new_row = current[0] + direction[0]
        new_col = current[1] + direction[1]
        
        # Check if the neighbor is within bounds and not a wall (1)
        if (0 <= new_row < rows and 
            0 <= new_col < cols and 
            grid[new_row][new_col] != 1):
            neighbors.append((new_row, new_col))
    
    return neighbors

def astar(grid, start, goal):
    """A* pathfinding algorithm implementation."""
    rows, cols = len(grid), len(grid[0])
    
    # Priority queue to store nodes to visit
    # Format: (f_score, current_cost, position)
    frontier = PriorityQueue()
    frontier.put((0, 0, start))
    
    # Keep track of where we came from
    came_from = {start: None}
    
    # Cost to reach each node
    cost_so_far = {start: 0}
    
    while not frontier.empty():
        # Get the node with lowest f_score
        _, current_cost, current = frontier.get()
        
        # If we reached the goal, reconstruct and return the path
        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            return path[::-1]
        
        # Check all neighbors
        for next_pos in get_neighbors(grid, current):
            # Movement cost is always 1 for cardinal directions
            new_cost = cost_so_far[current] + 1
            
            # If we haven't visited this node or found a better path
            if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                cost_so_far[next_pos] = new_cost
                # Calculate f_score (g_score + h_score)
                f_score = new_cost + euclidean_distance(next_pos, goal)
                frontier.put((f_score, new_cost, next_pos))
                came_from[next_pos] = current
    
    # No path found
    return None

# Example usage
grid = [
    [0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 0]
]

start = (0, 0)
goal = (4, 5)

# Find the path
path = astar(grid, start, goal)

# Print the path
if path:
    print("Path found:", path)
    
    # Visualize the path in the grid
    visual_grid = [row[:] for row in grid]
    for row, col in path:
        visual_grid[row][col] = '*'
    
    # Print the grid with the path
    for row in visual_grid:
        print(' '.join(['*' if cell == '*' else '#' if cell == 1 else '.' for cell in row]))
else:
    print("No path found!")