from queue import PriorityQueue
import math
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import animation
import numpy as np

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
        
        if (0 <= new_row < rows and 
            0 <= new_col < cols and 
            grid[new_row][new_col] != 1):
            neighbors.append((new_row, new_col))
    
    return neighbors

def astar(grid, start, goal):
    """A* search that prioritizes paths moving closer to the goal."""
    frontier = PriorityQueue()
    frontier.put((0, 0, start))  # (f_score, g_score, position)
    came_from = {start: None}
    cost_so_far = {start: 0}
    visited = set()
    steps = 0
    
    while not frontier.empty():
        _, current_g, current = frontier.get()
        
        if current in visited:
            continue
            
        visited.add(current)
        steps += 1
        
        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            yield path[::-1], visited, [], steps
            return
        
        # Get all neighbors
        neighbors = get_neighbors(grid, current)
        
        # Calculate f_scores for all neighbors
        neighbor_scores = []
        for next_pos in neighbors:
            if next_pos in visited:
                continue
                
            new_g = current_g + 1
            h = euclidean_distance(next_pos, goal)
            f = new_g + h
            neighbor_scores.append((f, new_g, next_pos))
        
        # Sort neighbors by f_score
        neighbor_scores.sort()
        
        # Only keep the best neighbor(s)
        current_frontier = []
        best_f = float('inf') if not neighbor_scores else neighbor_scores[0][0]
        
        for f, new_g, next_pos in neighbor_scores:
            # Only consider neighbors that are within a small threshold of the best f_score
            if f <= best_f + 0.5:  # Small threshold to allow some exploration
                if next_pos not in cost_so_far or new_g < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_g
                    frontier.put((f, new_g, next_pos))
                    current_frontier.append(next_pos)
                    came_from[next_pos] = current
        
        # Yield current state for visualization
        current_path = []
        curr = current
        while curr is not None:
            current_path.append(curr)
            curr = came_from[curr]
        yield current_path[::-1], visited, current_frontier, steps

def visualise_grid(grid, start, goal, search_generator, title="A* Search"):
    cmap = LinearSegmentedColormap.from_list('black_white', ['white', 'black'], N=256)
    grid_array = np.array(grid)
    
    fig, ax = plt.subplots()
    ax.imshow(grid_array, cmap=cmap)
    
    ax.set_xticks([x - 0.5 for x in range(1, len(grid[0]))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(grid))], minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', size=0)
    
    def update(frame):
        path, visited, frontier, steps = frame
        ax.clear()
        ax.imshow(grid_array, cmap=cmap)
        ax.set_xticks([x - 0.5 for x in range(1, len(grid[0]))], minor=True)
        ax.set_yticks([y - 0.5 for y in range(1, len(grid))], minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        ax.tick_params(which='minor', size=0)
        
        # Display visited nodes
        for position in visited:
            ax.scatter(position[1], position[0], c='yellow', alpha=0.5, 
                      label='Visited' if position == list(visited)[0] else "")
            
        # Display frontier nodes
        for node in frontier:
            ax.scatter(node[1], node[0], c='orange', alpha=0.7, 
                      label='Frontier' if node == frontier[0] else "")
            
        # Display path
        for position in path:
            ax.scatter(position[1], position[0], c='red', 
                      label='Path' if position == path[0] else "")
            
        ax.scatter(start[1], start[0], c='green', label='Start')
        ax.scatter(goal[1], goal[0], c='blue', label='Goal')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_title(f"{title} (Steps: {steps})")
    
    ani = animation.FuncAnimation(fig, update, frames=search_generator, repeat=False)
    plt.show()

if __name__ == "__main__":
    grid = [
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0]
    ]

    start = (0, 0)
    goal = (4, 5)
    
    search = astar(grid, start, goal)
    visualise_grid(grid, start, goal, search)
