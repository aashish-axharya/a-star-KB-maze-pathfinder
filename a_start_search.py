from queue import PriorityQueue
import math
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
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
        
        # Check if the neighbor is within bounds and not a wall (1)
        if (0 <= new_row < rows and 
            0 <= new_col < cols and 
            grid[new_row][new_col] != 1):
            neighbors.append((new_row, new_col))
    
    return neighbors

def astar(grid, start, goal):
    frontier = PriorityQueue()
    frontier.put((0, 0, start))  # (f_score, g_score, position)
    came_from = {start: None}
    cost_so_far = {start: 0}
    visited = set()

    steps = 0
    while not frontier.empty():
        _, current_cost, current = frontier.get()
        if current in visited:
            continue
        visited.add(current)  # Mark node as visited
        steps += 1

        # If the goal is reached, reconstruct the path
        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            yield path[::-1], visited, [], steps
            return

        neighbors = get_neighbors(grid, current)
        current_frontier = []
        for next_pos in neighbors:
            new_cost = cost_so_far[current] + 1  # Assume uniform cost for moving to a neighbor
            if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                cost_so_far[next_pos] = new_cost
                h_score = euclidean_distance(next_pos, goal)
                f_score = new_cost + h_score
                frontier.put((f_score, new_cost, next_pos))
                current_frontier.append(next_pos)
                came_from[next_pos] = current

        # Yield the current state for visualization
        yield [], visited, current_frontier, steps


def visualise_grid(grid, start, goal, search_generator, title="Grid"):
    # Define a colormap: smooth gradient from white to black
    cmap = LinearSegmentedColormap.from_list('black_white', ['white', 'black'], N=256)

    # Convert grid to a numpy array for easier manipulation
    grid_array = np.array(grid)

    # Find the maximum non-inf value in the grid
    max_value = np.max(grid_array[grid_array != float('inf')])
    inf_replacement = max_value + 1
    grid_array[grid_array == float('inf')] = inf_replacement

    # Normalize the grid values for the colormap
    norm = Normalize(vmin=0, vmax=inf_replacement)

    # Create a figure and axis
    fig, ax = plt.subplots()
    cax = ax.imshow(grid_array, cmap=cmap, norm=norm)

    # Set grid lines
    ax.set_xticks([x - 0.5 for x in range(1, len(grid[0]))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(grid))], minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', size=0)

    def update(frame):
        path, visited, frontier, steps = frame
        ax.clear()
        ax.imshow(grid_array, cmap=cmap, norm=norm)
        ax.set_xticks([x - 0.5 for x in range(1, len(grid[0]))], minor=True)
        ax.set_yticks([y - 0.5 for y in range(1, len(grid))], minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        ax.tick_params(which='minor', size=0)

        # Display visited nodes
        for position in visited:
            ax.scatter(position[1], position[0], c='yellow')

        # Display frontier nodes
        for node in frontier:
            ax.scatter(node[1], node[0], c='orange')

        # Display path
        for position in path:
            ax.scatter(position[1], position[0], c='red')

        ax.scatter(start[1], start[0], c='green', label='Start')
        ax.scatter(goal[1], goal[0], c='blue', label='Goal')
        ax.legend()
        ax.set_title(f"{title} (Steps: {steps})")

    ani = animation.FuncAnimation(fig, update, frames=search_generator, repeat=False)
    plt.show()

grid = [
    [0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 0]
]

start = (0, 0)
goal = (4, 5)

a_star_search = astar(grid, start, goal)
visualise_grid(grid, start, goal, a_star_search)
