from dataclasses import dataclass
from typing import List, Set, Tuple, Optional
import math

@dataclass(frozen=True)
class Predicate:
    """Base class for FOL predicates"""
    name: str
    arguments: tuple
    
    def __str__(self):
        return f"{self.name}({', '.join(map(str, self.arguments))})"

@dataclass
class Rule:
    """Represents a FOL rule with premises and conclusion"""
    premises: List[Predicate]
    conclusion: Predicate

class FOLMazeSolver:
    def __init__(self, grid: List[List[int]], start: Tuple[int, int], goal: Tuple[int, int]):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.start = start
        self.goal = goal
        self.knowledge_base: Set[Predicate] = set()
        self.rules: List[Rule] = []
        self.initialize_knowledge_base()
        
    def initialize_knowledge_base(self):
        """Initialize the knowledge base with facts about the maze"""
        # Add cell existence and properties
        for x in range(self.rows):
            for y in range(self.cols):
                # Cell existence
                self.knowledge_base.add(Predicate("Cell", (x, y)))
                
                # Cell properties (Free or Wall)
                if self.grid[x][y] == 0:
                    self.knowledge_base.add(Predicate("Free", (x, y)))
                else:
                    self.knowledge_base.add(Predicate("Wall", (x, y)))
        
        # Add start and goal
        self.knowledge_base.add(Predicate("Start", self.start))
        self.knowledge_base.add(Predicate("Goal", self.goal))
        
        # Initialize rules
        self.initialize_rules()
    
    def initialize_rules(self):
        """Initialize the FOL rules for maze solving"""
        # Rule 1: Adjacency rules for all possible directions
        self.add_adjacency_rules()
        
        # Rule 2: CanMove rules based on Free cells and Adjacency
        self.add_movement_rules()
        
        # Rule 3: PreferMove rules based on distance to goal
        self.add_preference_rules()
    
    def add_adjacency_rules(self):
        """Add rules for cell adjacency"""
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        for x in range(self.rows):
            for y in range(self.cols):
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.rows and 0 <= ny < self.cols:
                        self.knowledge_base.add(
                            Predicate("Adjacent", (x, y, nx, ny))
                        )
    
    def add_movement_rules(self):
        """Add rules for possible movements"""
        for p1 in self.knowledge_base:
            if p1.name == "Adjacent":
                x, y, x2, y2 = p1.arguments
                # Rule: If cell is free and adjacent, you can move there
                self.rules.append(Rule(
                    premises=[
                        Predicate("Adjacent", (x, y, x2, y2)),
                        Predicate("Free", (x2, y2))
                    ],
                    conclusion=Predicate("CanMove", (x, y, x2, y2))
                ))
    
    def calculate_distance(self, x1: int, y1: int, x2: int, y2: int) -> float:
        """Calculate squared Euclidean distance between two points"""
        return (x1 - x2)**2 + (y1 - y2)**2
    
    def add_preference_rules(self):
        """Add rules for move preferences based on distance to goal"""
        gx, gy = self.goal
        
        for p1 in self.knowledge_base:
            if p1.name == "CanMove":
                x, y, nx, ny = p1.arguments
                # Only add preference if move gets closer to goal
                curr_dist = self.calculate_distance(x, y, gx, gy)
                new_dist = self.calculate_distance(nx, ny, gx, gy)
                
                if new_dist < curr_dist:
                    self.rules.append(Rule(
                        premises=[Predicate("CanMove", (x, y, nx, ny))],
                        conclusion=Predicate("PreferMove", (x, y, nx, ny))
                    ))
    
    def forward_chain(self) -> Set[Predicate]:
        """Apply forward chaining to derive new facts"""
        new_facts = set()
        changed = True
        
        while changed:
            changed = False
            for rule in self.rules:
                if all(premise in self.knowledge_base for premise in rule.premises):
                    if rule.conclusion not in self.knowledge_base:
                        new_facts.add(rule.conclusion)
                        self.knowledge_base.add(rule.conclusion)
                        changed = True
        
        return new_facts
    
    def find_path(self) -> Optional[List[Tuple[int, int]]]:
        """Find path using FOL inference"""
        # Apply forward chaining to derive all possible moves
        self.forward_chain()
        
        # Use derived facts to construct path
        path = [self.start]
        current = self.start
        visited = {self.start}
        
        while current != self.goal:
            next_move = None
            best_distance = float('inf')
            
            # Look for preferred moves first
            for fact in self.knowledge_base:
                if (fact.name == "PreferMove" and 
                    fact.arguments[:2] == current and 
                    fact.arguments[2:] not in visited):
                    nx, ny = fact.arguments[2:]
                    dist = self.calculate_distance(nx, ny, self.goal[0], self.goal[1])
                    if dist < best_distance:
                        best_distance = dist
                        next_move = (nx, ny)
            
            # If no preferred move, look for any valid move
            if next_move is None:
                for fact in self.knowledge_base:
                    if (fact.name == "CanMove" and 
                        fact.arguments[:2] == current and 
                        fact.arguments[2:] not in visited):
                        nx, ny = fact.arguments[2:]
                        dist = self.calculate_distance(nx, ny, self.goal[0], self.goal[1])
                        if dist < best_distance:
                            best_distance = dist
                            next_move = (nx, ny)
            
            if next_move is None:
                return None  # No path found
            
            current = next_move
            path.append(current)
            visited.add(current)
            
            if len(path) > self.rows * self.cols:  # Prevent infinite loops
                return None
        
        return path

    def explain_reasoning(self, path: List[Tuple[int, int]]) -> List[str]:
        """Explain the logical reasoning for the path"""
        explanations = []
        if not path:
            return ["No valid path found using logical inference."]
        
        for i in range(len(path) - 1):
            current, next_pos = path[i], path[i + 1]
            
            # Check if this was a preferred move
            preferred = Predicate("PreferMove", (current[0], current[1], 
                                               next_pos[0], next_pos[1]))
            
            if preferred in self.knowledge_base:
                explanations.append(
                    f"Step {i+1}: Move from {current} to {next_pos} was preferred "
                    f"because it reduces distance to goal"
                )
            else:
                explanations.append(
                    f"Step {i+1}: Move from {current} to {next_pos} was valid "
                    f"based on adjacency and free cell rules"
                )
        
        return explanations

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
    
    solver = FOLMazeSolver(grid, start, goal)
    path = solver.find_path()
    
    if path:
        print("Path found:", path)
        print("\nReasoning:")
        for explanation in solver.explain_reasoning(path):
            print(explanation)
    else:
        print("No path found")