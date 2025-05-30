import math
import random
import heapq
import matplotlib.pyplot as plt

# Utility: Check if two line segments (p1, p2) and (q1, q2) intersect
def segments_intersect(p1, p2, q1, q2):
    def ccw(a, b, c):
        return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])
    return (ccw(p1, q1, q2) != ccw(p2, q1, q2)) and (ccw(p1, p2, q1) != ccw(p1, p2, q2))

# Check if a segment intersects a polygon (polygon is a list of (x, y) vertices)
def segment_polygon_collision(p1, p2, polygon):
    n = len(polygon)
    for i in range(n):
        q1 = polygon[i]
        q2 = polygon[(i+1) % n]
        if segments_intersect(p1, p2, q1, q2):
            return True
    return False

# Returns True if the segment from start to end does NOT intersect any polygon obstacle
def collision_free(start, end, obstacles):
    for poly in obstacles:
        if segment_polygon_collision(start, end, poly):
            return False
    return True

class Node:
    def __init__(self, x, y, parent=None, cost=0.0):
        self.x = x
        self.y = y
        self.parent = parent
        self.cost = cost

class PathPlannerBase:
    def __init__(self, obstacles, start, goal,
                 step_size=1.0, neighbor_radius=2.0, max_iter=1000):
        self.obstacles = obstacles
        self.start = start
        self.goal = goal
        self.step_size = step_size
        self.neighbor_radius = neighbor_radius
        self.max_iter = max_iter

    def collision_free(self, start, end):
        return collision_free(start, end, self.obstacles)

class AStarPlanner(PathPlannerBase):
    """A* planner: systematic, optimal, on a fine grid for continuous space."""
    def __init__(self, obstacles, start, goal,
                 grid_res=0.2, max_iter=10000):
        super().__init__(obstacles, start, goal, step_size=grid_res, max_iter=max_iter)
        self.grid_res = grid_res

    def plan(self):
        # Discretize the space for A* (fine grid)
        margin = 5
        xs = [self.start[0], self.goal[0]] + [v[0] for poly in self.obstacles for v in poly]
        ys = [self.start[1], self.goal[1]] + [v[1] for poly in self.obstacles for v in poly]
        xmin, xmax = min(xs)-margin, max(xs)+margin
        ymin, ymax = min(ys)-margin, max(ys)+margin

        def to_grid(p):
            gx = int(round((p[0] - xmin) / self.grid_res))
            gy = int(round((p[1] - ymin) / self.grid_res))
            return gx, gy

        def from_grid(gp):
            x = gp[0] * self.grid_res + xmin
            y = gp[1] * self.grid_res + ymin
            return x, y

        start_g = to_grid(self.start)
        goal_g = to_grid(self.goal)

        class AStarNode:
            def __init__(self, gx, gy):
                self.gx = gx
                self.gy = gy
                self.g = float('inf')
                self.h = 0
                self.parent = None
            def __lt__(self, other):
                return (self.g + self.h) < (other.g + other.h)

        open_list = []
        start_node = AStarNode(*start_g)
        goal_node = AStarNode(*goal_g)
        heapq.heappush(open_list, (0, start_node))
        start_node.g = 0
        start_node.h = math.hypot(start_node.gx - goal_node.gx, start_node.gy - goal_node.gy)
        closed_set = set()
        nodes_map = {}

        while open_list:
            _, current = heapq.heappop(open_list)
            if (current.gx, current.gy) == (goal_node.gx, goal_node.gy):
                # Reconstruct path
                path = []
                while current:
                    path.append(from_grid((current.gx, current.gy)))
                    current = current.parent
                return [], path[::-1]
            closed_set.add((current.gx, current.gy))
            nodes_map[(current.gx, current.gy)] = current
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = current.gx + dx, current.gy + dy
                    neighbor_pos = from_grid((nx, ny))
                    if not self.collision_free(from_grid((current.gx, current.gy)), neighbor_pos):
                        continue
                    if (nx, ny) in closed_set:
                        continue
                    tentative_g = current.g + math.hypot(dx, dy)
                    neighbor = nodes_map.get((nx, ny), AStarNode(nx, ny))
                    if tentative_g < neighbor.g:
                        neighbor.g = tentative_g
                        neighbor.h = math.hypot(nx - goal_node.gx, ny - goal_node.gy)
                        neighbor.parent = current
                        nodes_map[(nx, ny)] = neighbor
                        heapq.heappush(open_list, (neighbor.g + neighbor.h, neighbor))
        return [], []

class RRTPlanner(PathPlannerBase):
    """RRT: rapid, feasible, not optimal, continuous."""
    def plan(self):
        nodes = [Node(*self.start)]
        goal_node = Node(*self.goal)
        for _ in range(self.max_iter):
            rand_x = random.uniform(self.start[0]-5, self.goal[0]+5)
            rand_y = random.uniform(self.start[1]-5, self.goal[1]+5)
            nearest = min(nodes, key=lambda n: math.hypot(n.x - rand_x, n.y - rand_y))
            theta = math.atan2(rand_y - nearest.y, rand_x - nearest.x)
            new_x = nearest.x + self.step_size * math.cos(theta)
            new_y = nearest.y + self.step_size * math.sin(theta)
            if self.collision_free((nearest.x, nearest.y), (new_x, new_y)):
                new_node = Node(new_x, new_y, parent=nearest)
                nodes.append(new_node)
                if math.hypot(new_x - goal_node.x, new_y - goal_node.y) < self.step_size:
                    if self.collision_free((new_x, new_y), (goal_node.x, goal_node.y)):
                        goal_node.parent = new_node
                        nodes.append(goal_node)
                        break
        path = []
        current = goal_node if goal_node.parent else None
        while current:
            path.append((current.x, current.y))
            current = current.parent
        return nodes, path[::-1] if path else []

class RRTStarPlanner(PathPlannerBase):
    """RRT*: rewires tree for optimality, continuous."""
    def plan(self):
        nodes = [Node(*self.start, cost=0.0)]
        goal_node = Node(*self.goal)
        for _ in range(self.max_iter):
            rand_x = random.uniform(self.start[0]-5, self.goal[0]+5)
            rand_y = random.uniform(self.start[1]-5, self.goal[1]+5)
            nearest = min(nodes, key=lambda n: math.hypot(n.x - rand_x, n.y - rand_y))
            theta = math.atan2(rand_y - nearest.y, rand_x - nearest.x)
            new_x = nearest.x + self.step_size * math.cos(theta)
            new_y = nearest.y + self.step_size * math.sin(theta)
            if self.collision_free((nearest.x, nearest.y), (new_x, new_y)):
                new_node = Node(new_x, new_y)
                new_node.cost = nearest.cost + math.hypot(nearest.x - new_x, nearest.y - new_y)
                neighbors = [n for n in nodes if math.hypot(n.x - new_x, n.y - new_y) < self.neighbor_radius]
                min_cost = new_node.cost
                best_parent = nearest
                for neighbor in neighbors:
                    dist = math.hypot(neighbor.x - new_x, neighbor.y - new_y)
                    cost = neighbor.cost + dist
                    if cost < min_cost and self.collision_free((neighbor.x, neighbor.y), (new_x, new_y)):
                        min_cost = cost
                        best_parent = neighbor
                new_node.parent = best_parent
                new_node.cost = min_cost
                nodes.append(new_node)
                for neighbor in neighbors:
                    dist = math.hypot(new_node.x - neighbor.x, new_node.y - neighbor.y)
                    cost = new_node.cost + dist
                    if cost < neighbor.cost and self.collision_free((new_node.x, new_node.y), (neighbor.x, neighbor.y)):
                        neighbor.parent = new_node
                        neighbor.cost = cost
                        self.update_descendants(neighbor, nodes)
                if math.hypot(new_x - goal_node.x, new_y - goal_node.y) < self.step_size:
                    if self.collision_free((new_x, new_y), (goal_node.x, goal_node.y)):
                        goal_node.parent = new_node
                        goal_node.cost = new_node.cost + math.hypot(new_node.x - goal_node.x, new_node.y - goal_node.y)
                        nodes.append(goal_node)
                        break
        path = []
        current = goal_node if goal_node.parent else None
        while current:
            path.append((current.x, current.y))
            current = current.parent
        return nodes, path[::-1] if path else []

    def update_descendants(self, node, nodes):
        children = [n for n in nodes if n.parent == node]
        for child in children:
            if not self.collision_free((node.x, node.y), (child.x, child.y)):
                child.parent = None
                continue
            dist = math.hypot(node.x - child.x, node.y - child.y)
            child.cost = node.cost + dist
            self.update_descendants(child, nodes)