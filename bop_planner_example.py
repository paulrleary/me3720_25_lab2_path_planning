import numpy as np

from path_planners import AStarPlanner, RRTPlanner, RRTStarPlanner
from matplotlib import pyplot as plt


def plot_planning_results(obstacles, start, goal,
                          astar_path, rrt_nodes, rrt_path,
                          rrt_star_nodes, rrt_star_path):
    plt.figure(figsize=(10, 10))
    # Draw each polygon obstacle
    for poly in obstacles:
        px, py = zip(*(poly + [poly[0]]))  # close the polygon
        plt.fill(px, py, color='k', alpha=0.3)
    def plot_tree(nodes, color, alpha=0.3):
        for node in nodes:
            if node.parent:
                plt.plot([node.x, node.parent.x], [node.y, node.parent.y], color=color, alpha=alpha)
    plot_tree(rrt_nodes, 'blue')
    plot_tree(rrt_star_nodes, 'green')
    if astar_path:
        px, py = zip(*astar_path)
        plt.plot(px, py, 'm-', linewidth=3, label='A* Path')
    if rrt_path:
        px, py = zip(*rrt_path)
        plt.plot(px, py, 'r-', linewidth=2, label='RRT Path')
    if rrt_star_path:
        px, py = zip(*rrt_star_path)
        plt.plot(px, py, 'y-', linewidth=2, label='RRT* Path')
    plt.plot(start[0], start[1], 'go', markersize=12, label='Start')
    plt.plot(goal[0], goal[1], 'ro', markersize=12, label='Goal')
    plt.legend()
    plt.title("Path Planning: A*, RRT, RRT* with Polygonal Obstacle")
    plt.grid(True)
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    # Define the rectangular obstacle as a polygon (counterclockwise or clockwise)
    BOPstacle_inflated = [(-4, -4), (-4,7), (7,7), (7, -4)]
    rect_poly = [(-3, 0), (3, 0), (3, 6), (-3, 6)]
    obstacles = [BOPstacle_inflated]
    start = (-10, 2)
    goal = (10, 10)
    step_size = 1.0
    neighbor_radius = 2.0
    max_iter = 2000
    grid_res = 0.2
    
    import random
    random_seed = 42  # Or any integer you want
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    astar_planner = AStarPlanner(obstacles, start, goal,
                                 grid_res=grid_res, max_iter=10000)
    rrt_planner = RRTPlanner(obstacles, start, goal,
                             step_size=step_size, neighbor_radius=neighbor_radius, max_iter=max_iter)
    rrt_star_planner = RRTStarPlanner(obstacles, start, goal,
                                      step_size=step_size,
                                      neighbor_radius=neighbor_radius,
                                      max_iter=max_iter)
    astar_tree, astar_path = astar_planner.plan()
    rrt_nodes, rrt_path = rrt_planner.plan()
    rrt_star_nodes, rrt_star_path = rrt_star_planner.plan()
    plot_planning_results(obstacles, start, goal,
                          astar_path, rrt_nodes, rrt_path,
                          rrt_star_nodes, rrt_star_path)
