from planners_open import RRTPlanner, RRTStarPlanner, AStarPlanner

def setup_planners(q_init, q_goal):
    obstacles = [[(-4, -4), (-4, 7), (7, 7), (7, -4)]]
    start = (q_init[0], q_init[1])
    goal = (q_goal[0], q_goal[1])
    astar_planner = AStarPlanner(obstacles, start, goal, grid_res=0.2, max_iter=2000)
    rrt_star_planner = RRTStarPlanner(obstacles, start, goal, step_size=1.0, neighbor_radius=2.0, max_iter=2000)
    astar_tree, astar_path = astar_planner.plan()
    rrt_star_nodes, rrt_star_path = rrt_star_planner.plan()
    return astar_path, rrt_star_path

def get_track_list(rrt_star_path):
    return [np.array([*wp, 0]) for wp in rrt_star_path]