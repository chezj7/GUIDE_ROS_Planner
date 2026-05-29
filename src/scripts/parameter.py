CELL_SIZE = 0.4  # meter
NODE_RESOLUTION = 2.4 # meter

FREE = 0
OCCUPIED = 100
UNKNOWN = -1

SENSOR_RANGE = 20  # meter
UTILITY_RANGE = 0.5 * SENSOR_RANGE  # for each node, frontiers in this range will be considered as utility
MIN_UTILITY = 3  # if the number observable frontiers is less than this value, consider it is zero utility
FRONTIER_CELL_SIZE = 1 * CELL_SIZE  # downsample the frontiers based on this value

UPDATING_MAP_SIZE = 4 * SENSOR_RANGE + 4 * NODE_RESOLUTION  # the minimal map that contains all possible updating nodes

NODE_INPUT_DIM = 4
EMBEDDING_DIM = 128
K_SIZE = 25
NODE_PADDING_SIZE = 360  # the number of nodes will be padded to this value

THR_TO_WAYPOINT = 0.1 # meter, the waypoint will be considered as arrived if the robot is closer than this value  fael4——0.1m
THR_NEXT_WAYPOINT = 5 # meter, the planner will try to plan a waypoint farther than this value
THR_GRAPH_HARD_UPDATE = 8 # meter, node and edges in this range will be fully updated

CLUSTER_RANGE = 10 # meter, frontiers will be clustered based on this range

AVOID_OSCILLATION = True # if the planner outputs back and forth waypoints, move to one of them
ENABLE_SAVE_MODE = True # if the planner outputs waypoints in loop, move to the nearest frontier
ENABLE_DSTARLITE = False # Use D*-lite for graph rarefaction instead of A*

BLOCK_SIZE_IN_CELLS = 50  # the number of cells in a block cell_size
UPDATE_WINDOW_SIZE = 100 # the number of cells in a window
BOX_MIN_X = -10.0  # the minimum x coordinate of the box meter
BOX_MAX_X = 130.0  # the maximum x coordinate of the box   
BOX_MIN_Y = -30.0  # the minimum y coordinate of the box
BOX_MAX_Y = 80.0  # the maximum y coordinate of the box