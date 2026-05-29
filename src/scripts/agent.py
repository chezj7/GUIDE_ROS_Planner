import time
from copy import deepcopy

import numpy as np
import torch
import matplotlib.pyplot as plt
import copy
import matplotlib.colors as colors
import parameter
import collections

from utils import *
# from parameter import *
from node_manager import NodeManager


class Agent:
    def __init__(self, policy_net, device='cpu', plot=False):
        self.device = device
        self.policy_net = policy_net
        self.plot = plot

        # location and map
        self.location = None
        self.map_info = None
        self.obs_horizon = policy_net.n_obs_steps
        self.action_horizon = policy_net.n_action_steps

        # map related parameters
        self.cell_size = parameter.CELL_SIZE
        self.node_resolution = parameter.NODE_RESOLUTION
        self.updating_map_size = parameter.UPDATING_MAP_SIZE

        # map and updating map
        self.map_info = None
        self.updating_map_info = None

        # frontiers
        self.frontier = set()

        # hgrid
        self.regions = None
        self.regions_state = None
        self.unknown_centers = None

        # node managers
        self.node_manager = NodeManager()

        # graph
        self.node_coords, self.utility, self.guidepost = None, None, None
        self.node_value = None
        self.raw_node_coords = None
        self.current_index, self.adjacent_matrix, self.neighbor_indices = None, None, None
        self.is_unknown_node = None

        # rarefied graph
        self.key_node_coords, self.key_utility, self.key_guidepost = None, None, None
        self.key_current_index, self.key_adjacent_matrix, self.key_neighbor_indices = None, None, None
        self.is_unknown_node = None
        self.key_occupancy = None
        
        self.planned_path_x = []
        self.planned_path_y = []

    def update_map(self, map_info):
        self.map_info = map_info

    def update_updating_map(self, location):
        # the updating map is the part of the global map that maybe affected by new measurements
        self.updating_map_info = self.get_updating_map(location)

    def update_location(self, location):
        self.location = location
        node = self.node_manager.nodes_dict.find(location.tolist())
        if self.node_manager.nodes_dict.__len__() == 0:
            pass
        else:
            node.data.set_visited()

    def update_frontiers(self):
        self.frontier = get_frontier_in_map(self.updating_map_info)

    def get_updating_map(self, location):
        # the map includes all nodes that may be updating
        updating_map_origin_x = (location[
                                     0] - self.updating_map_size / 2)
        updating_map_origin_y = (location[
                                     1] - self.updating_map_size / 2)

        updating_map_top_x = updating_map_origin_x + self.updating_map_size
        updating_map_top_y = updating_map_origin_y + self.updating_map_size

        min_x = self.map_info.map_origin_x
        min_y = self.map_info.map_origin_y
        max_x = (self.map_info.map_origin_x + self.cell_size * (self.map_info.map.shape[1] - 1))
        max_y = (self.map_info.map_origin_y + self.cell_size * (self.map_info.map.shape[0] - 1))

        if updating_map_origin_x < min_x:
            updating_map_origin_x = min_x
        if updating_map_origin_y < min_y:
            updating_map_origin_y = min_y
        if updating_map_top_x > max_x:
            updating_map_top_x = max_x
        if updating_map_top_y > max_y:
            updating_map_top_y = max_y

        updating_map_origin_x = (updating_map_origin_x // self.cell_size + 1) * self.cell_size
        updating_map_origin_y = (updating_map_origin_y // self.cell_size + 1) * self.cell_size
        updating_map_top_x = (updating_map_top_x // self.cell_size) * self.cell_size
        updating_map_top_y = (updating_map_top_y // self.cell_size) * self.cell_size

        updating_map_origin_x = np.round(updating_map_origin_x, 1)
        updating_map_origin_y = np.round(updating_map_origin_y, 1)
        updating_map_top_x = np.round(updating_map_top_x, 1)
        updating_map_top_y = np.round(updating_map_top_y, 1)

        updating_map_origin = np.array([updating_map_origin_x, updating_map_origin_y])
        updating_map_origin_in_global_map = get_cell_position_from_coords(updating_map_origin, self.map_info)

        updating_map_top = np.array([updating_map_top_x, updating_map_top_y])
        updating_map_top_in_global_map = get_cell_position_from_coords(updating_map_top, self.map_info)

        updating_map = self.map_info.map[
                       updating_map_origin_in_global_map[1]:updating_map_top_in_global_map[1] + 1,
                       updating_map_origin_in_global_map[0]:updating_map_top_in_global_map[0] + 1]

        updating_map_info = MapInfo(updating_map, updating_map_origin_x, updating_map_origin_y, self.cell_size)

        return updating_map_info

    def update_planning_state(self, map_info, location):
        self.update_map(map_info)
        self.update_location(location)
        self.update_updating_map(self.location)
        self.divide_map_into_regions(self.location)
        self.update_frontiers()
        self.node_manager.update_graph(self.location,
                                       self.frontier,
                                       self.updating_map_info,
                                       self.map_info,
                                       self.regions_state,
                                       self.unknown_centers)
        t1 = time.time()
        self.node_manager.get_rarefied_graph(self.location, self.map_info)
        t2 = time.time()
        # print("graph rarefaction", t2 - t1)
        # self.node_coords, self.utility, self.guidepost, self.adjacent_matrix, self.current_index, self.neighbor_indices = \
        #     self.update_observation()
        t1 = time.time()
        self.key_node_coords, self.key_utility, self.key_guidepost,self.key_occupancy,self.key_adjacent_matrix, self.key_current_index, self.key_neighbor_indices, self.is_unknown_node = \
            self.update_key_node_observation(self.map_info,self.location,self.regions_state, self.unknown_centers)
        t2 = time.time()
        # print("update key node graph", t2 - t1)
    
    def divide_map_into_regions(self,location):  
        self.regions,self.regions_state,self.unknown_centers = get_map_into_regions(self.map_info,location)
        # print(f"[INFO] 当前 unknown_centers 数量: {len(self.unknown_centers)}")
        # print("=== Region States ===")
        # for i, row in enumerate(self.regions_state):
        #     row_str = "Row {}: ".format(i)
        #     row_str += ", ".join(str(cell_state) for cell_state in row)
        #     print(row_str)

        # print("\n=== Unknown Region Centers (meters) ===")
        # for idx, coord in enumerate(self.unknown_centers):
        #     print("Unknown center {}: x = {:.2f}, y = {:.2f}".format(idx, coord[0], coord[1]))

        # print("\n=== Region Shapes ===")
        # for i, row in enumerate(self.regions):
        #     for j, block in enumerate(row):
        #         print("Block ({}, {}): shape = {}".format(i, j, block.shape))


        # print(f"[INFO] Regions State: {self.regions_state}")  # 通常是每个区域的状态，例如FREE / OCCUPIED / UNKNOWN

        # print(f"[INFO] Total Unknown Centers: {len(self.unknown_centers)}")
        # for idx, center in enumerate(self.unknown_centers):
        #     print(f"  Unknown Center {idx}: {center}")
        # neighbor_centers= get_neighbor_region_centers_from_point(self.map_info, location, self.regions_state, self.unknown_centers)
        # print(f"Current location: {location}")
        # print(f"Neighbor unknown region centers count: {len(neighbor_centers)}")
        # print("Neighbor unknown region centers coordinates:")
        # for center in neighbor_centers:
        #     print(f"  {center}")
        # print(f"Regions size: rows={len(self.regions)}, cols={len(self.regions[0]) if len(self.regions) > 0 else 0}")
        # print(f"Unknown centers count: {len(self.unknown_centers)}")

        # # 取 location 所在大格子索引
        # row_idx,col_idx = get_region_index_from_point(self.map_info,location)
        # print(f"Location {location} is in region index: {row_idx}, {col_idx}")

        # # 取该大格子八邻域邻居索引
        # neighbors = get_neighboring_regions(self.map_info, row_idx, col_idx)
        # print(f"Neighbors of region {neighbors}")
        # total_regions = sum(len(row) for row in self.regions_state)
        # print(f"Total number of regions: {total_regions}")

        # for i, row in enumerate(self.regions_state):
        #     for j, state in enumerate(row):
        #         print(f"Region ({i},{j}) state: {'FREE' if state == FREE else 'UNKNOWN'}")

        # print(f"Total UNKNOWN region centers: {len(self.unknown_centers)}")
        # for idx, center in enumerate(self.unknown_centers):
        #     print(f"Unknown Center {idx}: {center}")


    def update_observation(self):
        all_node_coords = []
        for node in self.node_manager.nodes_dict.__iter__():
            all_node_coords.append(node.data.coords)
        all_node_coords = np.array(all_node_coords).reshape(-1, 2)
        utility = []
        guidepost = []

        n_nodes = all_node_coords.shape[0]
        adjacent_matrix = np.ones((n_nodes, n_nodes)).astype(int)
        node_coords_to_check = all_node_coords[:, 0] + all_node_coords[:, 1] * 1j
        for i, coords in enumerate(all_node_coords):
            node = self.node_manager.nodes_dict.find((coords[0], coords[1])).data
            utility.append(node.utility)
            guidepost.append(node.visited)
            for neighbor in node.neighbor_set:
                index = np.argwhere(node_coords_to_check == neighbor[0] + neighbor[1] * 1j)
                assert index is not None
                index = index[0][0]
                adjacent_matrix[i, index] = 0

        utility = np.array(utility)
        guidepost = np.array(guidepost)

        current_index = np.argwhere(node_coords_to_check == self.location[0] + self.location[1] * 1j)[0][0]
        neighbor_indices = np.argwhere(adjacent_matrix[current_index] == 0).reshape(-1)
        return all_node_coords, utility, guidepost, adjacent_matrix, current_index, neighbor_indices

    def update_key_node_observation(self,map_info,robot_location,regions_state, unknown_centers):
        t_start = time.time()
        all_key_node_coords = []
        region_to_centers = build_region_to_centers_map(unknown_centers, map_info, parameter.BLOCK_SIZE_IN_CELLS)
        for key_node_coords in self.node_manager.key_node_dict.keys():
            all_key_node_coords.append(np.array(key_node_coords))
        all_key_node_coords = np.array(all_key_node_coords).reshape(-1, 2)

        utility = []
        is_unknown_list = []
        guidepost = []

        n_nodes = all_key_node_coords.shape[0]
        adjacent_matrix = np.ones((n_nodes, n_nodes)).astype(int)
        node_coords_to_check = all_key_node_coords[:, 0] + all_key_node_coords[:, 1] * 1j
        t2 = time.time()
        for i, coords in enumerate(all_key_node_coords):
            node = self.node_manager.key_node_dict[(coords[0], coords[1])]
            utility.append(node.utility)
            is_unknown_list.append(node.is_unknown_node)
            guidepost.append(node.visited)
            for neighbor in node.neighbor_set:
                neighbor = np.array([neighbor[0], neighbor[1]])
                index = np.argwhere(node_coords_to_check == neighbor[0] + neighbor[1] * 1j)
                index = index[0][0]
                adjacent_matrix[i, index] = 0
            
            if node.utility != 0:
                neighbor_centers = get_neighbor_region_centers_from_point_fast(
                map_info, coords, regions_state, region_to_centers)
                for nb_center in neighbor_centers:
                    index_nb = np.argwhere(node_coords_to_check == nb_center[0] + nb_center[1] * 1j)
                    if index_nb.size > 0:
                        index_nb = index_nb[0][0]
                        # 检查是否碰撞（只考虑占据）
                        collision = check_collision_only_occupied(coords, nb_center, map_info)
                        if not collision:
                            adjacent_matrix[i, index_nb] = 2  # 连为2表示utility有效点的邻居连边
                            adjacent_matrix[index_nb, i] = 2
                            # print(f"[KeyNodeObs] Connected key node {coords} to neighbor {nb_center} with utility {node.utility}")

            # if node.is_unknown_node:
            #     neighbor_centers = get_neighbor_region_centers_from_point_fast(
            #     map_info, coords, regions_state, region_to_centers)
            #     for nb_center in neighbor_centers:
            #         index_nb = np.argwhere(node_coords_to_check == nb_center[0] + nb_center[1] * 1j)
            #         if index_nb.size > 0:
            #             index_nb = index_nb[0][0]
            #             # 这里不检查碰撞，直接连边为3
            #             adjacent_matrix[i, index_nb] = 3
        t3 = time.time()
        # print(f"[KeyNodeObs] Constructed adjacency in {t3 - t2:.4f}s")
        t4 = time.time()
        utility = np.array(utility)
        guidepost = np.array(guidepost)
        is_unknown_list = np.array(is_unknown_list)

        current_index = np.argwhere(node_coords_to_check == self.location[0] + self.location[1] * 1j)[0][0]
        neighbor_indices = np.argwhere(adjacent_matrix[current_index] == 0).reshape(-1)
        
        occupancy = np.zeros((n_nodes,1)) 
        # current_index = np.argwhere(node_coords_to_check == self.location[0] + self.location[1] * 1j)[0][0]
        occupancy[current_index] = -1   
        t_end = time.time()
        print(f"[KeyNodeObs] Total update time: {t_end - t_start:.4f}s")
        

        return all_key_node_coords, utility, guidepost, occupancy, adjacent_matrix, current_index, neighbor_indices,is_unknown_list
    

    
    def get_observation(self, robot_location):

        node_coords = deepcopy(self.key_node_coords)
        node_utility = self.key_utility.reshape(-1, 1)
        node_guidepost = self.key_guidepost.reshape(-1, 1)
        node_occupancy = self.key_occupancy.reshape(-1, 1)
        node_is_unknown = self.is_unknown_node.reshape(-1, 1)
        current_index = self.key_current_index
        edge_mask = self.key_adjacent_matrix
        current_edge = self.key_neighbor_indices
        n_node = node_coords.shape[0]

        node_coords[current_index] = robot_location

        current_node_coords = robot_location
        node_coords = np.concatenate((node_coords[:, 0].reshape(-1, 1) - current_node_coords[0],
                                      node_coords[:, 1].reshape(-1, 1) - current_node_coords[1]),
                                     axis=-1) / parameter.UPDATING_MAP_SIZE / 2
        node_utility = node_utility / (parameter.UTILITY_RANGE * 3.14 // parameter.FRONTIER_CELL_SIZE)
        node_inputs = np.concatenate((node_coords, node_utility, node_guidepost, node_occupancy,node_is_unknown), axis=1)#,node_is_unknown)
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)

        node_padding_size = parameter.NODE_PADDING_SIZE
        assert node_coords.shape[0] < node_padding_size
        padding = torch.nn.ZeroPad2d((0, 0, 0, node_padding_size - n_node))
        node_inputs = padding(node_inputs)
        node_padding_mask = torch.zeros((1, 1, n_node), dtype=torch.int16).to(self.device)
        node_padding = torch.ones((1, 1, node_padding_size - n_node), dtype=torch.int16).to(
            self.device)
        node_padding_mask = torch.cat((node_padding_mask, node_padding), dim=-1)

        current_index = torch.tensor([current_index]).reshape(1, 1, 1).to(self.device)

        edge_mask = torch.tensor(edge_mask).unsqueeze(0).to(self.device)
        
        padding = torch.nn.ConstantPad2d(
            (0, node_padding_size - n_node, 0, node_padding_size - n_node), 1)
        edge_mask = padding(edge_mask)

        current_in_edge = np.argwhere(current_edge == current_index)[0][0]
        
        current_edge = torch.tensor(current_edge).unsqueeze(0)
        
        K_SIZE = 25 
        k_size = current_edge.size()[-1]
        
        padding = torch.nn.ConstantPad1d((0, K_SIZE - k_size), 0)
        current_edge = padding(current_edge)
        current_edge = current_edge.unsqueeze(-1)

        
        edge_padding_mask = torch.zeros((1, 1, k_size), dtype=torch.int16).to(self.device)
        edge_padding_mask[0, 0, current_in_edge] = 1
        padding = torch.nn.ConstantPad1d((0, K_SIZE - k_size), 1)
        edge_padding_mask = padding(edge_padding_mask)
        # print("node_coords:", node_coords.shape)
        # print("node_utility:", node_utility.shape)
        # print("node_guidepost:", node_guidepost.shape)
        # print("node_occupancy:", node_occupancy.shape)
        # print("node_inputs shape:", node_inputs.shape)


        return [node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask]

    def get_next_observation(self, next_node_index, observation):
        node_inputs, _, edge_mask, curren_index, _, _ = observation
        next_edge = torch.argwhere(edge_mask[0, next_node_index] == 0).flatten()
        next_in_edge = torch.argwhere(next_edge == next_node_index).item()
        curren_in_edge = torch.argwhere(next_edge == curren_index.item()).item()
        k_size = next_edge.size()[-1]
        next_edge = next_edge.unsqueeze(-1).unsqueeze(0)
        next_node_index = torch.tensor([next_node_index]).reshape(1, 1, 1).to(self.device)
        edge_padding_mask = torch.zeros((1, 1, k_size), dtype=torch.int16).to(self.device)
        edge_padding_mask[0, 0, next_in_edge] = 1
        edge_padding_mask[0, 0, curren_in_edge] = 1
        return node_inputs, None, edge_mask, next_node_index, next_edge, edge_padding_mask

    # def select_next_waypoint(self, observation, greedy=True):
    #     _, _, _, _, current_edge, _ = observation
    #     with torch.no_grad():
    #         logp = self.policy_net(*observation)

    #     if greedy:
    #         action_index = torch.argmax(logp, dim=1).long()
    #     else:
    #         action_index = torch.multinomial(logp.exp(), 1).long().squeeze(1)
    #     next_node_index = current_edge[0, action_index.item(), 0].item()
    #     next_position = self.key_node_coords[next_node_index]
    #     # print("available next positions:", self.key_node_coords[current_edge[0].numpy()].reshape(-1, 2))

    #     return next_position, next_node_index
    
    def select_next_waypoint(self, obs_dict,robot_node_location, greedy=True):
       
        t_start = time.time()
        with torch.no_grad():
            action_dict = self.policy_net.predict_action(obs_dict)
        
        t_policy = time.time()
        
        action_pred = action_dict['action_pred'].squeeze(0).cpu().numpy()
        action_pred = np.round(action_pred / parameter.NODE_RESOLUTION) * parameter.NODE_RESOLUTION  #NODE_RESOLUTION
        self.planned_path_x.clear()
        self.planned_path_y.clear()

        
        # print("Processed action_pred (after rounding):", action_pred)
        # print("robot_current_position", robot_node_location)


        
        start = self.obs_horizon - 1
        end = start + self.action_horizon + 1
        action = action_pred[start:end,:] # (action_horizon, action_dim)
        # print("=== Predicted Action Sequence ===")
        # for i, act in enumerate(action_pred[start:end]):
        #     print(f"Step {i}: Δx = {act[0]:.2f}, Δy = {act[1]:.2f}")
        # print("=================================")
        planned_location = deepcopy(robot_node_location)
        self.planned_path_x.append(planned_location[0])
        self.planned_path_y.append(planned_location[1])
        for i in range(start, len(action_pred)):
            planned_location += action_pred[i]
            self.planned_path_x.append(planned_location[0])
            self.planned_path_y.append(planned_location[1])
        # print("Planned Path X:", self.planned_path_x)
        # print("Planned Path Y:", self.planned_path_y)
        t_path_construct = time.time()
        

        next_positions = []
        next_node_indices = []
        current_pos = robot_node_location.copy()
        current_node = self.node_manager.nodes_dict.find(robot_node_location.tolist()).data
        for i in range(self.action_horizon + 1):
            candidate_pos = current_pos + action[i]
            # print(f"robot_node_location: {robot_node_location}")
            # print(f"selected_coord: {action[0]}")
            
            ## Collision avoidance
            # check if selected_coord is a valid neighbour of current node
            if not any(np.all(candidate_pos == neighbor) for neighbor in current_node.neighbor_set):
                # print("Collision Detected!")
                # Vectors of 3 future positions from current position # HACK fixed number here
                direction_vectors = np.cumsum(action_pred[start + i: start + i + 3], axis=0)
                best_neighbor = None
                best_average_angle = float('inf')
                # print(f"Direction Vectors: {direction_vectors}")
                for neighbor_coords in current_node.neighbor_set:
                    # skip current robot location
                    if np.all(neighbor_coords == current_pos):
                        continue 
                    neighbor_direction = neighbor_coords - current_pos
                    # print(f"Neighbor Direction: {neighbor_direction}")
                    angles = []
                    for direction_vector in direction_vectors:
                        direction_magnitude = np.linalg.norm(direction_vector)
                        neighbor_magnitude = np.linalg.norm(neighbor_direction)
                        if direction_magnitude == 0 or neighbor_magnitude == 0: # skip zero vectors
                            continue
                        angle = np.arctan2(np.linalg.det([direction_vector, neighbor_direction]), np.dot(direction_vector, neighbor_direction))
                        angles.append(angle)
                    weights = np.arange(len(angles), 0, -1)
                    weighted_average_angle = np.average(np.abs(angles), weights=weights)  # Use absolute values for magnitude
                    # print(f"Weighted Average Angle: {weighted_average_angle}")
                    if weighted_average_angle < best_average_angle:
                        best_average_angle = weighted_average_angle
                        best_neighbor = neighbor_coords
                # print(f"Best Neighbor: {best_neighbor}, action: {best_neighbor - self.env.robot_locations[0]}")
                if best_neighbor is not None:
                    candidate_pos = best_neighbor
            
            # current_pos = robot_location
            # next_position = current_pos + action
            # next_node_index = np.argwhere(self.key_node_coords == next_position).flatten()
            # if next_node_index.size == 0:
            #     next_node_index = -1
            # else:
            #     next_node_index = next_node_index[0]
            next_positions.append(np.array(candidate_pos))
            current_pos = np.array(candidate_pos)
            current_node = self.node_manager.nodes_dict.find(current_pos.tolist()).data

        # ---- 方向一致性檢查 ----
        if len(next_positions) >= 2:
            dir1 = next_positions[0] - robot_node_location
            dir2 = next_positions[1] - next_positions[0]

            norm1 = np.linalg.norm(dir1)
            norm2 = np.linalg.norm(dir2)

            if norm1 > 1e-3 and norm2 > 1e-3:
                cos_angle = np.dot(dir1, dir2) / (norm1 * norm2)
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

                if angle > np.deg2rad(60):  # 超過60度則丟棄第二個點
                    next_positions = next_positions[:1]
            next_node_indices = [np.argwhere(self.key_node_coords == pos).flatten() for pos in next_positions]
        t_end = time.time()
        # print(f"\n=== Timing Info ===")
        # print(f"Policy inference time      : {t_policy - t_start:.4f}s")
        # print(f"Path construction time     : {t_path_construct - t_policy:.4f}s")
        # print(f"Next waypoint selection time: {t_end - t_path_construct:.4f}s")
        # print(f"Total select_next_waypoint time: {t_end - t_start:.4f}s")
        # print("====================\n")
            

        return next_positions, next_node_indices, self.planned_path_x, self.planned_path_y
    
    def plot_env(self, step, robot_location):
        # quite slow, only use it to debug

        # plt.switch_backend('TKAgg')
        plt.ion()
        plt.clf()

        plt.subplot(1, 2, 1)
        nodes = get_cell_position_from_coords(self.key_node_coords, self.map_info).reshape(-1, 2)
        if len(self.frontier) > 0:
            frontiers = get_cell_position_from_coords(np.array(list(self.frontier)), self.map_info).reshape(-1, 2)
            plt.scatter(frontiers[:, 0], frontiers[:, 1], c='r', s=2)
        robot = get_cell_position_from_coords(robot_location, self.map_info)
        # plt.imshow(self.map_info.map, cmap='gray')
        plt.imshow(self.map_info.map + 1.1, cmap='gray_r', norm=colors.LogNorm())
        plt.axis('off')
        plt.scatter(nodes[:, 0], nodes[:, 1], c=self.key_utility, zorder=2)
        for node, utility in zip(nodes, self.key_utility):
            plt.text(node[0], node[1], str(utility), zorder=3)
        plt.plot(robot[0], robot[1], 'mo', markersize=16, zorder=5)
        for coords in self.key_node_coords:
            node = self.node_manager.key_node_dict[(coords[0], coords[1])]
            for neighbor_coords in node.neighbor_set:
                end = (np.array(neighbor_coords) - coords) / 2 + coords
                plt.plot((np.array([coords[0], end[0]]) - self.map_info.map_origin_x) / self.cell_size,
                         (np.array([coords[1], end[1]]) - self.map_info.map_origin_y) / self.cell_size, 'tan', zorder=1)

        plt.subplot(1, 2, 2)
        nodes = get_cell_position_from_coords(self.node_coords, self.map_info)
        if len(self.frontier) > 0:
            frontiers = get_cell_position_from_coords(np.array(list(self.frontier)), self.map_info).reshape(-1, 2)
            plt.scatter(frontiers[:, 0], frontiers[:, 1], c='r', s=2)
        robot = get_cell_position_from_coords(robot_location, self.map_info)
        plt.imshow(self.map_info.map + 1.1, cmap='gray_r', norm=colors.LogNorm())
        plt.axis('off')
        plt.scatter(nodes[:, 0], nodes[:, 1], c=self.utility, zorder=2)
        for node, utility in zip(nodes, self.utility):
            plt.text(node[0], node[1], str(utility), zorder=3)
        plt.plot(robot[0], robot[1], 'mo', markersize=16, zorder=5)
        for coords in self.node_coords:
            node = self.node_manager.nodes_dict.find(coords.tolist()).data
            for neighbor_coords in node.neighbor_set:
                end = (np.array(neighbor_coords) - coords) / 2 + coords
                plt.plot((np.array([coords[0], end[0]]) - self.map_info.map_origin_x) / self.cell_size,
                         (np.array([coords[1], end[1]]) - self.map_info.map_origin_y) / self.cell_size, 'tan', zorder=1)

        plt.pause(1e-3)

        plt.savefig('{}/{}_samples.png'.format(f'gifs', step), dpi=150)
        # plt.close()
