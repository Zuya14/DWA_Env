import math
from collections import deque

import numpy as np

EXTEND_AREA = 0.0

class GridScan:

    def __init__(self, lidar_maxLen, resolution, scale=1.0):
        self.lidar_maxLen = lidar_maxLen
        self.resolution = resolution 

        self.min_w = round(-self.lidar_maxLen - EXTEND_AREA / 2.0)
        self.max_w = round(self.lidar_maxLen + EXTEND_AREA / 2.0)
        self.grid_num = int(round(self.max_w - self.min_w) / resolution)

    def bresenham(self, start, end):
        """
        Implementation of Bresenham's line drawing algorithm
        See en.wikipedia.org/wiki/Bresenham's_line_algorithm
        Bresenham's Line Algorithm
        Produces a np.array from start and end (original from roguebasin.com)
        >>> points1 = bresenham((4, 4), (6, 10))
        >>> print(points1)
        np.array([[4,4], [4,5], [5,6], [5,7], [5,8], [6,9], [6,10]])
        """
        # setup initial conditions
        x1, y1 = start
        x2, y2 = end
        dx = x2 - x1
        dy = y2 - y1
        is_steep = abs(dy) > abs(dx)  # determine how steep the line is
        if is_steep:  # rotate line
            x1, y1 = y1, x1
            x2, y2 = y2, x2
        # swap start and end points if necessary and store swap state
        swapped = False
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            swapped = True
        dx = x2 - x1  # recalculate differentials
        dy = y2 - y1  # recalculate differentials
        error = int(dx / 2.0)  # calculate error
        y_step = 1 if y1 < y2 else -1
        # iterate over bounding box generating points between start and end
        y = y1
        points = []
        for x in range(x1, x2 + 1):
            coord = [y, x] if is_steep else (x, y)
            points.append(coord)
            error -= abs(dy)
            if error < 0:
                y += y_step
                error += dx
        if swapped:  # reverse the list if the coordinates were swapped
            points.reverse()
        points = np.array(points)
        return points

    def generate_maps(self, ox, oy):
        
        space_map = np.ones((self.grid_num, self.grid_num)) 
        obstacle_map = np.zeros((self.grid_num, self.grid_num)) 

        center_x = int(round(-self.min_w / self.resolution)) 
        center_y = center_x 

        ixs = [int(round((x - self.min_w) / self.resolution)) for x in ox]
        iys = [int(round((y - self.min_w) / self.resolution)) for y in oy]

        for ix, iy in zip(ixs, iys):
            laser_beams = self.bresenham((center_x, center_y), (ix, iy))  # line form the lidar to the occupied point

            for laser_beam in laser_beams:
                if 0<=laser_beam[0] and laser_beam[0]<self.grid_num and 0<=laser_beam[1] and laser_beam[1]<self.grid_num:
                    space_map[laser_beam[0]][laser_beam[1]] = 0.0  

            if 0<=ix and ix<self.grid_num and 0<=iy and iy<self.grid_num:
                obstacle_map[ix][iy] = 1.0  

        return space_map, obstacle_map

    def merge_maps(self, space_map, obstacle_map):
        return np.maximum(space_map * 0.5, obstacle_map)

    def generate_local_map(self, ox, oy):
        return self.merge_maps(*self.generate_maps(ox, oy))

    def roll_map(self, local_map, x, y):
        ix = round(x / self.resolution)
        iy = round(y / self.resolution)

        g_map = np.roll(local_map, ix, axis=1)
        g_map = np.roll(g_map, iy, axis=0)

        return g_map

    def generate_map(self, ox, oy, x, y):

        occupancy_map = self.generate_local_map(ox, oy)

        ix = round(x / self.resolution)
        iy = round(y / self.resolution)

        occupancy_map = np.roll(occupancy_map, ix, axis=1)
        occupancy_map = np.roll(occupancy_map, iy, axis=0)

        return occupancy_map

    def generate_ray_casting_grid_map(self, ox, oy):
        
        # default 0.5
        occupancy_map = np.ones((self.grid_num, self.grid_num)) / 2
        
        center_x = int(round(-self.min_w / self.resolution))  # center x coordinate of the grid map
        center_y = center_x # center y coordinate of the grid map

        # # occupancy grid computed with bresenham ray casting
        # for (x, y) in zip(ox, oy):
        #     # x coordinate of the the occupied area
        #     ix = int(round((x - self.min_w) / self.resolution))
        #     # y coordinate of the the occupied area
        #     iy = int(round((y - self.min_w) / self.resolution))
        #     laser_beams = self.bresenham((center_x, center_y), (ix, iy))  # line form the lidar to the occupied point

        #     for laser_beam in laser_beams:
        #         occupancy_map[laser_beam[0]][laser_beam[1]] = 0.0  # free area 0.0
        #     occupancy_map[ix][iy] = 1.0  # occupied area 1.0
        #     # occupancy_map[ix + 1][iy] = 1.0  # extend the occupied area
        #     # occupancy_map[ix][iy + 1] = 1.0  # extend the occupied area
        #     # occupancy_map[ix + 1][iy + 1] = 1.0  # extend the occupied area

        ixs = [int(round((x - self.min_w) / self.resolution)) for x in ox]
        iys = [int(round((y - self.min_w) / self.resolution)) for y in oy]

        for ix, iy in zip(ixs, iys):
            laser_beams = self.bresenham((center_x, center_y), (ix, iy))  # line form the lidar to the occupied point

            for laser_beam in laser_beams:
                occupancy_map[laser_beam[0]][laser_beam[1]] = 0.0  # free area 0.0

        # for ix, iy in zip(ixs, iys):
        #     occupancy_map[ix][iy] = 1.0  # occupied area 1.0

        return occupancy_map
        
    def generate_ray_casting_grid_map_roll(self, ox, oy, x, y):

        occupancy_map = self.generate_ray_casting_grid_map(ox, oy)

        ix = round(x / self.resolution)
        iy = round(y / self.resolution)

        occupancy_map = np.roll(occupancy_map, ix, axis=1)
        occupancy_map = np.roll(occupancy_map, iy, axis=0)

        return occupancy_map