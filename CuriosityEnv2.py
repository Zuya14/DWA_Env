# import sys

import gym
# import numpy as np
# import gym.spaces
from gym.utils import seeding
# import torch

import pybullet as p
import cv2
import math

import random
# import copy

import bullet_lidar as bullet_lidar
# import sim  

from sim_abs import sim_abs
import numpy as np

from NoveltyBuffer import NoveltyBuffer
from GridScan import GridScan

class sim_CuriosityEnv2(sim_abs):

    def calcInitPos(self, initPos=None):
        if initPos is None:
            
            x = random.uniform(0.25, 9.0-0.25)
            y = random.uniform(0.25, 9.0-0.25)
            theta = random.uniform(-math.pi, math.pi)
            self.loadRobot(x, y, theta)

            while self.isContacts():
                x = random.uniform(0.25, 9.0-0.25)
                y = random.uniform(0.25, 9.0-0.25)
                theta = random.uniform(-math.pi, math.pi)
                self.loadRobot(x, y, theta)

            initPos = np.array([x, y])
            # print(initPos)

            return np.concatenate([initPos, [0.0]])
        else:
            self.loadRobot(*initPos)
            if self.isContacts():
                print("contact:",initPos)
                exit()
            return np.array(initPos)

    def calcTgtPos(self, tgtPos):
        if tgtPos is None:
            x = random.uniform(0.25, 9.0-0.25)
            y = random.uniform(0.25, 9.0-0.25)
            self.loadGoal(x, y, 0)

            while self.goal_isContacts():
                x = random.uniform(0.25, 9.0-0.25)
                y = random.uniform(0.25, 9.0-0.25)
                self.loadGoal(x, y, 0)

            tgtPos = np.array([x, y])
            # print(tgtPos)

            return np.concatenate([tgtPos, [0.0]])
        else:
            self.loadGoal(*tgtPos)
            if self.goal_isContacts():
                print("contact:",tgtPos)
                exit()
            return np.array(tgtPos)

    def loadObstacle(self):
        self.bodyUniqueIds += [self.phisicsClient.loadURDF("urdf/CuriosityEnv2/walls.urdf", basePosition=(0, 0, 0))]

    ''' action = [v, cos, sin, theta] '''
    def calcAction(self, action):
        # theta = np.arctan2(action[2], action[1])

        # _, ori = self.getRobotPosInfo()
        # yaw = p.getEulerFromQuaternion(ori)[2]
        # dtheta = yaw + theta
        
        # self.vx = action[0] * np.sin(dtheta)
        # self.vy = action[0] * np.cos(dtheta)
        # # self.w = action[3]
        # self.w = 0.0

        # dx = action[0] #+ random.uniform(-0.05, 0.05)
        # dy = action[1] #+ random.uniform(-0.05, 0.05)

        # v = math.sqrt(dx*dx + dy*dy)

        # theta = np.arctan2(action[1], action[0])
        # _, ori = self.getRobotPosInfo()
        # yaw = p.getEulerFromQuaternion(ori)[2]
        # dtheta = yaw + theta

        # if v > 1.0:
        #     self.vx = 1.0*np.sin(dtheta)
        #     self.vy = 1.0*np.cos(dtheta)
        # else:
        #     self.vx = v * np.sin(dtheta)
        #     self.vy = v * np.cos(dtheta)

        # self.w = 0.0


        v = (action[0] + 1.0) / 2.0

        theta = 0
        _, ori = self.getRobotPosInfo()
        yaw = p.getEulerFromQuaternion(ori)[2]
        dtheta = yaw + theta

        self.vx = v * np.cos(dtheta)
        self.vy = v * np.sin(dtheta)

        self.w = action[1]

    def isArrive(self, tgtPos=None, pos=None):
        if tgtPos is None:
            tgtPos = self.tgt_pos
        if pos is None:
            pos = self.getState()

        # return  np.linalg.norm(tgtPos - pos, ord=2) < 0.5
        return  np.linalg.norm(tgtPos[:2] - pos[:2], ord=2) < 0.1
        # return False

    def getWallPoints(self):
        return  (np.array([
            [ 465,  465],
            [-465,  465],
            [ 465, -465],
            [-465, -465],
        ]) + np.array([ 465, 465]))/100.0

    def getWidthWallPoints(self):
        return  (np.array([
            [-465,  -465, 930],
            [-465,  450, 930],

            [-450,  -57.5, 200],
            [-150,  -57.5, 107.5],

            [-57.5,  -57.5, 207.5],
            [250,  -57.5, 200],

        ]) + np.array([ 465-15, 465, 0]))/100.0

    def getHeightWallPoints(self):
        return  (np.array([
            [-465,  -465, 930],
            [ 450,  -465, 930],

            [-57.5,  -450, 100],
            [-57.5,  -250, 207.5],

            [-57.5,  -57.5, 207.5],
            [-57.5,   250, 200],

        ]) + np.array([ 465, 465-15, 0]))/100.0

    def renderWall(self, img, center, maxLen):

        points = self.getWallPoints()
        points -= self.renderOffset()

        pts = np.zeros((points.shape[0], 2))
        k = min(center[0], center[1])

        for a, b in zip(pts, points):
            a[0] =  b[0] * (k/maxLen)
            a[1] = -b[1] * (k/maxLen)

        pts += center + self.getRenderMarginOffset() * (k/maxLen)
        pts = pts.astype(np.int)



        points = self.getWidthWallPoints()[:, :2]
        points -= self.renderOffset()

        pts = np.zeros((points.shape[0], 2))

        for a, b in zip(pts, points):
            a[0] =  b[0] * (k/maxLen)
            a[1] = -b[1] * (k/maxLen)

        length = self.getWidthWallPoints()[:, 2] * (k/maxLen)
        length = length.astype(np.int)

        pts += center + self.getRenderMarginOffset() * (k/maxLen)
        pts = pts.astype(np.int)

        for p, l in zip(pts, length):
            cv2.rectangle(img, tuple(p), (p[0] + l, int(p[1] + 0.15*(k/maxLen))), (184, 196, 153), thickness=-1)



        points = self.getHeightWallPoints()[:, :2]
        points -= self.renderOffset()

        pts = np.zeros((points.shape[0], 2))

        for a, b in zip(pts, points):
            a[0] =  b[0] * (k/maxLen)
            a[1] = -b[1] * (k/maxLen)

        length = -self.getHeightWallPoints()[:, 2] * (k/maxLen)
        length = length.astype(np.int)

        pts += center + self.getRenderMarginOffset() * (k/maxLen)
        pts = pts.astype(np.int)

        for p, l in zip(pts, length):
            cv2.rectangle(img, tuple(p), (p[0] - int(0.15*(k/maxLen)), p[1] + l), (184, 196, 153), thickness=-1)

        return img

    def setImageSize(self):
        self.iw = 800
        self.ih = 800

    def createImage(self):
        # w = 1000
        # h = 800
        # img = np.zeros((h, w, 3), np.uint8)
        img = np.full((self.ih, self.iw, 3), 145, dtype=np.uint8)
        center = (self.iw//2, self.ih//2)
        # center = (h//2, w//2)

        return img, center

    def renderRobotAndGoal(self, img, center, maxLen):
        points = np.array([self.getState()[:2], self.tgt_pos[:2]])
        points -= self.renderOffset()

        pts = np.zeros((2, 2))
        k = min(center[0], center[1])

        for a, b in zip(pts, points):
            a[0] =  b[0] * (k/maxLen)
            a[1] = -b[1] * (k/maxLen)

        pts += center + self.getRenderMarginOffset() * (k/maxLen)
        pts = pts.astype(np.int)

        cv2.circle(img, tuple(pts[0]), radius=int(0.18 * (k/maxLen)), color=(255,0,0), thickness=-1, lineType=cv2.LINE_8, shift=0)
        cv2.circle(img, tuple(pts[1]), radius=int(0.18 * (k/maxLen)), color=(0,0,255), thickness=2, lineType=cv2.LINE_8, shift=0)

        yaw = self.getState()[2]
        # dp = (0.1 + np.sqrt(self.vx**2 + self.vy**2)/2.0) * np.array([np.sin(yaw), -np.cos(yaw)]) * (k/maxLen)
        dp = (0.1 + np.sqrt(self.vx**2 + self.vy**2)/2.0) * np.array([np.cos(yaw), -np.sin(yaw)]) * (k/maxLen)
        dp = dp.astype(np.int)

        cv2.line(img, tuple(pts[0]), tuple(pts[0]+dp), color=(0,0,128), thickness=2, lineType=cv2.LINE_8, shift=0)

        return img


class CuriosityEnv2(gym.Env):
    global_id = 0

    def __init__(self):
        super().__init__()
        self.seed(seed=random.randrange(10000))
        self.sim = None

        self.name = "CuriosityEnv2"
        self.center = (4.3, 3.8)

    def setting(self, _id=-1, mode=p.DIRECT, sec=0.1):
        if _id == -1:
            self.sim = sim_CuriosityEnv2(CuriosityEnv2.global_id, mode, sec)
            CuriosityEnv2.global_id += 1
        else:
            self.sim = sim_CuriosityEnv2(_id, mode, sec)

        # self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        # self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.lidar = self.createLidar()

        # self.state_space = gym.spaces.Box(low=0.0, high=9.0, shape=(3,))
        # self.velocity_space = gym.spaces.Box(low=0.0, high=9.0, shape=(3,))
        # self.observation_space = gym.spaces.Box(low=0.0, high=9.0, shape=(self.lidar.shape[0],))

        # self.observation_space = gym.spaces.Box(low=-math.inf, high=math.inf, shape=(3+3+self.lidar.shape[0]+3,))
        self.observation_space = gym.spaces.Box(low=-math.inf, high=math.inf, shape=(3+3+self.lidar.shape[0]+1+1,))
        # self.observation_space = gym.spaces.Box(low=-math.inf, high=math.inf, shape=(3+3+self.lidar.shape[0]+2+1+1+1,))
        # self.observation_space = gym.spaces.Box(low=-math.inf, high=math.inf, shape=(3+3+self.lidar.shape[0]+3+1+1,))

        self.sec = sec

        # self._max_episode_steps = 300
        # self._max_episode_steps = 500
        self._max_episode_steps = 1000
        # self._max_episode_steps = 100

        self.setParams()

        self.loadWorldMap()        
        self.gridScan = GridScan(lidar_maxLen=10.0, resolution=0.1)
        self.completeRate = 0

        self.reset()

    def copy(self, _id=-1):
        new_env = CuriosityEnv2()
        
        if _id == -1:
            new_env.sim = self.sim.copy(CuriosityEnv2.global_id)
            CuriosityEnv2.global_id += 1
        else:
            new_env.sim = self.sim.copy(_id)

        # new_env.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        new_env.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        new_env.lidar = new_env.createLidar()

        # new_env.state_space = gym.spaces.Box(low=0.0, high=9.0, shape=(3,))
        # new_env.velocity_space = gym.spaces.Box(low=0.0, high=9.0, shape=(3,))
        # new_env.observation_space = gym.spaces.Box(low=0.0, high=9.0, shape=(self.lidar[0],))

        new_env.observation_space = self.observation_space

        new_env.sec = self.sec

        return new_env

    def reset(self, initpos=None, tgtpos=None):
        assert self.sim is not None, print("call setting!!")
        # initpos=[1.8, 1.8, 0.0], tgtpos=[1.8, 1.8, 0.0] 
        self.sim.reset(sec=self.sec, initPos=initpos, tgtPos=tgtpos)
        # self.sim.reset(sec=self.sec, initPos=[1.5, 1.5, 0.0], tgtPos=[7.5, 1.5, 0.0])
        
        # index = random.randint(0, 3)

        # initPoslist =[
        #     [1.0, 1.0, 0.0],
        #     [1.0, 8.0, 0.0],
        #     [8.0, 1.0, 0.0],
        #     [8.0, 8.0, 0.0]
        # ]
        # self.sim.reset(sec=self.sec, initPos=initPoslist[index] if initpos is None else initpos, tgtPos=initPoslist[index] if tgtpos is None else tgtpos)

        self.createNoveltyBuffer()
        self.noveltyBuffer.add_if_far(self.sim.getState()[:2])
        self.updateGridMap()
        self.old_completeRate = self.completeRate
        return self.observe_all()

    def test_reset(self):
        assert self.sim is not None, print("call setting!!") 
        # self.reset(initpos=[1.5, 1.5, 0.0], tgtpos=[7.5, 1.5, 0.0])
        # self.reset(initpos=[1.5, 1.5, 0.0], tgtpos=[4.5, 7.5, 0.0])
        # r = self.reset(initpos=[1.8, 1.8, 0.0], tgtpos=[1.8, 1.8, 0.0])
        r = self.reset(initpos=[1.0, 1.0, 0.0], tgtpos=[1.0, 1.0, 0.0])
        # self.sim.reset(sec=self.sec)

        # self.createNoveltyBuffer()
        # self.noveltyBuffer.add_if_far(self.sim.getState()[:2])
        # self.updateGridMap()
        # self.old_completeRate = self.completeRate
        return r

    # def createNoveltyBuffer(self, buffer_size=100, thr_distance=1.0):
    def createNoveltyBuffer(self, buffer_size=100, thr_distance=1.8):
        self.noveltyBuffer = NoveltyBuffer(buffer_size, thr_distance)

        self.space_map = None
        self.obstacle_map = None
        self.occupancy_map = None

    def loadWorldMap(self):
        file_name = 'gridmap_CuriosityEnv2.npy'
        self.world_map = np.load(file_name)
    
    def createLidar(self):
        # resolusion = 90
        # resolusion = 30
        # resolusion = 22.5
        # resolusion = 12
        # resolusion = 36
        # resolusion = 6
        resolusion = 4.5
        # resolusion = 1
        deg_offset = 90.
        rad_offset = deg_offset*(math.pi/180.0)
        startDeg = -180. + deg_offset
        endDeg = 180. + deg_offset

        # maxLen = 20.
        maxLen = 10.
        # maxLen = 5.
        minLen = 0.
        return bullet_lidar.bullet_lidar(startDeg, endDeg, resolusion, maxLen, minLen)

    def step(self, action):

        done = self.sim.step(action)

        # if self.sim.steps % 5 == 0:
        self.updateGridMap()

        # done = done or (self.sim.steps == self._max_episode_steps)
        # done = done or (self.sim.steps == self._max_episode_steps) or (self.completeRate > self.params['curiosity_thr'])
        done = done or (self.sim.steps == self._max_episode_steps) or (self.completeRate > self.params['curiosity_thr']) or self.sim.isArrive()

        reward = self.get_reward(terminal=(self.sim.steps == self._max_episode_steps))

        self.noveltyBuffer.add_if_far(self.sim.getState()[:2])

        self.old_completeRate = self.completeRate

        return self.observe_all(), reward, done, {}

    def updateGridMap(self):
        state = self.sim.getState()
        x, y = state[0], state[1]
        yaw = state[2]

        cx = self.center[0]
        cy = self.center[1]

        rads, dist = self.scan()
        ox = np.sin(rads+yaw) * dist
        oy = np.cos(rads+yaw) * dist

        if self.occupancy_map is None:
            self.space_map, self.obstacle_map = self.gridScan.generate_maps(ox, oy)
            self.space_map = self.gridScan.roll_map(self.space_map, x-cx, y-cy)
            self.obstacle_map = self.gridScan.roll_map(self.obstacle_map, x-cx, y-cy)
    
            self.occupancy_map = self.gridScan.merge_maps(self.space_map, self.obstacle_map)
            
            self.occupancy_map_space_num = np.count_nonzero((self.occupancy_map[self.occupancy_map == self.world_map]) == 0)
            self.old_occupancy_map_space_num = self.occupancy_map_space_num
            self.world_map_space_num = np.count_nonzero(self.world_map == 0)
        else: 
            _space_map, _obstacle_map = self.gridScan.generate_maps(ox, oy)
            _space_map = self.gridScan.roll_map(_space_map, x-cx, y-cy)
            _obstacle_map = self.gridScan.roll_map(_obstacle_map, x-cx, y-cy)

            self.space_map = np.minimum(_space_map, self.space_map)
            self.obstacle_map = np.maximum(_obstacle_map, self.obstacle_map)

            self.occupancy_map = self.gridScan.merge_maps(self.space_map, self.obstacle_map)

            self.old_occupancy_map_space_num = self.occupancy_map_space_num
            self.occupancy_map_space_num = np.count_nonzero((self.occupancy_map[self.occupancy_map == self.world_map]) == 0)

        self.completeRate = self.occupancy_map_space_num / self.world_map_space_num

        return self.completeRate

    def get_occupancy_map(self):
        return self.gridScan.merge_maps(self.space_map, self.obstacle_map)

    def get_left_steps(self):
        return self._max_episode_steps - self.sim.steps

    def observe_all(self):
        state = self.sim.getState()
        left_steps = self._max_episode_steps - self.sim.steps
        # return np.concatenate([state, self.sim.getVelocity(), self.sim.getObserve(self.lidar), [self.completeRate], [left_steps]])
        return np.concatenate([state, self.sim.getVelocity(), self.sim.getObserve(self.lidar), [1.0-self.completeRate], [left_steps]])
        # return np.concatenate([state, self.sim.getVelocity(), self.sim.getObserve(self.lidar), self.noveltyBuffer.buffer[-1], [self.noveltyBuffer.calc_limited_mean_distance(state[:2])], [self.completeRate], [left_steps]]) 
        # return np.concatenate([state, self.sim.getVelocity(), self.sim.getObserve(self.lidar), state-self.sim.tgt_pos, [self.completeRate], [left_steps]])
        # return np.concatenate([state, self.sim.getVelocity(), self.sim.getObserve(self.lidar), state-self.sim.tgt_pos, [(self.world_map_space_num - self.occupancy_map_space_num)*self.gridScan.resolution*self.gridScan.resolution], [left_steps]])
        # return np.concatenate([state, self.sim.getVelocity(), self.sim.getObserve(self.lidar), [len(self.noveltyBuffer.buffer)], [self.noveltyBuffer.calc_limited_mean_distance(state[:2])], [left_steps]])

    def scan(self):
        return self.lidar.rads, self.sim.getObserve(self.lidar)

    def get_reward(self, terminal):
        return self.calc_reward(self.sim.isContacts(), self.sim.getState(), self.sim.tgt_pos, self.sim.getOldState(), terminal)

    def calc_reward(self, contact, pos, tgt_pos, old_pos=None, terminal=False):

        rewardArrive = (not contact) * self.params['arrive'] * self.sim.isArrive(tgt_pos[:2], pos[:2])

        rewardContact = -self.params['contact'] if contact else 0.0

        rewardCuriosity = 0.0
        # rewardCuriosity = self.completeRate * self.params['curiosity']
        # rewardCuriosity = self.completeRate
        # rewardCuriosity = (not contact) * self.params['curiosity'] * self.noveltyBuffer.calc_mean_distance(pos[:2])

        # rewardCuriosity += (not contact) * self.noveltyBuffer.calc_mean_distance(pos[:2])
        # rewardCuriosity += (not contact) * self.noveltyBuffer.calc_limited_mean_distance(pos[:2])# * self.params['curiosity']
        rewardCuriosity += (not contact) * self.noveltyBuffer.calc_limited_mean_distance(pos[:2]) * 0.1
        # rewardCuriosity += (not contact) * ((self.occupancy_map_space_num - self.old_occupancy_map_space_num)*self.gridScan.resolution*self.gridScan.resolution)# * self.params['curiosity']
        # rewardCuriosity +=  (not contact) *(self.completeRate - self.old_completeRate)* self.params['curiosity']
        # rewardCuriosity +=  (not contact) *(self.completeRate - self.old_completeRate) 
        if self.completeRate > self.params['curiosity_thr']:
            rewardCuriosity += (not contact) * self.params['curiosity']

        if rewardCuriosity <= 0:
            rewardCuriosity -= 0.01

        # T = tgt_pos[:2] - pos[:2]
        # T_norm = math.sqrt(T[0]**2 + T[1]**2)
        # vec = self.sim.getObserveLocalVec(self.lidar)
        # targetCapture = []

        # for v in vec:
        #     v_norm = math.sqrt(v[0]**2 + v[1]**2)
        #     if v_norm > T_norm:
        #         v2 = v * T_norm / v_norm
        #     else: 
        #         v2 = v

        #     prd = T[0]*v2[0] + T[1]*v2[1]

        #     if prd > 0:
        #         targetCapture.append(1.0-prd)

        # rewardTargetCapture = self.params['target_capture'] * np.average(targetCapture)
        rewardTargetCapture = 0
        
        # reward = rewardArrive + rewardContact 
        reward = rewardArrive + rewardContact + rewardCuriosity + rewardTargetCapture
        # print(rewardArrive, rewardContact, rewardCuriosity, rewardTargetCapture)
        
        if old_pos is None:
            self.params['distance'] = 0.0
            self.params['log_distance'] = 0.0
            self.params['move'] = 0.0
            self.params['forward'] = 0.0
            self.params['close'] = 0.0
            self.params['close_thr'] = 0.0

            rewardDistance = 0.0
            rewardMove = 0.0
            rewardForward = 0.0
            rewardClose = 0.0
        else:
            d1 = np.linalg.norm(tgt_pos[:2] - old_pos[:2], ord=2)
            d2 = np.linalg.norm(tgt_pos[:2] - pos[:2], ord=2)

            rewardDistance = (not contact) * self.params['distance'] * (d1 - d2) 
            rewardDistance += - self.params['log_distance'] * np.log(1.0 + d2)

            rewardMove = self.params['move'] * np.abs(d1 - d2)
            
            rewardForward = self.params['forward'] * np.abs(d1 - d2) *((d1 - d2) >0)

            rewardClose = (-self.params['close'] * d2 / self.params['close_thr'] + self.params['close'])  *(d2 < self.params['close_thr'])

            min_distance = min(self.sim.getObserve(self.lidar))
            rewardWarning = (self.params['warning'] * min_distance / self.params['warning_thr'] - self.params['warning'])  *(min_distance < self.params['warning_thr'])

            if terminal:
                rewardLastDistance = - self.params['last_distance'] * d2
            else:
                rewardLastDistance = 0.0

            reward += rewardDistance + rewardMove + rewardForward + rewardClose + rewardWarning + rewardLastDistance

        return reward

    def setParams(self):
        self.params = {
            'arrive': 0.0,
            'contact': 10.0,
            'last_distance': 0.0,
            'distance': 0.0,
            'log_distance': 0.0,
            'move': 0.0,
            'forward': 0.0,
            'close': 0.0,
            'close_thr': 1.0,
            'curiosity':100.0,
            'curiosity_thr':0.98,
            'warning':0.0,
            'warning_thr':0.3,
            'target_capture':0.0,
            }

    def getParams(self):
        return self.params

    def getEnvParams(self):
        return {
            'lidar_points': self.lidar._shape[0],
            'lidar_max': self.lidar.maxLen,
            'lidar_min': self.lidar.minLen,
            }

    def render(self, mode='human', close=False):
        return self.sim.render(self.lidar)
        # return self.sim.render()

    def close(self):
        self.sim.close()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def sample_random_action(self):
        return self.action_space.sample()

    def getState(self):
        return self.sim.getState()


if __name__ == '__main__':
    
    import time
    env = CuriosityEnv2()
    # env.setting()
    env.setting(mode=p.GUI)

    i = 0

    while True:
        i += 1
        # env.reset()
        # action = np.array([1.0, 1.0, 0.0, 0.0])

        # state, _, done, _ = env.step(action)
        state, _, done, _ = env.step(env.sample_random_action())

        # print(state)

        cv2.imshow("env", env.render())
        cv2.imshow("map", ((1.0-env.get_occupancy_map()[::-1])*255).astype(np.uint8))
        
        # cv2.waitKey(0)
        # time.sleep(1)
        # if done or cv2.waitKey(1000) >= 0:
        if done:
            env.reset()
        
        if cv2.waitKey(10) >= 0:
            # print(i)
            break