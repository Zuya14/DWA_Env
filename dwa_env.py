import gym

import pybullet as p
import cv2
import math

import random
from enum import Enum

import numpy as np
import matplotlib.pyplot as plt

class RobotType(Enum):
    circle = 0
    rectangle = 1

class Config:
    """
    simulation parameter class
    """

    def __init__(self):
        # robot parameter
        self.max_speed = 1.0  # [m/s]
        self.min_speed = 0.0  # [m/s]
        self.max_yaw_rate = 40.0 * math.pi / 180.0  # [rad/s]
        # self.max_accel = 0.2  # [m/ss]
        self.max_accel = 0.5  # [m/ss]
        self.max_delta_yaw_rate = 40.0 * math.pi / 180.0  # [rad/ss]
        self.max_delta_yaw_rate = math.pi # [rad/ss]
        # self.v_resolution = 0.01  # [m/s]
        self.v_resolution = 0.02  # [m/s]
        self.yaw_rate_resolution = 0.1 * math.pi / 180.0  # [rad/s]
        self.dt = 0.1  # [s] Time tick for motion prediction
        # self.predict_time = 3.0  # [s]
        self.predict_time = 1.5  # [s]
        # self.predict_time = 5.0  # [s]
        self.to_goal_cost_gain = 0.15
        self.speed_cost_gain = 1.0
        self.obstacle_cost_gain = 1.0
        # self.obstacle_cost_gain = 0.01
        self.robot_stuck_flag_cons = 0.001  # constant to prevent robot stucked
        self.robot_type = RobotType.circle

        # if robot_type == RobotType.circle
        # Also used to check if goal is reached in both types
        # self.robot_radius = 1.0  # [m] for collision check
        self.robot_radius = 0.25  # [m] for collision check

        # if robot_type == RobotType.rectangle
        # self.robot_width = 0.5  # [m] for collision check
        # self.robot_length = 1.2  # [m] for collision check

    @property
    def robot_type(self):
        return self._robot_type

    @robot_type.setter
    def robot_type(self, value):
        if not isinstance(value, RobotType):
            raise TypeError("robot_type must be an instance of RobotType")
        self._robot_type = value

config = Config()


def plot_arrow(x, y, yaw, length=0.5, width=0.1):  # pragma: no cover
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
    # plt.arrow(x, y, length * math.sin(yaw), length * math.cos(yaw),
              head_length=width, head_width=width)
    plt.plot(x, y)

def plot_robot(x, y, yaw, config):  # pragma: no cover
    if config.robot_type == RobotType.rectangle:
        outline = np.array([[-config.robot_length / 2, config.robot_length / 2,
                             (config.robot_length / 2), -config.robot_length / 2,
                             -config.robot_length / 2],
                            [config.robot_width / 2, config.robot_width / 2,
                             - config.robot_width / 2, -config.robot_width / 2,
                             config.robot_width / 2]])
        Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                         [-math.sin(yaw), math.cos(yaw)]])
        outline = (outline.T.dot(Rot1)).T
        outline[0, :] += x
        outline[1, :] += y
        plt.plot(np.array(outline[0, :]).flatten(),
                 np.array(outline[1, :]).flatten(), "-k")
    elif config.robot_type == RobotType.circle:
        circle = plt.Circle((x, y), config.robot_radius, color="b")
        plt.gcf().gca().add_artist(circle)
        out_x, out_y = (np.array([x, y]) +
                        np.array([np.cos(yaw), np.sin(yaw)]) * config.robot_radius)
        plt.plot([x, out_x], [y, out_y], "-k")

class DWA_Env:

    def __init__(self, env):
        self.env = env
        self.name = env.name + "_DWA"

    def reset(self, initpos=None, tgtpos=None):
        self.env.reset()

        # return slam_map

    def test_reset(self):
        self.env.test_reset()

        # return slam_map

    def step(self, goal):

        done = False

        x = np.concatenate([self.env.sim.getState(), [0.0, 0.0]])

        while not done:

            rads, dist = self.env.scan()
            # cos = np.cos(rads+x[2])
            # sin = np.sin(rads+x[2])
            ox = np.cos(rads+x[2]) * dist
            # ox = ox + x[0] 
            oy = np.sin(rads+x[2]) * dist
            # oy = oy + x[1] 
            ob = np.column_stack([ox, oy])
            # print(x[:2], goal)
            # ob = np.array([[l*c, l*s] for l, c, s in zip(dist, cos, sin)])
            ob += x[:2]

            u, predicted_trajectory = self.dwa_control(x, config, goal, ob)
            
            x = np.concatenate([self.env.sim.getState(), u])

            u[0] = (u[0]*2.0) - 1.0
            # u = self.env.sample_random_action()
            state, _, done, _ = self.env.step(u)

            # plt.cla()
            # plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], "-g")
            # plt.plot(x[0], x[1], "xr")
            # plt.plot(goal[0], goal[1], "xb")
            # plt.plot(ob[:, 0], ob[:, 1], "ok")
            # plot_robot(x[0], x[1], x[2], config)
            # plot_arrow(x[0], x[1], x[2])
            # plt.axis("equal")
            # plt.grid(True)

            cv2.imshow("env", self.env.render())
            cv2.imshow("map", ((1.0-self.env.get_occupancy_map()[::-1])*255).astype(np.uint8))

            plt.pause(0.0001)
            # plt.pause(0.2)

            if cv2.waitKey(10) >= 0:
                break

        reward = 0

        return None, reward, done, {}
        # return Slam_map, reward, done, {}


    def dwa_control(self, x, config, goal, ob):
        """
        Dynamic Window Approach control
        """
        dw = self.calc_dynamic_window(x, config)

        u, trajectory = self.calc_control_and_trajectory(x, dw, config, goal, ob)

        return u, trajectory

    def motion(self, x, u, dt):
        """
        motion model
        """

        x[2] += u[1] * dt
        x[0] += u[0] * math.cos(x[2]) * dt
        x[1] += u[0] * math.sin(x[2]) * dt
        x[3] = u[0]
        x[4] = u[1]

        return x

    def calc_dynamic_window(self, x, config):
        """
        calculation dynamic window based on current state x
        """

        # Dynamic window from robot specification
        Vs = [config.min_speed, config.max_speed,
            -config.max_yaw_rate, config.max_yaw_rate]

        # Dynamic window from motion model
        Vd = [x[3] - config.max_accel * config.dt,
            x[3] + config.max_accel * config.dt,
            x[4] - config.max_delta_yaw_rate * config.dt,
            x[4] + config.max_delta_yaw_rate * config.dt]

        #  [v_min, v_max, yaw_rate_min, yaw_rate_max]
        dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
            max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

        return dw

    def predict_trajectory(self, x_init, v, y, config):
        """
        predict trajectory with an input
        """

        x = np.array(x_init)
        trajectory = np.array(x)
        time = 0
        while time <= config.predict_time:
            x = self.motion(x, [v, y], config.dt)
            trajectory = np.vstack((trajectory, x))
            time += config.dt

        return trajectory

    def calc_control_and_trajectory(self, x, dw, config, goal, ob):
        """
        calculation final input with dynamic window
        """

        x_init = x[:]
        min_cost = float("inf")
        best_u = [0.0, 0.0]
        best_trajectory = np.array([x])

        # evaluate all trajectory with sampled input in dynamic window
        for v in np.arange(dw[0], dw[1], config.v_resolution):
            for y in np.arange(dw[2], dw[3], config.yaw_rate_resolution):

                trajectory = self.predict_trajectory(x_init, v, y, config)
                # calc cost
                to_goal_cost = config.to_goal_cost_gain * self.calc_to_goal_cost(trajectory, goal)
                speed_cost = config.speed_cost_gain * (config.max_speed - trajectory[-1, 3])
                ob_cost = config.obstacle_cost_gain * self.calc_obstacle_cost(trajectory, ob, config)

                final_cost = to_goal_cost + speed_cost + ob_cost

                # search minimum trajectory
                if min_cost >= final_cost:
                    min_cost = final_cost
                    best_u = [v, y]
                    best_trajectory = trajectory
                    if abs(best_u[0]) < config.robot_stuck_flag_cons \
                            and abs(x[3]) < config.robot_stuck_flag_cons:
                        # to ensure the robot do not get stuck in
                        # best v=0 m/s (in front of an obstacle) and
                        # best omega=0 rad/s (heading to the goal with
                        # angle difference of 0)
                        best_u[1] = -config.max_delta_yaw_rate
        return best_u, best_trajectory

    def calc_obstacle_cost(self, trajectory, ob, config):
        """
        calc obstacle cost inf: collision
        """
        ox = ob[:, 0]
        oy = ob[:, 1]
        dx = trajectory[:, 0] - ox[:, None]
        dy = trajectory[:, 1] - oy[:, None]
        r = np.hypot(dx, dy)

        if config.robot_type == RobotType.rectangle:
            yaw = trajectory[:, 2]
            rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            rot = np.transpose(rot, [2, 0, 1])
            local_ob = ob[:, None] - trajectory[:, 0:2]
            local_ob = local_ob.reshape(-1, local_ob.shape[-1])
            local_ob = np.array([local_ob @ x for x in rot])
            local_ob = local_ob.reshape(-1, local_ob.shape[-1])
            upper_check = local_ob[:, 0] <= config.robot_length / 2
            right_check = local_ob[:, 1] <= config.robot_width / 2
            bottom_check = local_ob[:, 0] >= -config.robot_length / 2
            left_check = local_ob[:, 1] >= -config.robot_width / 2
            if (np.logical_and(np.logical_and(upper_check, right_check),
                            np.logical_and(bottom_check, left_check))).any():
                return float("Inf")
        elif config.robot_type == RobotType.circle:
            if np.array(r <= config.robot_radius).any():
                return float("Inf")

        min_r = np.min(r)
        return 1.0 / min_r  # OK

    def calc_to_goal_cost(self, trajectory, goal):
        """
            calc to goal cost with angle difference
        """

        dx = goal[0] - trajectory[-1, 0]
        dy = goal[1] - trajectory[-1, 1]
        error_angle = math.atan2(dy, dx)
        cost_angle = error_angle - trajectory[-1, 2]
        cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))

        return cost

if __name__ == '__main__':
    from CuriosityEnv2 import CuriosityEnv2
    env = CuriosityEnv2()
    # from CuriosityEnv3 import CuriosityEnv3
    # env = CuriosityEnv3()
    # from CuriosityEnv4 import CuriosityEnv4
    # env = CuriosityEnv4()
    env.setting()

    goal = env.sim.tgt_pos[:2]

    dwa_env = DWA_Env(env)
    dwa_env.step(goal)
