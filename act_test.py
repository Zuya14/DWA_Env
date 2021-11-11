from CuriosityEnv4 import CuriosityEnv4
import cv2
import math
import numpy as np

if __name__ == '__main__':
    
    env = CuriosityEnv4()
    env.setting()

    while True:


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


        diff = env.sim.tgt_pos[:2] - env.sim.getState()[:2]
        diff = diff / math.sqrt(diff[0]**2 + diff[1]**2)

        # diff = -diff
        # print(diff)

        # diff = np.clip(diff, -1.0, 1.0)

        state, _, done, _ = env.step(diff)

        print(env.sim.tgt_pos[:2],  env.sim.getState()[:2], diff, env.sim.vx, env.sim.vy)

        cv2.imshow("env", env.render())
        cv2.imshow("map", ((1.0-env.get_occupancy_map()[::-1])*255).astype(np.uint8))

        if done:
            env.reset()
        
        if cv2.waitKey(10) >= 0:
            break