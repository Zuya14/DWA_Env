import math
from collections import deque

class NoveltyBuffer:

    def __init__(self, buffer_size, thr_distance):
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.thr_distance = thr_distance

    def add_if_far(self, pos):
        length = len(self.buffer)

        if length > 0:
            min_distance = float('inf')
            for p in self.buffer:  
                distance = math.sqrt((p[0] - pos[0])**2 + (p[1] - pos[1])**2)        
                min_distance = min(distance, min_distance)

            if min_distance > self.thr_distance:
                self.buffer.append(pos)
        else:
            self.buffer.append(pos)

    def calc_mean_distance(self, pos):
        length = len(self.buffer)
        mean_distance = 0

        if length > 0:
            for p in self.buffer:  
                mean_distance += math.sqrt((p[0] - pos[0])**2 + (p[1] - pos[1])**2)        
            mean_distance = mean_distance / length

        return mean_distance

    def calc_limited_mean_distance(self, pos):
        length = len(self.buffer)
        mean_distance = 0

        if length > 0:
            for p in self.buffer:  
                d = math.sqrt((p[0] - pos[0])**2 + (p[1] - pos[1])**2)

                if d > self.thr_distance:
                    mean_distance += d
                # else:
                #     mean_distance = 0

            mean_distance = mean_distance / length

        return mean_distance

if __name__ == '__main__':
    bf = NoveltyBuffer(5, 1)

    import numpy as np

    for _ in range(10):
        bf.add_if_far(np.random.randint(0, 100, (2)))
        print(bf.buffer[-1], bf.buffer)