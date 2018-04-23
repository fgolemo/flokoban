import itertools
import random
import time
import numpy as np

from matplotlib import pyplot as plt


class PushingObjects(object):
    """The Sokoban environment.
    """

    def __init__(self, *args, grid_size=10, object_count=2, **kwargs):

        self._grid_size = grid_size
        self._object_count = object_count

        self._noop = np.array([0, 0])
        self._move_down = np.array([1, 0])
        self._move_up = np.array([-1, 0])
        self._move_right = np.array([0, 1])
        self._move_left = np.array([0, -1])
        self._moves = [self._noop, self._move_down, self._move_up, self._move_right, self._move_left]

        self._action_space = np.array([[0, 4]])

        self._reward = None
        self._observation = None
        self._objects = None
        self._goals = None
        self._grid = None
        self._agent_pos = None

    def env_seed(self, seed):
        # TODO: fix it
        np.random.seed(seed)

    def reset(self):

        self._agent_pos = np.random.randint(0, self._grid_size, (2))

        # create a list of all available cells in the grid
        cells = list(itertools.product(range(self._grid_size), range(self._grid_size)))
        available_cells = list(itertools.product(range(self._grid_size), range(self._grid_size)))

        available_cells.remove(tuple(self._agent_pos))

        self._objects = []
        self._goals = []
        for obj_idx in range(self._object_count):
            obj_pos = random.sample(available_cells, 1)[0]
            available_cells.remove(obj_pos)
            self._objects.append(obj_pos)
        self._objects = np.array(self._objects)
        for goal_idx in range(self._object_count):
            goal = random.sample(cells, 1)[0]
            self._goals.append(goal)
        self._goals = np.array(self._goals)

        self.compute_grid()
        self.compute_observation()

        # We compute the initial reward.
        self.compute_reward()

    def compute_reward(self):
        self._reward = 0
        for obj in self._objects:
            for goal in self._goals:
                if (obj == goal).all():
                    self._reward += 1
        return self._reward

    def act(self, action=np.array([0])):
        """Perform an agent action in the Environment
        """

        assert action.shape == (1,)

        # We compute the new positions
        move = self._moves[action[0]]
        if self._in_grid(self._agent_pos + move):
            occupied, object_pose = self._is_occupied(self._agent_pos + move)
            if occupied and self._in_grid(object_pose + move):
                blocked, _ = self._is_occupied(object_pose + move)
                if not blocked:
                    object_pose += move
                    self._agent_pos += move
            if not occupied:
                self._agent_pos += move

        if self._render:
            self.compute_observation()
            self._ax.imshow(self.observation.transpose(1, 2, 0))
            plt.draw()
            plt.pause(0.01)

    def _is_occupied(self, pos):
        for obj in self._objects:
            if (pos == obj).all():
                return True, obj
        return False, False

    def _in_grid(self, pose):
        return (pose >= 0).all() and (pose < self._grid_size).all()

    def compute_grid(self):
        self._grid = np.zeros((self._grid_size, self._grid_size), dtype=np.uint8)
        self._grid[self._agent_pos[0], self._agent_pos[1]] = 1

        for obj_pos in self._objects:
            self._grid[obj_pos[0], obj_pos[1]] = 2
        for goal_pos in self._goals:
            self._grid[goal_pos[0], goal_pos[1]] = 3

    def compute_observation(self):
        obs_agent = np.zeros((self._grid_size, self._grid_size), dtype=np.float32)
        obs_objects = np.zeros((self._grid_size, self._grid_size), dtype=np.float32)
        obs_goals = np.zeros((self._grid_size, self._grid_size), dtype=np.float32)

        obs_agent[self._agent_pos[0], self._agent_pos[1]] = 1
        for obj_pos in self._objects:
            obs_objects[obj_pos[0], obj_pos[1]] = 1
        for goal_pos in self._goals:
            obs_goals[goal_pos[0], goal_pos[1]] = 1
        self._observation = np.array([obs_agent, obs_objects, obs_goals])

    def render(self):
        plt.ion()
        self._render = True
        self._figure = plt.figure()
        self._ax = self._figure.add_subplot(111)

    def terminate(self):

        # We reset stuffs to None to generate errors if called
        self._reward = None
        self._action_space = None
        self._observation = None

    @property
    def action_space(self):

        return self._action_space

    @property
    def observation(self):

        return self._observation

    @property
    def reward(self):

        return self._reward

    @classmethod
    def test(cls):

        pass


if __name__ == '__main__':
    actor = PushingObjects()
    actor.env_seed(0)
    actor.reset()
    actor.compute_grid()
    print(actor._grid)
    actor.compute_reward()
    actor._agent_pos = np.array([7, 0])
    actor._objects[0] = np.array([7, 1])
    actor._objects[1] = np.array([7, 2])
    actor._goals[1] = np.array([7, 2])
    actor.compute_grid()
    print(actor._grid)
    actor.render()
    for _ in range(10):
        actor.act(np.random.randint(5, size=1))
    actor.compute_grid()
    print(actor._grid)
    actor.compute_reward()
    print(actor.reward)
    actor.compute_observation()
    actor.act(np.array([0]))
    print(actor.observation)
