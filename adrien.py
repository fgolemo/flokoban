import itertools
import random

import numpy as np

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

    def reset(self):

        self._grid = np.zeros((self._grid_size, self._grid_size), dtype=np.uint8)
        self._agent_pos = np.random.randint(0,self._grid_size,(2))
        self._grid[self._agent_pos[0],self._agent_pos[1]] = 1

        # create a list of all available cells in the grid
        available_cells = list(itertools.combinations(range(self._grid_size), 2))

        available_cells.remove(tuple(self._agent_pos))

        for obj_idx in range(self._object_count):
            obj_pos = random.sample(available_cells, 1)[0]
            available_cells.remove(obj_pos)
            #TODO: randomly sample goal in the same way
            #TODO: then add both goal and obj to the internal list of objects/goals


        # TODO: create observation function
        # TODO: call observation function here and set it to initial obs

        # self._observation = np.concatenate([self._agent_pose, self._objects_initial_poses])

        # We compute the initial reward.
        # TODO: move reward into separate function
        self._reward = np.linalg.norm(self._actual_objects_poses - self._objects_rewarding_poses, ord=2)

    def act(self, action=np.array([0])):
        """Perform an agent action in the Environment
        """

        assert action.shape == (1,)

        # We compute the new positions
        move = self._moves[action[0]]
        if self._in_grid(self._agent_pose + move):
            occupied, object_pose = self._is_occupied(self._agent_pose + move)
            if occupied == True and self._in_grid(object_pose + move):
                raise NotImplementedError

            for object_pose in self._objects_initial_poses:
                if object_pose[0] == self._agent_pose[0] + 1 and object_pose[1] == self._agent_pose[1] + 1:
                    raise NotImplementedError
            # Move down
            self._agent_pose[0] += 1

    def _is_occupied(self, pose):
        for object_pose in self._actual_objects_poses:
            if (pose == object_pose):
                return True, object_pose
        return False, False

    def _in_grid(self, pose):
        return (pose >= 0).all() and (pose < self._grid_size).all()

    def terminate(self):

        # We reset stuffs to None to generate errors if called
        self._objects_initial_poses = None
        self._objects_rewarding_poses = None
        self._actual_objects_poses = None
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