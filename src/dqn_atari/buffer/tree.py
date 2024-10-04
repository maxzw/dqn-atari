class SumTree:
    """SumTree data structure for prioritized experience replay.

    The SumTree data structure is used to store the priorities of transitions in the Prioritized
    Experience Replay (PER) buffer. The SumTree allows for efficient sampling of transitions based
    on their priority, as well as updating the priorities of sampled transitions. The SumTree is
    implemented as a binary tree, where the parent node contains the sum of its children nodes.

    Args:
        size (int): The maximum number of transitions that the tree can store.
    """

    def __init__(self, size: int) -> None:
        self.nodes = [0] * (2 * size - 1)
        self.data = [None] * size

        # set counters
        self.size = size
        self.count = 0
        self.real_size = 0

    @property
    def total(self):
        return self.nodes[0]

    def update(self, data_idx: int, value: float) -> None:
        idx = data_idx + self.size - 1  # child index in tree array
        change = value - self.nodes[idx]

        self.nodes[idx] = value

        parent = (idx - 1) // 2
        while parent >= 0:
            self.nodes[parent] += change
            parent = (parent - 1) // 2

    def add(self, value: float, data: object) -> None:
        self.data[self.count] = data
        self.update(self.count, value)

        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def get(self, cumsum: float) -> tuple[int, float, object]:
        assert cumsum <= self.total

        idx = 0
        while 2 * idx + 1 < len(self.nodes):
            left, right = 2 * idx + 1, 2 * idx + 2

            if (cumsum <= self.nodes[left]) | (self.nodes[right] == 0):
                idx = left
            else:
                idx = right
                cumsum -= self.nodes[left]

        data_idx = idx - self.size + 1

        return data_idx, self.nodes[idx], self.data[data_idx]
