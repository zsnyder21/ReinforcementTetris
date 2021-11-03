from collections import deque


class MaxQueue(deque):
    """
    A deque object that only holds the largest elements. If no maxlen parameter
    is specified, this will behave exactly as deque does
    """
    def __init__(self, maxlen: int = None, key=lambda x: x, metric=lambda x, y: x > y):
        """
        Init sequence

        :param maxlen: Maximum number of items to hold
        :param key: The key to use when finding the minimum element
        :param metric: The metric used to compare two elements
        """
        super(MaxQueue, self).__init__(maxlen=maxlen)
        self.key = key
        self.metric = metric

    def append(self, val) -> None:
        """
        Append, but keep only maxlen largest elements

        :param val: The value to append
        :return: None
        """
        if self.maxlen:
            length = len(self)
            if length:
                minimum = min(self, key=self.key)
            else:
                minimum = None

            if minimum is None:
                super(MaxQueue, self).append(val)
            elif length < self.maxlen:
                super(MaxQueue, self).append(val)
            elif length >= self.maxlen and self.metric(val, minimum):
                self.remove(minimum)
                super(MaxQueue, self).append(val)
            else:
                pass
        else:
            super(MaxQueue, self).append(val)
