class DistanceErrorMonitor:
    def __init__(self, threshold = 2):
        self.threshold = threshold
        self.reset()

    def update(self, distance):
        self.prev_distance = self.current_distance
        self.current_distance = distance

    def is_growing(self):
        if self.prev_distance is None or self.current_distance is None:
            return False  # Not enough data yet
        return self.current_distance > self.prev_distance

    def is_above_threshold(self):
        if self.current_distance is None:
            return False
        return self.current_distance > self.threshold
    
    def reset(self):
        self.prev_distance = None
        self.current_distance = None