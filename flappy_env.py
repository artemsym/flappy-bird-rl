from flappy_bird import Bird, PipeManager, check_environment, screen_width, screen_height, pipe_width


class FlappyEnv:
    def __init__(self):
        self.bird = Bird()
        self.pipe_manager = PipeManager()
        self.score = 0
        self.ticks = 0

    def reset(self):
        self.bird.reset(screen_height // 2)
        self.pipe_manager.reset()
        self.score = 0
        self.ticks = 0
        return self.get_state()

    def get_state(self):
        target_pipe = None
        for pipe in self.pipe_manager.pipes:
            if pipe.x + pipe.width >= self.bird.rect.left:
                target_pipe = pipe
                break

        if target_pipe is None and len(self.pipe_manager.pipes) > 0:
            target_pipe = self.pipe_manager.pipes[0]

        if target_pipe is None:
            gap_center_y = screen_height / 2.0
            pipe_dist_x = float(screen_width)
        else:
            gap_center_y = (target_pipe.top_rect.bottom + target_pipe.bottom_rect.top) / 2.0
            pipe_dist_x = float(target_pipe.x - self.bird.rect.left)

        return [
            float(self.bird.y) / screen_height,
            float(self.bird.velocity) / 10.0,
            float(pipe_dist_x) / screen_width,
            float(gap_center_y - self.bird.y) / screen_height,
        ]

    def step(self, action):
        flap = (action == 1)

        self.bird.update(flap=flap)
        self.pipe_manager.update(self.score)
        self.ticks += 1

        is_dead, new_score = check_environment(self.bird, self.pipe_manager.pipes, self.score)

        reward = 0.0
        if new_score > self.score:
            reward = 1.0
            self.score = new_score

        if is_dead:
            reward = -1.0

        done = is_dead
        info = {"score": self.score, "ticks": self.ticks}
        next_state = self.get_state()

        return next_state, reward, done, info
