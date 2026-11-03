from flappy_bird import (
    Bird,
    PipeManager,
    check_environment,
    screen_width,
    screen_height,
)


class FlappyEnv:
    """Gym-совместимое окружение Flappy Bird для обучения с подкреплением."""

    def __init__(self):
        self.bird = Bird()
        self.pipe_manager = PipeManager()
        self.score = 0
        self.ticks = 0
        self.best_score = 0
        self.has_ever_passed_pipe = False
        self.has_passed_pipe_this_episode = False

    def reset(self):
        self.bird.reset(screen_height // 2)
        self.pipe_manager.reset()
        self.score = 0
        self.ticks = 0
        self.has_passed_pipe_this_episode = False
        return self.get_state()

    def get_state(self):
        # Находим ближайшую впереди идущую трубу
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
            pipe_dist_x = target_pipe.x - self.bird.rect.left

        # Нормализованный 4-мерный вектор наблюдений
        return [
            float(self.bird.y) / screen_height,
            float(self.bird.velocity) / 10.0,
            float(pipe_dist_x) / screen_width,
            float(gap_center_y - self.bird.y) / screen_height,
        ]

    def step(self, action):
        flap = (action == 1)
        self.bird.update(flap=flap)
        self.ticks += 1
        self.pipe_manager.update(self.ticks)

        is_dead, new_score, scored = check_environment(
            self.bird, self.pipe_manager.pipes, self.score
        )

        reward = 0.1  # Небольшое вознаграждение за выживание
        first_pipe_passed = False
        first_time_ever_passed_pipe = False
        new_record = False

        if scored:
            reward = 5.0
            self.score = new_score

            if not self.has_passed_pipe_this_episode:
                self.has_passed_pipe_this_episode = True
                first_pipe_passed = True

            if not self.has_ever_passed_pipe:
                self.has_ever_passed_pipe = True
                first_time_ever_passed_pipe = True

            if self.score > self.best_score:
                self.best_score = self.score
                new_record = True

        if is_dead:
            reward = -5.0

        done = is_dead
        info = {
            "score": self.score,
            "ticks": self.ticks,
            "scored": scored,
            "first_pipe_passed": first_pipe_passed,
            "first_time_ever_passed_pipe": first_time_ever_passed_pipe,
            "new_record": new_record,
            "best_score": self.best_score,
        }
        next_state = self.get_state()

        return next_state, reward, done, info
