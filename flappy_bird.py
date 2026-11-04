import pygame
import random
import time
import hd_visualizer

screen_width = 80
screen_height = 80
gravity = 0.4
flap_strength = -3.5
pipe_frequency = 40
pipe_width = 10.0

sub_ticks = 4
dt = 1.0 / sub_ticks

initial_pipe_speed = 1.0
max_pipe_speed = 2.0
initial_pipe_gap = 28
min_pipe_gap = 7
max_difficulty_score = 100

scale = 5
window_width = int(screen_width * scale)
window_height = int(screen_height * scale)


class Bird:
    def __init__(self):
        self.x = 20
        self.y = 40.0
        self.velocity = 0.0
        self.rect = pygame.Rect(self.x, int(self.y), 6, 6)

    def sub_step(self, flap=False, dt_val=dt):
        if flap:
            self.velocity += flap_strength # НЕ пугаемся, это специально :3
        self.velocity += gravity * dt_val
        self.y += self.velocity * dt_val
        self.rect.y = int(self.y)
        return self.y <= 0.0 or (self.y + 6.0) >= screen_height

    def update(self, flap=False):
        for i in range(sub_ticks):
            hit = self.sub_step(flap=(flap if i == 0 else False), dt_val=dt)
            if hit:
                return True
        return False

    def reset(self, start_y):
        self.y = float(start_y)
        self.velocity = 0.0
        self.rect.y = int(self.y)

    def draw(self, surface):
        pygame.draw.rect(surface, (255, 255, 0), self.rect)


class Pipe:
    def __init__(self, x, gap=initial_pipe_gap, gap_y=None):
        self.x = float(x)
        self.width = pipe_width
        self.gap = gap
        self.passed = False

        if gap_y is not None:
            self.gap_y = float(gap_y)
        else:
            min_y = 2
            max_y = max(min_y, screen_height - self.gap - 2)
            self.gap_y = float(random.randint(min_y, max_y))

        self.top_rect = pygame.Rect(int(self.x), 0, int(self.width), int(self.gap_y))
        bottom_height = screen_height - int(self.gap_y + self.gap)
        self.bottom_rect = pygame.Rect(int(self.x), int(self.gap_y + self.gap), int(self.width), bottom_height)

    def sub_step(self, speed, dt_val=dt):
        self.x -= speed * dt_val
        self.top_rect.x = int(self.x)
        self.bottom_rect.x = int(self.x)

    def draw(self, surface):
        pygame.draw.rect(surface, (0, 255, 0), self.top_rect)
        pygame.draw.rect(surface, (0, 255, 0), self.bottom_rect)


class PipeManager:
    def __init__(self):
        self.speed = initial_pipe_speed
        self.gap = initial_pipe_gap
        self.preloaded_gaps = None
        self.reset()

    def reset(self, preloaded_gaps=None):
        self.speed = initial_pipe_speed
        self.gap = initial_pipe_gap
        self.pipes = []
        self.spawn_distance = 0.0
        self.preloaded_gaps = list(preloaded_gaps) if preloaded_gaps else None
        gap_y = self.preloaded_gaps.pop(0) if self.preloaded_gaps else None
        self.pipes.append(Pipe(screen_width, gap=self.gap, gap_y=gap_y))

    def update_difficulty(self, score):
        progress = min(1.0, score / max_difficulty_score)
        self.gap = int(initial_pipe_gap - progress * (initial_pipe_gap - min_pipe_gap))
        self.speed = round(initial_pipe_speed + progress * (max_pipe_speed - initial_pipe_speed), 2)

    def sub_step(self, score=0, dt_val=dt):
        self.update_difficulty(score)
        self.spawn_distance += self.speed * dt_val
        if self.spawn_distance >= pipe_frequency:
            if self.preloaded_gaps:
                gap_y = self.preloaded_gaps.pop(0)
            else:
                min_y = 2
                max_y = max(min_y, screen_height - self.gap - 2)
                if self.pipes:
                    last_gap = self.pipes[-1].gap_y
                    frames = (pipe_frequency - (pipe_width + 6)) / self.speed
                    substeps = max(1, int(round(frames * sub_ticks)))
                    max_fall = 0.0
                    v = 0.0
                    for _ in range(substeps):
                        v += gravity * dt
                        max_fall += v * dt
                    allowed_down = max_fall * 0.85
                    allowed_up = 45.0
                    low = max(min_y, int(last_gap - allowed_up))
                    high = min(max_y, int(last_gap + allowed_down))
                    gap_y = random.randint(low, high) if low <= high else min(low, max_y)
                else:
                    gap_y = random.randint(min_y, max_y)

            self.pipes.append(Pipe(screen_width, gap=self.gap, gap_y=gap_y))
            self.spawn_distance -= pipe_frequency

        for pipe in self.pipes[:]:
            pipe.sub_step(self.speed, dt_val=dt_val)
            if pipe.x + pipe.width < 0:
                self.pipes.remove(pipe)

    def update(self, score=0):
        for _ in range(sub_ticks):
            self.sub_step(score=score, dt_val=dt)

    def draw(self, surface):
        for pipe in self.pipes:
            pipe.draw(surface)


def check_environment(bird, pipes, current_score):
    if bird.y <= 0.0 or (bird.y + 6.0) >= screen_height:
        return True, current_score

    for pipe in pipes:
        if bird.rect.colliderect(pipe.top_rect) or bird.rect.colliderect(pipe.bottom_rect):
            return True, current_score
        if not pipe.passed and (pipe.x + pipe.width < bird.rect.left):
            pipe.passed = True
            current_score += 1

    return False, current_score


if __name__ == '__main__':
    pygame.init()
    pygame.display.set_caption("Flappy Bird") # шаманство мне под никсос, блядска хуйня не запускается нормально :(((
    screen = pygame.display.set_mode((window_width, window_height))
    game_surface = pygame.Surface((screen_width, screen_height))
    clock = pygame.time.Clock()

    bird = Bird()
    pipe_manager = PipeManager()

    score = 0
    high_score = 0
    ticks = 0
    running = True

    while running:
        flap = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    flap = True

        bird.update(flap)
        pipe_manager.update(score)
        ticks += 1

        is_dead, score = check_environment(bird, pipe_manager.pipes, score)
        if is_dead:
            if score > high_score:
                high_score = score
            bird.reset(screen_height // 2)
            pipe_manager.reset()
            score = 0
            ticks = 0

        game_surface.fill((0, 0, 0))
        bird.draw(game_surface)
        pipe_manager.draw(game_surface)

        scaled_surface = pygame.transform.scale(game_surface, (window_width, window_height))
        screen.blit(scaled_surface, (0, 0))
        pygame.display.flip()

        clock.tick(30)

    pygame.quit()
