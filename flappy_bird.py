import sys
import random
import pygame
import hd_visualizer  # декоративный HD-слой

# ---------------------------------------------------------------------------
# Глобальные параметры симуляции (80x80)
# ---------------------------------------------------------------------------
screen_width = 80
screen_height = 80
gravity = 0.4
flap_strength = -3.5
initial_pipe_speed = 1.0
pipe_speed = initial_pipe_speed
pipe_gap = 28
pipe_frequency = 40
scale = 5
window_width = screen_width * scale
window_height = screen_height * scale

game_surface = pygame.Surface((screen_width, screen_height))


class Bird:
    def __init__(self, x=20, y=40.0):
        self.x = x
        self.y = float(y)
        self.velocity = 0.0
        self.rect = pygame.Rect(self.x, int(self.y), 6, 6)

    def update(self, flap=False):
        if flap:
            self.velocity += flap_strength
        self.velocity += gravity
        # Ограничиваем скорость падения для стабильности физики
        if self.velocity > 8.0:
            self.velocity = 8.0
        self.y += self.velocity
        self.rect.y = int(self.y)

    def reset(self, start_y=None):
        if start_y is None:
            start_y = screen_height // 2
        self.y = float(start_y)
        self.velocity = 0.0
        self.rect.y = int(self.y)

    def draw(self, surface):
        pygame.draw.rect(surface, (255, 255, 0), self.rect)


class Pipe:
    def __init__(self, x, width=10, gap=pipe_gap):
        self.x = float(x)
        self.width = width
        self.passed = False
        # Безопасный отступ от краев экрана (минимум 8px сверху и снизу)
        min_gap_y = 8
        max_gap_y = max(min_gap_y, screen_height - gap - 8)
        self.gap_y = random.randint(min_gap_y, max_gap_y)
        self.top_rect = pygame.Rect(int(self.x), 0, self.width, self.gap_y)
        self.bottom_rect = pygame.Rect(
            int(self.x),
            self.top_rect.height + gap,
            self.width,
            screen_height - (self.top_rect.height + gap),
        )

    def update(self, speed=None):
        if speed is None:
            speed = pipe_speed
        self.x -= speed
        self.top_rect.x = int(self.x)
        self.bottom_rect.x = int(self.x)

    def draw(self, surface):
        pygame.draw.rect(surface, (0, 255, 0), self.top_rect)
        pygame.draw.rect(surface, (0, 255, 0), self.bottom_rect)


class PipeManager:
    def __init__(self):
        self.speed = initial_pipe_speed
        self.pipes: list[Pipe] = []
        self.spawn_distance = 0.0
        self.reset()

    def reset(self):
        global pipe_speed
        self.speed = initial_pipe_speed
        pipe_speed = initial_pipe_speed
        self.pipes = [Pipe(screen_width)]
        self.spawn_distance = 0.0

    def update(self, ticks=0):
        global pipe_speed
        # Плавное увеличение сложности каждые 300 тиков
        if ticks > 0 and ticks % 300 == 0:
            self.speed = round(self.speed + 0.05, 2)
            pipe_speed = self.speed

        self.spawn_distance += self.speed
        if self.spawn_distance >= pipe_frequency:
            self.pipes.append(Pipe(screen_width))
            self.spawn_distance -= pipe_frequency

        for pipe in self.pipes[:]:
            pipe.update(self.speed)
            if pipe.x + pipe.width < 0:
                self.pipes.remove(pipe)

    def draw(self, surface):
        for pipe in self.pipes:
            pipe.draw(surface)


def check_environment(bird, pipes, current_score):
    """Проверка столкновений и прохождения труб."""
    # Столкновение с потолком или полом
    if bird.rect.top <= 0 or bird.rect.bottom >= screen_height:
        return True, current_score, False

    scored = False
    for pipe in pipes:
        if bird.rect.colliderect(pipe.top_rect) or bird.rect.colliderect(pipe.bottom_rect):
            return True, current_score, False
        if not pipe.passed and (pipe.x + pipe.width < bird.rect.left):
            pipe.passed = True
            current_score += 1
            scored = True

    return False, current_score, scored


# ---------------------------------------------------------------------------
# Единый класс состояния игры для переиспользования
# ---------------------------------------------------------------------------
class FlappyGame:
    def __init__(self, headless=False):
        self.bird = Bird()
        self.pipe_manager = PipeManager()
        self.score = 0
        self.high_score = 0
        self.ticks = 0
        self.headless = headless
        self.has_passed_first_pipe = False

    def reset(self):
        self.bird.reset(screen_height // 2)
        self.pipe_manager.reset()
        self.score = 0
        self.ticks = 0
        self.has_passed_first_pipe = False

    def step(self, flap=False):
        self.bird.update(flap)
        self.ticks += 1
        self.pipe_manager.update(self.ticks)

        is_dead, self.score, scored = check_environment(self.bird, self.pipe_manager.pipes, self.score)
        first_pipe_event = False
        new_record_event = False

        if scored:
            if self.score == 1 and not self.has_passed_first_pipe:
                self.has_passed_first_pipe = True
                first_pipe_event = True
            if self.score > self.high_score:
                self.high_score = self.score
                new_record_event = True

        return is_dead, self.score, scored, first_pipe_event, new_record_event


# ---------------------------------------------------------------------------
# Совместимость с глобальным контекстом модуля
# ---------------------------------------------------------------------------
clock = pygame.time.Clock()
bird = Bird()
pipe_manager = PipeManager()
score = 0
high_score = 0
ticks = 0
hud_title = "HUMAN PLAYER"
running = True


def draw_pixel_hud(surface, current_score, best_score):
    """Отрисовка очков на случай отключения HD-слоя."""
    if not pygame.font.get_init():
        try:
            pygame.font.init()
        except Exception:
            return
    font = pygame.font.Font(None, 24)
    text = font.render(f"Score: {current_score}", True, (255, 255, 255))
    surface.blit(text, (5, 5))


def run_interactive():
    global score, high_score, ticks, running, pipe_speed
    pygame.init()
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Flappy Bird")

    print("\n🎮 Запуск Flappy Bird! Управление: [ПРОБЕЛ] - прыжок, [Q/ESC] - выход.\n")
    first_pipe_announced = False

    while running:
        flap = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    flap = True
                elif event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False

        bird.update(flap)
        ticks += 1
        pipe_manager.update(ticks)

        is_dead, score, scored = check_environment(bird, pipe_manager.pipes, score)

        if scored:
            if score == 1 and not first_pipe_announced:
                first_pipe_announced = True
                print(f"🚩 Первая труба успешно пройдена! (Счет: 1, Тиков: {ticks})")
            else:
                print(f"🚩 Пройдена труба {score}!")

            if score > high_score:
                high_score = score
                print(f"🏆 Новый рекорд: {high_score} очков!")

        if is_dead:
            print(f"💥 Столкновение на тике {ticks}. Итоговый счет: {score}")
            bird.reset(screen_height // 2)
            pipe_manager.reset()
            score = 0
            ticks = 0
            first_pipe_announced = False

        game_surface.fill((0, 0, 0))
        bird.draw(game_surface)
        pipe_manager.draw(game_surface)

        scaled_surface = pygame.transform.scale(game_surface, (window_width, window_height))
        screen.blit(scaled_surface, (0, 0))
        draw_pixel_hud(screen, score, high_score)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == '__main__':
    run_interactive()
