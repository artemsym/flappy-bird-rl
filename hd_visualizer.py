"""HD-визуализация для Flappy Bird.

Слой полностью декоративный: он только читает
    bird.rect, bird.velocity, pipe.top_rect, pipe.bottom_rect
и ничего не пишет в объекты симуляции. Ядро (физика, AABB, сетка 80x80)
остаётся нетронутым и детерминированным.

Подключение (одна строка в основном файле):

    import hd_visualizer

Модуль сам перехватывает pygame.display.flip и рисует HD-сцену поверх
старого кадра. Отключить авто-подключение: FLAPPY_HD=0.

Либо вручную, без авто-хука:

    from hd_visualizer import HDVisualizer
    hd = HDVisualizer(window_width / screen_width,
                      window_height / screen_height)
    ...
    hd.draw(screen, bird, pipe_manager)
"""

from __future__ import annotations

import math
import os
import random

import pygame


# ---------------------------------------------------------------------------
# палитра (цвета, приближённые к оригинальной Flappy Bird)
# ---------------------------------------------------------------------------
SKY = (78, 192, 202)  # #4EC0CA

CITY_FAR = (219, 235, 190)
CITY_NEAR = (186, 216, 155)
WINDOW_LIGHT = (238, 246, 214)

GROUND_GRASS = (115, 191, 46)  # #73BF2E
GROUND_GRASS_DARK = (88, 155, 34)
GROUND_SAND = (222, 216, 149)
GROUND_SAND_DARK = (206, 198, 128)

PIPE_BODY = (115, 191, 46)
PIPE_LIGHT = (161, 222, 96)
PIPE_CORE = (192, 236, 132)
PIPE_DARK = (84, 148, 36)
PIPE_OUTLINE = (60, 104, 24)
PIPE_OPENING = (46, 86, 20)

BIRD_BODY = (248, 216, 0)  # #F8D800
BIRD_BELLY = (255, 252, 224)
BIRD_WING = (240, 172, 24)
BIRD_OUTLINE = (60, 44, 30)
BIRD_BEAK = (244, 122, 32)
BIRD_BEAK_DARK = (200, 82, 12)
BIRD_EYE = (255, 255, 255)
BIRD_PUPIL = (24, 24, 28)


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


class HDVisualizer:
    """Рисует HD-сцену по координатам симуляции 80x80.

    Параметры
    ---------
    scale_x, scale_y:
        Множители перевода координат сетки в пиксели окна.
        Например, для окна 600x600 при сетке 80x80 это 7.5 и 7.5.

    Все методы только читают переданные объекты. Ни один rect/velocity
    не изменяется.
    """

    def __init__(self, scale_x: float, scale_y: float) -> None:
        self.sx = float(scale_x)
        self.sy = float(scale_y)

        self._size = (0, 0)
        self._t = 0.0

        self._sky: pygame.Surface | None = None
        self._city_far: pygame.Surface | None = None
        self._city_near: pygame.Surface | None = None
        self._ground: pygame.Surface | None = None
        self._ground_h = 0
        self._clouds: list[dict] = []
        self._frame: pygame.Surface | None = None

        self._bird_cache: dict[tuple[int, int, int], pygame.Surface] = {}

    # ------------------------------------------------------------------
    # публичный API
    # ------------------------------------------------------------------
    def draw(
        self,
        screen: pygame.Surface,
        bird,
        pipes,
        score: int | None = None,
        high_score: int | None = None,
        title: str | None = None,
    ) -> None:
        """Нарисовать кадр.

        screen     -- целевая поверхность (обычно display surface);
        bird       -- объект с .rect и .velocity;
        pipes      -- PipeManager (с .pipes) либо обычный список Pipe;
        score      -- текущий счет (если None, пытается прочитать из bird или pipes);
        high_score -- лучший счет;
        title      -- заголовок или статус (например 'ITERATION 100').
        """
        width, height = screen.get_size()
        self._ensure_scene(width, height)

        self._t += 1.0 / 60.0

        frame = self._frame
        frame.blit(self._sky, (0, 0))
        self._draw_clouds(frame, width, height)
        self._draw_city(frame, width, height)

        pipe_list = getattr(pipes, "pipes", pipes)
        for pipe in pipe_list:
            self._draw_pipe(frame, pipe)

        # Земля перекрывает низ труб — как в оригинале.
        if self._ground is not None:
            frame.blit(self._ground, (0, height - self._ground_h))

        self._draw_bird(frame, bird)

        # Отрисовка HUD (счет, рекорд, бейдж агента)
        self._draw_hud(frame, score, high_score, title)

        screen.blit(frame, (0, 0))

        # Гарантируем непрозрачность окна
        screen.fill((0, 0, 0, 255), special_flags=pygame.BLEND_RGBA_MAX)

    def _draw_hud(
        self,
        surface: pygame.Surface,
        score: int | None,
        high_score: int | None,
        title: str | None = None,
    ) -> None:
        if not pygame.font.get_init():
            try:
                pygame.font.init()
            except Exception:
                return

        # Крупный счет по центру сверху
        if score is not None:
            font_size = max(28, int(38 * self.sy / 5.0))
            font = pygame.font.Font(None, font_size)
            text = str(score)
            text_shadow = font.render(text, True, (0, 0, 0))
            text_surf = font.render(text, True, (255, 255, 255))
            cx = surface.get_width() // 2
            cy = int(24 * self.sy / 5.0)
            rect = text_surf.get_rect(center=(cx, cy))
            surface.blit(text_shadow, rect.move(2, 2))
            surface.blit(text_surf, rect)

        # Рекорд в левом верхнем углу
        small_size = max(16, int(18 * self.sy / 5.0))
        small_font = pygame.font.Font(None, small_size)
        if high_score is not None:
            hs_text = f"RECORD: {high_score}"
            hs_shadow = small_font.render(hs_text, True, (0, 0, 0))
            hs_surf = small_font.render(hs_text, True, (255, 220, 40))
            surface.blit(hs_shadow, (11, 11))
            surface.blit(hs_surf, (10, 10))

        # Бейдж агента/игрока в правом верхнем углу
        if title:
            t_shadow = small_font.render(title, True, (0, 0, 0))
            t_surf = small_font.render(title, True, (210, 235, 255))
            tr = t_surf.get_rect(topright=(surface.get_width() - 10, 10))
            surface.blit(t_shadow, tr.move(1, 1))
            surface.blit(t_surf, tr)

    # ------------------------------------------------------------------
    # геометрия
    # ------------------------------------------------------------------
    def _rect(self, rect: pygame.Rect) -> pygame.Rect:
        """Перевод rect из сетки 80x80 в пиксели. Только чтение."""
        return pygame.Rect(
            round(rect.x * self.sx),
            round(rect.y * self.sy),
            max(1, round(rect.w * self.sx)),
            max(0, round(rect.h * self.sy)),
        )

    # ------------------------------------------------------------------
    # фон
    # ------------------------------------------------------------------
    def _ensure_scene(self, width: int, height: int) -> None:
        if (width, height) == self._size and self._sky is not None:
            return

        self._size = (width, height)
        self._sky = self._build_sky(width, height)
        self._ground_h = max(10, int(height * 0.085))
        self._ground = self._build_ground(width, self._ground_h)
        self._city_far = self._build_city(
            width, height, int(height * 0.20), CITY_FAR, seed=11, windows=False
        )
        self._city_near = self._build_city(
            width, height, int(height * 0.13), CITY_NEAR, seed=27, windows=True
        )
        self._build_clouds(width, height)
        self._frame = pygame.Surface((width, height))
        self._bird_cache.clear()

    def _build_ground(self, width: int, height: int) -> pygame.Surface:
        ground = pygame.Surface((width, height))
        ground.fill(GROUND_SAND)

        grass_h = max(3, int(height * 0.42))
        pygame.draw.rect(ground, GROUND_GRASS, (0, 0, width, grass_h))
        pygame.draw.rect(
            ground, GROUND_GRASS_DARK, (0, grass_h - max(1, grass_h // 4), width, max(1, grass_h // 4))
        )

        # вертикальные травинки
        blade_w = max(1, width // 120)
        step = max(3, width // 40)
        for x in range(0, width, step):
            pygame.draw.rect(ground, GROUND_GRASS_DARK, (x, 0, blade_w, grass_h))

        # диагональная штриховка песка
        stripe = max(1, height // 12)
        for x in range(-height, width, stripe * 3):
            pygame.draw.line(
                ground,
                GROUND_SAND_DARK,
                (x, height - 1),
                (x + height, grass_h),
                max(1, stripe // 2),
            )

        return ground

    def _build_sky(self, width: int, height: int) -> pygame.Surface:
        # Оригинальное небо — ровный бирюзовый цвет, без градиента.
        sky = pygame.Surface((width, height))
        sky.fill(SKY)
        return sky

    def _build_clouds(self, width: int, height: int) -> None:
        rng = random.Random(20261103)
        self._clouds = []

        for _ in range(9):
            cloud_w = int(rng.uniform(0.12, 0.26) * width)
            cloud_h = max(8, int(cloud_w * rng.uniform(0.42, 0.58)))
            surface = self._make_cloud(cloud_w, cloud_h)

            depth = rng.random()  # 0 = далеко, 1 = близко
            self._clouds.append(
                {
                    "surface": surface,
                    "x": rng.random(),
                    "y": rng.uniform(0.05, 0.50) * height,
                    "speed": (0.006 + 0.020 * depth) * width,
                    "alpha": int(170 + 85 * depth),
                }
            )

    @staticmethod
    def _make_cloud(width: int, height: int) -> pygame.Surface:
        # Плотное белое облако с плоским низом — как в оригинале.
        surface = pygame.Surface((width, height), pygame.SRCALPHA)
        base_h = max(2, int(height * 0.34))
        base = pygame.Rect(0, height - base_h, width, base_h)
        pygame.draw.rect(surface, (255, 255, 255), base, border_radius=base_h // 2)

        puffs = (
            (0.24, 0.60, 0.42, 0.72),
            (0.50, 0.40, 0.48, 0.90),
            (0.76, 0.60, 0.40, 0.72),
        )
        for fx, fy, fw, fh in puffs:
            rect = pygame.Rect(
                int((fx - fw / 2) * width),
                int((fy - fh / 2) * height),
                int(fw * width),
                int(fh * height),
            )
            pygame.draw.ellipse(surface, (255, 255, 255), rect)
        return surface

    def _draw_clouds(self, screen: pygame.Surface, width: int, height: int) -> None:
        for cloud in self._clouds:
            surface = cloud["surface"]
            span = width + surface.get_width()
            x = (cloud["x"] * span + self._t * cloud["speed"]) % span - surface.get_width()
            surface.set_alpha(cloud["alpha"])
            screen.blit(surface, (int(x), int(cloud["y"])))

    def _build_city(
        self,
        width: int,
        height: int,
        base_height: int,
        color,
        seed: int,
        windows: bool,
    ) -> pygame.Surface:
        base_height = max(12, base_height)
        surface = pygame.Surface((width, base_height), pygame.SRCALPHA)
        rng = random.Random(seed)

        x = -rng.randint(0, 10)
        while x < width:
            building_w = rng.randint(max(6, int(width * 0.025)), max(10, int(width * 0.075)))
            building_h = rng.randint(int(base_height * 0.22), base_height)
            top = base_height - building_h

            pygame.draw.rect(surface, color, (x, top, building_w, building_h))

            if rng.random() < 0.45:
                pygame.draw.rect(
                    surface,
                    color,
                    (x + building_w // 3, top - rng.randint(2, 6), max(2, building_w // 4), 6),
                )

            if windows and building_w >= 10:
                for wy in range(top + 5, base_height - 4, 7):
                    for wx in range(x + 3, x + building_w - 3, 6):
                        if rng.random() < 0.30:
                            pygame.draw.rect(surface, (*WINDOW_LIGHT, 150), (wx, wy, 2, 3))

            x += building_w + rng.randint(2, 9)

        return surface

    def _draw_city(self, screen: pygame.Surface, width: int, height: int) -> None:
        baseline = height - self._ground_h
        far = self._city_far
        near = self._city_near
        if far is not None:
            screen.blit(far, (0, baseline - far.get_height()))
        if near is not None:
            screen.blit(near, (0, baseline - near.get_height()))

    # ------------------------------------------------------------------
    # трубы
    # ------------------------------------------------------------------
    def _draw_pipe(self, screen: pygame.Surface, pipe) -> None:
        top = self._rect(pipe.top_rect)
        bottom = self._rect(pipe.bottom_rect)

        if top.h > 0:
            self._draw_pipe_body(screen, top)
            self._draw_pipe_cap(screen, top, at_bottom=True)

        if bottom.h > 0:
            self._draw_pipe_body(screen, bottom)
            self._draw_pipe_cap(screen, bottom, at_bottom=False)

    def _draw_pipe_body(self, screen: pygame.Surface, rect: pygame.Rect) -> None:
        pygame.draw.rect(screen, PIPE_BODY, rect)

        edge = max(2, rect.w // 6)
        pygame.draw.rect(screen, PIPE_LIGHT, (rect.x, rect.y, edge, rect.h))
        pygame.draw.rect(
            screen,
            PIPE_CORE,
            (rect.x + edge, rect.y, max(1, rect.w // 10), rect.h),
        )
        pygame.draw.rect(
            screen,
            PIPE_DARK,
            (rect.right - edge, rect.y, edge, rect.h),
        )

        outline = max(1, round(self.sx * 0.35))
        pygame.draw.rect(screen, PIPE_OUTLINE, rect, width=outline)

    def _draw_pipe_cap(self, screen: pygame.Surface, rect: pygame.Rect, at_bottom: bool) -> None:
        extra = max(4, int(rect.w * 0.18))
        cap_h = max(8, min(int(rect.w * 0.46), max(10, rect.h // 2)))
        cap = pygame.Rect(
            rect.x - extra // 2,
            rect.bottom - cap_h if at_bottom else rect.y,
            rect.w + extra,
            cap_h,
        )
        radius = max(2, cap_h // 3)
        outline = max(1, round(self.sx * 0.4))

        pygame.draw.rect(screen, PIPE_BODY, cap, border_radius=radius)

        inset = max(2, extra // 2)
        inner = cap.inflate(-inset * 2, -max(2, cap_h // 3))
        pygame.draw.rect(screen, PIPE_LIGHT, inner, border_radius=max(1, radius - 2))

        core = inner.inflate(-max(2, inner.w // 4), -max(2, inner.h // 2))
        pygame.draw.rect(screen, PIPE_CORE, core, border_radius=max(1, radius - 3))

        pygame.draw.rect(screen, PIPE_OUTLINE, cap, width=outline, border_radius=radius)

        # тёмное отверстие трубы
        opening_h = max(2, cap_h // 4)
        opening = pygame.Rect(
            cap.x + inset,
            cap.bottom - opening_h if at_bottom else cap.y,
            max(2, cap.w - inset * 2),
            opening_h,
        )
        pygame.draw.rect(screen, PIPE_OPENING, opening, border_radius=max(1, opening_h // 2))

    # ------------------------------------------------------------------
    # птица
    # ------------------------------------------------------------------
    def _draw_bird(self, screen: pygame.Surface, bird) -> None:
        rect = self._rect(bird.rect)
        bw = max(3, rect.w)
        bh = max(3, rect.h)

        phase = int(self._t * 9.0) % 3
        sprite = self._bird_sprite(bw, bh, phase)

        angle = -_clamp(float(bird.velocity) * 7.0, -28.0, 65.0)
        rotated = pygame.transform.rotozoom(sprite, angle, 1.0)

        center = (rect.centerx, rect.centery)
        screen.blit(rotated, rotated.get_rect(center=center))

    def _bird_sprite(self, width: int, height: int, phase: int) -> pygame.Surface:
        key = (width, height, phase)
        cached = self._bird_cache.get(key)
        if cached is not None:
            return cached

        surface = pygame.Surface((width, height), pygame.SRCALPHA)
        w, h = float(width), float(height)
        outline = max(1, round(w * 0.055))

        # хвост
        tail = (
            (w * 0.00, h * 0.44),
            (w * 0.27, h * 0.30),
            (w * 0.27, h * 0.66),
        )
        pygame.draw.polygon(surface, BIRD_WING, tail)
        pygame.draw.polygon(surface, BIRD_OUTLINE, tail, width=outline)

        # тело
        body = pygame.Rect(int(w * 0.08), int(h * 0.05), int(w * 0.86), int(h * 0.90))
        pygame.draw.ellipse(surface, BIRD_BODY, body)

        # белый живот
        belly = pygame.Rect(int(w * 0.33), int(h * 0.52), int(w * 0.44), int(h * 0.36))
        pygame.draw.ellipse(surface, BIRD_BELLY, belly)

        # крыло: три фазы взмаха
        wing_w = max(3, int(w * 0.44))
        wing_h = max(2, int(h * 0.26))
        wing = pygame.Surface((wing_w, wing_h), pygame.SRCALPHA)
        wing_rect = wing.get_rect()
        pygame.draw.ellipse(wing, BIRD_WING, wing_rect)
        pygame.draw.ellipse(wing, BIRD_OUTLINE, wing_rect, width=outline)
        pygame.draw.ellipse(
            wing,
            (255, 226, 120),
            wing_rect.inflate(-max(2, wing_w // 2), -max(2, wing_h // 2)),
        )

        wing_angles = (26.0, 0.0, -24.0)
        wing = pygame.transform.rotozoom(wing, wing_angles[phase], 1.0)
        wing_y = h * (0.34 + 0.08 * (1 - abs(phase - 1)))
        surface.blit(wing, (int(w * 0.10), int(wing_y)))

        # контур тела поверх крыла
        pygame.draw.ellipse(surface, BIRD_OUTLINE, body, width=outline)

        # глаз
        eye = (int(w * 0.67), int(h * 0.33))
        eye_r = max(2, int(w * 0.16))
        pygame.draw.circle(surface, BIRD_EYE, eye, eye_r)
        pygame.draw.circle(surface, BIRD_OUTLINE, eye, eye_r, width=outline)
        pygame.draw.circle(
            surface,
            BIRD_PUPIL,
            (int(w * 0.705), int(h * 0.335)),
            max(1, int(w * 0.072)),
        )
        pygame.draw.circle(
            surface,
            BIRD_EYE,
            (int(w * 0.735), int(h * 0.305)),
            max(1, int(w * 0.026)),
        )

        # клюв: верхняя и нижняя половины
        beak_upper = (
            (int(w * 0.78), int(h * 0.44)),
            (int(w * 1.00), int(h * 0.54)),
            (int(w * 0.78), int(h * 0.55)),
        )
        beak_lower = (
            (int(w * 0.78), int(h * 0.55)),
            (int(w * 1.00), int(h * 0.54)),
            (int(w * 0.78), int(h * 0.64)),
        )
        pygame.draw.polygon(surface, BIRD_BEAK, beak_upper)
        pygame.draw.polygon(surface, BIRD_BEAK_DARK, beak_lower)
        pygame.draw.polygon(surface, BIRD_OUTLINE, beak_upper, width=outline)
        pygame.draw.polygon(surface, BIRD_OUTLINE, beak_lower, width=outline)

        self._bird_cache[key] = surface
        return surface


# ---------------------------------------------------------------------------
# автоматическое подключение к существующему игровому циклу
# ---------------------------------------------------------------------------
_AUTO = os.environ.get("FLAPPY_HD", "1") != "0"
_INSTALLED = False


def _auto_install() -> None:
    """Оборачивает pygame.display.flip, не меняя основной код игры."""
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    original_flip = pygame.display.flip
    visualizer: HDVisualizer | None = None
    visualizer_key: tuple[float, float] | None = None

    def flip(*args, **kwargs):
        nonlocal visualizer, visualizer_key

        import __main__
        import sys

        # 1. Берём активную поверхность дисплея Pygame напрямую
        screen = pygame.display.get_surface()
        if screen is None:
            return original_flip(*args, **kwargs)

        game = vars(__main__)
        fb = sys.modules.get("flappy_bird")
        fb_vars = vars(fb) if fb else {}

        # 2. Игровые объекты: ищем в __main__ или в модуле flappy_bird
        bird = game.get("bird") or fb_vars.get("bird")
        pipes = game.get("pipe_manager") or fb_vars.get("pipe_manager")

        if bird is not None and pipes is not None:
            surface_w, surface_h = screen.get_size()
            try:
                ww = game.get("window_width") or fb_vars.get("window_width") or surface_w
                sw = game.get("screen_width") or fb_vars.get("screen_width") or 80.0
                wh = game.get("window_height") or fb_vars.get("window_height") or surface_h
                sh = game.get("screen_height") or fb_vars.get("screen_height") or 80.0
                scale_x = float(ww) / float(sw)
                scale_y = float(wh) / float(sh)
            except (KeyError, TypeError, ZeroDivisionError):
                scale_x = surface_w / 80.0
                scale_y = surface_h / 80.0

            key = (scale_x, scale_y)
            if visualizer is None or visualizer_key != key:
                visualizer = HDVisualizer(scale_x, scale_y)
                visualizer_key = key

            score = game.get("score") if game.get("score") is not None else fb_vars.get("score")
            high_score = game.get("high_score") if game.get("high_score") is not None else fb_vars.get("high_score")
            title = game.get("hud_title") or fb_vars.get("hud_title")
            visualizer.draw(screen, bird, pipes, score=score, high_score=high_score, title=title)

        return original_flip(*args, **kwargs)

    pygame.display.flip = flip


if _AUTO:
    _auto_install()
