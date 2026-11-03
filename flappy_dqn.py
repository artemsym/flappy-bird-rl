import argparse
import os
import sys
import time
from pathlib import Path
import pygame
import torch

from dqn import DQNAgent
from flappy_bird import (
    Bird,
    PipeManager,
    check_environment,
    screen_width,
    screen_height,
    window_width,
    window_height,
    draw_pixel_hud,
)
from flappy_env import FlappyEnv
import hd_visualizer


def resolve_checkpoint_path(checkpoint_arg: str, checkpoint_dir: str = "checkpoints") -> Path:
    """Определяет путь к чекпоинту по номеру итерации или ключевому слову."""
    cdir = Path(checkpoint_dir)
    arg_lower = str(checkpoint_arg).strip().lower()

    # 1. Прямой путь к существующему файлу
    direct_path = Path(checkpoint_arg)
    if direct_path.is_file():
        return direct_path

    in_dir = cdir / checkpoint_arg
    if in_dir.is_file():
        return in_dir

    if (cdir / f"{checkpoint_arg}.pth").is_file():
        return cdir / f"{checkpoint_arg}.pth"

    # 2. Ключевые слова
    if arg_lower in ("best", "b"):
        best_p = cdir / "flappy_dqn_best.pth"
        if best_p.is_file():
            return best_p
        # Если нет в директории checkpoints, проверяем корень
        if Path("flappy_dqn_best.pth").is_file():
            return Path("flappy_dqn_best.pth")

    if arg_lower in ("latest", "last", "l"):
        last_p = cdir / "flappy_dqn_latest.pth"
        if last_p.is_file():
            return last_p
        if Path("flappy_dqn_final.pth").is_file():
            return Path("flappy_dqn_final.pth")

    # 3. Числовой номер итерации (эпизода), например "50", "100"
    if checkpoint_arg.isdigit():
        num = int(checkpoint_arg)
        candidate = cdir / f"checkpoint_ep{num}.pth"
        if candidate.is_file():
            return candidate

    # 4. Поиск частичных совпадений среди чекпоинтов
    if cdir.is_dir():
        matches = list(cdir.glob(f"*{checkpoint_arg}*.pth"))
        if matches:
            return sorted(matches)[-1]

    raise FileNotFoundError(
        f"Контрольная точка '{checkpoint_arg}' не найдена в каталоге {cdir.resolve()}.\n"
        f"Используйте --list, чтобы посмотреть доступные итерации."
    )


def list_checkpoints(checkpoint_dir: str = "checkpoints"):
    """Выводит список всех сохраненных итераций и чекпоинтов."""
    cdir = Path(checkpoint_dir)
    print(f"\n📂 Доступные контрольные точки в каталоге '{cdir.resolve()}':\n")
    if not cdir.is_dir():
        print("   (каталог с чекпоинтами пока не создан)")
        return

    checkpoints = sorted(cdir.glob("*.pth"))
    if not checkpoints:
        print("   (нет сохраненных файлов .pth)")
        return

    print(f" {'#':<3} | {'Имя файла':<32} | {'Эпизод':<8} | {'Рекорд':<7} | {'Размер':<9} | {'Дата':<19}")
    print("-" * 88)

    for i, cp in enumerate(checkpoints, 1):
        try:
            data = torch.load(cp, map_location="cpu")
            if isinstance(data, dict):
                ep = str(data.get("episode", "-"))
                score = str(data.get("best_score", "-"))
            else:
                ep = "-"
                score = "-"
        except Exception:
            ep = "?"
            score = "?"

        size_kb = f"{cp.stat().st_size / 1024:.1f} KB"
        mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(cp.stat().st_mtime))
        print(f" {i:<3} | {cp.name:<32} | {ep:<8} | {score:<7} | {size_kb:<9} | {mtime:<19}")
    print()


def train(episodes: int = 1000, save_interval: int = 50, target_sync: int = 10, checkpoint_dir: str = "checkpoints"):
    """Обучение DQN агента с подробным выводом событий в терминал."""
    cdir = Path(checkpoint_dir)
    cdir.mkdir(parents=True, exist_ok=True)

    env = FlappyEnv()
    agent = DQNAgent(state_dim=4, action_dim=2)

    best_score = 0
    total_pipes_passed = 0
    first_pipe_overall_notified = False

    print("\n" + "=" * 70)
    print("🚀 СТАРТ ОБУЧЕНИЯ DQN АГЕНТА ДЛЯ FLAPPY BIRD")
    print(f"   Эпизодов: {episodes} | Сохранение каждые {save_interval} эп. | Синхронизация target: каждые {target_sync} эп.")
    print(f"   Устройство: {agent.device} | Папка чекпоинтов: {cdir.resolve()}")
    print("=" * 70 + "\n")

    for episode in range(1, episodes + 1):
        state = env.reset()
        episode_reward = 0.0
        done = False
        last_loss = 0.0
        loss_count = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            # Сохраняем переход и делаем шаг обучения
            agent.memory.push(state, action, reward, next_state, done)
            loss = agent.train_step()
            if loss is not None:
                last_loss += loss
                loss_count += 1

            # Оповещение: впервые за всё обучение пройдена труба
            if info.get("first_time_ever_passed_pipe") and not first_pipe_overall_notified:
                first_pipe_overall_notified = True
                print(
                    f"\n🌟 [Эпизод {episode:4d}] ВЕХА ОБУЧЕНИЯ: Агент ВПЕРВЫЕ преодолел первую трубу! "
                    f"(Тик: {info['ticks']}, Награда: {reward:+.1f})\n"
                )
            # Оповещение: первая труба в текущем эпизоде
            elif info.get("first_pipe_passed"):
                print(f"🚩 [Эпизод {episode:4d}] Пройдена 1-я труба! (Тик: {info['ticks']})")
            # Пройдена последующая труба
            elif info.get("scored") and info["score"] > 1:
                print(f"🚩 [Эпизод {episode:4d}] Преодолена труба #{info['score']}! (Тик: {info['ticks']})")

            # Оповещение: новый рекорд
            if info.get("new_record"):
                best_score = info["score"]
                record_path = cdir / f"checkpoint_record_{best_score}.pth"
                best_path = cdir / "flappy_dqn_best.pth"
                agent.save(record_path, episode=episode, best_score=best_score)
                agent.save(best_path, episode=episode, best_score=best_score)
                print(
                    f"🏆 [Эпизод {episode:4d}] НОВЫЙ ВСЕОБЩИЙ РЕКОРД: {best_score} очков! "
                    f"(Тик: {info['ticks']}) -> Сохранен чекпоинт: {record_path.name}"
                )

            state = next_state
            episode_reward += reward

        # Уменьшаем epsilon раз в эпизод
        agent.decay_epsilon()

        # Периодическая синхронизация target-сети
        if episode % target_sync == 0:
            agent.update_target()

        # Сохранение периодической контрольной точки (любая итерация)
        if episode % save_interval == 0:
            cp_path = cdir / f"checkpoint_ep{episode}.pth"
            latest_path = cdir / "flappy_dqn_latest.pth"
            agent.save(cp_path, episode=episode, best_score=best_score)
            agent.save(latest_path, episode=episode, best_score=best_score)
            print(f"💾 [Чекпоинт] Сохранена итерация {episode} в {cp_path.name}")

        # Логирование каждые 10 эпизодов
        if episode % 10 == 0 or episode == 1:
            avg_loss_str = f"{last_loss / max(1, loss_count):.4f}" if loss_count > 0 else "---"
            print(
                f"📊 Эпизод: {episode:4d}/{episodes} | "
                f"Счет: {info['score']:2d} (Рекорд: {best_score:2d}) | "
                f"Тики: {info['ticks']:4d} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Loss: {avg_loss_str} | "
                f"Буфер: {len(agent.memory):5d}"
            )

    # Финальное сохранение
    final_path = cdir / "flappy_dqn_latest.pth"
    agent.save(final_path, episode=episodes, best_score=best_score)
    print("\n" + "=" * 70)
    print(f"✅ Обучение успешно завершено! Итоговый рекорд: {best_score} очков.")
    print(f"   Финальная модель: {final_path.resolve()}")
    print("=" * 70 + "\n")


def play(checkpoint_arg: str, fps: int = 30, checkpoint_dir: str = "checkpoints"):
    """Запуск визуальной игры с выбранной итерацией/моделью агента."""
    model_path = resolve_checkpoint_path(checkpoint_arg, checkpoint_dir)

    agent = DQNAgent(state_dim=4, action_dim=2)
    saved_ep, saved_best = agent.load(model_path)
    agent.set_eval_mode()

    pygame.init()
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption(f"Flappy Bird — Agent [{model_path.name}]")
    clock = pygame.time.Clock()
    game_surface = pygame.Surface((screen_width, screen_height))

    bird = Bird()
    pipe_manager = PipeManager()

    # Глобальные переменные модуля для HD-визуализатора
    flappy_globals = sys.modules["flappy_bird"]
    flappy_globals.screen = screen
    flappy_globals.bird = bird
    flappy_globals.pipe_manager = pipe_manager
    flappy_globals.hud_title = f"AGENT [Ep {saved_ep}]"

    print("\n" + "=" * 70)
    print(f"🎮 ЗАПУСК АГЕНТА ИЗ ИТЕРАЦИИ: {model_path.name}")
    print(f"   Обучен на эпизоде: {saved_ep} | Рекорд чекпоинта: {saved_best}")
    print("   Управление в окне: [ПРОБЕЛ] - пауза, [ВВЕРХ/ВНИЗ] - изменить скорость (FPS), [Q/ESC] - выход.")
    print("=" * 70 + "\n")

    session_record = 0
    game_episode = 1
    ticks = 0
    score = 0
    paused = False
    running = True
    first_pipe_announced = False

    def get_agent_state():
        target_pipe = None
        for p in pipe_manager.pipes:
            if p.x + p.width >= bird.rect.left:
                target_pipe = p
                break
        if target_pipe is None and len(pipe_manager.pipes) > 0:
            target_pipe = pipe_manager.pipes[0]

        if target_pipe is None:
            gap_center_y = screen_height / 2.0
            dist_x = float(screen_width)
        else:
            gap_center_y = (target_pipe.top_rect.bottom + target_pipe.bottom_rect.top) / 2.0
            dist_x = target_pipe.x - bird.rect.left

        return [
            float(bird.y) / screen_height,
            float(bird.velocity) / 10.0,
            float(dist_x) / screen_width,
            float(gap_center_y - bird.y) / screen_height,
        ]

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    print("⏸️ Пауза" if paused else "▶️ Возобновление")
                elif event.key == pygame.K_UP:
                    fps = min(240, fps + 15)
                    print(f"⚡ Скорость: {fps} FPS")
                elif event.key == pygame.K_DOWN:
                    fps = max(10, fps - 15)
                    print(f"🐢 Скорость: {fps} FPS")

        if not paused:
            # Агент выбирает действие
            state = get_agent_state()
            action = agent.select_action(state, eval_mode=True)
            flap = (action == 1)

            bird.update(flap)
            ticks += 1
            pipe_manager.update(ticks)

            is_dead, score, scored = check_environment(bird, pipe_manager.pipes, score)

            # Синхронизация глобалов для HD-слоя
            flappy_globals.score = score
            flappy_globals.high_score = session_record

            if scored:
                if score == 1 and not first_pipe_announced:
                    first_pipe_announced = True
                    print(f"🚩 [Раунд {game_episode:3d}] Агент преодолел 1-ю трубу! (Счет: 1, Тик: {ticks})")
                else:
                    print(f"🚩 [Раунд {game_episode:3d}] Пройдена труба #{score}!")

                if score > session_record:
                    session_record = score
                    flappy_globals.high_score = session_record
                    print(f"🏆 [Раунд {game_episode:3d}] НОВЫЙ РЕКОРД ТЕКУЩЕЙ СЕССИИ: {session_record} очков!")

            if is_dead:
                print(f"💥 [Раунд {game_episode:3d}] Столкновение на тике {ticks}. Итоговый счет раунда: {score}")
                game_episode += 1
                bird.reset(screen_height // 2)
                pipe_manager.reset()
                score = 0
                ticks = 0
                first_pipe_announced = False
                flappy_globals.score = score

        # Отрисовка
        game_surface.fill((0, 0, 0))
        bird.draw(game_surface)
        pipe_manager.draw(game_surface)

        scaled_surface = pygame.transform.scale(game_surface, (window_width, window_height))
        screen.blit(scaled_surface, (0, 0))
        draw_pixel_hud(screen, score, session_record)

        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()
    print(f"\n👋 Просмотр завершен. Лучший счет в сессии: {session_record}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Flappy Bird DQN — обучение и визуализация любых итераций агента."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--train",
        action="store_true",
        help="Запустить тренировку DQN агента с терминальными оповещениями.",
    )
    group.add_argument(
        "--play",
        metavar="ITERATION_OR_PATH",
        nargs="?",
        const="best",
        help="Открыть и посмотреть игру агента. Примеры: --play best, --play latest, --play 100, --play checkpoints/checkpoint_ep50.pth",
    )
    group.add_argument(
        "--list",
        action="store_true",
        help="Показать список всех сохраненных итераций и чекпоинтов.",
    )

    parser.add_argument("--episodes", type=int, default=1000, help="Количество эпизодов обучения (по умолчанию: 1000)")
    parser.add_argument("--save-interval", type=int, default=50, help="Интервал сохранения чекпоинтов (по умолчанию: 50)")
    parser.add_argument("--fps", type=int, default=30, help="Частота кадров при воспроизведении (по умолчанию: 30)")
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints", help="Папка с чекпоинтами (по умолчанию: checkpoints)")

    args = parser.parse_args()

    if args.list:
        list_checkpoints(args.checkpoints_dir)
    elif args.play is not None:
        play(args.play, fps=args.fps, checkpoint_dir=args.checkpoints_dir)
    else:
        # По умолчанию запускаем обучение
        train(
            episodes=args.episodes,
            save_interval=args.save_interval,
            checkpoint_dir=args.checkpoints_dir,
        )


if __name__ == "__main__":
    main()
