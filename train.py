import json
import random
import sys
from pathlib import Path
import pygame
from flappy_bird import (
    Bird,
    PipeManager,
    check_environment,
    screen_width,
    screen_height,
    window_width,
    window_height,
)
from flappy_core import RustFlappyEnv
from agent import DQNAgent
import hd_visualizer


def train(episodes=1000, target_sync=10, patience=50):
    replays_dir = Path("replays")
    replays_dir.mkdir(exist_ok=True)

    env = RustFlappyEnv()
    agent = DQNAgent(episodes=episodes)
    high_score = 0

    step_count = 0
    scores_window = []

    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        actions_history = []
        episode_transitions = []
        score = 0

        while not done:
            action = agent.select_action(state)
            actions_history.append(action)

            next_state, reward, done, score, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            episode_transitions.append((state, action, reward, next_state, done))
            step_count += 1
            if step_count % 4 == 0:
                agent.train_step(high_score=high_score)
            state = next_state

        if episode >= 300 or high_score >= 2:
            agent.epsilon = 0.0
        else:
            agent.decay_epsilon()

        scores_window.append(score)
        if len(scores_window) > 20:
            scores_window.pop(0)
        avg_score = sum(scores_window) / len(scores_window)
        agent.check_plateau(avg_score, patience=patience)

        replay_data = {
            "episode": episode,
            "score": score,
            "actions": actions_history,
            "pipe_gaps": env.get_pipe_gaps(),
        }
        with open(replays_dir / f"ep_{episode}.json", "w") as f:
            json.dump(replay_data, f)

        if score > high_score:
            high_score = score
            agent.update_best_trajectory(episode_transitions)
            with open(replays_dir / "best.json", "w") as f:
                json.dump(replay_data, f)

        drift = agent.critic_drift()
        print(f"№{episode:<4d} critic {drift:+5.1f}% lr {agent.lr:.4f} hr {high_score:<3d} score {score:<3d}")


def run_replay(replay_data, screen, clock, game_surface, fps_state):
    speeds = [30, 60, 80, 100, 120, 240]

    bird = Bird()
    pipe_manager = PipeManager()
    pipe_manager.reset(replay_data.get("pipe_gaps"))
    score = 0

    flappy_globals = sys.modules["flappy_bird"]
    flappy_globals.screen = screen
    flappy_globals.bird = bird
    flappy_globals.pipe_manager = pipe_manager
    flappy_globals.score = score
    flappy_globals.high_score = replay_data.get("score", 0)
    flappy_globals.hud_title = f"EP {replay_data.get('episode', '?')}"

    for action in replay_data["actions"]:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE)):
                return False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                fps_state["idx"] = min(len(speeds) - 1, fps_state["idx"] + 1)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                fps_state["idx"] = max(0, fps_state["idx"] - 1)

        dead_bound = bird.update(flap=(action == 1))
        pipe_manager.update(score)
        is_dead, score = check_environment(bird, pipe_manager.pipes, score)
        flappy_globals.score = score

        game_surface.fill((0, 0, 0))
        bird.draw(game_surface)
        pipe_manager.draw(game_surface)

        screen.blit(pygame.transform.scale(game_surface, (window_width, window_height)), (0, 0))
        pygame.display.flip()
        clock.tick(speeds[fps_state["idx"]])

        if dead_bound or is_dead:
            break

    return True


def play(replay_target="best"):
    replays_dir = Path("replays")
    if str(replay_target).isdigit():
        replay_path = replays_dir / f"ep_{replay_target}.json"
    else:
        replay_path = replays_dir / f"{replay_target}.json"

    if not replay_path.is_file():
        print(f"реплей {replay_path} не найден")
        return

    with open(replay_path, "r") as f:
        replay_data = json.load(f)

    pygame.init()
    pygame.display.set_caption("Flappy Bird")
    screen = pygame.display.set_mode((window_width, window_height))
    clock = pygame.time.Clock()
    game_surface = pygame.Surface((screen_width, screen_height))
    fps_state = {"idx": 0}

    run_replay(replay_data, screen, clock, game_surface, fps_state)
    pygame.quit()


def play_all(filter_score=None):
    replays_dir = Path("replays")
    files = sorted(replays_dir.glob("ep_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    if not files:
        print("нет реплеев для воспроизведения")
        return

    # filter_score может быть:
    # int (например 14 -> exact 14)
    # tuple (например (12, 14) -> диапазон с 12 по 14 включительно)
    selected_replays = []
    for f in files:
        try:
            with open(f, "r") as fp:
                data = json.load(fp)
            sc = data.get("score", 0)
            if filter_score is not None:
                if isinstance(filter_score, tuple):
                    low, high = filter_score
                    if not (low <= sc <= high):
                        continue
                elif isinstance(filter_score, int):
                    if sc != filter_score:
                        continue
            selected_replays.append(data)
        except Exception:
            pass

    if not selected_replays:
        print(f"нет реплеев под фильтр: {filter_score}")
        return

    print(f"Найдено реплеев: {len(selected_replays)}")

    pygame.init()
    pygame.display.set_caption("Flappy Bird")
    screen = pygame.display.set_mode((window_width, window_height))
    clock = pygame.time.Clock()
    game_surface = pygame.Surface((screen_width, screen_height))
    fps_state = {"idx": 0}

    for data in selected_replays:
        print(f"Воспроизведение: Эпизод {data.get('episode')}, Скор: {data.get('score')}")
        keep_running = run_replay(data, screen, clock, game_surface, fps_state)
        if not keep_running:
            break

    pygame.quit()


if __name__ == "__main__":
    # Проверка аргументов вида --playall14 или --playall12_14 или playall12_14
    playall_arg = None
    for arg in sys.argv[1:]:
        cleaned = arg.lstrip("-")
        if cleaned.startswith("playall"):
            playall_arg = cleaned[len("playall"):]
            break

    if playall_arg is not None:
        if not playall_arg:
            play_all()
        elif "_" in playall_arg:
            parts = playall_arg.split("_")
            low, high = int(parts[0]), int(parts[1])
            play_all(filter_score=(low, high))
        elif playall_arg.isdigit():
            play_all(filter_score=int(playall_arg))
        else:
            play_all()
    elif "--play" in sys.argv:
        idx = sys.argv.index("--play")
        target = sys.argv[idx + 1] if len(sys.argv) > idx + 1 else "best"
        play(target)
    else:
        episodes_input = int(input("эпизоды: "))
        train(episodes=episodes_input)
