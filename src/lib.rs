use pyo3::prelude::*;
use rand::Rng;

const SCREEN_WIDTH: f32 = 80.0;
const SCREEN_HEIGHT: f32 = 80.0;
const GRAVITY: f32 = 0.4;
const FLAP_STRENGTH: f32 = -3.5;
const PIPE_WIDTH: f32 = 10.0;
const PIPE_FREQUENCY: f32 = 40.0;

const BIRD_WIDTH: f32 = 6.0;
const BIRD_HEIGHT: f32 = 6.0;

const SUB_TICKS: usize = 4;
const DT: f32 = 1.0 / 4.0;

const INITIAL_PIPE_SPEED: f32 = 1.0;
const MAX_PIPE_SPEED: f32 = 2.0;
const INITIAL_PIPE_GAP: f32 = 28.0;
const MIN_PIPE_GAP: f32 = 7.0;
const MAX_DIFFICULTY_SCORE: f32 = 100.0;

#[derive(Clone, Copy)]
struct Bird {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    velocity: f32,
}

impl Bird {
    fn new() -> Self {
        Self {
            x: 20.0,
            y: 40.0,
            width: 6.0,
            height: 6.0,
            velocity: 0.0,
        }
    }

    fn reset(&mut self, start_y: f32) {
        self.y = start_y;
        self.velocity = 0.0;
    }

    fn sub_step(&mut self, flap: bool, dt: f32) {
        if flap {
            self.velocity += FLAP_STRENGTH;
        }
        self.velocity += GRAVITY * dt;
        self.y += self.velocity * dt;
    }

    fn update(&mut self, flap: bool) -> bool {
        for i in 0..SUB_TICKS {
            let sub_flap = if i == 0 { flap } else { false };
            self.sub_step(sub_flap, DT);
            if self.y <= 0.0 || (self.y + self.height) >= SCREEN_HEIGHT {
                return true;
            }
        }
        false
    }
}

#[derive(Clone, Copy)]
struct Pipe {
    x: f32,
    width: f32,
    gap: f32,
    gap_y: f32,
    passed: bool,
}

impl Pipe {
    fn new(x: f32, gap: f32) -> Self {
        let mut rng = rand::thread_rng();
        let min_y = 2.0;
        let max_y = (SCREEN_HEIGHT - gap - 2.0).max(min_y);
        let gap_y = rng.gen_range(min_y..=max_y);

        Self {
            x,
            width: PIPE_WIDTH,
            gap,
            gap_y,
            passed: false,
        }
    }

    fn sub_step(&mut self, speed: f32, dt: f32) {
        self.x -= speed * dt;
    }
}

struct PipeManager {
    speed: f32,
    gap: f32,
    pipes: Vec<Pipe>,
    spawn_distance: f32,
    gaps_history: Vec<f32>,
}

impl PipeManager {
    fn new() -> Self {
        let mut pm = Self {
            speed: INITIAL_PIPE_SPEED,
            gap: INITIAL_PIPE_GAP,
            pipes: Vec::with_capacity(4),
            spawn_distance: 0.0,
            gaps_history: Vec::with_capacity(64),
        };
        pm.reset();
        pm
    }

    fn reset(&mut self) {
        self.speed = INITIAL_PIPE_SPEED;
        self.gap = INITIAL_PIPE_GAP;
        self.pipes.clear();
        self.spawn_distance = 0.0;
        self.gaps_history.clear();
        self.spawn_pipe();
    }

    fn spawn_pipe(&mut self) {
        let min_y = 2.0;
        let max_y = (SCREEN_HEIGHT - self.gap - 2.0).max(min_y);

        let gap_y = match self.pipes.last() {
            Some(last_pipe) => {
                let frames = (PIPE_FREQUENCY - (PIPE_WIDTH + BIRD_WIDTH)) / self.speed;
                let substeps = (frames * SUB_TICKS as f32).round().max(1.0) as i32;
                let mut max_fall = 0.0f32;
                let mut v = 0.0f32;
                for _ in 0..substeps {
                    v += GRAVITY * DT;
                    max_fall += v * DT;
                }

                let allowed_down = max_fall * 0.85;
                let allowed_up = 45.0;

                let low = (last_pipe.gap_y - allowed_up).max(min_y);
                let high = (last_pipe.gap_y + allowed_down).min(max_y);

                let mut rng = rand::thread_rng();
                if low <= high {
                    rng.gen_range(low..=high)
                } else {
                    low.min(max_y)
                }
            }
            None => {
                let mut rng = rand::thread_rng();
                rng.gen_range(min_y..=max_y)
            }
        };

        let p = Pipe {
            x: SCREEN_WIDTH,
            width: PIPE_WIDTH,
            gap: self.gap,
            gap_y,
            passed: false,
        };
        self.gaps_history.push(p.gap_y);
        self.pipes.push(p);
    }

    fn update_difficulty(&mut self, score: i32) {
        let progress = (score as f32 / MAX_DIFFICULTY_SCORE).min(1.0);
        self.gap = (INITIAL_PIPE_GAP - progress * (INITIAL_PIPE_GAP - MIN_PIPE_GAP)).floor();
        self.speed = INITIAL_PIPE_SPEED + progress * (MAX_PIPE_SPEED - INITIAL_PIPE_SPEED);
    }

    fn sub_step(&mut self, score: i32, dt: f32) {
        self.update_difficulty(score);

        self.spawn_distance += self.speed * dt;
        if self.spawn_distance >= PIPE_FREQUENCY {
            self.spawn_pipe();
            self.spawn_distance -= PIPE_FREQUENCY;
        }

        for pipe in self.pipes.iter_mut() {
            pipe.sub_step(self.speed, dt);
        }

        self.pipes.retain(|p| p.x + p.width >= 0.0);
    }

    fn update(&mut self, score: i32) {
        for _ in 0..SUB_TICKS {
            self.sub_step(score, DT);
        }
    }
}

#[pyclass]
pub struct RustFlappyEnv {
    bird: Bird,
    pipe_manager: PipeManager,
    score: i32,
    ticks: u32,
}

#[pymethods]
impl RustFlappyEnv {
    #[new]
    pub fn new() -> Self {
        let mut env = Self {
            bird: Bird::new(),
            pipe_manager: PipeManager::new(),
            score: 0,
            ticks: 0,
        };
        env.reset();
        env
    }

    pub fn reset(&mut self) -> Vec<f32> {
        self.bird.reset(SCREEN_HEIGHT / 2.0);
        self.pipe_manager.reset();
        self.score = 0;
        self.ticks = 0;
        self.get_state()
    }

    pub fn get_pipe_gaps(&self) -> Vec<f32> {
        self.pipe_manager.gaps_history.clone()
    }

    pub fn get_state(&self) -> Vec<f32> {
        let target_pipe = self.pipe_manager.pipes.iter().find(|p| p.x + p.width >= self.bird.x);

        let (gap_center_y, dist_x) = match target_pipe {
            Some(p) => {
                let center = (p.gap_y + (p.gap_y + p.gap)) / 2.0;
                let dist = p.x - self.bird.x;
                (center, dist)
            }
            None => (SCREEN_HEIGHT / 2.0, SCREEN_WIDTH),
        };

        vec![
            self.bird.y / SCREEN_HEIGHT,
            self.bird.velocity / 10.0,
            dist_x / SCREEN_WIDTH,
            (gap_center_y - self.bird.y) / SCREEN_HEIGHT,
        ]
    }

    pub fn step(&mut self, action: u8) -> (Vec<f32>, f32, bool, i32, u32) {
        let flap = action == 1;

        let boundary_dead = self.bird.update(flap);
        self.pipe_manager.update(self.score);
        self.ticks += 1;

        let mut is_dead = boundary_dead;

        let mut scored = false;
        if !is_dead {
            for pipe in self.pipe_manager.pipes.iter_mut() {
                let bird_right = self.bird.x + self.bird.width;
                let bird_bottom = self.bird.y + self.bird.height;
                let pipe_right = pipe.x + pipe.width;

                if bird_right > pipe.x && self.bird.x < pipe_right {
                    if self.bird.y < pipe.gap_y || bird_bottom > (pipe.gap_y + pipe.gap) {
                        is_dead = true;
                        break;
                    }
                }

                if !pipe.passed && pipe_right < self.bird.x {
                    pipe.passed = true;
                    self.score += 1;
                    scored = true;
                }
            }
        }

        let mut reward = 0.0;
        if scored {
            reward = 1.0;
        }
        if is_dead {
            reward = -1.0;
        }

        let next_state = self.get_state();
        (next_state, reward, is_dead, self.score, self.ticks)
    }
}

#[pymodule]
fn flappy_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustFlappyEnv>()?;
    Ok(())
}
