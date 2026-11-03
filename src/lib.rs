use pyo3::prelude::*;
use rand::Rng;

const SCREEN_WIDTH: f32 = 80.0;
const SCREEN_HEIGHT: f32 = 80.0;
const GRAVITY: f32 = 0.4;
const FLAP_STRENGTH: f32 = -3.5;
const PIPE_SPEED: f32 = 1.0;
const PIPE_GAP: f32 = 28.0;
const PIPE_FREQUENCY: f32 = 40.0;

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
            y: SCREEN_HEIGHT / 2.0,
            width: 6.0,
            height: 6.0,
            velocity: 0.0,
        }
    }

    fn reset(&mut self, start_y: f32) {
        self.y = start_y;
        self.velocity = 0.0;
    }

    fn update(&mut self, flap: bool) {
        if flap {
            self.velocity += FLAP_STRENGTH;
        }
        self.velocity += GRAVITY;
        self.y += self.velocity;
    }

    #[inline(always)]
    fn top(&self) -> f32 { self.y }
    #[inline(always)]
    fn bottom(&self) -> f32 { self.y + self.height }
    #[inline(always)]
    fn left(&self) -> f32 { self.x }
    #[inline(always)]
    fn right(&self) -> f32 { self.x + self.width }
}

#[derive(Clone)]
struct Pipe {
    x: f32,
    width: f32,
    top_height: f32,
    bottom_y: f32,
    passed: bool,
}

impl Pipe {
    fn new(x: f32) -> Self {
        let mut rng = rand::thread_rng();
        let gap_y = rng.gen_range(8.0..(SCREEN_HEIGHT - PIPE_GAP - 8.0));
        Self {
            x,
            width: 10.0,
            top_height: gap_y,
            bottom_y: gap_y + PIPE_GAP,
            passed: false,
        }
    }

    fn update(&mut self) {
        self.x -= PIPE_SPEED;
    }
}

#[pyclass]
pub struct RustFlappyEnv {
    bird: Bird,
    pipes: Vec<Pipe>,
    spawn_distance: f32,
    score: i32,
    ticks: u32,
}

#[pymethods]
impl RustFlappyEnv {
    #[new]
    pub fn new() -> Self {
        let mut env = Self {
            bird: Bird::new(),
            pipes: Vec::with_capacity(4),
            spawn_distance: 0.0,
            score: 0,
            ticks: 0,
        };
        env.reset();
        env
    }

    pub fn reset(&mut self) -> Vec<f32> {
        self.bird.reset(SCREEN_HEIGHT / 2.0);
        self.pipes.clear();
        self.spawn_distance = 0.0;
        self.pipes.push(Pipe::new(SCREEN_WIDTH));
        self.score = 0;
        self.ticks = 0;
        self.get_state()
    }

    pub fn get_state(&self) -> Vec<f32> {
        let target_pipe = self.pipes.iter().find(|p| p.x + p.width >= self.bird.left());

        let (gap_center_y, pipe_dist_x) = match target_pipe {
            Some(pipe) => {
                let center = (pipe.top_height + pipe.bottom_y) / 2.0;
                let dist = pipe.x - self.bird.left();
                (center, dist)
            }
            None => {
                if let Some(pipe) = self.pipes.first() {
                    let center = (pipe.top_height + pipe.bottom_y) / 2.0;
                    let dist = pipe.x - self.bird.left();
                    (center, dist)
                } else {
                    (SCREEN_HEIGHT / 2.0, SCREEN_WIDTH)
                }
            }
        };

        vec![
            self.bird.y / SCREEN_HEIGHT,
            self.bird.velocity / 10.0,
            pipe_dist_x / SCREEN_WIDTH,
            (gap_center_y - self.bird.y) / SCREEN_HEIGHT,
        ]
    }

    /// Шаг симуляции: принимает action (0 или 1)
    /// Возвращает кортеж: (next_state, reward, done, score, ticks)
    pub fn step(&mut self, action: u8) -> (Vec<f32>, f32, bool, i32, u32) {
        let flap = action == 1;

        // 1. Физический шаг
        self.bird.update(flap);

        self.spawn_distance += PIPE_SPEED;
        if self.spawn_distance >= PIPE_FREQUENCY {
            self.pipes.push(Pipe::new(SCREEN_WIDTH));
            self.spawn_distance -= PIPE_FREQUENCY;
        }

        for pipe in self.pipes.iter_mut() {
            pipe.update();
        }
        self.pipes.retain(|p| p.x + p.width >= 0.0);
        self.ticks += 1;

        // 2. Коллизии (AABB) и подсчет очков
        let mut is_dead = false;
        if self.bird.top() <= 0.0 || self.bird.bottom() >= SCREEN_HEIGHT {
            is_dead = true;
        }

        let mut scored = false;
        if !is_dead {
            for pipe in self.pipes.iter_mut() {
                if self.bird.right() > pipe.x && self.bird.left() < pipe.x + pipe.width {
                    if self.bird.top() < pipe.top_height || self.bird.bottom() > pipe.bottom_y {
                        is_dead = true;
                        break;
                    }
                }
                if !pipe.passed && pipe.x + pipe.width < self.bird.left() {
                    pipe.passed = true;
                    self.score += 1;
                    scored = true;
                }
            }
        }

        // 3. Формирование награды по выверенной формуле
        let mut reward = 0.1;
        if scored {
            reward = 5.0;
        }
        if is_dead {
            reward = -5.0;
        }

        let next_state = self.get_state();
        (next_state, reward, is_dead, self.score, self.ticks)
    }
}

#[pymodule]
fn flappy_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<RustFlappyEnv>()?;
    Ok}