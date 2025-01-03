use std::{
    io::{self, Read},
    time::{Duration, Instant},
};

use draw::{draw_link, draw_point, vec2};
use faer::{prelude::SpSolver, sparse::SparseColMat, Col};
use macroquad::{
    color::BLACK,
    window::{clear_background, next_frame, Conf},
};
use raddy::{sparse::objective::Objective, types::advec, Ad};

pub mod draw;
extern crate nalgebra as na;

const g: f64 = 9.8;
const K: f64 = 100.0;
const N: usize = 32;
const LEN: f64 = 1.0;
const DT: f64 = 0.01;
const RESTLEN: f64 = LEN / N as f64;

struct SpringEnergy {
    k: f64,
}

fn wait_for_keypress() {
    println!("Press any key to continue...");
    let _ = io::stdin().read(&mut [0u8]).unwrap();
}

// 2d * 2nodes = 4dof
impl Objective<4> for SpringEnergy {
    /// rest length
    type EvalArgs = f64;

    fn eval(&self, variables: &raddy::types::advec<4, 4>, args: &f64) -> raddy::Ad<4> {
        let restlen = *args;
        let p1 = advec::<4, 2>::new(variables[0].clone(), variables[1].clone());
        let p2 = advec::<4, 2>::new(variables[2].clone(), variables[3].clone());

        let len = (p2 - p1).norm();
        // Hooke's law
        let potential =
            Ad::inactive_scalar(0.5 * self.k) * (len - Ad::inactive_scalar(restlen)).powi(2);

        potential
    }
}

struct SystemEnergy {
    pub spring: SpringEnergy,
    pub v: Col<f64>,
}

impl SystemEnergy {
    /// *** potential at xnew given previous position x
    fn system_energy(
        &self,
        x: &Col<f64>,
        xnew: &Col<f64>,
        spring: &[[usize; 4]],
    ) -> (f64, Col<f64>, Vec<(usize, usize, f64)>) {
        let n = x.nrows();

        // compute xhat
        let xhat = {
            let mut xhat = x.clone();
            xhat += DT * &self.v;
            for i in 0..N {
                xhat[2 * i + 1] += DT * DT * g;
            }
            xhat
        };

        let dx = xnew - &xhat;

        let inertial = 0.5 * dx.squared_norm_l2();
        let computed = self.spring.compute(xnew, spring, &RESTLEN);
        let e = inertial + computed.value;

        let mut grad = computed.grad + dx;
        // enforce boundary
        grad[0] = 0.0;
        grad[1] = 0.0;

        let mut trips = computed.hess_trips;
        for i in 0..N {
            trips.push((i * 2, i * 2, 1.0));
            trips.push((i * 2 + 1, i * 2 + 1, 1.0));
        }
        // enforce boundary
        for trip in &mut trips {
            if trip.0 == 0 || trip.1 == 0 || trip.0 == 1 || trip.1 == 1 {
                trip.2 = 0.0;
            }
        }

        (e, grad, trips)
    }
}

fn window_conf() -> Conf {
    Conf {
        window_title: String::from("Springs"),
        window_width: 1000,
        window_height: 1000,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let spring = (0..N - 1)
        .into_iter()
        .map(|i| [i * 2, i * 2 + 1, i * 2 + 2, i * 2 + 3])
        .collect::<Vec<_>>();
    let nodes: Vec<_> = (0..N)
        .into_iter()
        .flat_map(|i| [RESTLEN * i as f64, 0.0].into_iter())
        .collect();

    let x0 = faer::col::from_slice(&nodes).to_owned();
    let mut x = x0.clone();

    let mut sys = SystemEnergy {
        spring: SpringEnergy { k: K },
        v: Col::zeros(N * 2),
    };

    let mut last_time = Instant::now();
    let mut last_iframe = 0;
    let mut dir: Col<f64>;

    for iframe in 0.. {
        clear_background(BLACK);
        draw_point(&vec2::new(0.0, 0.0));

        let mut xnew = x.clone();
        let mut iter = 0;
        while {
            let egh = sys.system_energy(&x, &xnew, &spring);

            let grad = egh.1;
            let mut trips = egh.2;

            // Enforce boundary conditions: remove node 0 (idx 0 and 1) from DOFs
            let grad = grad.subrows(2, N * 2 - 2);
            for (r, c, val) in &mut trips {
                if *r == 0 || *c == 0 || *r == 1 || *c == 1 {
                    *val = 0.0;
                } else {
                    *r -= 2;
                    *c -= 2;
                }
            }
            let hess = SparseColMat::try_new_from_triplets(N * 2 - 2, N * 2 - 2, &trips).unwrap();

            dir = hess.sp_lu().unwrap().solve(-&grad);

            // let dirnorm = dir.norm_l2();
            // dbg!(dirnorm);

            dir.norm_l2() > 1e-4
        } {
            let mut dof = xnew.subrows_mut(2, N * 2 - 2);
            dof += dir;

            // println!("Frame {} iter {}", iframe, iter);
            // iter += 1;
        }

        sys.v = (&xnew - &x) / DT;
        x = xnew;

        for i in 0..N {
            let pos = vec2::new(x[i * 2], x[i * 2 + 1]);
            draw_point(&pos);
        }

        for i in 0..N - 1 {
            let pos1 = vec2::new(x[i * 2], x[i * 2 + 1]);
            let pos2 = vec2::new(x[i * 2 + 2], x[i * 2 + 3]);
            draw_link(&pos1, &pos2);
        }

        // println!("Simulated frame {iframe}");

        next_frame().await;

        // 检查是否已经过去 1 秒
        let elapsed = last_time.elapsed();
        if elapsed >= Duration::from_secs(1) {
            let fps = (iframe - last_iframe) as f64 / elapsed.as_secs_f64();
            println!("FPS: {:.2}", fps);

            // 重置计数器和时间
            last_iframe = iframe;
            last_time = Instant::now();
        }
    }
}
