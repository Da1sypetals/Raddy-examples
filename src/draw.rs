#![allow(non_camel_case_types)]
use macroquad::prelude::*;

pub type vec<const D: usize> = na::SVector<f64, D>;
pub type mat<const RC: usize> = na::SMatrix<f64, RC, RC>;
pub type vec2 = vec<2>;

pub fn draw_point(p: &vec2) {
    let _p = (p + vec2::new(1.0, 1.0)) / 2.0;
    let p_unnorm = vec2::new(
        _p[0] * screen_width() as f64,
        _p[1] * screen_height() as f64,
    );
    // dbg!(&p_unnorm);
    draw_circle(p_unnorm[0] as f32, p_unnorm[1] as f32, 3.0, YELLOW);
}

pub fn draw_link(p1: &vec2, p2: &vec2) {
    let _p1 = (p1 + vec2::new(1.0, 1.0)) / 2.0;
    let _p2 = (p2 + vec2::new(1.0, 1.0)) / 2.0;
    let p1_unnorm = vec2::new(
        _p1[0] * screen_width() as f64,
        _p1[1] * screen_height() as f64,
    );
    let p2_unnorm = vec2::new(
        _p2[0] * screen_width() as f64,
        _p2[1] * screen_height() as f64,
    );
    draw_line(
        p1_unnorm.x as f32,
        p1_unnorm.y as f32,
        p2_unnorm.x as f32,
        p2_unnorm.y as f32,
        1.,
        WHITE,
    );
}
