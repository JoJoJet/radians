#![feature(test)]
extern crate test;
use test::Bencher;

use radians::Rad64;
use rand::Rng;

fn all_floats() -> impl Iterator<Item = f64> {
    (0_u64..).map(f64::from_bits)
}

fn rand_floats(rng: &mut impl Rng) -> impl Iterator<Item = (f64, f64)> + '_ {
    std::iter::repeat_with(|| (rng.gen(), rng.gen()))
}

fn rand_pairs(rng: &mut impl Rng) -> impl Iterator<Item = (Rad64, Rad64)> + '_ {
    std::iter::repeat_with(|| (Rad64::new(rng.gen()), Rad64::new(rng.gen())))
}

const BENCH_SIZE: usize = 10_000_000;

#[bench]
fn eq_float(bench: &mut Bencher) {
    let pairs: Vec<_> = rand_floats(&mut rand::thread_rng())
        .take(BENCH_SIZE)
        .collect();
    bench.iter(|| {
        for &(a, b) in &pairs {
            let _ = test::black_box(a == b);
        }
    });
}
#[bench]
fn eq_rad(bench: &mut Bencher) {
    let pairs: Vec<_> = rand_pairs(&mut rand::thread_rng())
        .take(BENCH_SIZE)
        .collect();
    bench.iter(|| {
        for &(a, b) in &pairs {
            let _ = test::black_box(a == b);
        }
    });
}

#[bench]
fn eq_same_float(bench: &mut Bencher) {
    let vals: Vec<_> = all_floats().take(BENCH_SIZE).collect();
    bench.iter(|| {
        for &a in &vals {
            let _ = test::black_box(a == a);
        }
    })
}
#[bench]
fn eq_same_rad(bench: &mut Bencher) {
    let vals: Vec<_> = all_floats().map(Rad64::new).take(BENCH_SIZE).collect();
    bench.iter(|| {
        for &a in &vals {
            let _ = test::black_box(a == a);
        }
    })
}

#[bench]
fn cmp_float(bench: &mut Bencher) {
    let pairs: Vec<_> = rand_floats(&mut rand::thread_rng())
        .take(BENCH_SIZE)
        .collect();
    bench.iter(|| {
        for &(a, b) in &pairs {
            let _ = test::black_box(a.partial_cmp(&b));
        }
    });
}
#[bench]
fn cmp_rad(bench: &mut Bencher) {
    let pairs: Vec<_> = rand_pairs(&mut rand::thread_rng())
        .take(BENCH_SIZE)
        .collect();
    bench.iter(|| {
        for &(a, b) in &pairs {
            let _ = test::black_box(a.partial_cmp(&b));
        }
    });
}

#[bench]
fn lt_float(bench: &mut Bencher) {
    let pairs: Vec<_> = rand_floats(&mut rand::thread_rng())
        .take(BENCH_SIZE)
        .collect();
    bench.iter(|| {
        for &(a, b) in &pairs {
            let _ = test::black_box(a < b);
        }
    });
}
#[bench]
fn lt_rad(bench: &mut Bencher) {
    let pairs: Vec<_> = rand_pairs(&mut rand::thread_rng())
        .take(BENCH_SIZE)
        .collect();
    bench.iter(|| {
        for &(a, b) in &pairs {
            let _ = test::black_box(a < b);
        }
    });
}
