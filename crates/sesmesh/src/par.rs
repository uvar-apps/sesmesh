// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Parallel iteration facade (Design §11.1): rayon with the `parallel`
//! feature, plain sequential iteration otherwise. Every helper returns results
//! in index order, so output never depends on scheduling.

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// `(0..n).map(f).collect()`, in parallel when enabled.
pub(crate) fn map<T, F>(n: usize, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Sync + Send,
{
    #[cfg(feature = "parallel")]
    {
        (0..n).into_par_iter().map(f).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        (0..n).map(f).collect()
    }
}

/// Like [`map`], with per-thread scratch state created by `init`.
pub(crate) fn map_init<S, T, I, F>(n: usize, init: I, f: F) -> Vec<T>
where
    T: Send,
    I: Fn() -> S + Sync + Send,
    F: Fn(&mut S, usize) -> T + Sync + Send,
{
    #[cfg(feature = "parallel")]
    {
        (0..n).into_par_iter().map_init(init, f).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut scratch = init();
        (0..n).map(|i| f(&mut scratch, i)).collect()
    }
}

/// Items per task when `n` items are processed in chunks with flat
/// per-chunk arrays: about 256 tasks for large inputs, whose chunk arrays
/// are then big allocations the allocator returns to the system when they
/// are freed (per-item vectors kept peak memory high, issue #21); one item
/// per task below 512 items, so small inputs keep their parallelism. It
/// depends on `n` only; callers concatenate in item order, so results do
/// not depend on it either.
pub(crate) fn chunk_len(n: usize) -> usize {
    (n / 256).clamp(1, 4096)
}

/// The index range of chunk `c` (see [`chunk_len`]).
pub(crate) fn chunk_range(n: usize, chunk: usize, c: usize) -> std::ops::Range<usize> {
    c * chunk..n.min((c + 1) * chunk)
}

/// Unstable sort, in parallel when enabled.
pub(crate) fn sort_unstable<T: Ord + Send>(v: &mut [T]) {
    #[cfg(feature = "parallel")]
    v.par_sort_unstable();
    #[cfg(not(feature = "parallel"))]
    v.sort_unstable();
}

#[cfg(test)]
mod tests {
    #[test]
    fn order_is_preserved() {
        let v = super::map(10_000, |i| i * 2);
        assert!(v.iter().enumerate().all(|(i, &x)| x == 2 * i));
        let w = super::map_init(1000, Vec::<usize>::new, |s, i| {
            s.push(i);
            i
        });
        assert_eq!(w, (0..1000).collect::<Vec<_>>());
        let mut u = vec![5, 3, 9, 1];
        super::sort_unstable(&mut u);
        assert_eq!(u, [1, 3, 5, 9]);
    }
}
