use crate::pool::Ptr;

/// A partir de cuánto trabajo conviene la GPU, MEDIDO (GTX 1650, `eva gpu`):
///
/// ```text
///   128³ = 2.1M   CPU gana 1.25x     el costo fijo de subir/encolar/bajar
///   192³ = 7.1M   GPU gana 1.29x     todavía no se amortiza abajo de esto
///   512³ = 134M   GPU gana 8.68x
/// ```
///
/// El cruce está entre esos dos; 4M es el punto medio. **Es el número de una
/// GTX 1650**: en la RX 580 hay que volver a medirlo, no heredarlo.
const GPU_FROM: usize = 4_000_000;

pub fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, out: &mut [f32]) {
    crate::prof::time(crate::prof::P::Matmul, || matmul_inner(a, b, m, k, n, out))
}

fn matmul_inner(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, out: &mut [f32]) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(out.len(), m * n);

    if m == 0 || k == 0 || n == 0 {
        return;
    }

    let work = m * k * n;
    if work >= GPU_FROM && gpu_matmul(a, b, m, k, n, out) {
        return;
    }
    if work < 200_000 {
        matmul_rows(a, b, k, n, out, 0, 0, m);
        return;
    }

    let pool = crate::pool::global();
    if std::env::var("EVA_SERIAL").is_ok() {
        matmul_rows(a, b, k, n, out, 0, 0, m);
        return;
    }
    let chunks = (pool.workers() + 1).min(m);
    if chunks <= 1 {
        matmul_rows(a, b, k, n, out, 0, 0, m);
        return;
    }

    let rows_per = m.div_ceil(chunks);
    let a_ptr = Ptr(a.as_ptr());
    let b_ptr = Ptr(b.as_ptr());
    let out_ptr = Ptr(out.as_mut_ptr());
    let f = move |t: usize| {
        let bs = t * rows_per;
        // `rows_per * chunks` can overshoot `m`, which leaves the last chunks
        // with no rows at all. Without this guard `be - bs` underflows in usize
        // to ~2^64 and `as_mut` builds a slice over the whole address space.
        if bs >= m {
            return;
        }
        let be = (bs + rows_per).min(m);
        // SAFETY: chunk `t` owns the disjoint band [bs, be) of `out`; bands of
        // distinct chunks never alias. The `'static` slices are sound only
        // because `pool::run` does not return until every piece has finished,
        // so nothing here outlives the borrow that produced these pointers.
        let band = out_ptr.add(bs * n).as_mut((be - bs) * n);
        matmul_rows(a_ptr.as_ref(m * k), b_ptr.as_ref(k * n), k, n, band, bs, 0, be - bs);
    };
    pool.run(chunks, f);
}

/// Intenta hacerlo en la GPU. Devuelve `false` si no hay o si falló, y en ese
/// caso el que llama sigue por CPU como si nada.
///
/// SE PRENDE CON `EVA_GPU=1`, a propósito apagado por defecto: la GPU suma en
/// otro orden, así que dos entrenamientos con y sin ella no dan bit a bit lo
/// mismo. Que eso pase tiene que ser una decisión, no una sorpresa.
///
/// Por hilo, y no global, porque los handles de Vulkan no son `Send`. En la
/// práctica sólo el hilo principal llega hasta acá: los workers del pool
/// entran por `matmul_rows`, más abajo.
fn gpu_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, out: &mut [f32]) -> bool {
    use std::cell::OnceCell;
    thread_local! {
        static GPU: OnceCell<Option<crate::gpu::Gpu>> = const { OnceCell::new() };
    }
    GPU.with(|cell| {
        let gpu = cell.get_or_init(|| {
            if std::env::var("EVA_GPU").as_deref() != Ok("1") {
                return None;
            }
            match crate::gpu::Gpu::init() {
                Ok(g) => {
                    eprintln!("[eva] matmul por GPU: {}", g.info().name);
                    Some(g)
                }
                Err(e) => {
                    eprintln!("[eva] sin GPU ({e}); sigo por CPU");
                    None
                }
            }
        });
        match gpu {
            None => false,
            Some(g) => match g.matmul_into(a, b, m, k, n, out) {
                Ok(()) => true,
                // Un error de Vulkan a mitad de un entrenamiento no puede
                // tirarlo abajo: se avisa y se sigue por CPU.
                Err(e) => {
                    eprintln!("[eva] la GPU falló ({e}); sigo por CPU");
                    false
                }
            },
        }
    })
}

fn matmul_rows(a: &[f32], b: &[f32], k: usize, n: usize, out: &mut [f32], base: usize, start: usize, end: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
            unsafe {
                matmul_rows_avx2(a, b, k, n, out, base, start, end);
            }
            return;
        }
    }
    matmul_rows_scalar(a, b, k, n, out, base, start, end);
}

fn matmul_rows_scalar(a: &[f32], b: &[f32], k: usize, n: usize, out: &mut [f32], base: usize, start: usize, end: usize) {
    const BK: usize = 32;
    const BN: usize = 32;

    for i in start..end {
        out[i * n..(i + 1) * n].fill(0.0);
    }

    let mut kb = 0;
    while kb < k {
        let ke = (kb + BK).min(k);
        let mut nb = 0;
        while nb < n {
            let ne = (nb + BN).min(n);
            let bw = ne - nb;
            for i in start..end {
                let row = i * n;
                for p in kb..ke {
                    let av = a[(base + i) * k + p];
                    if av == 0.0 {
                        continue;
                    }
                    let br = p * n + nb;
                    for j in 0..bw {
                        out[row + nb + j] += av * b[br + j];
                    }
                }
            }
            nb = ne;
        }
        kb = ke;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn matmul_rows_avx2(
    a: &[f32],
    b: &[f32],
    k: usize,
    n: usize,
    out: &mut [f32],
    base: usize,
    start: usize,
    end: usize,
) {
    use std::arch::x86_64::*;

    for i in start..end {
        let row = i * n;
        out[row..row + n].fill(0.0);

        let ai = a.as_ptr().add((base + i) * k);
        let oi = out.as_mut_ptr().add(row);

        let mut j = 0;
        while j + 8 <= n {
            let mut acc = _mm256_setzero_ps();
            for p in 0..k {
                let av = _mm256_set1_ps(*ai.add(p));
                let bv = _mm256_loadu_ps(b.as_ptr().add(p * n + j));
                acc = _mm256_fmadd_ps(av, bv, acc);
            }
            _mm256_storeu_ps(oi.add(j), acc);
            j += 8;
        }

        for jj in j..n {
            let mut s = 0.0;
            for p in 0..k {
                s += *ai.add(p) * b[p * n + jj];
            }
            *oi.add(jj) = s;
        }
    }
}

pub fn transpose(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut t = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            t[j * rows + i] = x[i * cols + j];
        }
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn matmul_threaded_matches_scalar() {
        let mut rng = Rng::new(42);
        let (m, k, n) = (512, 512, 512);
        let a: Vec<f32> = (0..m * k).map(|_| rng.uniform(-1.0, 1.0)).collect();
        let b: Vec<f32> = (0..k * n).map(|_| rng.uniform(-1.0, 1.0)).collect();
        let mut c1 = vec![0.0f32; m * n];
        matmul(&a, &b, m, k, n, &mut c1);
        let mut c2 = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for p in 0..k {
                    s += a[i * k + p] * b[p * n + j];
                }
                c2[i * n + j] = s;
            }
        }
        let mut bad = 0;
        for i in 0..m * n {
            if (c1[i] - c2[i]).abs() > 1e-2 {
                bad += 1;
            }
        }
        assert_eq!(bad, 0, "{} elementos difieren entre threaded y scalar", bad);
    }
}
