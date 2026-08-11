use crate::model::EvaModel;
use crate::rng::Rng;

pub struct LogitsMask {
    allowed: Vec<bool>,
}

impl LogitsMask {
    pub fn all(vocab: usize) -> Self {
        LogitsMask { allowed: vec![true; vocab] }
    }

    pub fn disallow(&mut self, tok: usize) {
        if let Some(a) = self.allowed.get_mut(tok) {
            *a = false;
        }
    }

    pub fn allow_only(&mut self, toks: &[usize]) {
        self.allowed.fill(false);
        for &t in toks {
            if let Some(a) = self.allowed.get_mut(t) {
                *a = true;
            }
        }
    }

    pub fn is_allowed(&self, tok: usize) -> bool {
        self.allowed.get(tok).copied().unwrap_or(false)
    }
}

pub trait Constraint {
    fn allowed(&mut self, so_far: &[usize], vocab: usize) -> LogitsMask;
}

pub struct NoConstraint;

impl Constraint for NoConstraint {
    fn allowed(&mut self, _so_far: &[usize], vocab: usize) -> LogitsMask {
        LogitsMask::all(vocab)
    }
}

pub fn sample(logits: &[f32], temp: f32, top_k: usize, rng: &mut Rng) -> usize {
    sample_masked(logits, &LogitsMask::all(logits.len()), temp, top_k, rng)
}

pub fn sample_masked(
    logits: &[f32],
    mask: &LogitsMask,
    temp: f32,
    top_k: usize,
    rng: &mut Rng,
) -> usize {
    let v = logits.len();
    let mut cand: Vec<(f32, usize)> = (0..v)
        .filter(|&i| mask.is_allowed(i))
        .map(|i| (logits[i], i))
        .collect();
    if cand.is_empty() {
        cand = (0..v).map(|i| (logits[i], i)).collect();
    }
    let k = top_k.min(cand.len());
    let n = if k == cand.len() {
        cand.len()
    } else {
        cand.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        k
    };
    let mut max = f32::NEG_INFINITY;
    for &(val, _) in cand.iter().take(n) {
        if val > max {
            max = val;
        }
    }
    let mut sum = 0.0;
    let mut probs = Vec::with_capacity(n);
    for &(val, idx) in cand.iter().take(n) {
        let p = ((val - max) / temp).exp();
        probs.push((p, idx));
        sum += p;
    }
    let mut r = rng.next_f32() * sum;
    for (p, idx) in probs {
        r -= p;
        if r <= 0.0 {
            return idx;
        }
    }
    cand[0].1
}

pub fn generate_constrained(
    model: &EvaModel,
    prompt: &[usize],
    max_tokens: usize,
    temp: f32,
    top_k: usize,
    rng: &mut Rng,
    seq: usize,
    constraint: &mut dyn Constraint,
) -> Vec<usize> {
    let _ = seq; // el recorte de ventana ahora lo maneja el estado
    let mut ctx: Vec<usize> = prompt.to_vec();
    let mut out: Vec<usize> = Vec::with_capacity(max_tokens);
    if ctx.is_empty() {
        return out;
    }

    // Antes esto rehacía la pasada completa sobre toda la ventana para CADA
    // token y descartaba todas las filas menos la última: hasta 64x de trabajo
    // tirado. Ahora el estado viaja y cada token cuesta un token.
    let mut st = crate::stream::Streamer::new(model);
    let mut logits = Vec::new();
    for &id in &ctx {
        logits = st.next(id);
    }

    for _ in 0..max_tokens {
        let mask = constraint.allowed(&ctx, logits.len());
        let tok = sample_masked(&logits, &mask, temp, top_k, rng);
        ctx.push(tok);
        out.push(tok);
        logits = st.next(tok);
    }
    out
}

pub fn generate(
    model: &EvaModel,
    prompt: &[usize],
    max_tokens: usize,
    temp: f32,
    top_k: usize,
    rng: &mut Rng,
    seq: usize,
) -> Vec<usize> {
    let mut nc = NoConstraint;
    generate_constrained(model, prompt, max_tokens, temp, top_k, rng, seq, &mut nc)
}

/// Qué se hizo en cada posición, para poder decir cuánto se ahorró en vez de
/// suponerlo.
#[derive(Default, Debug)]
pub struct Cuentas {
    /// La estructura dejaba una sola opción: no se consultó al modelo.
    pub forzadas: usize,
    /// Se calcularon sólo algunas columnas de la cabeza.
    pub parciales: usize,
    /// El modelo decidió libre, con la cabeza entera.
    pub libres: usize,
    /// Columnas de la cabeza calculadas, contra las que se habrían calculado
    /// sin restricción. Es la medida honesta del ahorro en la salida.
    pub columnas: usize,
    pub columnas_sin_restriccion: usize,
}

/// Genera respetando una restricción **consultada antes de cada paso**.
///
/// La diferencia con `generate_constrained` no es qué produce sino qué gasta:
/// allá la máscara se aplicaba sobre logits ya calculados, acá donde la
/// estructura no deja elegir el modelo directamente no se consulta.
pub fn generate_shaped(
    model: &EvaModel,
    prompt: &[usize],
    max_tokens: usize,
    temp: f32,
    top_k: usize,
    rng: &mut Rng,
    forma: &mut dyn crate::constrain::Constraint,
) -> (Vec<usize>, Cuentas) {
    use crate::constrain::Allowed;
    let mut c = Cuentas::default();
    let mut ctx: Vec<usize> = prompt.to_vec();
    let mut out = Vec::with_capacity(max_tokens);
    if ctx.is_empty() {
        return (out, c);
    }

    let vocab = model.cfg.vocab;
    let mut st = crate::stream::Streamer::new(model);
    // El prompt se consume sin cabeza: de esas posiciones no sale nada.
    for &id in &ctx[..ctx.len() - 1] {
        st.consume(id);
    }
    let mut ultimo = ctx[ctx.len() - 1];

    for _ in 0..max_tokens {
        c.columnas_sin_restriccion += vocab;
        let tok = match forma.allowed(&ctx) {
            Allowed::Only(v) if v.len() == 1 => {
                // Nada que decidir. Se avanza el estado y listo.
                st.consume(ultimo);
                c.forzadas += 1;
                v[0]
            }
            Allowed::Only(v) => {
                let logits = st.next_among(ultimo, &v);
                c.parciales += 1;
                c.columnas += v.len();
                let i = pick(&logits, temp, top_k, rng);
                v[i]
            }
            Allowed::Any => {
                let logits = st.next(ultimo);
                c.libres += 1;
                c.columnas += vocab;
                let mask = LogitsMask::all(vocab);
                sample_masked(&logits, &mask, temp, top_k, rng)
            }
        };
        ctx.push(tok);
        out.push(tok);
        ultimo = tok;
    }
    (out, c)
}

/// Elige un índice dentro de un puñado de logits ya restringido.
fn pick(logits: &[f32], temp: f32, top_k: usize, rng: &mut Rng) -> usize {
    let mask = LogitsMask::all(logits.len());
    sample_masked(logits, &mask, temp, top_k, rng)
}
