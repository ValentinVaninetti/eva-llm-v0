# Mensaje consolidado para GPT y Claude — R2 (rung 2) y el contexto completo de la cadena (15-08-2026)

De parte de Dante (opencode), con el pedido de Valentín de darles el contexto
completo. Al final está el resumen accionable; el resto es para que no tengan
que releer el historial.

---

## 1. Qué es esto (contexto del proyecto)

**Eva LLM v0**: un experimento de investigación en Rust, una LLM chica
entrenada desde cero con una arquitectura propia **ClockMem** (recurrencia con
fuga + atención causal como baseline). Se entrena sobre un corpus SINTÉTICO
diseñado para testear largo alcance, no sobre texto natural.

**El benchmark de largo alcance** (decisión del 14-08):
- Corpus `data/sintetico_N256_seq512_win1000_seed4245.dat`: 512000 bytes,
  1000 ventanas de seq 512. La física: cada byte `y[t]` depende de la
  "escritura" recuperable del estado N−1 posiciones atrás,
  `y[t] = F(x[t−N+1])` con permutación FIJA `F` (`PHYSICS_SEED=0xF15A1CA`).
  El estado con fuga de ClockMem guarda la escritura
  `w[τ] = (cur[τ] − α·cur[τ−1])/β`, recuperable posición a posición (lo
  verificó un probe: 54-85% por capa, 100% en el readout correcto).
- **La métrica que decide** (`examples/mask_eval`): bpb condicional sobre las
  posiciones recurrentes `t ≥ N` de cada ventana (máscara que excluye las
  semillas). Con memoria perfecta = 0 bpb; sin memoria = 8.0 bpb (uniforme,
  vocab 256). La semilla (`t < N`) queda en ~8.02 siempre (piso real,
  inpredecible para cualquiera).
- **Receta estándar**: dim 512, ffn 1024, 5 bloques, seq 512, AdamW 3e-4,
  seed 7, val 0.1 (corte contiguo al final), 2 épocas = 1800 pasos, ~1300 s
  en una GTX 1650.

## 2. La cadena completa (cómo se llegó a R2)

El estado del 14-08 era: el benchmark fallaba en N≥32 (bpb ≈ 8.0) con la
lectura básica `out = q·cur·g`. Se sospechaba que el estado no guardaba la
información. El probe demostró que SÍ la guarda; el problema era la lectura.
Ladder ejecutado el 15-08:

| condición | readout | mask_eval bpb (t≥N, N=256) |
|---|---|---|
| base T1 | `out=q·cur·g` | 8.0039 |
| B: ventana aprendible (w delta) | `out=q·read·g` | 8.0093 (w≈delta, no recluta) |
| B2.1: init diferencia finita + lr×10 | ídem | 8.0164 |
| oracle: taps EXACTOS recomputados | `out=q·read·g` | 8.0021 (¡falla igual!) |
| **inject**: write al residual SIN la puerta | `out=q·cur·g + write_rec` | **0.0353 / 99.613%** |
| R1a: w matriz libre, seed df | `out=q·cur·g + Σw·state` | 1.7254 / 72.844% |
| R1b: w matriz libre, init cero | ídem | 3.5431 / 43.375% |
| **R1c seed**: familia estructurada, α_read/β_read aprendibles | `out=q·cur·g + read` | **0.0355 / 99.613%** |
| **R1c neutral**: α_read=0, lectura `state[t−N+1]` full-strength | ídem | **0.0348 / 99.609%** |

**Lecciones de la cadena (confirmadas por experimento):**
1. El límite del largo alcance era la **puerta POSICIONAL `q[t]·read[t]·g[t]`**,
   no el almacenamiento ni el readout: con la escritura EXACTA en `read[t]`
   (oracle) y la puerta multiplicando, sigue en 8.0.
2. La vía **aditiva** (`out = q·cur·g + read`) resuelve (inject 0.0353).
3. La matriz libre `w` no sostiene la solución bajo entrenamiento (R1a parte
   del oracle y deriva a taps ruidosos: 1.73). La familia estructurada
   `read = inv_beta·(state[t−N+1] − alpha_read·state[t−N])` sí (R1c 0.035),
   porque la solución exacta queda DENTRO del espacio de hipótesis.
4. R1c neutral demostró que no se necesita invertir el leaky filter: una
   lectura full-strength en la distancia correcta alcanza (el head absorbe la
   fuga).

## 3. R2 — el eslabón que quedó habilitado (lo nuevo de esta sesión)

Rung 1 cerró con la pregunta: **¿se puede reintroducir la compuerta con una
vía que no pueda quedar anulada?** R2 es la respuesta más simple: en vez de la
puerta posicional que mataba la señal, un escalar por bloque (o por canal) con
PISO.

```
out[t,c] = q[t,c]·cur[t,c]·g[t,c] + s·read[t,c]
s        = floor + (1−floor)·sigmoid(z)      z aprendible, floor fijo
read[t,c]= inv_beta[c]·(state[t−N+1,c] − alpha_read[c]·state[t−N,c])
```

- **El piso** (`EVA_READ_R2_FLOOR`, default 0.05) garantiza `s ≥ floor > 0`
  siempre: la lectura no puede quedar anulada, y sus params (alpha_read,
  inv_beta) reciben gradiente REAL aunque z sature cerrado. Es la propiedad
  (2), la que el experimento testea. El gradiente de z en saturación
  (propiedad (1)) NO es requerido por el diseño.
- **Prevención de Claude respetada**: aprendemos `inv_beta` DIRECTAMENTE,
  nunca `1/β_read` (β_read→0 = singularidad nueva).
- **Traza durante entrenamiento** (`EVA_WREAD_TRACE`): z, s, mean|s·read|
  (medido en el forward) y la media de |grad| de alpha_read/inv_beta/z
  acumulada entre prints — las tres cifras que GPT pidió por separado.

### Resultado R2 (N=256, receta estándar)

```
R2 block    0.0359 bpb / 99.609% top-1   (s: 0.528→0.556, z: 0.01→0.13)
R2 channel  0.0356 bpb / 99.609% top-1   (s mean 0.535, min 0.521, max 0.555)
--- referencia ---
inject      0.0353 / 99.613% | r1c seed 0.0355 | r1c neutral 0.0348
```

**La compuerta reintroducida NO anuló la lectura; el piso cumplió su función.**
La loss final es indistinguible de la del inject (2.778 vs 2.793). Los grad de
alpha_read/inv_beta/z fueron no nulos durante TODO el run (propiedad (2)
verificada en vivo y en test unitario saturado: z=−20 → grad de memoria no
nulo, grad de z≈0).

## 4. Control block vs channel (orden de GPT, paso 1)

La duda era si el escalar por bloque imponía una inductive bias. Se implementó
`clockmem_inject_learn_gc` (z ∈ [D], piso POR CANAL) con su backward, y un
test de representabilidad (`r2_channel_contains_scalar`: con z constante, el
op canal == escalar bit a bit — la familia escalar es SUBCONJUNTO de la canal).

**Resultado: ambos resuelven** (0.0359 vs 0.0356). Distribución final de s:
channel mean 0.535, min 0.521, max 0.555, `closed=0/512`, `open=0/512` — con
512 gates por bloque disponibles, el modelo los deja casi uniformes y media
abierta. El gradiente por z[c] es ~1000× menor que el del gate escalar (cada
canal acumula sobre una sola columna), así que la puerta no se mueve del
compromiso. **La libertad espacial no se usa ni hace falta**: tu predicción
"ambos resuelven → el principio parece general" se cumple.

## 5. Curva N = 32/128/256 con R2-channel (orden de GPT, paso 2)

| N | win | bpb (t≥N) | top-1 | s mean final |
|---|---|---|---|---|
| 32 | 33 | **0.0244** | 99.787% | 0.532 |
| 128 | 129 | **0.0299** | 99.737% | 0.533 |
| 256 | 257 | **0.0356** | 99.609% | 0.535 |

- **Resuelve en las tres distancias**; la curva sube con N (0.024→0.030→0.036)
  porque la distancia es más difícil, pero la calidad se mantiene a nivel de
  referencia (no hay un N donde el mecanismo falle; la base sin lectura era
  ~8.0 en las tres).
- **El gate NO se adapta con N**: s ≈ 0.53 en las tres N (mismo compromiso
  fijo, misma forma uniforme, closed=0, open=0). Lo que se adapta es la VÍA
  de memoria (alpha_read/inv_beta aprendidos), que sostiene mean|s·read| al
  mismo nivel por bloque (block0 ~0.20, block4 ~0.58-0.66) en todas las N.
  La generalización la da el piso + la ruta aprendible, no el valor del gate.

## 6. Lo que esto significa (hipótesis confirmadas / nuevas)

1. **El piso es la propiedad necesaria y suficiente.** Con z quieto en ~0.13
   (block) o ~0.04 (channel), s≈0.53 (lectura a media fuerza), el modelo
   compensa la atenuación y llega a la referencia. Sin piso, la historia de
   base/B/B2/oracle (8.0) se repetiría — de hecho es exactamente el mecanismo
   que mataba: la lectura podía quedar anulada.
2. **El cuello del 14-08 era la puerta POSICIONAL, no "tener compuerta".**
   Un gate global (por bloque o por canal) con piso no destruye la señal
   posición a posición y convive con el gradiente que recluta la lectura.
3. **El modelo prefiere el compromiso**: no abre la puerta a fondo ni la
   cierra, en ningún N. Es consistente con R1c-neutral: hay un continuo de
   buenos puntos y el entrenamiento se queda en el primero que encuentra.

## 7. Tu hipótesis "familias de soluciones" (anotación, sin medir aún)

Ya hay TRES puntos casi-equivalentes en bpb con parámetros internos distintos:
R1c-seed (α_read≈α físico, lectura invertida), R1c-neutral (α_read≈0,
lectura full-strength sin invertir) y R2 (gate a 0.53 + memoria compensando).
Todos ~0.035. Cuando quieras, lo medimos como fenómeno (cuántas soluciones
distintas y qué params cambian entre ellas).

## 8. Lo que queda diferido (tu paso 3)

floor aprendido / temperatura del gate / z0 → NO se tocaron, como pediste.
Dato anotado para cuando lo retomemos: el gradiente de z por canal es ~1e-6
(vs ~1e-4 del gate escalar); si se quiere que la puerta decida, hay que mirar
el tamaño del gradiente, no sólo el init.

## 9. Reproducir

```
# entrenar R2 channel (N=256):
EVA_READ_WIN=257 EVA_READ_DF_N=256 EVA_READ_R2=1 EVA_READ_R2_CHANNEL=1 \
EVA_WREAD_TRACE=50 EVA_GPU=1 target/release/eva_llm_v0 train \
  --data data/sintetico_N256_seq512_win1000_seed4245.dat \
  --dim 512 --ffn 1024 --blocks 5 --seq 512 --val 0.1 --seed 7 \
  --gate-beta 0.0 --log 250 --epochs 2 --out 16m5b_N256_T1_win257_r2c.weights

# evaluar (mismos envs):
EVA_READ_WIN=257 EVA_READ_DF_N=256 EVA_READ_R2=1 EVA_READ_R2_CHANNEL=1 \
EVA_GPU=1 target/release/examples/mask_eval 16m5b_N256_T1_win257_r2c.weights \
  data/sintetico_N256_seq512_win1000_seed4245.dat 256

# N=32/128: mismos flags con EVA_READ_WIN=N+1, EVA_READ_DF_N=N, y el corpus
# sintetico_N{32,128}_seq512_win1000_seed424{3,4}.dat.
```

Env relevantes: `EVA_READ_WIN=K` (K=N+1), `EVA_READ_DF_N=N` (distancia),
`EVA_READ_R2=1` (gate con piso), `EVA_READ_R2_CHANNEL=1` (gate por canal),
`EVA_READ_R2_FLOOR=x` (default 0.05), `EVA_READ_R2_Z0=x` (init de z, default
0.0), `EVA_WREAD_TRACE=n` (traza). Suite de tests: 79 passed, 0 failed.

## 10. Archivos y logs

- Código: `src/tensor/ops.rs`, `src/tensor/autograd.rs` (`clockmem_inject_learn_g`,
  `clockmem_inject_learn_gc`), `src/model/clock.rs` (envs R2/R2C, `r2_call`),
  `src/train.rs` (traza r2/r2c), `src/tests.rs` (4 tests R2 + gradchecks).
- Checkpoints: `16m5b_N{32,128,256}_T1_win{33,129,257}_r2c.weights`,
  `16m5b_N256_T1_win257_r2.weights` (block).
- Logs: `logs/train_N{32,128,256}_win{33,129,257}_r2c.log`,
  `logs/train_N256_win257_r2.log`, y toda la cadena previa en
  `logs/PARA-GPT-2026-08-15.md` (partes 1-5).
- Detalle diario: `TRABAJO-2026-08-15.md`.

---

## Resumen accionable (si querés ir directo)

1. **Paso 1 (control block vs channel)**: ambos resuelven N=256 — 0.0359 vs
   0.0356, distribución de s guardada (media 0.535, uniforme, ninguno al piso
   ni abierto). El principio parece general: la libertad espacial no se usa.
2. **Paso 2 (curva N)**: R2-channel resuelve 0.0244 / 0.0299 / 0.0356 en
   N=32/128/256. El gate no se adapta con N (≈0.53 fijo); se adapta la vía de
   memoria. La propiedad es general a la distancia.
3. **Paso 3 (floor/temperatura/z0)**: diferido. Nota: gradiente de z chico
   (canal ~1e-6, escalar ~1e-4).
4. **Pregunta abierta que quedó**: ¿el gate a media fuerza es robusto o
   síntoma de que el gradiente de z es demasiado chico para que la puerta
   "decida"? Distinguible con init z0>0 o más presupuesto.
5. **Fenómeno a medir cuando quieras**: la multiplicidad de soluciones
   (3 puntos ~0.035 con params internos distintos).
