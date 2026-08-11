# EVA v0

LLM propio, escrito **desde cero** en Rust. Cero dependencias (solo `std`), cero CUDA,
pensado para correr en hardware AMD común (la RX 580 de Martha) y para ser **totalmente
modificable**: vos cambiás cualquier pieza sin pelear contra un framework cerrado.

La idea de fondo: las LLM comerciales consumen mucho, y parte de ese consumo es lógica de
negocio (hardware más caro). Acá la prioridad es eficiencia y apertura.

## Qué hay en esta carpeta (v0)

- **`src/math.rs`** — hot path. `matmul` con AVX2+FMA (detección en runtime) y multithreading
  con `std::thread::scope`; fallback escalar con tiling para cache.
- **`src/tensor/`** — tensores contiguos (`Vec<f32>` + shape) y **autograd propio**:
  grafo de nodos, `saved_t`/`saved_f`/`saved_u` por operación, backward por op.
- **`src/tensor/ops.rs`** — ops con backward: `add`, `mul`, `scale`, `matmul`, `silu`,
  `sigmoid`, `rms_norm`, `gather`, `cross_entropy` (fused, estable), `conv1d` depthwise
  causal, y `clockmem` (el corazón experimental).
- **`src/nn/`** — trait `Module` + `Linear`, `RMSNorm`, `GLUFFN` (SwiGLU), `Embedding`,
  `DepthwiseConv1d`. Todo reporta parámetros nombrados → serialización genérica.
- **`src/model/`** — la arquitectura **EvaClock**: bloques apilables.
- **`src/optim.rs`** — AdamW (con weight decay decoupled).
- **`src/data.rs` / `src/tokenizer.rs`** — dataset por ventanas + tokenizer byte-level (vocab 256).
- **`src/gen.rs`** — sampling top-k con temperatura.
- **`src/save.rs`** — formato de pesos propio y abierto (`EVAV0`): config + parámetros nombrados.
- **`src/cli.rs`** — `eva train`, `eva gen`, `eva info` (sin clap, cero deps).

## La arquitectura: EvaClock (memoria multi-escala)

Nada de atención cuadrática. Cada bloque es:

```
x ──► +──► RMSNorm ─► Conv1D depthwise causal ──┐
      │                                          │
      └──────────────────────────────────────────┘  (mezcla local)
      │
      ▼
      +──► RMSNorm ─► ClockMem ──┐
      │                          │
      └──────────────────────────┘  (memoria recurrente multi-escala)
      │
      ▼
      +──► RMSNorm ─► GLU-FFN (SwiGLU) ──┐
      │                                   │
      └───────────────────────────────────┘
```

**ClockMem** (la parte novedosa):

```
s_t = α ⊙ s_{t-1} + β · (k_t ⊙ v_t)      // estado recurrente por canal
o_t = q_t ⊙ s_t ⊙ g_t                     // lectura con gate
α = sigmoid(log_clock)   // reloj aprendido POR CANAL, en (0,1)
```

- **Cada canal del estado tiene su propio decaimiento** aprendido: algunos canales olvidan
  rápido (sintaxis, formato), otros integran la secuencia entera (tema, entidad). La red elige
  la escala de tiempo que necesita, canal por canal.
- Compute **O(S·D)**, inferencia con estado **O(D)** (no depende de la longitud).
- Backward analítico en el op `clockmem` (backprop a través del tiempo, en un solo nodo).

Es una mezcla de linear attention + RNN multi-escala per-channel. No es el transformer de
siempre: es barato, pequeño y fácil de tocar.

## Roadmap

- [x] Núcleo tensor + autograd + arquitectura EvaClock
- [ ] Validación numérica del autograd (gradcheck contra diferencias finitas)
- [ ] BPE/tokenizer multilingüe
- [ ] Backend GPU **Vulkan** (compute shaders) para la RX 580 — nunca CUDA
- [ ] Scan asociativo paralelo para entrenar ClockMem con O(log n)
- [ ] Thread pool persistente (no spawn por matmul)
- [ ] CUDA: **no**, a propósito

## Uso

```sh
cargo run --release -- train --data data.txt --epochs 20 --dim 256 --seq 64
cargo run --release -- gen --weights eva.weights --prompt "hola "
cargo run --release -- info --weights eva.weights
```

## Cómo modificarlo

- ¿Otra arquitectura? Implementá `Module` (en `src/nn/`) y enchufala en `src/model/block.rs`.
- ¿Otra op? Agregala a `backward_op` en `src/tensor/autograd.rs` con su forward en `ops.rs`.
- ¿Otra optimización del matmul? `src/math.rs` es un archivo chico y aislado.
