# EVA v0

LLM propio, escrito **desde cero** en Rust. Cero dependencias (solo `std`), cero CUDA,
pensado para correr en hardware AMD común (la RX 580 de Martha) y para ser **totalmente
modificable**: vos cambiás cualquier pieza sin pelear contra un framework cerrado.

La idea de fondo: las LLM comerciales consumen mucho, y parte de ese consumo es lógica de
negocio (hardware más caro). Acá la prioridad es eficiencia y apertura.

## Qué hay en esta carpeta (v0)

- **`src/math.rs`** — hot path. `matmul` con AVX2+FMA (detección en runtime), repartido en
  bandas de filas sobre el pool; fallback escalar con tiling para cache.
- **`src/pool.rs`** — pool de hilos persistente. Los hilos se crean una vez y se estacionan;
  el que llama trabaja también. `EVA_SERIAL=1` lo saltea entero (útil para aislar bugs).
- **`src/gpu/`** — backend Vulkan por FFI cruda (sin `ash`, sin `wgpu`): instancia, device,
  pipeline de cómputo y un shader de matmul con tiling en registros.
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

## Rendimiento medido

Todo lo de abajo salió de `eva gpu`, mejor de 3 corridas de 30, **en una GTX 1650**.
La RX 580 de Martha es la placa que importa y estos números **no** están tomados ahí
todavía: la máquina estaba apagada. Tomarlos es el primer pendiente.

| matmul  | GPU al empezar | GPU hoy    | CPU (8 hilos) |
|---------|----------------|------------|---------------|
| 512³    | 11.63 ms       | **1.10 ms**| 8.00 ms       |
| 1024³   | ~27 ms         | **5.98 ms**| 142.86 ms     |

A 1024³ eso es ~360 GFLOP/s: **13% del pico teórico** de la placa. Queda margen, pero
el próximo paso se elige con el perfil en la mano.

### Y sin embargo, entrenar casi no mejora

`EVA_GPU=1` manda a la GPU todo matmul con más de 4M de trabajo (umbral medido: el
cruce está entre 128³, donde la CPU gana 1.25x, y 192³, donde gana la GPU 1.29x).
Entrenamiento completo de punta a punta, mismo corpus, misma config:

| modelo             | CPU      | GPU      |         |
|--------------------|----------|----------|---------|
| dim 256 / ffn 512  | 174.3 s  | 157.9 s  | 1.10x   |
| dim 512 / ffn 1024 | 312.3 s  | 241.6 s  | 1.29x   |

Un matmul suelto va 8.68x más rápido y el entrenamiento entero apenas 1.29x.

> **CORRECCIÓN (2026-08-11).** Acá decía que el cuello era la recurrencia de
> ClockMem. **Era falso**, y estaba escrito como un hecho a partir de evidencia
> indirecta. Al medirlo de verdad con `EVA_PROFILE=1`, ClockMem resultó ser el
> **0.8%** del tiempo: 2.4 s de 310. Hacerlo infinitamente rápido no cambiaría
> nada. Tercera vez en el día que la intuición elige mal el cuello; la tabla de
> abajo es lo que hay que mirar en su lugar.

## Dónde se va el tiempo de verdad

`EVA_PROFILE=1` mide por etiqueta. Entrenamiento dim 512 / ffn 1024, 861 pasos:

| etiqueta         | al empezar | + optimizador | + sin allocs | + GPU      |
|------------------|-----------|---------------|--------------|------------|
| matmul           | 135.6 s   | 137.3 s       | 133.5 s      | 66.2 s     |
| backward (todo)  | 173.7 s   | 172.2 s       | 143.4 s      | 105.0 s    |
| optimizador      | 62.7 s    | **22.3 s**    | 17.5 s       | 19.8 s     |
| clockmem fwd+bwd | 2.4 s     | 2.4 s         | 2.4 s        | 2.7 s      |
| **reloj**        | 310.2 s   | 269.3 s       | 233.1 s      | **164.3 s**|

**310 s → 164 s, 1.90x**, sin tocar una sola línea de ClockMem.

Los dos arreglos, los dos aburridos:

1. **El optimizador se llevaba el 20%.** Dos pasadas sobre los 10.7M de parámetros,
   escalar y en un solo hilo. Fusionadas en una y repartidas en bandas sobre el pool:
   62.7 → 22.3 s. No dio 8x porque ahora choca contra el ancho de banda de memoria
   (171 MB por paso), que es el piso real.
2. **`broadcast_to` y `reduce` asignaban DOS `Vec` por elemento.** Se llaman en el
   backward de `add` y `mul`; para un tensor de 64x1024 eran 131 mil allocations en
   una sola llamada. Reemplazadas por un recorrido con odómetro y paso 0 para la
   dimensión que se difunde, sin tocar el heap: backward 172 → 143 s. De paso, el
   chequeo de compatibilidad que estaba adentro del lazo sólo dependía de las formas.

Lo que queda arriba de la lista, con los números en la mano: **el backward sin
contar matmul, ~96 s, casi la mitad del total**. Eso es maquinaria de autograd
—asignaciones y clones por operación—, no aritmética.

Está apagado por defecto a propósito: la GPU suma en otro orden, así que dos
entrenamientos con y sin ella no dan bit a bit lo mismo. Que eso pase tiene que ser
una decisión. Las pérdidas finales sí coinciden (0.112 vs 0.118 en el chico).

```sh
cargo run --release -- gpu --m 1024 --k 1024 --n 1024 --iters 30
EVA_GPU_PROFILE=1 cargo run --release -- gpu   # parte el tiempo en subir/calcular/bajar
cargo test --release                            # 12 tests
cargo test --release -- --ignored               # 2 más, piden una GPU con Vulkan
```

## ¿Es buena la arquitectura? Ahora hay un número

Mismo bloque, misma conv, mismo FFN, mismo optimizador, mismas semillas, mismo
corpus. **Lo único que cambia es el mezclador temporal.** ClockMem tiene
`wq/wk/wv/wg` y la atención `wq/wk/wv/wo`: cuatro matrices DxD cada uno.

250 KB de prosa real, dim 256 / ffn 512 / 4 bloques / seq 64, 1 época, 10%
reservado para validar (corte contiguo al final). Pérdida sobre texto no visto:

| mezclador                | s7     | s8     | s9     | media      | bits/byte | tiempo |
|--------------------------|--------|--------|--------|------------|-----------|--------|
| **ClockMem** (2.77 M)    | 1.8245 | 1.8307 | 1.8135 | **1.8229** | **2.630** | 319 s  |
| atención (2.77 M)        | 1.8585 | 1.8711 | 1.8593 | 1.8630     | 2.688     | 339 s  |
| atención + pos (2.78 M)  | 1.9253 | 1.9229 | 1.9166 | 1.9216     | 2.772     | 345 s  |

**ClockMem gana por 2.2% y corre 6% más rápido**, con estado O(D) en inferencia
contra O(S²) de cómputo. Los tres grupos no se solapan: la peor corrida de
ClockMem (1.8307) es mejor que la mejor de la atención (1.8585), y la brecha
entre arquitecturas es mayor que la dispersión dentro de cada una.

La tercera fila es una hipótesis mía que salió mal, y se deja porque ese es el
punto: sospeché que la comparación era injusta --ClockMem codifica la posición
gratis en el decaimiento `α^(t-i)` y la atención no tenía nada-- así que le di
posiciones absolutas aprendidas y 16 K parámetros de ventaja. **Empeoró en las
tres semillas.** La conv causal depthwise ya le daba estructura posicional; la
tabla sólo diluía. El resultado original no era un artefacto.

### Lo que este número NO dice

- **seq 64 es corto.** Lo que compra la atención es recuperación asociativa a
  distancia, y a 64 tokens casi no hay distancia. La comparación que falta es a
  contexto largo, donde ella paga O(S²) y ClockMem O(S).
- Un corpus, un tamaño, una época. Los dos modelos están muy poco entrenados.
- Una sola cabeza, y posiciones absolutas aprendidas (el esquema posicional más
  débil). Con RoPE o multi-cabeza el resultado podría moverse.

## Trampas que ya nos costaron caro

Están documentadas en el encabezado de cada archivo, pero conviene tenerlas juntas:

1. **`pool.rs`** — esperar en un condvar con `if` en vez de `while`, más `notify_all` al
   terminar, hacía que los workers se despertaran entre sí y **re-ejecutaran el trabajo
   viejo**: medido, 23 a 31 veces cada pieza por una sola llamada. Invisible mientras los
   buffers vivieran; SIGSEGV apenas el que llamaba liberaba los suyos.
2. **`gpu/mod.rs`** — el descriptor pool tenía lugar para un set y se pedía uno por
   llamada: **la segunda llamada de cualquier proceso fallaba**. No se notó nunca porque
   `eva gpu` multiplicaba una sola vez.
3. **`gpu/mod.rs`** — staging `HOST_VISIBLE|HOST_COHERENT` sin `HOST_CACHED` es memoria
   **sin caché**: escribirla va bien, leerla desde la CPU va a ~300 MB/s. Bajar el
   resultado se llevaba el 70% del tiempo total. Un flag.
4. **Cortar un tensor con `Tensor::new` para pasarlo al grafo.** El corte no
   tiene nodo de autograd: el gradiente nunca vuelve y el parámetro entrena
   contra nada, **en silencio y con el mismo aspecto que uno que aprende**. Por
   eso existe `slice_rows`, con gradcheck y con un test de que NO llegue
   gradiente a las filas no usadas.
5. **Creer que el cuello era ClockMem** por evidencia indirecta, escribirlo en este
   README como un hecho, y que al medirlo fuera el **0.8%**. La evidencia indirecta
   sirve para elegir qué medir, nunca para concluir.

La moraleja de las tres, y de la que más duele: **la intuición decía "optimizá el
shader" y eran los flags de memoria.** Medir primero, partido en fases.

## Roadmap

- [x] Núcleo tensor + autograd + arquitectura EvaClock
- [x] Validación numérica del autograd (gradcheck contra diferencias finitas)
- [x] Backend GPU **Vulkan** (compute shaders) — nunca CUDA
- [x] Thread pool persistente (no spawn por matmul)
- [ ] **Medir todo esto en la RX 580** (los números de arriba son de una GTX 1650)
- [x] Integrar el matmul de GPU en `math::matmul` como backend opcional (`EVA_GPU=1`), con umbral medido
- [ ] **Bajar las asignaciones del autograd** — el bucle real: ~96 s de 201, casi la mitad
- [ ] ~~Scan asociativo paralelo para ClockMem~~ — MEDIDO: ClockMem es el 0.8% del tiempo.
      Sólo valdría si D fuera chico o para un port de ClockMem a GPU; hoy no mueve la aguja
- [ ] BPE/tokenizer multilingüe
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
