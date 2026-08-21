# Para OpenCode

Hola. Soy Claude. Valentín me pasó el proyecto cuando se te murieron los
credenciales, seguí desde donde dejaste y esto es todo lo que pasó. Está
escrito para que puedas discutirlo, no para que lo aceptes: donde me parece que
me equivoqué o que algo quedó flojo, lo digo.

---

## Parte 1 — Dónde te trabaste, y qué era

Quedaste con `matmul_threaded` tirando **SIGSEGV** y un `[dbg]` que no imprimía
nada. Estabas buscando en `math.rs`, en las bandas de `out` y los punteros
crudos. **El matmul estaba bien.** El bug era de `pool.rs`.

`worker()` esperaba en el condvar con `if` en vez de `while`, y cada worker que
terminaba llamaba a `notify_all`. Eso despertaba a los que ya estaban
estacionados, que releían `st.job` --que nunca se ponía en `None`-- y **volvían
a ejecutar el trabajo viejo**. Lo medí antes de tocar nada, con `--nocapture`:

    pieza 0: 1 vez      (la que corre el que llama)
    piezas 1..7: entre 23 y 31 veces cada una

Recalcular una banda de matmul da el mismo resultado, así que era invisible
mientras los buffers vivieran. Cuando el test terminaba y liberaba sus
vectores, los rezagados seguían escribiendo ahí: **ese era el segfault**, y por
eso no aparecía siempre.

Tu `[dbg]` no imprimía porque el harness de test captura la salida y el proceso
moría antes de volcarla.

Lo cerré con tres propiedades, cada una con su test: todo `wait` revalida `gen`
en un `while`; los workers avisan al que espera por un condvar **separado** (así
terminar no despierta a un par, y de paso saca el *thundering herd*); y `run`
limpia el job bajo el lock antes de volver, así ningún puntero capturado
sobrevive a la llamada.

De paso, en `math.rs` había un underflow latente: `be - bs` en `usize` cuando
`rows_per * chunks` pasaba de `m`. No se había disparado --hacía falta un equipo
de 34+ hilos-- pero armaba un slice sobre todo el espacio de direcciones.

## Parte 2 — Tu lista de pendientes: toda hecha

- **Thread pool persistente** ✅ (`pool.rs`, reescrito por lo de arriba)
- **Buffers GPU reutilizables** ✅ — y ahí había otro bug tuyo/nuestro: el
  descriptor pool tenía lugar para **un** set y se pedía uno por llamada, así
  que **la segunda llamada de cualquier proceso fallaba**. Nunca se notó porque
  `eva gpu` multiplicaba una sola vez; un entrenamiento lo habría encontrado en
  el segundo paso.
- **Shader más rápido** ✅ pero con una sorpresa, ver abajo.
- **Integrar matmul GPU en `math::matmul`** ✅ (`EVA_GPU=1`, umbral medido en 4M
  de trabajo: el cruce está entre 128³ y 192³).
- **Suite completa + entrenamiento chico end-to-end** ✅

### La sorpresa del shader, que es la lección más cara

Hice el tiling en registros que proponías (cada hilo calcula 4x4 en vez de 1,
bajando de 2 lecturas de LDS por FMA a 0.5). Medido alternando viejo y nuevo,
mejor de 3:

    512³    5.06 -> 4.47 ms   (1.13x)
    1024³  27.25 -> 22.79 ms  (1.20x)

Mucho menos de lo que predecía la cuenta. **Eso no era ruido: era la pista.**
Partí el tiempo en fases (`EVA_GPU_PROFILE=1`):

    subir 0.63 | cálculo 4.85 | BAJAR 12.5-18.1 | total ~20 ms

Bajar el resultado era el 70%. La causa: el staging pedía
`HOST_VISIBLE|HOST_COHERENT` y nada más, que en una placa discreta es memoria
**sin caché** -- escribirla va bien, leerla desde la CPU va a ~300 MB/s.
Agregando `HOST_CACHED` sólo al buffer que se lee:

    512³   11.63 -> 1.10 ms   (10.5x contra el punto de partida)
    1024³  ~27   -> 5.98 ms

**La intuición decía "optimizá el shader" y eran los flags de memoria.**

## Parte 3 — Lo que faltaba y no estaba en la lista

**No había forma de saber si el modelo era bueno.** Sin held-out, sin baseline,
la única evidencia era que memorizaba un corpus de 55 KB, cosa que hace
cualquier cosa con suficientes parámetros. Agregué:

- Validación con corte **contiguo al final** (aleatorio filtraría: las ventanas
  vecinas comparten contexto), reportada en bits/byte.
- Una **atención causal de una cabeza** como rival, justa por construcción:
  ClockMem tiene `wq/wk/wv/wg` y la atención `wq/wk/wv/wo`, cuatro matrices DxD
  cada uno, mismo bloque, mismo FFN, mismas semillas.
- Corpus real de 405 KB (documentación de los proyectos), 100% de líneas
  únicas. El anterior eran 900 repeticiones de ocho frases.

Y apareció un bug de formato: al cargar, `eps` se leía **sin avanzar el
cursor**, así que `seq_len` releía esos bytes y volvía 925353388. Todo
checkpoint cargado tenía la ventana rota, y como sólo recorta, no fallaba nunca.
Formato subido a EVAV1.

## Parte 4 — Los experimentos, incluidos los que salieron mal

### ClockMem le gana a la atención (3 semillas, sin solapar)

    ClockMem        2.630 bits/byte   319 s
    atención        2.688             339 s
    atención + pos  2.772             345 s

La tercera fila es una hipótesis mía que salió mal y la dejo escrita: sospeché
que la comparación era injusta porque ClockMem codifica posición gratis en su
decaimiento, le di posiciones absolutas y 16 K parámetros de ventaja, y
**empeoró en las tres semillas**. La conv causal ya le daba estructura
posicional.

### Perfil del entrenamiento, y dos arreglos aburridos

`EVA_PROFILE=1`. Yo había afirmado en el README que el cuello era ClockMem. Al
medirlo: **0,8% del tiempo.** Lo que sí pesaba:

- El optimizador se llevaba el 20% recorriendo los parámetros **dos veces**, en
  un solo hilo. Fusionado y repartido en bandas: 62,7 → 22,3 s.
- `broadcast_to` y `reduce` asignaban **dos `Vec` en el heap por cada número**
  (`index_to_coords` devolvía uno y las coordenadas eran otro). Para 64x1024
  son 131 mil allocations en UNA llamada, en el backward de `add` y `mul`.
  Reemplazadas por un recorrido con odómetro: backward 172 → 143 s.

Entrenamiento completo: **310 → 164 s**, sin tocar una línea de ClockMem.

### Inferencia con estado: 37 → 750 tok/s

`generate` rehacía la pasada completa sobre toda la ventana **para cada token**
y descartaba 63 de las 64 filas. Ahora el estado viaja (`src/stream.rs`).

    64 tokens     37.5 -> 356.4 tok/s
    1024 tokens   se degradaba -> 750.6 tok/s

Lo importante no es el múltiplo: antes la velocidad **caía** con el contexto y
ahora **sube**. Implementé también el caché KV de la atención, que no
necesitábamos, para poder medir la diferencia en vez de afirmarla: el estado de
ClockMem no crece después de 200 tokens; el de la atención crece hasta el tope
y hay que recortarlo.

### Los que salieron mal, con su diagnóstico

**Aprender sólo de lo que sorprende** (saltear el backward si la pérdida está
bajo lo esperado): a igual presupuesto de backwards queda **1,2% peor pagando
28% más**. Falló porque filtré por *ventana* --64 tokens promediados-- cuando
el argumento es sobre *tokens*: al promediar, la varianza se lava. Y aun por
token no ahorraría: enmascarar la pérdida no abarata un backward que recorre el
grafo entero igual. **La sparsity en la pérdida no se cobra si el cómputo es
denso.**

**Crédito local** (cada bloque con su objetivo, sin gradiente que cruce): a 4
bloques cuesta 3,6% de calidad, es 12% más lento y ahorra sólo 14% de memoria.
El ahorro sí crece con la profundidad (45% a 16 bloques) y las activaciones van
de ~170 MB a ~29 MB, pero el costo de calidad a profundidad **está sin medir**.

**Estado persistente entre ventanas**: neutro. La primera corrida daba 11,9%
peor pero eso era **saturación, no la idea**: `log_clock` llegaba a
alpha=0.9999 y el estado alcanzó magnitud 880. Con techo en 0.999 baja a 27 y
se recupera. De ahí salió lo que más vale del experimento: **barajar cuesta
8,4%**, y persistir obliga a ir en orden. Cualquier esquema con memoria entre
ventanas arrastra esa mochila.

### El que sí importa: la hipótesis central

Modelo A de 2.767.108 parámetros contra modelo B de **1.413.433 (la mitad) más
una tabla de k-gramas exactos** --cero parámetros, cero entrenamiento--. Lambda
elegido en desarrollo, validación tocada una sola vez. Tres semillas:

    semilla   A solo    B+tabla         B+tabla (sólo órdenes >=6)
       7      1.8413    1.7262  -6,3%   1.8152  -1,4%
       8      1.8633    1.7360  -6,8%   1.8362  -1,5%
       9      1.8705    1.7394  -7,0%   1.8453  -1,3%

La mitad de los parámetros más una tabla tonta **le gana al modelo entero**. La
cobertura con todos los órdenes es 98,9%, así que sospeché suavizado de
n-gramas y medí de dónde vienen los aciertos (8:24% 6:25% 4:34% 3:12% 2:5%).
Restringiendo a órdenes >= 6, que disparan sólo el 41% de las veces, **sigue
ganando**.

**La cuenta que importa no es la de disco**: la tabla pesa 4,3 MB, A pesa 11,1
y B 5,7, así que B+tabla y A almacenan casi lo mismo. Se gana en lo que hay que
**leer por token**: A lee 11,1 MB enteros, B lee 5,7 MB más un hash de cien
bytes. La mitad del tráfico, que es el cuello de la generación.

---

## Parte 5 — Mi evaluación honesta, para discutir

**Hasta ahora no construimos nada nuevo.** Te lo digo derecho porque es lo que
Valentín quiere del proyecto y conviene no engañarnos:

- ClockMem es un SSM diagonal: la familia de S4D, RWKV, RetNet.
- La inferencia con estado es lo que ya venden Mamba y RWKV.
- La tabla es interpolación n-grama + red, práctica de los 2000 revivida por
  kNN-LM en 2020.
- Sorpresa es *hard example mining*; crédito local es *greedy layerwise*;
  estado persistente es TBPTT. Los tres, conocidos, y los tres fallaron.

Lo que sí es poco común es el rigor: held-out, baseline pareado, varias
semillas, y los negativos publicados al lado de los positivos.

Y hay un patrón: **todo lo que probé son perillas del bucle estándar**
--variaciones sobre descenso por gradiente prediciendo el próximo token--. La
única que se salió un poco fue la tabla, y es la que funcionó.

**Dónde creo que está la puerta:** usamos la tabla *por afuera*. El modelo no
sabe que existe, así que gasta capacidad memorizando lo que la tabla ya tiene.
Entrenar **con el almacén adentro de la pérdida** --retropropagando a través de
la mezcla-- haría que el gradiente se apague solo donde la tabla ya acierta, y
esa capacidad quede libre. Eso no es RAG ni kNN-LM: cambia qué son los pesos.
Está a un cambio chico de distancia, porque la mezcla ya está escrita.

## El marcador, que es la regla de la casa

Mi intuición va **0 de 6** en este proyecto: el shader que no era, ClockMem que
era el 0,8%, las posiciones que iban a emparejar y empeoraron, la sorpresa que
iba a rendir, el crédito local que iba a ahorrar mucho, el estado persistente
que iba a dar memoria gratis. **La evidencia indirecta sirve para elegir qué
medir, nunca para concluir.**

Si vas a tocar algo: medí antes, medí después, y publicá los dos números aunque
el segundo sea peor. Y si escribís un segundo camino para la misma matemática
(como `stream.rs`), el test que exige que los dos den lo mismo va **antes** que
el código -- se desincronizan solos y no avisan.

---

## Referencia rápida

```sh
cargo test --release                  # 33 tests
cargo test --release -- --ignored     # 2 más, piden GPU con Vulkan

cargo run --release -- train --data data/prosa250.txt --arch clock \
    --epochs 1 --dim 256 --ffn 512 --blocks 4 --seq 64 --val 0.1 --seed 7
cargo run --release -- recall --weights X.weights --data data/prosa250.txt
cargo run --release -- techo --weights X.weights --data data/prosa250.txt
cargo run --release -- gpu --m 1024 --k 1024 --n 1024 --iters 30
```

Interruptores: `EVA_GPU=1`, `EVA_GPU_PROFILE=1`, `EVA_PROFILE=1`,
`EVA_SERIAL=1`, `EVA_MIN_ORDER=6`, `EVA_ALPHA_MAX=0.999`.
Banderas: `--arch clock|attn`, `--val`, `--surprise`, `--local`, `--persist`,
`--inorder`.

**Todos los números son de una GTX 1650 + Comet Lake-H de 8 núcleos.** La placa
que importa es la RX 580 de Martha y ahí no se midió nada: el umbral de GPU de
`math.rs` es de NVIDIA y en GCN hay que rehacerlo.

Dónde tocar: `src/model/` arquitecturas, `src/learn.rs` reglas de aprendizaje
(trait intercambiable), `ops.rs` + `autograd.rs` + **gradcheck en `tests.rs`**
para operaciones nuevas, `math.rs`/`pool.rs`/`gpu/` para rendimiento.

Y leé `DISENO.md`: está el razonamiento entero de por qué las LLM son caras
hoy, qué de eso es física y qué es costumbre, y las hipótesis abiertas.

---

## Parte 6 — El experimento decisivo del largo alcance: la lectura con ventana (14-08)

**Contexto (dónde te habías trabado).** En `TRABAJO-2026-08-14.md` quedó el
corte: el benchmark de largo alcance (dependencia a N pasos, corpus sintético
autocontenido por ventana de 512) daba **bpb ≈ 8.0 (uniforme) para N=32, 128 y
256** con ClockMem, contra 0.0048 en N=8. El espectro α estaba CONGELADO en su
init (α_media 0.2156, gradientes de `log_clock` colapsados a ~1e-6, ver "el
espectro NO migra"). Antes de tocar `src/` corrimos un probe de memoria
(`examples/state_probe.rs`, pre-registrado).

**Lo que dijo el probe.** El estado con fuga `cur[t] = β·Σ α^j·k[t-j]·v[t-j]`
no puede recuperar `input[t-N+1]` a través de `out=q·cur·g` (probe A = 8.0000
bpb en toda N≥32). PERO la "escritura"
`w[τ] = (cur[τ] − α_c·cur[τ−1])/β` — la diferencia finita que una lectura con
ventana de tamaño ≥N+1 puede expresar — es **recuperable posición-a-posición
con 54-85% de acierto** incluso en modelos que nunca aprendieron (loss 8.0).
Es decir: el valor SÍ está en el estado; el límite era la lectura con fuga, no
el almacenamiento.

**Lo que implementé (condición B).** Un readout con ventana:
`read[t,c] = Σ_{j<min(K,t+1)} w[j,c]·state[t-j,c]`, `out[t] = q[t]·read[t]·g[t]`
— op `clockmem_readwin` (forward `src/tensor/ops.rs`, backward
`src/tensor/autograd.rs`), activado por `EVA_READ_WIN=K` en `src/model/clock.rs`
(init delta `w[0,c]=1`, resto 0, para que B arranque bit-idéntica a la base).
Gradcheck numérico en `tests.rs` (`gradcheck_clockmem_readwin`); suite 69 OK.

**El golpe de rendimiento.** La versión naive (lazos anidados j-outer, saltos
strided de `d`) corría a 4-5 s/paso. Con precompute `vals=g·q·gg` + partición de
los 3 lazos pesados (read, `direct`, `gw`) en bandas con `crate::pool::global()`
(8 núcleos, c-inner secuencial), el op quedó en 0.62 s/paso y el training
entero volvió a **1.4 steps/s = la tasa base**. Si tocas ese op: los lazos
tienen que ser paralelos y c-inner, o el costo es K·s·d strided y mata el step.

**Resultado: B no rompe el piso.** Entrené los 3 N con receta idéntica a la
base (solo `EVA_READ_WIN=K=N+1`):

| N | base bpb (máscara t≥N) | cond B | `wread` post-entrenamiento |
|---|------------------------|--------|----------------------------|
| 32 | 8.0012 | **8.0017** | ≈delta (w[0]≈0.996, taps ~0.007) |
| 128 | 8.0021 | **8.0018** | ≈delta (w[0]≈0.994, taps ~0.007) |
| 256 | 8.0039 | **8.0093** | ≈delta (w[0]≈1.00, taps ~0.009) |

Las 3 corridas fueron clones exactos de la base (losses idénticas a 4
decimales, mismos alpha/grad, misma tasa) — control limpio. El diagnóstico
(`examples/check_wread.rs`: `EVA_READ_WIN=K cargo run --release --example
check_wread -- <ckpt>`) mostró que **el modelo nunca movió los taps de la
ventana**: se quedó en el delta init.

**Interpretación (lo importante).** No es que el estado "no pueda": el probe
demostró que la información está y es extraíble. Es que **la optimización no
recluta la ventana** — el gradiente hacia `wread` es demasiado débil con la
receta base. El límite de largo alcance de ClockMem es de **aprendibilidad del
readout**, no de representación. Siguiente paso pre-registrable (B2):
arrancar `w` con la diferencia finita ya puesta (`w[N-1]=1/β`, `w[N]=-α_c/β`),
o supervisar la escritura directamente, o un lr separado para `wread`.

**Notas del entorno que dolieron (importantes para no repetir).**
- NUNCA usar `/tmp/opencode` para logs; todo va a `logs/` del repo.
- Los outputs de la sesión se corrompieron/duplicaron varias veces (reads
  stale, "exit=137" fantasma de una corrida que en realidad terminó OK, un log
  de N128 que "no avanzaba" y después apareció completo). Verificar siempre
  con timestamps y mirando el archivo en disco.
- El tool `edit` reportaba éxito sin cambiar el archivo: parchear con python
  heredoc y verificar con `rg`/`nl` en el mismo comando.
- `Tensor.data` es `Arc<Vec<f32>>` (en `saved_v` clonar `q.data` directo;
  `Arc::new` solo para vecs frescos).
- El binario es `target/release/eva_llm_v0`; rebuild tras tocar `src/`
  (binarios viejos se confundieron con nuevos antes).

**Archivos nuevos:** `examples/state_probe.rs`, `examples/bench_win.rs`,
`examples/check_wread.rs`, `run_windowed.sh`, `eval_windowed.sh`. Logs y
resultados: `logs/train_N*_win*.log`, `logs/eval_N*_win*.log`,
`logs/PARA-GPT-2026-08-14.md`, `TRABAJO-2026-08-14.md`.

---

## Parte 7 — La puerta q·g es el límite del largo alcance; inyección limpia lo resuelve (15-08)

**Contexto (dónde te habías trabado).** Condición B (readout con ventana,
`clockmem_readwin`, delta-init aprendible) no rompió el piso: `wread` quedó ≈
delta y mask_eval seguía en ~8.0 para N=32/128/256, pese a que el probe de
memoria mostró la escritura `(cur−α·cur)/β` recuperable 54-85%. Pre-registrada:
condición B2 (init con la diferencia finita + lr separado). Todo el detalle en
`TRABAJO-2026-08-15.md` y `logs/PARA-GPT-2026-08-15.md`.

**La cadena de esta sesión (N256, mask_eval bpb en t≥N):**

| condición | readout | bpb |
|---|---|---|
| base T1 | `out=q·cur·g` | 8.0039 |
| B (ventana aprendible) | `out=q·read·g` | 8.0093 (w≈delta) |
| B2.1 (init df exacta + lr×10, `EVA_READ_DF_N`+`EVA_READ_LR`) | ídem | 8.0164 |
| oracle (`EVA_READ_ORACLE`, taps exactos recomputados por forward) | `out=q·read·g` | 8.0021 |
| **inject** (`EVA_READ_INJECT`: `out=q·cur·g + write_rec[t]`) | **0.0353 / 99.6%** | |

**Conclusión: el límite de largo alcance de ClockMem es la puerta `q·g` en la
lectura, no el almacenamiento ni el readout.** Con los taps EXACTOS en `read`
(oracle) la escritura llega a `read[t]` y aun así `out=q·read·g` falla; la
única variante que resuelve es la inyección limpia del write recuperado al
residual, sin pasar por `q·g` (`ops::clockmem_inject`, sin parámetro `wread`).

**Probe (`examples/activation_probe.rs`, nuevo):** sobre el modelo inject, la
escritura recuperada decodifica el token **100% en `b0.read`** y sobrevive a
todo el stack hasta `final.head_in` 99.98%; **MODEL OWN top-1 = 99.980%**
(25595/25600).

**Bug de label que te va a morder si tocás estos probes:** el generador aplica
una permutación FIJA F (`y[t]=F(x[t−N])`, `PHYSICS_SEED=0xF15A1CA`, ver
`examples/generate_synthetic.rs`). El modelo emite `F(input[t−N+1])`, NO
`input[t−N+1]`. Un "MODEL OWN" o ridge que no aplique F da ~1% o decodifica
con sesgo (por eso el "1.098%" y el "84%" de ayer). La F se replica con
`Rng::new(PHYSICS_SEED).shuffle(0..256)`.

**Env relevantes (setear también en eval/load, o el checkpoint se carga como
base y da 8.0):** `EVA_READ_WIN=K` (K=N+1), `EVA_READ_DF_N=N`,
`EVA_READ_INJECT=1`, `EVA_READ_ORACLE=1` (diagnóstico), `EVA_READ_LR=x`
(lr×x de `wread`, `lr_mult` por parámetro en `src/optim.rs`). En modo
oracle/inject no existe `wread`.

**Pregunta que queda viva (siguiente paso honesto):** la inyección limpia gana
por construcción (la solución está puesta a mano, no aprendida). El paso que
valdría la pena: **convertir la inyección en aprendible** (p. ej. hacer que la
puerta tenga una vía aditiva o un `w_inj`), y ver si el gradiente ahora sí fluye
a la puerta en las posiciones con máscara. También falta: reproducción del
resultado inject (1 sola corrida), α no migra (14-08) y si eso cambia con la
inyección, y extender a N=128/32 con inyección limpia.

**Archivos nuevos:** `examples/activation_probe.rs`, cambios en
`src/tensor/ops.rs`/`autograd.rs` (`clockmem_inject`), `src/model/clock.rs`
(envs oracle/inject/df), `src/optim.rs` (`lr_mult`), `src/train.rs`
(`EVA_READ_LR`), `src/tests.rs`. Logs: `logs/train_N256_win257_inject.log`,
`logs/eval_N256_win257_*.log`, `logs/PARA-GPT-2026-08-15.md`,
`TRABAJO-2026-08-15.md`.

---

## Parte 8 — Benchmark de entidades + control carry 2×2 + baseline SSM (15-08, cierre)

Tres cosas encadenadas, todo en `examples/entity_retrieval.rs` + el env
`EVA_SSM_ALPHA` (baseline SSM):

1. **Benchmark dirigido (orden de GPT):** entidades del Quijote por bucket de
   distancia, modos independiente y carry. Resultado independiente NEGATIVO:
   hasta 512 tokens nadie recupera entidades por memoria larga (T1 y T1.3
   iguales; la recurrencia no aporta sobre la atención local). El bpb global
   no veía esto; el dirigido sí — la hipótesis de GPT confirmada como método.

2. **Control carry 2×2 (orden de GPT):** Claude reentrenó T1@Martha; yo, sin
   acceso a Martha, el espejo T13@Cristina. Quedó completo: **T1 colapsa en
   carry en ambas máquinas, T1.3 estable en ambas** (T13@Cristina
   numéricamente indistinguible de T13@Martha, bpb 1.720 = 1.720 → el
   entrenamiento es invariante a la máquina). La asimetría es efecto real de
   temperatura, no varianza de hardware. Primera propiedad de T=1.3 no nula.

3. **Baseline SSM alpha fijo 0.999 (orden de GPT, propuesta de Claude):**
   mismo `entity_retrieval`, mismos buckets, mismo carry, mismas métricas.
   Resultado: **el SSM también mantiene el estado entre ventanas (no colapsa
   como T1) → la estabilidad no es particular de T=1.3.** Y lo importante
   (GPT lo pidió separar): **estabilidad del estado ≠ memoria útil** — en
   carry la distancia >1024 nunca supera la referencia sin-memoria del mismo
   modelo (T1 7.1 vs 34.9; T1.3 33.8 vs 37.4; SSM 14.6 vs 29.4 top1%).
   Arrastrar estado nunca mejora la recuperación respecto de empezar de cero.
   La ventaja cuantitativa de T1.3 sobre SSM es solo retención (canales lentos
   más lentos), no memoria que sirva para lenguaje.

**Lectura honesta del cierre:** el benchmark de entidades (la métrica
principal desde Parte 7) es negativo para memoria larga en todos los modos y
todas las variantes. El bpb global queda descartado para conclusiones de
memoria. Lo único que quedó demostrado es que T=1.3 no colapsa el estado al
cruzar ventanas (y que eso no es exclusivo suyo). Siguiente pregunta viva
(para decidir con el equipo): ¿régimen (seq 512 / 1 época / 13M params) o
mecanismo? — y en su momento, seq más largo / más épocas, siempre sobre el
benchmark dirigido, no sobre bpb global.

**Archivos nuevos:** `examples/entity_retrieval.rs`, env `EVA_SSM_ALPHA` en
`src/model/clock.rs` (+ test `ssm_fixed_alpha_excludes_clock_and_is_constant`
en `src/tests.rs`, suite 80 passed), `src/train.rs` (trace vía
`alpha_physical`), `examples/entity_retrieval.rs` (header SSM). Checkpoints:
`16m5b_quijote_T1_martha.weights`, `16m5b_quijote_T13_Cristina.weights`,
`16m5b_quijote_SSM.weights`. Logs: `logs/entity_retrieval_quijote_*.log`,
`logs/train_quijote_{T13_cristina,SSM}.log`, `logs/PARA-GPT-2026-08-15.md`
(Partes 7-8), `logs/PARA-CLAUDE-2026-08-15.md` (§6-7).

---

## Parte 9 — Compuerta de escritura como detector de novedad: NEGATIVO, cerrado (16-08)

**El experimento (agregado al modelo en producción, no es un branch).** Probe
mínimo acordado con Valentín: la magnitud de escritura error-condicionada
dentro del forward normal, puerta *detach-eada* y sin parámetros nuevos.
Implementado en `src/model/clock.rs` y `src/train.rs` (envs `EVA_WRITE_ERROR`,
`EVA_WRITE_ERROR_P`, trace por `wprobe`). El mecanismo y el probe quedaron
documentados en los comentarios del código (la consigna de Valentín: agregar
mecanismo y medición a producción, sin tocar hiperparámetros). La idea era
detectar repeticiones de un token (novedad = clave nueva vs. repetida) desde
la magnitud de escritura, sin entrenar nada.

**Medición (Dante, dos condiciones).** `examples/wprobe.rs` (checksum `wprobe`,
se reutiliza el corpus `assoc_k2_clock` que dio el 93% top-1):

| condición | clave | e = \|d\|_2 | r = \|α·d\| | write = β·r | sat (σ(·)) |
|---|---|---|---|---|---|
| assoc_k2_clock, seq 512, dim 512, 5 bloques | nueva | 0.9606 | 0.983 | 1.0104 | 49.4% |
| assoc_k2_clock, idem | repetida | 0.9629 | 0.986 | 1.0136 | 54.2% |
| prosa (seed7), seq 64 | nueva | 0.9167 | 0.956 | 0.8385 | 47.6% |
| prosa (seed7), idem | repetida | 0.9170 | 0.970 | 0.8512 | 54.5% |

Baseline de la magnitud (media de β sobre las claves): assoc 1.0277, prosa
0.8775 → la señal bruta por token es un 3-9% de la magnitud media. La
repetición es un **artefacto puramente mecánico** de la puerta (r y write
correlacionan al 100% con la magnitud de `d`, no con la identidad del token):
el experimento está bien, la hipótesis de que el error condiciona la magnitud
en la dirección "menos para lo repetido" es falsa en producción.

**Verificación independiente (Valentín, ~4h después):** mismo patrón, ~2% de
separación entre repetida y nueva, y **en la dirección OPUESTA a la
esperada** (la repetida escribe MÁS, no menos). Los `e` de su corrida salen
más bajos (0.82/0.87 vs 0.96/0.96), probablemente por el checkpoint (grande
vs chico); el patrón y la conclusión son los mismos. Confirmado: **la puerta
no discrimina novedad.** El hilo se cierra como NEGATIVO.

**Diagnóstico mecánico (confirmado por Valentín, no es un bug).** El error se
computa contra el estado COMPLETO, que es una mezcla acumulada de todo lo
escrito. Comparar la escritura de un solo paso contra esa mezcla diluye
cualquier repetición individual: un único "token igual al de hace 5 pasos" es
una diferencia de 1/5,000 de la masa del estado (y en un estado de 512 canales,
aún menos). La señal plana era lo esperable matemáticamente — la pregunta
estaba mal planteada contra el objeto equivocado, no mal implementada.

**Autocrítica a mi propuesta de comparar contra "dónde se escribió la clave
la última vez" (circular, descartada).** Para saber dónde se escribió la clave
la última vez hay que identificarla como repetida de antemano → es el problema
de recuperación por contenido, no una señal más barata. Circular. No se
construye.

**Propuesta nueva, NO circular (queda documentada, NO se implementa).**
Comparar la escritura contra un **segundo estado auxiliar con decaimiento
mucho más rápido** (memoria de solo los últimos pasos): una "huella de lo muy
reciente", mucho menos diluida que el estado completo, sin necesidad de
resolver primero la recuperación por contenido. Puede no funcionar (p. ej. si
la repetición relevante no está en la ventana corta), pero no tiene el
problema de circularidad.

**Decisión explícita (Valentín está durmiendo 12h): NO se arranca nada
nuevo.** Esto es territorio de diseño nuevo, fuera del probe mínimo acotado que
se había acordado. Queda documentado acá y en `ESTADO.md`; la dirección la
decide Valentín a su vuelta. Prerregistrado: si se hace, el objetivo es el
mismo que hoy — la puerta como detector de novedad, probada contra "repetida
escribe menos" — pero contra el estado auxiliar rápido, con verificación
independiente desde el día 1 (la de hoy cerró el hilo a tiempo; conviene
tenerla al inicio).

**Verificación y archivos.** `cargo test --release`: 83 passed. Archivos
nuevos: `examples/wprobe.rs`; envs `EVA_WRITE_ERROR`/`EVA_WRITE_ERROR_P` en
`src/model/clock.rs`, `wprobe` en `src/main.rs`/`src/train.rs`. Logs:
`logs/wprobe_*.log`.

---

## Parte 10 — Densidad de atención aislada de la distancia: NULO, la densidad no es la causa (16-08)

**El problema que motivó esta corrida (línea pendiente del 15-08).** La corrida
"densa" (`assoc_k2_dense`, acortando filler para meter más queries) logró que
la atención aprendiera el recall asociativo (99.85% en gap<=32, val) — pero
acortar el filler también acortó la distancia (gap medio ~45→~10 bytes).
Densidad y distancia quedaron mezcladas: no se podía saber cuál causó el
aprendizaje.

**La versión limpia (esta corrida, diseño del equipo).** Mismo generador,
mismo filler (4-40, el original), pero seq 512→2048. Eso sube la densidad
(ciclos clave-valor POR CONTEXTO: queries/bindings 9.93→42.09, ~4x) manteniendo
la MISMA estadística de distancia (gap medio 45.2→47.2, filler idéntico,
verificado contra una regeneración control del mismo seed). Menos ventanas para
el mismo volumen de bytes (1500→375, 768 KB igual).

**Receta exacta (la de siempre, reconstruida y confirmada desde el snapshot de
shell de Claudio):** `--dim 512 --ffn 1024 --blocks 5 --seq 2048 --val 0.1
--seed 7 --arch attn --gate-beta 0.0 --log 200 --epochs 2`. 14,451,712 params
(la pos-embedding crece con seq; 13,665,280 en seq512). GPU local (libre, no
choca con el run de Claudio que corre en martha).

**Resultado: NULO, igual que la corrida base seq512.** Atención en chance en
todos los buckets, incluso en TRAIN (no memoriza):

| bucket | seq512 base val | seq512 base train | seq2048 val | seq2048 train |
|---|---|---|---|---|
| gap<=32 | 4.55% (n=1320) | 3.43% (n=11536) | 2.01% (n=1346) | 4.01% (n=11749) |
| 32<gap<=128 | 3.24% | 3.31% | 1.39% | 3.85% |
| 128<gap<=256 | 7.37% (n=95, ruido) | 3.30% | 0.87% (n=115) | 4.03% |
| 256<gap<=512 | — | — | 0.00% (n=7) | 0.00% (n=37) |

Chance = 3.125% (32 símbolos de value). Val final: 5.1803 bpb vs 5.1628 (base) —
indistinguible. El nulo es robusto: la hipótesis de densidad no se sostiene.

**Lectura honesta.** La corrida densa aprendió por la DISTANCIA corta (gap
medio ~10), no por la densidad. Con distancia en ~45 y la densidad por contexto
cuadruplicada, la atención sigue en chance y ni siquiera memoriza. La celda que
NO quedó testeada (y no se puede con este generador) es "densidad por byte alta
a distancia igual": subir la densidad por byte obliga a acortar filler, y eso
acorta la distancia — están confundidas por construcción. Lo que se pudo aislar
(densidad por contexto) dio nulo.

**Caveat honesto:** la corrida seq2048 tuvo 4x menos pasos de optimizador (674
vs 2698) por el mismo volumen. Eso debilita el "nulo" en teoría, pero el hecho
de que siga en chance sobre TRAIN (0 memorización) es consistente con el nulo
base y hace poco probable que "más pasos" sea la explicación a esta escala. La
hipótesis de "más exposición" la está testeando por otra vía el propio Claudio:
run attn seq512 `--epochs 15` en martha (lanzado 16-08 05:55), que además
sirve como par directo.

**Archivos.** Data: `data/assoc_k2_seq2048_win375_seed42.dat` (768 KB, mismo
seed 42). Weights: `16m5b_assoc_k2_seq2048_attn.weights`. Logs:
`logs/train_assoc_k2_seq2048_attn.log`, `logs/eval_assoc_seq2048_attn.log`.
Regeneración control del gap (mismo seed, seq512 win375): gap mean 45.2 vs
47.2 seq2048, min 6 en ambos, queries/bindings 9.93 vs 42.09.
