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
