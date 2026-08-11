# Para quien retoma esto sin haber estado en la conversación

Escrito para OpenCode y para cualquiera que agarre el proyecto en frío,
incluido yo dentro de un mes. Empezá por acá, después `DISENO.md`, después el
`README.md`.

## Qué es esto y qué NO es

Una LLM propia en Rust desde cero, cero dependencias, cero CUDA. **No es un
cerebro para EVA**: eso es otro proyecto (`marco`, que es un arnés que le
pregunta a un modelo ajeno y verifica lo que contesta). Acá se construye el
modelo en sí, y el objetivo es un **cambio de paradigma en cómo aprende**, con
eficiencia extrema de hardware. La consigna de Valentín: *"las LLM consumen al
pedo; tiene que poder entrenarse con algo mínimo"*.

## La regla que rige todo

**Nada se afirma sin medición contra la línea base, con validación held-out y
varias semillas.** No es formalismo: el marcador de la intuición en este
proyecto es **0 de 5**.

| se creyó que | resultó |
|---|---|
| el cuello era el shader de GPU | eran los flags de memoria (2.5x contra 1.13x) |
| el cuello era ClockMem | era el 0,8% del tiempo |
| la atención sin posiciones estaba en desventaja | dárselas la EMPEORÓ, 3 de 3 |
| aprender por sorpresa iba a rendir | empata pagando 28% más |
| el crédito local ahorraría mucha memoria | 14% a 4 bloques, y cuesta 3,6% de calidad |

Cinco veces seguidas la hipótesis razonable fue falsa. **La evidencia indirecta
sirve para elegir qué medir, nunca para concluir.** Si vas a tocar algo, medí
antes y medí después, y publicá los dos números aunque el segundo sea peor.

## El estado, en una tabla

| pieza | estado |
|---|---|
| núcleo tensor + autograd + gradchecks | anda |
| arquitectura EvaClock | anda, y **le gana a la atención** con los mismos params |
| backend Vulkan (matmul) | anda, `EVA_GPU=1`, umbral medido |
| pool de hilos persistente | anda |
| validación held-out + baseline de atención | anda |
| inferencia con estado | anda, **37 → 750 tok/s** |
| aprender por sorpresa | **descartado por medición** |
| crédito local | construido, **pierde a esta escala** |
| tabla de k-gramas (hipótesis central) | construido, **midiéndose** |

## Los números que importan, y de qué máquina son

**Todo está medido en una GTX 1650 + Intel Comet Lake-H de 8 núcleos.** La
placa que importa es la RX 580 de otra máquina (Martha) y **ahí no se midió
nada todavía**. No heredes estos números: el umbral de GPU de `math.rs` (4M de
trabajo) es de NVIDIA y en GCN hay que rehacerlo.

    ClockMem vs atención (mismos params, 3 semillas, sin solapar)
        ClockMem        2.630 bits/byte   319 s
        atención        2.688             339 s
        atención + pos  2.772             345 s

    matmul en GPU     512³  1.10 ms   |  1024³  5.98 ms
    generación        750 tok/s a 1024 tokens

## Cómo correr las cosas

```sh
cargo test --release                  # 33 tests
cargo test --release -- --ignored     # 2 más, piden GPU con Vulkan

cargo run --release -- train --data data/prosa250.txt --arch clock \
    --epochs 1 --dim 256 --ffn 512 --blocks 4 --seq 64 --val 0.1 --seed 7

cargo run --release -- gpu --m 1024 --k 1024 --n 1024 --iters 30
```

Interruptores: `EVA_GPU=1` (matmul por GPU), `EVA_GPU_PROFILE=1` (parte el
tiempo en subir/calcular/bajar), `EVA_PROFILE=1` (en qué se va el
entrenamiento), `EVA_SERIAL=1` (saltea el pool de hilos, para aislar bugs).
Banderas: `--arch clock|attn`, `--val`, `--surprise`, `--local`.

## Las trampas que ya nos costaron caro

Todas están documentadas en el encabezado de su archivo. Juntas, porque tienen
**la misma forma: fallan en silencio.**

1. **`pool.rs`** — esperar en un condvar con `if` en vez de `while`, más
   `notify_all` al terminar: los workers se despertaban entre sí y
   **re-ejecutaban el trabajo viejo**. Medido: 23 a 31 veces cada pieza por UNA
   llamada. Invisible mientras los buffers vivieran; SIGSEGV apenas el que
   llamaba liberaba los suyos.
2. **`gpu/mod.rs`** — el descriptor pool tenía lugar para un set y se pedía uno
   por llamada: **la segunda llamada de cualquier proceso fallaba**. Nadie lo
   vio porque `eva gpu` multiplicaba una sola vez.
3. **`gpu/mod.rs`** — staging `HOST_VISIBLE|HOST_COHERENT` sin `HOST_CACHED` es
   memoria **sin caché**: leerla desde la CPU va a ~300 MB/s. Bajar el
   resultado se llevaba el **70%** del tiempo total. Un flag.
4. **`save.rs`** — `eps` se leía sin avanzar el cursor, así que `seq_len`
   releía esos bytes y volvía 925353388. Todo checkpoint cargado tenía la
   ventana rota, y como sólo recorta, no fallaba: generaba fuera de rango en
   silencio. Por eso el formato es EVAV1.
5. **Cortar un tensor con `Tensor::new` para pasarlo al grafo** — el corte no
   tiene nodo de autograd: el gradiente nunca vuelve y el parámetro entrena
   contra nada, **con el mismo aspecto que uno que aprende**. De ahí
   `ops::slice_rows`.

**Si vas a escribir un segundo camino para la misma matemática** (como
`stream.rs`, que hace inferencia de a un token), escribí PRIMERO el test que
exige que los dos den lo mismo. Los dos caminos se desincronizan solos y no
avisan.

## Dónde tocar según qué quieras hacer

- **Otra arquitectura de mezclador**: `src/model/` — agregá una variante a
  `Mixer` y una entrada en `Arch`. La comparación justa exige mismos
  parámetros, mismo bloque, mismas semillas.
- **Otra regla de aprendizaje**: `src/learn.rs`, trait `LearnGate`. Es
  intercambiable a propósito.
- **Otra operación con gradiente**: `ops.rs` el forward, `autograd.rs` el
  backward, y **su gradcheck en `tests.rs`**. Sin gradcheck no entra: un
  backward mal no falla, sólo aprende peor, y arruina cualquier comparación.
- **Rendimiento**: `math.rs` (matmul, AVX2), `pool.rs`, `gpu/`. Medí con
  `EVA_PROFILE=1` antes de elegir qué optimizar — dos veces el cuello no
  estaba donde parecía.

## Lo que está abierto ahora

1. **La hipótesis central**: ¿se puede separar lo que necesita *generalizar* de
   lo que sólo necesita *recordarse*? Se está midiendo con `src/recall.rs`:
   un modelo de N params contra uno de N/2 **más una tabla de k-gramas**. Si el
   chico alcanza al grande, cambia la dirección del proyecto.
2. **¿El costo del crédito local crece con la profundidad?** A 4 bloques cuesta
   3,6% de calidad y ahorra 14% de memoria: mal negocio. A 16 bloques el ahorro
   es 45%. Si el costo se mantiene, el intercambio se vuelve interesante.
3. **Medir todo en la RX 580.**
4. **Bajar las asignaciones del autograd**: ~70 s de 164 en un entrenamiento.
   Cada operación asigna un `Vec` por gradiente y el forward clona sus entradas.

## Una cosa sobre cómo trabajar acá

Valentín pide, con razón, que se distinga **lo que se sabe de lo que se está
completando**. El modo de falla de los que trabajamos en este repo no es
equivocarse: es **estar seguros mientras nos equivocamos**, y que el error
cueste una medición entera en descubrirse. Si escribís algo que no verificaste,
decilo en el mismo renglón.
