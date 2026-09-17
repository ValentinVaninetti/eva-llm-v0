# PARA CLAUDE — leé esto primero al volver

Handoff de Claude para Claude. No está fechado a propósito: es el documento
vivo de "dónde estamos". Actualizarlo al cerrar cada ronda.

**Última actualización: 2026-08-24, debate filosófico memoria/comprensión; dos caminos propuestos (--persist y T=1.3).**

---

## 1. Quiénes somos y cómo trabajamos

**Valentín** es el dueño del proyecto y el único humano. QA engineer, de
Necochea, habla rioplatense. Vos sos "Claudio" para él.

**Cómo hablarle, esto importa:**
- En castellano rioplatense, directo, sin vueltas. Él putea y le gusta que
  se le hable de igual a igual.
- **Nada de adular.** Si le decís algo lindo sin sustento te lo va a marcar
  ("no me tires flores"). Elogio solo cuando hay evidencia concreta detrás.
- Cuando pide "en criollo", quiere analogías cotidianas, no jerga. Tenemos
  una que ya funciona y conviene reusar: **la oficina de 512 escritorios**
  (cada canal es un escritorio con su pizarra y su perilla de olvido).
  Está desarrollada en `~/Escritorio/EXPLICACION-EVALLM-PARA-VALENTIN.md`.
- **Decile cuando te equivocaste, explícitamente.** Lo valora más que
  cualquier acierto. En esta ronda me retracté de una afirmación fuerte y
  eso mejoró la conversación, no la empeoró.
- Él no sabe la técnica a fondo y lo dice sin drama, pero **tiene olfato de
  QA muy bueno**: pregunta "¿esto no estará confundido con aquello?" y
  suele tener razón. Tomale en serio esas objeciones, no las despaches.

**El equipo:** Dante (OpenCode, implementa y corre; tiene acceso al repo),
GPT (propone hipótesis y diseño experimental; **no** tiene acceso al repo,
hay que pasarle todo autocontenido), y vos. Valentín hace de puente: le
pedís "dame mensaje para X" y él lo copia y pega.

## 2. Reglas de la casa (no negociables, salieron de cagadas propias)

1. **Medir antes de construir.** La intuición eligió mal el cuello tres
   veces en un día, cada vez que no se midió primero.
2. **"Código que toca X" ≠ "X funciona, medido".** Son dos afirmaciones
   distintas y no se mezclan en los documentos.
3. **Una variable por experimento.** Y si hay un confundido inevitable,
   **declararlo por escrito**, y preferir el que juega EN CONTRA de tu
   propia hipótesis (así un resultado positivo es más fuerte, no más
   débil).
4. **Un resultado negativo cierra una línea** hasta que un número nuevo la
   reabra. No se re-litiga con corazonadas.
5. **Avisar antes de tocar `src/` o un archivo que otro tenga abierto.**
6. **Verificá los números de los demás contra los logs crudos** antes de
   aceptarlos. No por desconfianza: es la regla, y ya salvó errores reales.
   Lo mismo con los tuyos.

## 3. Dónde estamos, exactamente

La investigación viene de una cadena larga sobre memoria en ClockMem. El
estado al cierre de esta ronda:

**Lo que se descubrió (y es lo más limpio del proyecto):** el fracaso de
ClockMem en recall asociativo es una **falla de vínculo**, no de memoria.

```
  qué predice el modelo    Clock 2   Clock 8   Attn 2   Attn 8
  el valor correcto         99,65%    23,42%   47,23%    2,89%
  el valor de OTRA clave     0,00%    76,49%   42,45%   16,74%
  --- dentro del conjunto   99,65%    99,91%   89,68%   19,63%
```

Con 2 claves ClockMem liga perfecto (cero confusiones en 5.967 consultas).
Con 8 claves **sabe cuáles son los 8 candidatos (99,91%) pero elige el de la
clave equivocada el 76% de las veces**. No olvidó nada, no le falta
estructura: falla específicamente en atar clave con valor, justo donde una
partición de 512 canales entre claves se queda sin lugar.

**Por qué eso importa:** la escritura de ClockMem es elemento a elemento
(`cur[c] += β·k[t,c]·v[t,c]`, sin término cruzado entre canales), así que
solo permite *dedicar canales por clave*, no *guardar un vínculo*. **Toda la
investigación previa atacó la LECTURA** (ventana aprendible, init
estructurado, oracle, inyección, R1c, R2 con piso, compuerta por error) y
ninguna podía arreglar algo que se destruye al escribir.

> ## ⚠️⚠️⚠️ DÓNDE VIVE EL VÍNCULO (20-08): EN 4 CANALES DE 256
>
> Ablación de lectura por canal (`EVA_READ_MASK`) sobre el ganador:
> apagar **8 canales cualesquiera** fuera del grupo no hace nada (+0,88);
> apagar **los canales 0-3** destruye la capacidad entera (+0,0004). El
> canal 0 solo se lleva el 75%.
>
> Son los pocos que conservan memoria: las alfas aprendidas dan **251 de 256
> canales (98%) en la banda rápida** (<10 tokens), 4 medios y 1 muy lento.
> Los 251 rápidos NO PUEDEN sostener un vínculo a través de un hueco de 45
> bytes. Conecta directo con la §3 (el reloj se aplana solo).
>
> **Y los ciegos tienen las MISMAS alfas y usan los MISMOS canales**
> (apagar 0-3 los lleva del 43,59% al 3,85%, o sea al azar). Mismo sustrato,
> dos circuitos encima: el ciego escribe ahí *el conjunto de candidatos*, el
> ganador escribe *el vínculo*.
>
> **Esto mata toda explicación por capacidad**: la capacidad efectiva son 5
> canales en TODOS los modelos. Por eso el ancho no predecía nada.
>
> La pregunta pasa a ser: **¿por qué el gradiente escribe el conjunto de
> candidatos 7 de cada 8 veces y el vínculo sólo 1?**
> Detalle en `logs/PARA-DANTE-2026-08-20-INGENIERIA-INVERSA.md`.

> ## ⚠️⚠️ EL RESULTADO CENTRAL (19-08): ES UNA LOTERÍA DE OPTIMIZACIÓN
>
> Cinco corridas de `dim256/k2`. **Pesos iniciales idénticos** (`--seed` no
> toca la init, ver gotchas), mismo corpus, mismo binario. Lo único distinto
> es **el orden en que ve las mismas ventanas**:
>
> ```
>              val bpb    lee la consulta      acierto
>   seed 7      7,303        no  (+0,000)       ~50%  moneda
>   seed 8      7,345        no  (+0,000)        41,81%
>   seed 9      7,298        no  (+0,000)        43,91%
>   seed 10     7,235        SÍ  (+0,985)        99,77%
>   seed 11     7,309        no  (−0,001)        45,75%
> ```
>
> **1 de 5.** Y de ahí:
>
> 1. **El objetivo SÍ premia leer la consulta**: el ganador tiene la mejor
>    bpb por 0,063 sin solapamiento. **La val bpb sola ya delata quién ligó**
>    — filtro gratis desde el log, sin correr el diagnóstico.
> 2. **Se decide en el primer cuarto.** Al paso 5000: ganador +0,197, ciegos
>    +0,0002/+0,0000/+0,0002. En los ciegos NUNCA aparece, ni transitoria-
>    mente. Esto habilita un **protocolo corto** (cortar en 5000, 4x más
>    barato) verificado 4 de 4.
> 3. **La escritura elemento a elemento SÍ PUEDE LIGAR**, al 99,77%. Esto
>    **invalida el motivo declarado de #58** tal como estaba escrito. Puede;
>    casi nunca lo encuentra. El problema es de OPTIMIZACIÓN, no de
>    representación. Hay que reescribir esa motivación en el paper.
>
> **Ojo con el checkpoint final:** el del ganador es PEOR que el del paso
> 20000 (92,18% contra 99,77%; la val periódica también sube al final).
> Guardamos siempre el último y el último no es el mejor.
>
> **Consecuencia para cualquier A/B de arquitectura:** con base 1/5, una
> corrida por brazo mide la lotería, no la arquitectura. `query_swap` pasa a
> ser **criterio de admisión**: si un brazo no lee la consulta, no puede
> contestar nada sobre mecanismos de vínculo. Y #58 hay que medirlo como
> "¿cambia la PROBABILIDAD del sorteo?", con varios seeds por brazo.
>
> **CERRADO 20-08: la compuerta `g` NO es el culpable.** 12 pares apareados,
> `EVA_NO_GATE` dio **0 de 12** contra una base de **2 de 17 (11,8%)**.
> Criterio prefijado 5/12; Fisher p~0,34. No contradice la §4 (allí la
> lectura existía y la compuerta la tapaba; acá nunca se forma).
>
> **Mapa de ejes:** canales/clave REFUTADO · ancho no explica · nº de claves
> no explica · orden de datos ES LOTERÍA (11,8%) · compuerta DESCARTADA ·
> **inicialización NUNCA PROBADA** (imposible hasta el 19-08).
>
> **Lo único estructural sin tocar es la inicialización.** `EVA_INIT_SEED`
> variando pesos con `--seed` FIJO: 12 corridas cortas, ~8h30. Martha libre.

> ## LO MÁS IMPORTANTE DE LA RONDA DEL 18/19-08
>
> **La falla no es "no puede guardar vínculos": es que en varias
> configuraciones el modelo NI SIQUIERA LEE LA CONSULTA.** Medido con el
> diagnóstico `query_swap` (idea de GPT): se toma una consulta real y se
> vuelve a correr la MISMA ventana cambiando sólo a cuál clave se pregunta.
>
> ```
>               ancho claves correcto  ratio CLAVE/RELLENO  ventaja vínculo
>   dim512/k2    512    2     99,65%       696,94x           2,00x (tope 2x)
>   dim256/k4    256    4     48,56%        23,01x           2,10x (tope 4x)
>   dim512/k8    512    8     23,42%         6,65x           1,88x (tope 8x)
>   dim512/k4    512    4     28,82%         0,50x           1,16x ~= azar
>   dim256/k2    256    2     50,38%         0,28x           1,08x ~= azar
> ```
>
> El ratio es contra un control: cuánto se mueve la salida al perturbar un
> byte de RELLENO. `dim512/k4` con 0,50x mueve la salida **menos que el
> ruido** — le preguntás por otra clave y contesta lo mismo. Su 28,82% es 1/4
> de los candidatos, o sea una moneda de cuatro caras.
>
> **Leer la consulta es la variable que manda: ordenadas por ratio, las
> precisiones caen solas.** Y no lo predice ni el ancho ni la cantidad de
> claves (512 tiene el mejor Y uno de los peores).
>
> **La cuenta de "canales por clave" está REFUTADA** (dos celdas con 128
> canales/clave dan 50,38% y 28,82%), y con refuerzo de conteo: guardar cuál
> de 32 valores son 5 bits, y 64 canales sobran por un factor enorme. La
> capacidad nunca fue el límite.
>
> **Esto pone #58 en pausa, y con razón:** la vía de bajo rango lee por
> `qr = Σ_c q[t,c]·pq[c,i]`. Si `q` no lleva identidad de clave, hereda el
> problema, y medirla ahí daría un nulo imposible de interpretar.
>
> Corriendo ahora (Martha, PID 149320, termina ~13:30 del 19): 4 seeds de
> `dim256/k2`. Ver `logs/PARA-DANTE-2026-08-18.md`, apéndice del 19-08.

**#58 YA ESTÁ IMPLEMENTADO** (18-08, 02:15): la escritura de producto
externo de bajo rango vive en `ops::clockmem_lowrank_from`, detrás de
`EVA_WRITE_LOWRANK=r` (apagado por defecto). Gradcheck de los 11 tensores,
mutation-testeado, suite en 86. **Está implementado y verificado, NO está
medido** — no se sabe si sirve. El detalle entero, con las tres decisiones
de diseño y sus porqués, en `logs/PARA-DANTE-2026-08-18.md`.

**Antes de medirlo hay un control corriendo**, y es el que decide si #58
tiene sentido: toda la evidencia varió *la cantidad de claves*, y `dim`
nunca se tocó. Si el reparto de canales es la variable que manda,
`dim=1024/k=8` tiene que parecerse a `dim=512/k=4`. Además es el control que
#58 necesita igual (agrega parámetros; sin la curva de "parámetros solos" no
se puede atribuir una mejora al producto externo).

**Lo que NO hay que olvidar (la salvedad grande):** todo esto es un
**benchmark sintético de 2 a 8 claves**. En texto real (el benchmark de
entidades sobre el Quijote) **dio nulo**: ningún mecanismo recuperó nada
mejor que empezar de cero. El puente sintético → lenguaje real sigue sin
demostrarse. No lo vendas como resuelto.

## 4. Qué leer, en este orden

1. **`PAPER-DRAFT.md`, secciones 4.7 a 4.11** — el frente actual, con todos
   los números verificados y las salvedades. §4.10 tiene una nota de
   corrección arriba: sus números quedaron superados por §4.11 (se conserva
   a propósito, no se borra lo que se creyó).
2. **`logs/PARA-DANTE-2026-08-20.md`** — **la puesta al día completa del
   18 al 20**: los cambios de `src/`, las dos correcciones al paper, el
   diagnóstico `query_swap`, la lotería de optimización, la compuerta
   descartada, el mapa de ejes y los gotchas nuevos. Si volvés después de
   un corte, leé ése.
3. **`logs/PARA-DANTE-2026-08-17.md`** — el detalle operativo de la ronda
   anterior, secciones 7 a 10. La 10 es el cierre.
4. **`logs/PARA-DANTE-2026-08-18.md`** — bitácora cronológica cruda del 18
   y 19, con apéndices apilados. Detalle, no puesta al día.
3. **`TRABAJO-2026-08-15.md`** — el lado de Dante (R1/R2, la compuerta con
   piso, la escritura por error).
4. **`PARA-OPENCODE.md` partes 6 y 7** — el handoff canónico de Dante.
5. **`OVERVIEW.md`** — el panorama general si necesitás contexto de arriba.

Para hablar con Valentín en criollo:
`~/Escritorio/EXPLICACION-EVALLM-PARA-VALENTIN.md`.

## 5. Las máquinas

**Cristina** (local, GTX 1650, Intel 8 hilos): **limitada térmicamente**. Con
carga de entrenamiento se clava en 96°C y la frecuencia cae de 4.500 a 3.515
MHz. Al soltarla baja a 57°C en 25 segundos — o sea que ventiladores y
disipador andan bien; **lo gastado es la pasta térmica** (5 años sin
cambiar). Valentín la iba a cambiar; si ya lo hizo, **medir el antes/después
con la línea base de arriba**. Tiene `btop`, `s-tui` y `nvtop` instalados.
Los ventiladores marcan 0 RPM: es una limitación del driver `hp`, **no** es
que no giren (Valentín ya me corrigió eso una vez).

**Martha** (`ssh martha`, RX 580 + Ryzen 5 2600, 12 hilos, headless): la
máquina de trabajo. ~23% más lenta (1,03 contra 1,27 pasos/s en la receta
16m5b/seq512) pero **estable de principio a fin** y con 64°C de margen
térmico. Binario en `~/evallm_check/eva_llm_v0_bin_r2` (hay que copiarlo con
`scp` tras recompilar; Martha no tiene cargo ni gestor de paquetes
funcional). Para leerle la temperatura del CPU: `modprobe k10temp` (no
sobrevive a un reinicio).

**Decisión acordada:** las corridas **comparables** (las que van una al lado
de la otra en una tabla) van **todas a Martha, en serie**. Ya nos mordió una
vez entrenar dos condiciones en máquinas distintas: costó un 2×2 cruzado
entero descartar el confundido. Cristina para lo que **no** se compara:
evaluaciones, sondas, análisis.

## 6. Gotchas prácticos que ya me costaron tiempo

- **`--seed` NO varía la inicialización** (era la constante `0xE7A1` en
  `EvaModel::new`); mueve el orden de las ventanas y la generación de
  muestras. Toda corrida histórica descrita como "otra semilla" partió de
  pesos idénticos. Desde el 19-08 hay `EVA_INIT_SEED` para variarla de
  verdad (sin la variable, comportamiento idéntico al histórico).
- **Los env vars hay que setearlos también al evaluar**, no solo al
  entrenar (`EVA_READ_WIN`, `EVA_READ_DF_N`, `EVA_READ_R2`,
  `EVA_READ_R2_CHANNEL`, `EVA_ALPHA_TEMP`, `EVA_SSM_ALPHA`...). Sin ellos el
  checkpoint se carga como modelo base y da un resultado sin sentido.
- **`pgrep -f PATRON` dentro de un bucle de espera se encuentra a sí mismo**
  (el patrón está en la línea de comando de la shell que espera) → el bucle
  nunca termina. Dejé cuatro colgados 20 horas por esto. **Esperá por PID**,
  no por patrón.
- **`pkill -f PATRON` por ssh es peor: se mata a sí mismo.** El patrón viaja
  en la línea de comando de la shell remota. Me cortó la sesión en vivo
  (18-08). Matá por PID, y si el script es largo escribilo local y mandalo
  con `scp` en vez de meterlo por heredoc.
- **La validación con muchas ventanas tarda muchísimo** y desde afuera
  parece que el proceso se colgó: con `--val 0.1` sobre el corpus de 22.5k
  son 2.250 ventanas. Si estás haciendo humo, bajá `--val`.
- **Compilar un ejemplo no toca el binario principal**:
  `cargo build --release --example X` deja `target/release/eva_llm_v0`
  intacto — verificalo por timestamp si hay corridas encoladas usándolo.
- `examples/eval_associative.rs` acepta un **tercer argumento opcional**:
  máximo de ventanas por sección. Evaluar 22.5k ventanas completas tarda más
  de una hora; con 300 ya sobra muestra.
- Suite de tests: `cargo test --release` (estaba en 83 al cierre).
- Los checkpoints intermedios se guardan cada `log*10` pasos → **se puede
  evaluar una corrida a mitad de camino** sin esperar a que termine. Muy
  útil, lo usé para decidir cortar.

## 7. Tareas vivas

**CORRIENDO EN MARTHA** (18-08 01:45, PID 122674, termina ~07:15):
`dim=512, k=4, corpus grande` — el barrido de canales por clave. Log
`train_assoc_k4ctl_huge_clock.log`, temperaturas en `temps_k4ctl.log`.
Después va `dim=1024, k=8` (~22 h). Comando y razonamiento completos en
`logs/PARA-DANTE-2026-08-18.md`.

- **#58** — ~~implementar~~ **hecho y verificado; falta MEDIRLO.** Va después
  del barrido de `dim`, porque ése es su control de parámetros.
- **#31** — correr `eva gpu` en la RX 580; los números del README siguen
  siendo de la GTX 1650.
- **#32** — bajar las asignaciones del autograd (~70 s de 164). La
  optimización grande pendiente.
- **#36** — ¿el costo del crédito local crece con la profundidad?
- **#28** — reserva DHCP para Martha en el router (ya se le movió la IP una
  vez, a `.184`; si se mueve de nuevo se rompen los scripts).

---

## Conversación 24-08: Filosofía, memoria sin entendimiento, y giro de rumbo

### Contexto

Claude mandó el CONSEJO-2026-08-24.md (cierre de ronda, 7 partes, 9
preguntas). Valentín mandó a GPT el documento. GPT respondió con un plan
operativo. Dante verificó los números (regla 6). Después arrancó un debate
filosófico entre Valentín y Dante, con aportes de Claude.

### Verificación de números (Dante)

Todos los números verificables contra logs crudos son exactos:
- Lottery 5 seeds: bpb idénticos al 4to decimal ✓
- Width sweep (dim256/k2, dim256/k4, dim512/k2, dim512/k4): ✓
- Quijote 8 seeds: bpb ✓
- Entidades 8 seeds: Δlnp las 32 celdas ✓
- Binary verification r3 vs r2: ✓

No verificable: ablación de compuerta (BASE 2/17, NOGATE 0/12) — no hay
logs de query_swap ni eval_associative para los checkpoints cortos en
Martha. Pendiente de auditoría, no bloqueante.

### La idea de Valentín (núcleo del debate)

"La memoria sin entendimiento es un buffer, no una memoria."

Valentín propuso: en vez de intentar generar memoria por relación
(CLAVE→VALOR como tabla SQL), hay que lograr que el modelo primero tenga
comprensión, y recién después utilizar memoria persistente para ver si
aprovecha esa capa de comprensión.

Traducido a arquitectura: la comprensión vive en los pesos (memoria
semántica consolidada); el estado de ClockMem es memoria de trabajo.
Estuvimos midiendo la memoria de trabajo de un modelo que no tiene nada
consolidado que sostener.

### Punto clave de Valentín sobre la compuerta de escritura

ClockMem escribe TODO al estado sin preguntar si importa:
`cur[c] = α·cur[c] + β·k·v`. No filtra. Un buffer.

La propuesta: una compuerta de ESCRITURA que dependa del contenido,
para que el modelo decida QUÉ guardar:
```
should_write = σ(Wwrite · x)
cur[c] = α[c]·cur[c] + should_write · β·k[t,c]·v[t,c]
```
Eso es O(S·D), no O(S²). Y la compuerta de escritura ES la capa de
comprensión: para decidir si algo importa, hay que procesar el contenido.

### Respuesta de Dante (refutaciones)

1. **La regla de 20 tokens/param no aplica así**: es de transformers
   grandes para generalización. Un modelo chico puede memorizar con pocos
   tokens. El problema es optimización, no capacidad.

2. **Separares comprensión y memoria es hipótesis, no hecho**: un bebé no
   primero entiende y después recuerda. Son co-evolutivas. Entrenar sin
   memoria y después agregarla pone al modelo fuera de distribución.

3. **dim=48 puede ser demasiado chico**: si el umbral mínimo para
   comprensión de frase es dim=64 o dim=128, dim=48 nunca lo muestra.

4. **--persist no es solo prender un flag**: cambia orden de ventanas,
   propagación del estado, distribución de entrenamiento. Es otro régimen.

5. **La hipótesis original sigue viva**: el experimento B (N vs N/2 +
   búsqueda) la testea, no la da por muerta.

### El hallazgo del día (Claude)

La cantidad de canales con memoria es siempre ~2% de dim, en cualquier
corpus. No se gana memoria poniendo más ancho: se gana el 2% de lo que
pongás. Eso es la firma de la trampa de saturación de la §3.

Espectro de alfas medido:
```
dim=512 QUIJOTE   501 rápidos (98%)  10 medios  1 muy lento  → 11 con memoria
dim=512 SINTÉTICO 502 rápidos (98%)   9 medios  1 muy lento  → 10
dim=48  QUIJOTE    47 rápidos (98%)   0 medios  1 muy lento  →  1
```

La palanca NO es dim: es arreglar la trampa de saturación.

### Confirmación de Dante (dim=48)

Claude entrenó dim=48 en el Quijote (170 s, 143K params, 2.182 bpb):
- dim=48: fonotáctica correcta pero palabras inventadas ("legras",
  "desberlla", "padrerla"). No tiene léxico.
- dim=512: dice "Sancho", "el rucio", "cautiva", "encomendar".

dim=48 no existe como peldaño. No se puede preguntar por comprensión a
algo que todavía no tiene palabras.

### La propuesta de Claude (T=1.3)

La §3 del paper tiene un candidato archivado: temperatura T=1.3 en la
inicialización de alfas. Se descartó midiendo con bpb global, que es
ciego a efectos de memoria (0.006 bpb = 7.6 puntos de vínculo). Ahora
tenemos Δlnp con n=8. Volver a medir T=1.3 contra esa base es barato,
está motivado mecánicamente, y sería la primera intervención (no otro
diagnóstico).

### Los dos caminos sobre la mesa

1. **--persist desde el arranque** (idea de Valentín/Dante): entrenar con
   texto continuo, estado propagado. Los pesos aprenden a usar memoria
   mientras se forman. Prueba si el mecanismo puede aprender con memoria
   desde el vamos.

2. **T=1.3** (idea de Claude): atacar la trampa de saturación que limita
   el 2% de canales con memoria. Barato, una constante.

### Pendientes de Dante

- Verificar ablación de compuerta (BASE 2/17, NOGATE 0/12) — compilar
  eval_associative en Martha, correr sobre 24 checkpoints. No bloqueante.
- §4.7 del PAPER-DRAFT dice espectro ≈34% rápido / ≈39% medio / ≈26%
  lento para Quijote T=1. Claude mide 98/2/0. Alguna de las dos está
  mal o las bandas se definen distinto.
