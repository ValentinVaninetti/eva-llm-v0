# Trabajo — Claudio, ronda 2 (2026-08-12)

Archivo propio para no repetir el problema de orden de la ronda 1 (todos
escribiendo en el mismo `.md` al mismo tiempo). Tarea asignada: el Paso 0 de
mi propia propuesta ("diseño de la señal auxiliar", `TRABAJO-2026-08-12.md`)
-- la vara de `argmax(tabla) == verdad` restringida a `count ≥ c0`, sobre
`prosa250.txt`. No toqué `src/`.

## Qué hice

Extendí `examples/orden_cobertura.rs` (es mío, de la ronda 1) con
`paso_0_vara_por_count`: para cada posición de **validación** (no train --
es la condición real en la que tendría que funcionar cualquier mecanismo
nuevo, generalizando a texto que la tabla no memorizó), busca
`tabla.lookup_detail(ctx, 256)` -- ya pública, la agregó Dante hoy, no hizo
falta tocar `recall.rs` -- y compara `argmax(q)` contra el byte real, por
umbral de `count`.

## Un bug propio, encontrado antes de reportar

La primera corrida dio **5.4-5.9%** de acierto -- un número tan bajo que no
lo creí y me puse a revisar mi propio código antes de escribirlo acá.
Encontré el error: `TextDataset::window` arma `target[t] = ids[start+t+1]`
(el target va corrido uno respecto del input), y yo estaba usando
`pos_global = wi*seq+t` en vez de `wi*seq+t+1` -- el contexto que le pasaba
a la tabla le faltaba justo el byte más reciente, el más importante. Con esa
falta, la tabla predecía casi a ciegas y el número no significaba nada.

Corregido (`pos_global = wi*seq+t+1`), lo volví a correr.

## El número (corregido, sobre VALIDACIÓN)

```
=== PASO 0: vara de argmax(tabla) == verdad, sobre VALIDACIÓN ===
  count ≥ c0   n        acierto
         2    24807     53.3%
         5    17148     52.3%
        10    12173     51.6%
        50     3596     44.1%
```

## Lectura

**La vara real es ~50-53%, no el ~97% que todos -- yo incluido -- veníamos
asumiendo.** La confusión de fondo: la cobertura (¿la tabla tiene ALGO que
decir?, ~99.7%) y la exactitud (¿lo que dice es lo que realmente sigue?,
~53%) son dos números completamente distintos, y hasta ahora sólo se había
medido el primero. En texto que la tabla no memorizó, tener un contexto de
8 bytes que ya apareció en train no significa que esa vez haya continuado
igual -- la mayoría de las veces el CONTEXTO se repite (estructura,
puntuación, frases de documentación) pero el CONTENIDO específico que sigue
no.

**Dato menor pero real:** el acierto BAJA levemente con `count` más alto
(53.3% → 44.1%), al revés de la intuición "más repetido = más confiable".
Un contexto visto 50+ veces en train no necesariamente tiene una
continuación dominante -- puede estar repartida entre varias, y en val
gana la que no era. No lo investigo más por ahora, es la tarea de hoy, pero
queda anotado para quien siga esta línea.

**Para las dos candidatas que dependen de este número (mi cabecita de
acuerdo / la suma de logits de Séneca):** la vara bajó mucho, así que el
listón es más fácil de cruzar de lo que se pensaba -- pero también hay que
tener cuidado con lo simétrico: si un mecanismo nuevo mide un AUROC alto
prediciendo "la tabla tiene razón" contra una base de sólo ~53%, ese AUROC
puede ser mucho más fácil de conseguir que contra un ~97% -- no confundir
"más fácil de ganar" con "más valioso". El desbalance de clases ya no es
tan extremo como se temía (no es 97/3, es más cerca de 53/47), lo cual en
realidad es una condición más sana para entrenar cualquier clasificador
encima.

No toqué `src/`. Código en `examples/orden_cobertura.rs`, corrida real, sin
promesas.

---

## Ronda 3 — Paso 1: la sonda detached (tarea, la pasa Dante)

Tu propio Paso 1 del diseño (sección en el central), ahora contra la vara YA
medida (~53%, no ~97%).

**Qué hacer** (en un `examples/` nuevo, NO toques `src/`):
1. Sobre `16m5b_seed7.weights`, una sola pasada de `forward_hidden` (público,
   `src/model/mod.rs`) sobre las ventanas de validación, con las MISMAS
   posiciones y labels del Paso 0 (`pos_global = wi*seq+t+1`).
2. Dataset de sonda: pares `(hidden.detach(), table_correct)` con
   `table_correct_i = (hit_i=1 ∧ count_i ≥ 2 ∧ argmax(q_i) == target_i)`.
   Entrenás SÓLO la cabecita `g = sigmoid(w·h + b)` (detached, cero gradiente
   al modelo).
3. **Nuevo respecto de tu Paso 1:** estratificá el AUROC por banda de count
   {2–4, 5–9, 10–49, 50+}. Esto ataca la anomalía de la ronda 2 (la vara BAJA
   con count alto: 53.3 → 44.1). Si el AUROC sube donde la vara baja, el gate
   por count es contraproducente y tu cabecita debería ignorarlo.
4. Split de la sonda 50/50 de las ventanas de val (entrenar en la mitad,
   AUROC en la otra mitad), mismo corpus/splits que tu Paso 0.

**El número que la mata:** AUROC ≈ 0.5–0.55 en la mitad de eval (≈ azar
contra una base de ~53%) → el modelo NO puede saber cuándo la tabla tiene
razón → la cabecita muere acá, sin tocar `train.rs` una vez. Si el AUROC
supera ~0.6, la cabecita sigue viva y pasamos al entrenamiento conjunto
(β chico, tu plan).

Compartí el número por banda, no la promesa.

## Bloqueado: la lib no compila, no es cosa mía

Escribí `examples/sonda_tabla.rs` completo (reusa `forward_hidden`,
`Recall::lookup_detail`, `ops::stake_loss` con span_len=1 y `StakeHead` tal
cual -- cero op nueva, cero cambio a `src/`) y no lo pude correr ni una vez
porque **la librería entera no compila**, con código que no es mío,
sin commitear, y en dos estados distintos en menos de una hora:

1. **Primer intento:** `src/tensor/ops.rs:545`, `let (s, v) = x.shape;` --
   `x.shape` es `Vec<usize>`, no se puede desestructurar así en una tupla.
   Parece parte de un `softmax_rows` nuevo (comentario: "útil para mezclar
   logits del modelo con logits de la tabla" -- pinta a la idea de suma de
   logits en desarrollo).
2. **Segundo intento** (ya arreglado el `(s, v)`): `src/mix.rs` llama a
   `ops::mixed_ce` y `ops::mixed_ce_count`, que ya no están en `ops.rs`.
   Alguien las sacó o les cambió nombre al meter la versión de logits y no
   actualizó `mix.rs`.

**No toqué nada de esto** -- ni el `(s, v)` de una línea que hubiera sido
trivial arreglar, ni `mix.rs`. Es código de otra persona en curso, y me
pidieron explícitamente no tocar `src/` para esta tarea. Documento acá para
que Dante lo vea al ponerse al día: el Paso 1 está listo para correr apenas
la lib vuelva a compilar, no hace falta que yo haga nada más hasta entonces.

---

## URGENTE antes de seguir implementando: esto ya existe, verificado

Valentín pidió que esto quede escrito ahora, no al cierre, porque afecta
directo lo que se está construyendo en este momento (`mix.rs`, `softmax_rows`,
la idea de "suma de logits"). Lo busqué y lo confirmé -- no es de memoria.

**Es kNN-LM** (Khandelwal, Levy, Zettlemoyer, Lewis -- Stanford/FAIR, ICLR
2020, *"Generalization through Memorization: Nearest Neighbor Language
Models"*), y pega en las DOS cosas más grandes de la semana, no en una:

1. **Nuestra hipótesis central (tabla + modelo chico ≈ modelo entero) es su
   resultado fundacional.** Cita: *"entrenar un modelo con 100M de tokens y
   usar kNN sobre un datastore de 3 mil millones puede superar a entrenar ese
   mismo modelo con los 3 mil millones enteros."* Es nuestro punto 5, con más
   ceros.
2. **Lo que Dante encontró hoy a los golpes -- que mezclar la tabla en el
   TARGET de entrenamiento mata el gradiente (`mixed_ce`, NEGATIVO MONÓTONO)
   -- es la razón exacta por la que kNN-LM nunca lo hace así.** Su fórmula es
   idéntica a la nuestra, término a término: `p = λ·p_LM + (1−λ)·p_kNN`. Pero
   la usan **sólo en tiempo de evaluación** -- nunca entra a la pérdida de
   entrenamiento. Y el λ se elige "a nivel de corpus, sobre un dev set" --
   exactamente el barrido sobre desarrollo que ya usa `cmd_recall`.

**Pasamos un día entero, con GPU real y el chip tocando 100°C, redescubriendo
empíricamente algo publicado en 2019.** Si alguien leía el diseño de ese
paper antes de escribir `mixed_ce`, ese camino se descartaba en diez minutos
de lectura, no en tres corridas de 15 minutos cada una.

**Lo que SÍ es distinto, para no ser injusto:** kNN-LM busca sobre embeddings
densos con FAISS, un índice aproximado sobre un datastore gigante. Nosotros
usamos coincidencia exacta de k-gramas -- más simple, más barato, y a esta
escala probablemente suficiente. La idea es la misma; la implementación es
más cruda, a propósito.

**Lo que esto pide, concreto, antes de seguir con `softmax_rows`/la suma de
logits:** buscar si ESA variante específica (inyectar `log(q)` en los logits
en vez de interpolar en probabilidad) también tiene precedente -- huele a
Product of Experts (Hinton, ~1999-2002), que tiene el mismo modo de falla en
el caso extremo (un experto muy seguro fuerza la certeza del conjunto,
gradiente cero igual, por otro camino matemático). No lo confirmé todavía --
lo dejo como advertencia, no como veredicto, para que alguien lo chequee
antes de invertir otra corrida de GPU.

Fuentes: [Generalization through Memorization: Nearest Neighbor Language
Models](https://www.alphaxiv.org/overview/1911.00172) · [Nearest Neighbor
Language Models (OpenReview)](https://openreview.net/pdf?id=HklBjCEKvH)

---

## Paso 1 corrido: la señal está, pero no hacía falta construir nada para tenerla

La lib volvió a compilar, corrí `examples/sonda_tabla.rs` completo. Primera
vez en todo el proyecto que una sonda "antes del hecho" encuentra algo real
-- y el resultado completo cambia la recomendación, no sólo la confirma.

### El número principal

Cabecita `g=sigmoid(w·h+b)` (misma forma que `StakeHead`, span_len=1),
DETACHED, entrenada 60 pasos full-batch sobre la mitad de las ventanas de
validación (12.316 ejemplos con hit de tabla), evaluada en la otra mitad
nunca vista (12.491 ejemplos):

```
vara (aciertos de la tabla en eval): 55.0%
AUROC pooled: 0.742  (n=12491)

banda      n       vara       AUROC
2-4        3896    56.9%    0.745
5-9        2446    55.1%    0.756
10-49      4480    56.5%    0.735
50+        1669    46.5%    0.730
```

**0.742 supera holgado el umbral que mata (0.55) y el que confirma (0.6),
consistente en las cuatro bandas de count.** Por primera vez en este
proyecto, el estado oculto SÍ anticipa algo -- no "voy a acertar yo" (eso
falló tres veces: stake antes-de-gastar, magnitud, congelamiento), pero sí
"este contexto es de los que la tabla domina". Mi hipótesis del diseño
original se sostiene.

### El control que cambia la recomendación

Agregué lo que yo mismo había marcado como riesgo en la propuesta original:
¿esto es señal nueva, o la cabecita sólo redescubre la confianza que el
modelo YA declara gratis (`p[argmax]`, cero parámetros, ya calculado en
cada forward)?

```
AUROC de p[argmax] SOLO (bet::classify_row, cero parámetros nuevos): 0.797
AUROC de la cabecita entrenada:                                      0.742
```

**`p[argmax]` solo le GANA a la cabecita entrenada.** No es un empate que
justifique construir igual -- es peor, y no cuesta nada.

### Veredicto real, más preciso que "AUROC > 0.6"

- **La hipótesis subyacente está confirmada:** el estado / la salida del
  modelo SÍ anticipan si la tabla tiene razón, antes de generar. Es la
  primera señal "antes del hecho" que sobrevive en todo el proyecto.
- **El mecanismo que propuse (cabecita entrenada, `dim+1` parámetros) NO
  hace falta.** Lo que iba a aprender ya está, gratis, en `p[argmax]` --
  que además es EXACTAMENTE la misma vara post-hoc del punto 8 (media
  p[argmax] ≈0.67 a nivel de tramo) aplicada ahora a nivel de token. No es
  una señal nueva -- es la misma vara de siempre, mostrando que también
  sirve para esto.
- **Recomendación concreta:** si se quiere un gate "¿confío en la tabla
  acá?", usar `p[argmax]` directo como umbral. Cero entrenamiento, cero
  parámetros nuevos, cero riesgo de interferir con la CE (ni siquiera hay
  CE de por medio). No construir la cabecita.

Esto no mata la línea -- la vuelve más barata de lo que yo mismo la había
diseñado. Código en `examples/sonda_tabla.rs`, corrida real sobre
`16m5b_seed7.weights`, sin tocar `src/`.
