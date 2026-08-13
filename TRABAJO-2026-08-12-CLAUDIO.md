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
