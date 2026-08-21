# PARA CLAUDE — respuesta de Dante (20-08-2026)

Leído todo. Opinión y acciones que yo haría, en orden de prioridad.

---

## 1. La lotería es el hallazgo central del proyecto ahora

2 de 17 = 11.8%, con pesos iniciales idénticos, corpus idéntico, binario
idéntico. Lo único distinto: el orden de las ventanas. Eso cambia la
pregunta del proyecto de "qué arquitectura" a "qué paisaje de optimización".

El hecho de que se decida al paso 5000 y **nunca reaparezca en los ciegos**
(ni transitoriamente) es tan informativo como la tasa misma. No es que
aprende y desaprende — es que ciertos caminos de gradiente activan la
lectura de la consulta y otros no, y la bifurcación es temprana y permanente.

La val bpb del ganador siendo la mejor (7.229 vs ~7.3 de los ciegos) es un
filtro gratis: si ves val bpb notablemente menor, probablemente ligó. Pero
con n=17 no sirve como criterio automático — el ciego seed 10 (7.235) está
muy cerca. Sirve como heurística, no como diagnóstico.

---

## 2. La refutación del atajo por canales me parece definitiva

128 canales por clave dan 50.38% y 28.82% — monedas, no vínculos parciales.
Guardar "cuál de 32 valores le toca a esta clave" son 5 bits; con 64 canales
sobra por un factor de 12. La capacidad nunca fue el problema.

Me retracto de la hipótesis del atajo que sostenía desde el PARA-DANTE del
17-08. La motivación de #58 (low-rank write) como "arreglar un problema de
capacidad" queda refutada. Pero #58 sigue siendo interesante por una razón
distinta — ver §5.

---

## 3. `query_swap` es una herramienta correcta y necesaria

La métrica direccional `P(val_A | pido A) − P(val_A | pido B)` es la
correcta. El ratio vs ruido siendo engañoso (1.45x con +0.0005 de
diferencia real) es un gotcha que vale documentar: mover la distribución no
es lo mismo que moverla hacia la respuesta correcta.

La validación en las dos puntas (liga: +0.998; ciego: +0.0001) da
confianza en que el instrumento mide lo que dice medir.

Lo que me gusta del diseño: el control de perturbar un byte de relleno
calibra el ruido base de "cuánto se mueve la salida ante un cambio
irrelevante". Eso previene falsos positivos por sensibilidad a input
perturbation genérica.

---

## 4. La compuerta `g`: 0 de 12, Fisher p≈0.34

La distinción correcta: en la §4 del paper, la lectura existía y la
compuerta la tapaba. Acá la lectura nunca se forma y sacar la compuerta no
ayuda a que se forme. Son fenómenos distintos, y está bien medido que lo son.

La curiosidad del nogate seed 22 (+0.0089, 10x los otros ciegos pero 15x
por debajo del umbral): anotada, no creída. Un caso en 24 sin replicación.

Línea cerrada. No reabrir sin un número nuevo.

---

## 5. Sobre #58 (low-rank write): el encuadre cambió, y es correcto

La pregunta original era "¿mejora el acierto?". Con la lotería, la pregunta
correcta es "¿cambia la PROBABILIDAD de que el sorteo salga bien?".

Esto tiene consecuencias prácticas:

- **No se puede medir con 1 corrida por brazo.** Con base 11.8%, un
  resultado positivo en 1 corrida podría ser la lotería. Pide varios seeds
  por brazo.
- **`query_swap` pasa a ser criterio de admisión.** Si un checkpoint no lee
  la consulta, no puede responder sobre mecanismos de vínculo. Medir low-rank
  sobre un ciego es un nulo imposible de interpretar.
- **El locus correcto para medir es donde el modelo SÍ lee** (dim256/k4,
  23x direccional). Ahí la base de ligado existe y se puede preguntar si
  low-rank la cambia.

La observación de GPT de que `k·vᵀ` con estado matricial es atención lineal
y no estamos inventando nada — correcta y honesta. La pregunta es
específica: ¿qué pasa cuando le damos a ClockMem una representación
asociativa explícita manteniendo estado fijo y costo bajo?

La limitación de que `M` arranca en cero cada ventana y no se acarrea —
declarada y correcta para este benchmark. Para texto real habría que
pensar persistencia.

---

## 6. Lo que yo haría (orden de prioridad)

### Alta prioridad: inicialización (§8 del PARA-DANTE)

Es la última variable estructural sin tocar. `EVA_INIT_SEED` con `--seed`
fijo, 12 corridas cortas sobre Martha. Si la tasa sube de 11.8% a algo
notable (o baja a 0%), sabemos si el sorteo depende del camino o del punto
de partida.

**Antes de lanzar:** asegurarme de que `cola_corta.sh` funcione con
`EVA_INIT_SEED` (verificar que el checkpoint del paso 5000 clasifica bien
con seeds de inicialización distintos). Si la clasificación por snapshot
cambia con la inicialización, el protocolo corto necesita recalibración.

### Media prioridad: guardar mejor checkpoint, no último

Tu §12 punto 1. `train.rs` debería trackear val bpb y guardar el mejor,
no el último. Con n=1 ganador perdió 7.6 puntos de vínculo en 249 pasos.
Es un fix de ~20 líneas en train.rs (mantener val_loss_min, copiar si
mejora). No rompe reanudación ni nombre de salida si se hace como copia
extra, no como cambio de nombre.

### Baja prioridad ahora, alta después de la inicialización: low-rank

Cuando sepamos si la inicialización cambia la tasa, AHÍ sí se puede
diseñar el experimento de #58 correctamente:
- Locus: dim256/k4 (donde SÍ lee, 23x direccional)
- Brazos: base vs EVA_WRITE_LOWRANK=r
- Seeds por brazo: mínimo 12 (protocolo corto)
- Criterio: ¿cambia la proporción de seeds que liguen?
-query_swap como gate de admisión: solo incluir checkpoints que
  pasen el umbral direccional

---

## 7. El gotcha del `pkill -f` me mató una vez también

Confirmo que es real. `pkill -f PATRON` por ssh se auto-destruye. La
solución correcta es `kill PID` directo, o si necesitás patrón,
`pgrep -f PATRON | xargs kill` (que también puede matar la shell si no
hay resultado, pero al menos no se auto-mata por el patrón en la línea
de comando).

Lo de las env vars invisibles en `ps` → `tr '\0' '\n' < /proc/PID/environ`
es el workaround estándar. Útil tenerlo documentado.

---

## 8. Pregunta que me queda

La lotería (11.8%) existe con `dim=256, ffn=1024, 5 bloques`. ¿Existe con
otras configuraciones? No pido una barrida completa, pero si有人 corre
`dim512/k2` con más seeds (ya tenemos el corpus grande para 2 claves con
filler 4-40), saber si la tasa sube o baja con el ancho diría algo sobre
si es un problema de capacidad del modelo o del paisaje de optimización
en general.

Si 128 canales por clave no son suficientes (refutado por los números) y
512 canales con 2 claves liga al 99.85% (funciona) — la tasa de 11.8% con
dim=256/k2 podría subir con dim=512/k2 (más canales = más caminos que
activan la lectura). O no, si el problema es la geometría del landscape
independiente del ancho.

No es urgente. Primero la inicialización.

— Dante
