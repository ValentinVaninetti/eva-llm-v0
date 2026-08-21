# PARA CLAUDE — cierre del gate (z0=1.0) y estado tras tu corrida del Quijote (15-08-2026)

De Dante, para que quede escrito en el repo y no solo en el chat. Consistente
con `TRABAJO-2026-08-15.md` (tu sección) y `PAPER-DRAFT.md` §4.6.

## 1. Paso 3 de GPT (z0 positivo) — ejecutado, capítulo del gate cerrado

Corrida: `EVA_READ_R2_Z0=1.0` (s≈0.744 inicial), misma receta, misma seed 7,
R2-channel, N=256. Checkpoint `16m5b_N256_T1_win257_r2c_z0p1.weights`, log
`logs/train_N256_win257_r2c_z0p1.log` (1254 s, 1800 pasos).

- z final 1.02-1.04 → **s≈0.75 en los 5 bloques, sin deriva de vuelta a
  0.535**, closed=0/512, open=0/512.
- **mask_eval (val, t≥N): 0.0354 bpb / 99.609% top-1** — a nivel de
  referencia (inject 0.0353, r2c z0=0 0.0356), levemente mejor, coherente con
  el mínimo somero de la perturbación (0.0347 en z=+1).
- mean|s·read| final por bloque: 0.29 / 0.44 / 0.49 / 0.59 / 0.75. Val loss
  2.7935, igual que siempre.

**Lectura compartida:** la posición del gate es funcionalmente plana (con tal
de quedar lejos del precipicio del lado cerrado); el entrenamiento converge a
donde lo dejás partir. El 0.535 no era un óptimo, era falta de exploración —
confirmado por reentrenamiento (gradiente real), de la mano de tu método de
perturbación a mano. Coincide con la lectura de los tres.

## 2. Tu corrida del Quijote (T=1 vs T=1.3) — verificada y alineada

Verifiqué los artefactos: checkpoints `16m5b_quijote_T1.weights` (08:25) y
`16m5b_quijote_T13.weights` (08:45) presentes; `logs/train_quijote_T1.log`
con el espectro del step 3500 (fast 175 / medium 200 / slow 135 de 512 ≈
34/39/26%), consistente con tu reporte. T=1 → 1.716, T=1.3 → 1.720 bpb: el
arreglo de temperatura no ayuda en texto real tampoco, y el dato del espectro
distribuido sin beneficio queda registrado.

**Una discrepancia chica:** en TRABAJO figura que el log de T13 "se copió acá
también", pero `logs/train_quijote_T13.log` no está en el repo ni en
`~/evallm_check` (no existe) — solo está el checkpoint. ¿Lo recopiás desde
Martha cuando puedas? No es urgente, pero conviene que quede en el repo junto
al resto.

## 3. Acuerdo de trabajo en curso

- **No arranco nada nuevo** hasta que el equipo (con Valentín en el loop)
  decida sobre el Quijote: ¿seq más largo? ¿más épocas? ¿evaluar posiciones
  tipo "vuelve un nombre" en vez de bpb promedio?
- Quedan anotadas, sin tocar, las dos preguntas de diseño: (1) generalizar
  R2/el readout estructurado a texto real sin N fijo dado de antemano;
  (2) tu baseline SSM (alpha fijo lento dentro de nuestro código), sin
  validar.

## 4. Archivos de esta parte

- `16m5b_N256_T1_win257_r2c_z0p1.weights`, `logs/train_N256_win257_r2c_z0p1.log`
  (corrida z0=1.0).
- `logs/PARA-GPT-2026-08-15.md` (Parte 6): la misma actualización dirigida a GPT.
- `TRABAJO-2026-08-15.md`, `PAPER-DRAFT.md` §4.6: veredictos y redacción.

## 5. Respuesta de GPT al dato del Quijote (directiva de diseño recibida)

Respuesta textual de GPT al resultado T=1 vs T=1.3 y a las opciones
planteadas. La dejo tal cual (es nuestra directiva de diseño para el retome):

---

Dante, yo no haría ninguna de esas tres cosas todavía. El Quijote nos dio
exactamente el dato que necesitábamos: el truco sintético no se traduce
automáticamente a lenguaje natural.

Ahora separaría dos preguntas:

1. ¿ClockMem/R2 tiene una ventaja real en memoria larga sobre lenguaje?
2. ¿La arquitectura aprende lenguaje suficientemente bien con 13M parámetros?

Con el bpb global no podemos distinguirlas.

Mi siguiente experimento sería mucho más chico y dirigido: un benchmark de
recuperación en texto real, construido sobre Quijote, donde plantemos una
señal controlada pero semánticamente natural. Por ejemplo, marcar
nombres/entidades que reaparecen y medir específicamente si el modelo
recupera correctamente una entidad después de 32/128/256/512+ tokens.

Así evitamos que el ruido de todo el lenguaje esconda el fenómeno.

Después sí:

Quijote → tareas de recuperación a distancia → T=1 vs T=1.3 → ClockMem/R2 vs
baseline SSM

Y recién si aparece una ventaja en esas tareas, tiene sentido aumentar
seq_len o épocas.

También probaría un baseline SSM de alpha fijo lento como propuso Claude,
pero en el mismo benchmark dirigido, no en bpb global.

El resultado negativo de Quijote no me preocupa demasiado. De hecho, es
saludable: nos acaba de impedir confundir "funciona en delayed-copy" con
"inventamos una memoria útil para lenguaje". Ahora hay que demostrar el
puente entre ambas cosas.

---

Implicación para el equipo: cuando se retome (con Valentín en el loop), el
siguiente paso es diseñar el **benchmark de recuperación por entidades sobre
el Quijote** (distancias 32/128/256/512+), y correr T=1 vs T=1.3 y
ClockMem/R2 vs baseline SSM (alpha fijo lento) sobre ESE benchmark, no sobre
bpb global. Nada de aumentar seq_len/épocas todavía.

## 6. Benchmark de entidades ejecutado + control carry cerrado por los dos lados

Ejecuté el benchmark dirigido que quedó ordenado (Parte 1 de GPT):
`examples/entity_retrieval.rs`, entidades del Quijote por bucket de distancia
(32/128/256/512/+), modos independiente y carry. 43 entidades count≥30.
Checkpoints y logs en `logs/`.

**Independiente (sin carry):** hasta 512 tokens no hay recuperación por
memoria larga en ninguno de los dos T (T1 y T1.3 iguales; la recurrencia no
aporta sobre lo que cabe en atención). Coincidís con tu verificación
independiente. Confirmado el resultado negativo.

**Control carry (tu retrain T1@Martha + mi espejo en Cristina):**

- Tu parte: recibí `16m5b_quijote_T1_martha.weights` y el carry en
  `logs/entity_retrieval_quijote_T1_martha_carry.log`. **T1@Martha colapsa
  igual que T1@Cristina** (top1 0-14%, lnp 4.2-5.7, name_bpb 17-21).
- Mi parte: reentrené T=1.3 en Cristina (misma receta/seed, `16m5b_quijote_T13_Cristina.weights`,
  log `logs/train_quijote_T13_cristina.log`, 2877s). Val 1.1922 / **1.720 bpb
  = idéntico a tu T13@Martha**. Carry en `logs/entity_retrieval_quijote_T13_cristina_carry.log`:
  **numéricamente indistinguible del T13@Martha** (top1 34-51%, lnp 2.4-3.5,
  name_bpb 0.9-2.3), ver tabla en `logs/PARA-GPT-2026-08-15.md` Parte 7.

**El 2×2 quedó completo: T1 colapsa en carry en ambas máquinas; T1.3 es
estable en ambas.** Mismo hardware, solo T varía → la asimetría es un efecto
real de temperatura, no varianza de máquina (además, los dos T1.3 son
indistinguibles entre GTX 1650 y RX 580: el entrenamiento es invariante a la
máquina). Primera propiedad de T=1.3 sobre lenguaje natural que no es nula:
robustez del estado recurrente al cruzar ventanas. Caveat: es estabilidad, no
rendimiento (no mejoró bpb ni recuperación).

Queda habilitado (tu secuencia + Parte 6 de GPT): baseline SSM de alpha fijo
lento sobre ESTE benchmark de entidades. No lo arranco sin OK explícito.

## 7. Baseline SSM ejecutado (tu propuesta): estabilidad ≠ memoria útil

Ejecuté tu baseline tal cual la propusiste: alpha FIJO lento dentro de
nuestro código (`EVA_SSM_ALPHA=0.999`, reemplaza el clock aprendido per-
channel por 0.999 fijo; todo lo demás idéntico al base ClockMem, misma
receta/seed, mismo `entity_retrieval`, mismos buckets/carry/métricas).
`16m5b_quijote_SSM.weights`, log `logs/train_quijote_SSM.log` (3129s).
Suite 80 passed (test nuevo `ssm_fixed_alpha_excludes_clock_and_is_constant`).

Resultados (val):
- Global: SSM 1.988 bpb vs 1.716/1.720 (T1/T1.3). El alpha lento fijo en
  todo cuesta en modelado local, como cabía esperar.
- Independiente: sin memoria larga, igual que todos (outside 29.4% vs
  34.9/37.4).
- Carry: SSM NO colapsa (top1 15-37% vs el 0-14% de T1) pero DECAE con la
  distancia: >1024 14.6% top1 / 2.99 name_bpb, vs T1.3 33.8% / 2.00.

Lectura para el equipo:
1. **Mantener el estado entre ventanas es propiedad de CUALQUIER recurrencia
   lenta, no de T=1.3.** El SSM fijo lo hace sin aprender nada. El colapso de
   T1 era patología propia (α extremos → estado OOD).
2. **Estabilidad ≠ memoria útil (tu advertencia, confirmada):** en carry, la
   distancia >1024 nunca supera la referencia sin-memoria del mismo modelo
   (T1 7.1 vs 34.9; T1.3 33.8 vs 37.4; SSM 14.6 vs 29.4 top1%). Nadie
   recupera entidades por memoria larga en ningún modo.
3. La diferencia cuantitativa T1.3 vs SSM (>1024) es solo retención (canales
   lentos más lentos), no memoria útil para lenguaje.

El benchmark de entidades sigue siendo la métrica principal y sigue negativo.
No vuelvo a bpb global para conclusiones de memoria. Detalle en
`logs/PARA-GPT-2026-08-15.md` Parte 8 y `TRABAJO-2026-08-15.md`.
