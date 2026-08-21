# Dónde estamos — EVA v0

> Para quien se incorpora (Orfeo, Qwen3, quien venga): **leé esto primero**.
> La historia, la conversación y cada medición están en `ONCE-DECISIONES.md`
> (3000+ líneas). Acá está el estado al día, con punteros al gigante para
> profundizar. Actualizado: 14-08-2026.

> ⚠️ **ANTES DE SEGUIR CON `mix.rs` / `softmax_rows` / la suma de logits:**
> nuestra hipótesis central Y el fracaso de hoy de `mixed_ce` **ya están
> publicados** -- es **kNN-LM** (Khandelwal et al., ICLR 2020). Ellos
> interpolan `p = λ·p_LM + (1−λ)·p_kNN` **sólo en inferencia, nunca en el
> target de entrenamiento** -- exactamente la lección que costó un día de
> GPU hoy. Detalle completo, fuentes, y la sospecha sin confirmar sobre la
> "suma de logits" (Product of Experts, mismo modo de falla por otro
> camino): `TRABAJO-2026-08-12-CLAUDIO.md`, sección "URGENTE antes de
> seguir implementando". Leer antes de escribir la próxima línea.

## Reglas de la casa

1. **Medir barato antes de construir.** Si no hay número, es propuesta, no hecho.
2. **"Existe código que toca X" ≠ "X funciona, medida".** Son cosas distintas; este repo entero se sostiene en no mezclarlas.
3. Se escribe **agregando al final** (orden cronológico), no al principio.
4. Nada se ajusta contra la validación; se toca una sola vez.
5. Un resultado negativo cierra la línea hasta que haya un número nuevo que la reabra.

## Quién es quién

- **Valentín** — dueño; decide rumbo y prioridades.
- **Dante** (opencode) — código, mediciones, auditorías.
- **Claude** — diseño y revisiones; mantiene el orden del doc.
- **GPT** — propuestas y auditorías desde afuera.
- **Orfeo** y **Qwen3** — incorporados el 12-08-2026.

## Las once decisiones, al día

| # | Idea | Estado | En una línea | Dónde |
|---|------|--------|--------------|-------|
| 8 | Que apueste | **VIVO** (recortado) | La confianza declarada rastrea el acierto (r=0.998 por decil, ECE 0.022). La vara post-hoc del tramo (media p[argmax] ≈ 0.67) gobierna **después** de generar. La apuesta **antes** de gastar perdió (cabeza r≈0.05–0.16) → se hace gratis con la vara. | 567, 1969 |
| 5 | Tabla de k-gramas | **VIVO** | Lo único que ganó: modelo chico + tabla externa le gana al modelo entero (16M: 6.0% vs 6.3%). OJO: es la pieza VIEJA; el "cuaderno adentro" del punto 5 NO está construido. | 696, 2998 |
| 11 | Esqueleto primero | **PARCIAL** | `constrain.rs:Skeleton` + `generate_shaped` existen y funcionan, pero con vocab 256 la cabeza es ~2.5% del costo → ahorra poco hoy. | 679, 3007 |
| 4 | Largo como confianza | **NEGATIVO, medido** | La magnitud no rastrea (r≈−0.06 pre y post-norma). "Gratis no viene"; habría que entrenar la señal. Pospuesto. | 622, 2986 |
| 9 | Congelar/descongelar | **NEGATIVO, medido** | 0.0% de pesos cruza el umbral estricto en 6 épocas. No hay centinela. | 1479, 1629 |
| 10 | Crédito local | **NEGATIVO, medido** | Perdió 3.6% de calidad. | 794 |
| 6 | Sobresaltos | NO construido | Vecinos + sobresaltos (no en su lugar). Sin medir. | 660 |
| 1 | Corte dinámico | NO construido | Después del 8. | 669 |
| 7 | Borrado dirigido | NO construido | Versión de Valentín: consolidar (el "sueño de EVA"), no borrar en caliente. | 688 |
| 3 | Bloques sin orden fijo | NO construido | Alto riesgo de colapso; la familia del cómputo condicional ya murió medido (abajo). | 709 |
| 2 | Embeddings que derivan | **NO HACER** | Descartada. | 714 |

## Cerrado con medición — no reabrir sin un número nuevo

- **Saltear bloques / cómputo por influencia:** techo con cero margen a 2.7M; control: 5 bloques entrenados desde cero **empatan** con 6 (2.594 vs 2.5985) → el bloque 3 sobra por arquitectura, no hay señal que un centinela pueda anticipar. | 2216, 2598
- **Cabeza de stake antes de gastar:** r≈0.05–0.16, cuatro veces por debajo de la vara 0.67. | 1969
- **Congelamiento como señal:** 0.0% bajo umbral estricto. | 1479
- **Autocontraste como segunda parte:** es MC dropout, con abuela. | 1517
- **Gradient checkpointing (de BitVMX):** el grafo es 8 MB de 282 → no vale. | 1244
- **Crédito local por aporte:** −3.6% de calidad. | 794
- **Achatamiento del espectro de `alpha` — mecanismo explicado (artefacto, no elección):** el colapso era saturación del sigmoid (trampa absorbente cerca de alpha≈0), no preferencia de la loss. Benchmark causal: corr(alpha, grad_firmado)≈−0.001, perturbaciones ~0, |grad| 20-100x menor en banda rápida explicado por α(1−α). `sigmoid(z/T)` con T=1,3: bpb 2.588 vs 2.594 (sin costo), espectro final 16/54/30 estable desde un init 16/53/30 (T=1 colapsa desde init balanceado 35/39/26 a 98/2/0). Predicción pre-registrada, cumplida. T=1,3 queda como flag `EVA_ALPHA_TEMP`, no default. Pregunta viva: ¿el espectro distribuido paga con data de largo alcance? | `TRABAJO-2026-08-12.md` (final) |
- **Largo alcance N=256 — límite identificado con cirugía (MILESTONE):** la cadena B (ventana aprendible, w≈delta) → B2.1 (init diferencia finita + lr×10) → oracle (taps exactos por forward) quedan TODOS en ~8.0 bpb con máscara (8.0093 / 8.0164 / 8.0021). Con el oracle, `read[t]=write[t-N+1]` EXACTO llega a la lectura y aun así `out=q·read·g` falla → **el límite es la puerta q·g, no el almacenamiento ni el readout.** La única variante que resuelve es saltarse la puerta: inyección limpia del write recuperado al residual (`EVA_READ_INJECT`: `out=q·cur·g+write_rec[t]`) → **0.0353 bpb / 99.6% top-1 en val** (seed ~8.02, piso real). `activation_probe`: la escritura decodifica 100% en `b0.read` y sobrevive hasta `head_in` 99.98%; MODEL OWN 99.98% (label corregido con la permutación F del generador). Milestone: la información existe y viaja; falta que el modelo APRENDA la ruta. | `TRABAJO-2026-08-15.md` |
- **Compuerta de escritura como detector de novedad — NEGATIVO, cerrado (16-08):** probe mínimo agregado a producción (`EVA_WRITE_ERROR`/`EVA_WRITE_ERROR_P`, `examples/wprobe.rs`, puerta detach-eada, sin params nuevos) contra `assoc_k2_clock` y prosa seed7. La magnitud de escritura NO discrimina clave repetida de nueva: separación ~1-2% y en la dirección opuesta a la esperada (la repetida escribe MÁS). Verificación independiente confirmó el patrón. La señal plana es lo esperable: el error se computa contra el estado COMPLETO (mezcla acumulada), y una repetición individual es ~1/5000 de la masa → pregunta mal planteada contra el objeto equivocado, no bug. Propuesta NO circular (documentada, NO implementada, espera decisión de Valentín): comparar contra un estado auxiliar de decaimiento rápido, huella de lo muy reciente, menos diluido. | `PARA-OPENCODE.md` Parte 9 |
- **Densidad de atención aislada de la distancia — NULO, cerrado (16-08):** la corrida "densa" (filler corto) había hecho que atención aprendiera el recall asociativo, pero mezclaba densidad con distancia (gap ~45→~10). Versión limpia: mismo generador/filler (4-40), seq 512→2048 → densidad por contexto ×4 (queries/bindings 9.93→42.09) a la MISMA distancia (gap medio 45.2→47.2). Receta de siempre (seed 7, dim 512/ffn 1024/5 bloques/2 épocas, 768 KB). Resultado: atención en chance en TODOS los buckets, incluso en TRAIN (no memoriza) — idéntico al nulo base. La corrida densa aprendió por la distancia corta, no por la densidad. Celda no testeable con este generador: densidad por byte alta a distancia igual (confundidas por construcción). Caveat: 4x menos pasos (674 vs 2698); Claudio testeando "más exposición" en paralelo (attn seq512, 15 épocas, en martha). | `PARA-OPENCODE.md` Parte 10 |

## Hilos abiertos hoy

1. **¿El espectro de `alpha` distribuido (T=1,3) paga con data de largo
   alcance?** El objetivo es indiferente a la distribución de alpha (bpb
   igual 2.588 vs 2.594); la pregunta es si la palanca data (corpus con
   estructura de largo alcance, o sintético con dependencias controladas a
   distancia N) premia el espectro distribuido. Par a medir: T=1 vs T=1,3
   sobre esa data, no sobre prosa250.
2. **Aprender la ruta `write_rec → residual` sin los taps exactos.** El
   oracle+inject resuelve N256 (0.0353) pero por construcción (la solución
   está puesta a mano). La pregunta viva que GPT y Dante acordaron: cuál es
   la modificación MÍNIMA para que el modelo aprenda por sí mismo una ruta
   equivalente. Ladder de menor a mayor intervención: (1) ruta de memoria
   aprendible NO multiplicativa `read → linear/projection → residual`
   (sin `q·read·g`) — si aprende N256, el problema era el gating, no la
   lectura; (2) reintroducir la compuerta con una vía que no pueda quedar
   anulada durante el aprendizaje; (3) recién después ROUTE/READ/STORE. El
   oracle queda como "profesor" temporal para estudiar el mecanismo mínimo.
   | `TRABAJO-2026-08-15.md` |
   - **Rung 1 ejecutado (15-08): CONFIRMADO con matiz.** Ruta aprendible SIN
     la puerta (`out=q·cur·g+Σw·state`, `w` K×d) — réplica del inject
     idéntica (0.0353/99.6%), y **R1b desde cero aprende la lectura larga**
     (3.54 bpb / 43.4% top-1 vs 8.0/0.4% base) → la puerta era el cuello.
     PERO el tramo aprendido no llega al oracle: R1a (seed df) 1.73 bpb / 73%,
     R1b 3.54 bpb / 43%. Trace: R1a se aleja de la relación df exacta y
     contamina la lectura; R1b construye los taps lento y **no convergió al
     cierre** (loss aún cayendo). Candidatos R1c: más pasos, α_c/β_c
     aprendibles por canal (familia df), o supervisión de `w` hacia el df.
     Hilo Sigue abierto en el rung 2 (reintroducir la compuerta sin vía
     anulable). | `TRABAJO-2026-08-15.md`, `logs/PARA-GPT-2026-08-15.md` |
   - **R1c (mismo día): la familia estructurada RESUELVE — rung 1 CERRADO.**
     `read = inv_beta[c]·(state[t−N+1] − alpha_read[c]·state[t−N])`, ~2
     params/canal, `inv_beta=1/β` aprendido directo (sin división por
     parámetro libre, prevención de Claude). Seed en el oracle: **0.0355 bpb
     / 99.6%** (se queda pegado, |da|≤0.071, inv_beta≤1.033 — sin taps
     residuales donde escapar, a diferencia de R1a). Neutral (α_read=0):
     **0.0348 bpb / 99.6%** — converge a otro punto de la familia (lectura
     ~`1.04·state[t−N+1]`, sin invertir la fuga) que también resuelve. La
     hipótesis de GPT quedó confirmada: la matriz libre tiene demasiados
     grados de libertad y encuentra soluciones ruidosas; la familia correcta
     obliga a una lectura full-strength en la distancia correcta desde el
     init. Geometría entendida. Próximo: rung 2 del ladder. |
     `TRABAJO-2026-08-15.md`, `logs/PARA-GPT-2026-08-15.md` |
3. **Pregunta de Orfeo a Valentín:** prioridad entre "no inventar", "gastar
   menos" y "modular". Es la única que le toca decidir a Valentín. | 2853
4. **Bisector de BitVMX** — en pausa; el análisis de granularidad (ronda del
   12) cerró con change-point AUROC 0.672 pero payoff real insuficiente
   (14,6% de cobertura). No construir encima sin número nuevo.

## Cómo se usa este archivo

- Se actualiza cuando cambia un veredicto, se abre o cierra un hilo, o entra/sale un actor.
- Los números y su metodología viven en `ONCE-DECISIONES.md`; acá solo el estado y el puntero.
- Un modelo nuevo lee ESTO primero, y va al gigante solo si necesita profundizar en un punto.
