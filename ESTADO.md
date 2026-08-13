# Dónde estamos — EVA v0

> Para quien se incorpora (Orfeo, Qwen3, quien venga): **leé esto primero**.
> La historia, la conversación y cada medición están en `ONCE-DECISIONES.md`
> (3000+ líneas). Acá está el estado al día, con punteros al gigante para
> profundizar. Actualizado: 12-08-2026.

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

## Hilos abiertos hoy

1. **Bisector de BitVMX** — candidata para pensar, nada construido. Localizar *dónde* arranca a fallar una respuesta usando la traza por token de `scan()` (`bet.rs:265`). Siguiente paso: análisis de granularidad (tramo entero / mitad / change-point por token) sobre datos que ya existen. La corrección ya acordada: se biseca la **traza**, no el texto. | 2684, 2743
2. **Pregunta de Orfeo a Valentín:** prioridad entre "no inventar", "gastar menos" y "modular". Es la única que le toca decidir a Valentín. | 2853

## Cómo se usa este archivo

- Se actualiza cuando cambia un veredicto, se abre o cierra un hilo, o entra/sale un actor.
- Los números y su metodología viven en `ONCE-DECISIONES.md`; acá solo el estado y el puntero.
- Un modelo nuevo lee ESTO primero, y va al gigante solo si necesita profundizar en un punto.
