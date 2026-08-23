# CRITERIO DE #58, FIJADO ANTES DE CORRER (24-08-2026)

Se escribe **antes** de la corrida para que no se pueda acomodar después.
Corrección de GPT aceptada: el umbral sale de la dispersión del **mismo**
bucket entre semillas, no del bucket vecino.

## Base: 8 semillas del Quijote, ya corridas (Δlnp, negativo = ayuda)

```
  bucket      media    desvío   SE de la diferencia (8 vs 8)
  128-256    -0,168    0,092         0,046
  256-512    -0,026    0,075         0,037
```

## Umbrales

Dos brazos de n=8, t de dos muestras, 14 grados de libertad, p≈0,05
(2,15 SE):

```
  #58 mejora en 128-256   si su media llega a  <= -0,267
  #58 despierta 256-512   si su media llega a  <= -0,107
```

**Positivo** = cualquiera de los dos. El segundo es el más interesante:
significaría que el producto externo **estira el alcance** de la memoria más
allá de donde hoy se muere, no sólo que la profundiza donde ya funciona.

**Negativo** = ninguno de los dos. En ese caso #58 queda medido y cerrado, y
no se re-litiga sin un número nuevo (regla 4).

## Diseño

- 8 semillas (las mismas 7, 21-27) con `EVA_WRITE_LOWRANK=32`
- Todo lo demás idéntico a la base: `dim=512, ffn=1024, 5 bloques, seq=512`,
  1 época, mismo corpus, mismo binario r3
- No hace falta correr el brazo de control: la base ya existe
- Comparación apareada por semilla, además de la de medias

**Confundido declarado:** #58 agrega 4·d·r = 65.536 parámetros (~0,5% de
13,4 M). Si mejora, parte podría ser capacidad y no el producto externo. Se
mitiga mirando la sonda `lowrank_probe`: si `M` no se usa, cualquier mejora
es de los parámetros extra.

## Por qué acá y no en el sintético

La lotería (2 de 17) no aparece en texto real: 8 semillas, dos métricas
independientes, las dos unimodales. Medir si #58 cambia la tasa de la
lotería sería medir una intervención sobre un fenómeno de laboratorio. Esto
mide si mejora **lo único que tenemos funcionando en texto real**.
