# CRITERIO DEL REFERENTE EXTERNO, FIJADO ANTES DE CORRER (2026-08-31)

Se escribe **antes** de la corrida para que no se pueda acomodar despues.
Mismo formato que `CRITERIO-58.md`: el umbral sale de la dispersion del
**mismo** bucket entre semillas, no de una intuicion.

## La pregunta

Todo lo medido hasta hoy compara ClockMem contra baselines **de esta misma
casa**: la attention minima de la §2 y el SSM de `alpha=0.999` fijo. El propio
draft lo declara, §6 y §4.12:

> *"this attention baseline is deliberately minimal -- single-head, learned
> absolute positions, described in its own source header as existing 'to have
> something to measure against'... These results therefore support 'ClockMem
> substantially outperforms this reference baseline at matched budget', not
> 'attention cannot do this'."*

> *"a head-to-head benchmark against a modern SSM baseline has not yet been run."*

**Pregunta:** a igualdad de parametros, corpus y presupuesto, .como queda esta
arquitectura contra un transformer estandar bien implementado?

Hoy no hay respuesta. No es que la evidencia este en contra: **el examen no se
tomo.**

## Base: 8 semillas del Quijote, ya corridas

Receta: `dim=512, ffn=1024, 5 bloques, seq=512`, 1 epoca, corpus 2.205.169
bytes, 4.306 ventanas (3.875 train / 431 val), 13.405.701 params.
Tiempo: **2.678,8 s (~45 min) en la GTX 1650.**

```
  semilla      7      26      25      27      21      24      22      23
  val bpb   1,716  1,717  1,719  1,721  1,722  1,724  1,726  1,743
```

```
  metrica                media    sd (n=8)
  val bpb               1,7235    0,0086
  dlnp 32-128 bytes    -0,6979    0,1225
```

**La 1,743 (semilla 23) NO se saca.** Sin ella la sd cae a 0,0036 y el umbral
se afloja a menos de la mitad -- o sea que sacarla haria **mas facil** declarar
que ganamos. Regla 3: ante un confundido, quedarse con el que juega en contra
de la hipotesis propia.

## Umbrales (t de dos muestras, p~0,05, brazo base n=8)

```
  n del referente    umbral bpb    umbral dlnp 32-128
       4              0,0117           0,167
       6              0,0101           0,144
       8              0,0092           0,131
```

**Lectura, con el brazo base fijo en 1,7235:**

```
  GANA  ClockMem     el referente queda >= 1,7352 bpb   (n=4)
  EMPATA            el referente cae dentro de la banda
  PIERDE ClockMem    el referente queda <= 1,7118 bpb   (n=4)
```

**Los tres resultados sirven y los tres se aceptan.** Un negativo cierra la
linea hasta que un numero nuevo la reabra (regla 4). No se re-litiga.

**Condicion sobre los umbrales, tambien fijada ahora:** valen si el referente
tiene dispersion comparable. Si su sd sale mas de 2x la de la base, se
recalculan los umbrales con sd apareada **antes de leer el signo del
resultado**, no despues.

## Las dos metricas, y por que dos

1. **val bpb global.** Barata, y es el numero con el que se compara todo el
   mundo. Pero **es ciega a lo que nos importa**: ya esta medido que 0,006 bpb
   equivalen a 7,6 puntos de vinculo, porque las consultas son una fraccion
   chica de los bytes.
2. **dlnp en el bucket 32-128** (`examples/entity_retrieval.rs`), que es el
   unico resultado positivo robusto en texto real que tiene el draft (n=8, las
   ocho negativas). **Esta es la comparacion interesante:** .un transformer
   comun tambien se lleva ese beneficio de memoria en contexto entre 32 y 256
   bytes, o es propio del mecanismo?

   - Si el transformer saca **mas** beneficio -> el mecanismo no tiene nada
     especial, y el unico positivo en texto real se cae.
   - Si empatan -> paridad, y la ventaja habria que buscarla en costo.
   - Si saca **menos** -> primera evidencia real de la tesis del paradigma.

   **Costo de implementacion declarado:** `entity_retrieval.rs` hoy no sabe
   leer un checkpoint que no sea de esta casa. Adaptarlo es parte del trabajo
   y hay que hacerlo **antes** de mirar ningun numero.

## Que es el referente, exactamente

Un transformer estandar a nivel byte, tipo nanoGPT: multi-cabeza, pre-norm,
posiciones aprendidas o RoPE, mismo vocabulario de 256 bytes.

- **Mismo corpus, mismo split, byte identico.** Las 431 ventanas de validacion
  tienen que ser LAS MISMAS. Si no, la comparacion es nula, no "aproximada".
- **Mismo presupuesto de parametros**: 13,4 M +-2%. Se ajusta capas/ancho hasta
  llegar; se reporta el conteo exacto.
- **Misma cantidad de pasos**: 3.875, 1 epoca.
- Corre en PyTorch. Es un **referente**, no entra al repo ni le agrega
  dependencias a evallm.

## Confundidos declarados (regla 3)

1. **La receta esta afinada para ClockMem.** Semanas de ajuste implicito de un
   lado, defaults del otro. Juega **a favor** de la hipotesis propia -> se
   compensa dandole al referente un barrido chico de learning rate (3 valores)
   y quedandose con **su mejor** resultado. Si igual gana ClockMem, el
   resultado es mas fuerte, no mas debil.
2. **Rust contra PyTorch**: init, numerica y optimizador difieren. No se
   mitiga; se declara y se reporta el conteo de parametros de los dos lados.
3. **La velocidad NO se compara aca.** Un numero Rust-contra-Python mide dos
   implementaciones, no dos arquitecturas. Afirmar "mi arquitectura es mas
   rapida" con eso seria deshonesto. Si se quiere medir costo, es otro
   experimento con las dos del mismo lado.

## Diseno ESCALONADO. Reglas de parada fijadas ANTES

**Etapa 0 -- 1 semilla, ~1 h.** No mira bpb todavia: verifica el arnes.
- .el split de validacion es byte-identico? (hash de las 431 ventanas)
- .el conteo de parametros entra en el +-2%?
- .la bpb del referente es sana (entre ~1,5 y ~2,2)?

  Si sale absurda, **es un bug del arnes, no un resultado.** Se arregla, no
  se publica. Y ojo con la trampa 1 de traza: **un numero identico al de la
  base no es "empato", es "no corrio"** -- se verifica por hash del
  checkpoint, no por la cara del numero.

**Etapa 1 -- 3 semillas mas (n=4), ~3 h.** Solo si la etapa 0 pasa limpia.
-> si cae claramente de un lado del umbral de n=4, se corta aca.

**Etapa 2 -- 4 semillas mas (n=8), ~4 h.** Solo si queda en el borde.

Caso probable: 4 horas. Peor caso: 8. Mismas semillas de la base (7, 21-27)
para poder aparear.

## Que NO decide este experimento

- No dice si se le puede competir a Qwen ni a ningun modelo grande. El
  referente es un transformer chico al mismo presupuesto.
- No dice nada sobre traza. Es otro proyecto y otra pregunta.
- No dice nada sobre el 62% de los nombres. Eso ya esta medido aparte y
  no lo toca.

Dice **una** cosa: si esta arquitectura, en igualdad de condiciones, esta a la
altura del estandar aburrido. Que es la pregunta que faltaba.
