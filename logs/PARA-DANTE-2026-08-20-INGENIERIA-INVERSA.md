# INGENIERÍA INVERSA DEL GANADOR (20-08) — dónde vive el vínculo

Idea de GPT: antes de tocar la arquitectura, descubrir qué hace el modelo
que resuelve la tarea. Todo esto es **sólo inferencia** sobre checkpoints
que ya existían: no se entrenó nada.

## Herramienta nueva

`EVA_READ_MASK=c1,c2,...` en `src/model/clock.rs`: anula la LECTURA de esos
canales poniendo `g[c]=0`, sin tocar la escritura ni la dinámica del estado.
Se pudo hacer sin op nuevo ni backward porque `g` se calcula fuera del op.
Sin la variable, comportamiento idéntico.

**OJO (gotcha que me costó una vuelta):** `cargo build --release` NO
reconstruye los `examples/`. Hay que usar `--examples`, o el ejemplo queda
enlazado contra la librería vieja y el flag "no hace nada". Verificado por
el control de máscara total.

## El resultado

Modelo: `16m5b_assoc_k2_d256_s10` (el ganador, dim 256, 2 claves, 99,77%).
Métrica: desplazamiento direccional de `query_swap`.

```
  sin máscara                      +0,8849
  ── CONTROLES: 8 canales, en otro lado ──
  8 del extremo rápido (248-255)   +0,8755    intacto
  8 del medio (124-131)            +0,8894    intacto
  8 al azar, dispersos             +0,8879    intacto
  ── los canales lentos ──
  primera mitad (0-127)            +0,0025    destruido
  segunda mitad (128-255)          +0,6724    sobrevive
  0-63                             −0,0004
  0-31                             +0,0014
  0-15                             +0,0008
  0-7                              −0,0001
  0-3                              +0,0004    destruido con CUATRO canales
  0-1                              +0,0279
  sólo el canal 0                  +0,2230    cae 75%
```

**Especificidad perfecta.** Ocho canales cualesquiera fuera del grupo: efecto
cero. Cuatro canales específicos: destrucción total. **Toda la capacidad de
vínculo de un modelo de 5,4 M de parámetros vive en 4 canales de 256**, y el
75% en uno solo.

## Por qué esos cuatro: conecta con la §3 del paper

El índice de canal NO es arbitrario: `log_clock` se inicializa geométrico,
canal 0 con α≈0,9999 (memoria más larga) y canal 255 con α≈0,01. Las alfas
APRENDIDAS del ganador:

```
  banda                              canales
  rápida (<10 tokens de memoria)     251 (98%)
  media (10-100)                       4
  muy lenta (>1000)                    1   <- α=0,9998, ~5744 tokens
```

La §3 ya documentó que el reloj se aplana solo. Un vínculo tiene que
sobrevivir un hueco de ~45 bytes; **los 251 canales rápidos no pueden
físicamente**. Quedan 5, y son exactamente los que la ablación señala.

## LO DECISIVO: los ciegos tienen las MISMAS alfas

```
                    rápida(<10)  media(10-100)  muy lenta(>1000)
  GANADOR seed 10     251 (98%)      4              1
  ciego   seed  9     251 (98%)      4              1
  ciego   seed 11     251 (98%)      4              1
```

Hasta los percentiles coinciden. **La lotería NO es sobre qué canales
sobreviven la trampa de saturación.** Todos terminan con el mismo sustrato.

Y más todavía — los ciegos **usan** esos mismos canales:

```
                       sin máscara    canales 0-3 apagados
  CIEGO   (seed 9)       43,59%           3,85%
  GANADOR (seed 10)      94,87%           3,85%
```

(3,85% = azar, 1 de 32 valores.)

**Mismo sustrato, mismos 4 canales, dos circuitos distintos encima:**

- el **ciego** escribe ahí *"estos son los valores en juego en esta ventana"*
  → tira una moneda entre dos y saca 44%
- el **ganador** escribe ahí *"la clave A va con el valor X"* → saca 95%

## Qué mata esto

Cualquier resto de explicación por capacidad. El modelo ciego usa
exactamente el mismo recurso de memoria para un cómputo peor. La capacidad
efectiva son **5 canales, no 256**, en todos los modelos, y por eso el ancho
no predecía nada y la cuenta de canales-por-clave estaba refutada.

## La pregunta que queda

> No es "¿cómo le damos memoria asociativa a ClockMem?". La memoria está,
> son 4 canales, y TODOS los modelos la usan. Es: **¿por qué el gradiente
> escribe ahí el conjunto de candidatos 7 de cada 8 veces, y el vínculo
> sólo 1?**

## Lo que NO está hecho

- La prueba de **intercambio de canales** que propuso GPT (tomar los canales
  importantes de A y B y permutarlos en una consulta por A, a ver si la
  respuesta se corre a `val_B`). Requiere manipular `cur`, no `g`, así que
  pide tocar el op. Es la confirmación directa de que la identidad de la
  clave vive en ese subespacio.
- Verificar si la concentración en 4 canales se repite en el OTRO ganador
  (`dim512/k2`) o es propia de este.
- n=1 ganador para todo esto. Las ablaciones son deterministas y los
  controles son limpios, pero el circuito está caracterizado sobre un solo
  modelo.

— Claude
