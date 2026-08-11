# Cómo aprende este cerebro

Documento de diseño y bitácora de razonamiento. Separa con cuidado tres cosas
que no valen lo mismo: **medido**, **construido sin medir**, y **propuesto**.
Todo lo que diga "propuesto" es una apuesta.

---

## Parte I — Por qué es caro hoy

No hay una respuesta: hay dos costos con causas distintas.

### Entrenar

El costo es **≈ 6 × N × D** (parámetros × tokens de entrenamiento). El 6 sale
de dos operaciones por parámetro en la ida, dos para el gradiente respecto de
las activaciones y dos para el gradiente respecto de los pesos. **El backward
cuesta el doble que el forward, por construcción.**

Por qué N y D son enormes:

- **D es enorme porque el descenso por gradiente es un modo absurdamente
  ineficiente de guardar algo.** Para aprender un hecho hay que verlo muchas
  veces, y cada vez se empujan un poquito millones de pesos. Escribirlo en una
  tabla cuesta una operación.
- **N es enorme porque la mayoría de los parámetros son almacén, no
  razonamiento.** Las estimaciones de la literatura rondan los 2 bits de
  conocimiento por parámetro.

**El nudo:** toleramos ese costo porque la representación distribuida
generaliza y una tabla no interpola. Generalizar exige representación
distribuida; la representación distribuida exige gradiente; el gradiente es
carísimo. Pagamos precio de generalización para guardar cosas que sólo
necesitan recordarse.

### Inferir

Otra cuenta y da distinto: generar un token obliga a **leer todos los pesos
activos**. No es cómputo, es tráfico.

Balance de la máquina de prueba: 8 núcleos con AVX2 dan del orden de
250 GFLOP/s en `f32`; la memoria, decenas de GB/s. El hardware está balanceado
alrededor de **~6 operaciones por byte leído**. Una generación densa pide dos
operaciones por peso de cuatro bytes: **0,5 operaciones por byte**. Se le está
pidiendo al procesador **una décima parte** de lo que sabe hacer por byte
traído.

*(El ancho de banda exacto de esta máquina está SIN medir bien: el benchmark
que hice medía su propio lazo. El desbalance de un orden de magnitud sí es
robusto; el factor preciso, no.)*

**De ahí sale la economía, sin conspiración:** un modelo denso con un usuario
lee miles de millones de pesos para producir un token. Con mil usuarios en
paralelo, esa misma lectura sirve para mil tokens. La estructura de costos
premia servir a muchos y castiga al usuario local. Es consecuencia de densidad
+ ancho de banda, y explica por qué nadie con incentivo de vender cómputo tiene
apuro en arreglarlo.

### Qué es física y qué es costumbre

| | |
|---|---|
| Leer los pesos activos para generar un token | **física** |
| Que "los activos" sean *todos* los pesos | elección (densidad) |
| Que el backward cueste 2× el forward | física **de backprop** |
| Que haga falta backprop | **elección** |
| Guardar activaciones de toda la red para retropropagar | consecuencia de esa elección |
| Que el conocimiento viva en el mismo sustrato que el cómputo | **elección** |
| Que aprender un hecho requiera verlo miles de veces | consecuencia de esa elección |
| Que generalizar requiera representación distribuida | **física**, hasta donde sabemos |

Sólo la última fila es irreductible. El resto son decisiones que se tomaron una
vez y se volvieron costumbre.

---

## Parte II — La hipótesis central

> **¿Se puede separar lo que necesita generalizar de lo que sólo necesita
> recordarse, sin perder la capacidad de razonar sobre lo recordado?**

Si la respuesta es sí, el costo se derrumba: la estructura del lenguaje y los
patrones de razonamiento son *poca* información y sí necesitan gradiente; los
hechos son *mucha* y no necesitan generalizar.

Si es no, el gasto de hoy está justificado y hay poco que hacer.

**No está saldado en la literatura** — es el debate paramétrico contra
recuperación. No se puede afirmar, hay que medirlo.

### El experimento que lo decide

Un modelo de N parámetros contra **un modelo de N/2 con acceso a una búsqueda
sobre el corpus**. Misma validación held-out, mismas semillas. Si el chico con
búsqueda llega a los mismos bits/byte, la hipótesis queda demostrada con
números. Si no llega, aprendimos que el conocimiento en los pesos hacía algo
que la búsqueda no reemplaza.

Es trabajo de verdad —hay que construir el mecanismo de recuperación— pero es
**decidible**, que es lo único que importa.

---

## Parte III — Lo que hicimos y lo que salió

### MEDIDO: ClockMem le gana a la atención

Mismo bloque, misma conv, mismo FFN, mismos parámetros exactos (cuatro matrices
DxD cada uno), mismas semillas. 250 KB de prosa real, seq 64, 3 semillas:

    ClockMem        1.8229   2.630 bits/byte   319 s
    atención        1.8630   2.688             339 s
    atención + pos  1.9216   2.772             345 s

Sin solapamiento entre grupos. La peor corrida de ClockMem es mejor que la
mejor de la atención. **No es ruido.**

Lo que **no** dice: seq 64 es corto y es justo donde la atención no puede
lucirse. Falta la comparación a contexto largo.

### MEDIDO Y NEGATIVO: aprender sólo de lo que sorprende

`src/learn.rs`. Forward siempre; **backward sólo si la pérdida supera lo que el
modelo venía esperando** (media móvil de la propia pérdida, umbral que se mueve
solo).

    línea base      1.8229   3515 backwards   319 s
    barato (1 ép)   2.0275   ~1598 backwards  ~196 s
    igual-bwd (2 ép) ~1.84   ~3220 backwards  ~410 s

**No sirve.** Con la mitad del aprendizaje pierde 11% de calidad; con el mismo
presupuesto de backwards empata en calidad pagando **28% más de cómputo total**
(porque hay que pagar el doble de forwards para juntar los mismos backwards).

**Por qué falló, que es lo valioso:** filtré por **ventana** —64 tokens
promediados— pero el argumento de "la mayoría del corpus es trivial" es sobre
**tokens**. Al promediar 64 la varianza se lava: todas las ventanas tienen una
mezcla parecida de fácil y difícil, y no queda nada que elegir.

**Y aunque hubiera filtrado por token, tampoco ahorraba:** enmascarar la
pérdida de los tokens fáciles no abarata el backward, porque el backward
recorre el grafo entero igual. **La sparsity en la pérdida no se cobra si el
cómputo es denso.**

Conclusión que deja: *el problema no es de qué aprendemos, es que backprop hace
que cada evento de aprendizaje cueste la red entera.* Ningún truco sobre los
datos lo arregla.

---

## Parte IV — Cómo aprende un animal, y qué de eso sirve

Cuatro cosas con traducción computacional directa:

1. **Dos sistemas a velocidades distintas.** Hipocampo: de un golpe, caro de
   mantener. Corteza: lento, generaliza. No es metáfora — es *Complementary
   Learning Systems*, y explica por qué una red sola sufre olvido catastrófico
   y un animal no.
2. **Consolidación durante el sueño.** El sistema lento no se entrena con el
   mundo en vivo sino **repitiendo** lo que el rápido guardó: datos ya
   filtrados por relevancia.
3. **Se aprende del error de predicción**, no de etiquetas.
4. **El crédito es local.** No hay mecanismo biológico conocido que transporte
   pesos hacia atrás por toda la red.

**Lo que NO se copia porque es cargo cult:** picos, neuronas de integración y
disparo, y cualquier cosa que se justifique sólo por parecerse a una neurona.
El criterio es si baja el costo o sube la calidad, medido.

---

## Parte V — El diseño

### Las tres memorias, por escala de tiempo

**1. Estado — milisegundos — CONSTRUIDO Y MEDIDO.** `ClockMem`. Lleva el
contexto en D números de tamaño fijo.

**2. Memoria rápida — minutos a horas — PROPUESTO.** Clave-valor escrita en
inferencia, de un solo golpe, sin gradientes (tipo Hebb), con olvido propio.
**Aprender algo nuevo cuesta O(D)** en vez de una pasada ida y vuelta por toda
la red. Riesgo real: interferencia entre claves parecidas.

**3. Pesos lentos — el sueño — PROPUESTO.** La red se entrena repitiendo lo que
la memoria rápida acumuló y sobrevivió a su propio olvido. El paso caro deja de
ser continuo y pasa a ser esporádico sobre un conjunto chico y curado.

### Las cuatro palancas

**A. Sorpresa — MEDIDA Y DESCARTADA.** Ver Parte III.

**B. Crédito local: nunca sostener el grafo entero — PROPUESTO. La grande.**
Que cada bloque tenga su propio objetivo local y ningún gradiente cruce de un
bloque a otro. La memoria de entrenamiento pasa de *profundidad × activaciones*
a **un bloque por vez**: una red de N bloques se entrena con la memoria de uno.
Es literalmente la diferencia entre necesitar una GPU y no necesitarla.

*Honestidad:* **las reglas locales rinden peor que backprop en todo intento
serio publicado.** Lo que da esperanza: siempre se las probó sobre
arquitecturas diseñadas *para* backprop. Nadie las probó sobre un recurrente
con decaimiento por canal, donde el crédito temporal ya es analítico y local.

**C. Cómputo condicional que entre en caché — PROPUESTO.** Que el conjunto
activo por token entre en L2/L3 (8 MB acá; el modelo entero son 11 MB en f32 y
2,8 MB en int8). Ataca el único piso físico: la palanca no es leer más rápido,
es leer menos.

**D. Inferencia con estado — PROPUESTO, desperdicio ya medido.** Hoy
`generate` rehace la pasada completa sobre toda la ventana para cada token y
**descarta 63 de las 64 filas** que calculó. Un transformer no puede evitarlo
sin un caché KV que crece; ClockMem sí, porque su estado es de tamaño fijo.

### Orden

1. **D** — una tarde, desperdicio medido, sin riesgo.
2. **B** — la apuesta grande. Decide si "entrenar con algo mínimo" es real.
3. **El experimento de la Parte II** — decide la hipótesis central.
4. **2 y 3** (memoria rápida y sueño) — dependen de B.
5. **C** — última: complica el entrenamiento y sin B no se sostiene.

---

## La regla de la casa

Nada entra sin medición contra la línea base, con validación held-out y varias
semillas.

**Marcador de la intuición en este proyecto: 0 de 4.**

| creí que | era |
|---|---|
| el cuello era el shader | los flags de memoria (2.5x contra 1.13x) |
| el cuello era ClockMem | el 0,8% del tiempo |
| la atención estaba lisiada sin posiciones | dárselas la empeoró, 3 de 3 |
| aprender por sorpresa iba a rendir | empata pagando 28% más |

Cuatro de cuatro en contra. **La evidencia indirecta sirve para elegir qué
medir, nunca para concluir.** Y una convicción no se gradúa a hecho sin pasar
por una medición — especialmente la hipótesis central, que es justo la que más
tentaría saltear.
