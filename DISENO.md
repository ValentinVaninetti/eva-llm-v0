# Cómo aprende este cerebro

Documento de diseño. Separa con cuidado tres cosas que no valen lo mismo:
**medido**, **construido pero sin medir**, y **propuesto**. Todo lo que diga
"propuesto" es una apuesta, no un plan cerrado.

## El problema, con la aritmética adelante

Entrenar una red hoy tiene un costo dominante que casi nadie nombra: **para
retropropagar hay que guardar las activaciones de toda la pasada hacia
adelante**. La memoria crece con profundidad × ancho × largo de secuencia, y es
lo que empuja a comprar hardware caro. No es el cómputo: es la memoria.

Nuestro autograd hace exactamente eso, y peor: cada operación **clona sus
entradas** en `saved_v`. `clockmem` sola clona cinco tensores de `s×d` por
llamada. Medido: la maquinaria de autograd sin contar matmul se lleva ~96 s de
201 en un entrenamiento de 861 pasos.

El resto del gasto son convenciones, no física:

- Todo token recibe el mismo esfuerzo, aunque la mayoría de un corpus sea
  trivialmente predecible.
- Todo parámetro se actualiza para todo token.
- Se reaprende lo mismo millones de veces, sin noción de "esto ya lo sé".
- Aprender e inferir son fases separadas, y lo aprendido en una conversación
  muere con ella.

## Cómo aprende un animal, y qué de eso se puede copiar

No todo lo que hace un cerebro es transferible ni conviene fetichizarlo. Pero
hay cuatro cosas que **sí** tienen traducción computacional directa:

1. **Dos sistemas con velocidades distintas.** El hipocampo aprende de un solo
   golpe y es caro de mantener; la corteza aprende lento y generaliza. Lo nuevo
   entra rápido y sin gradientes; lo que resulta valioso se destila después.
   Esto no es metáfora: es *Complementary Learning Systems*, y explica por qué
   una red sola sufre olvido catastrófico y un animal no.
2. **Consolidación durante el sueño.** El sistema lento no se entrena con el
   mundo en vivo: se entrena con **repeticiones** de lo que el sistema rápido
   guardó. O sea, con datos ya filtrados por relevancia.
3. **Se aprende del error de predicción**, no de etiquetas. Predecir siempre,
   propagar sólo lo que falló.
4. **El crédito es local.** No hay ningún mecanismo biológico conocido que
   transporte pesos hacia atrás por toda la red. Cada sinapsis se ajusta con lo
   que tiene cerca.

Lo que **no** voy a copiar porque es cargo cult: picos, neuronas de integración
y disparo, y cualquier cosa que se justifique sólo por parecerse a una neurona.
El criterio es si baja el costo o sube la calidad, medido.

## Las tres memorias

El diseño es una jerarquía por escala de tiempo. Cada pieza es un módulo con su
propio archivo, su propio test y la posibilidad de apagarse.

### 1. Estado — milisegundos — **CONSTRUIDO Y MEDIDO**

`ClockMem`: `s_t = α ⊙ s_{t-1} + β(k_t ⊙ v_t)`, con `α` por canal aprendido.
Lleva el contexto actual en **D números de tamaño fijo**.

Medido contra atención de una cabeza con los mismos parámetros exactos, tres
semillas, sin solapamiento: **2.630 vs 2.688 bits/byte**, y 6% más rápido.

### 2. Memoria rápida — minutos a horas — **PROPUESTO**

Una memoria clave-valor escrita **en inferencia, de un solo golpe, sin
gradientes**: cuando el modelo predice mal, escribe la asociación con una
actualización local tipo Hebb (producto externo), con su propio olvido.

Por qué importa para el hardware: **aprender algo nuevo cuesta O(D) en vez de
una pasada hacia adelante y otra hacia atrás por toda la red.** Tres órdenes de
magnitud menos por hecho nuevo. Y no hay fase de entrenamiento: aprende
mientras corre.

El riesgo real: interferencia. Dos claves parecidas se pisan y la memoria se
degrada. Por eso existe la tercera capa.

### 3. Pesos lentos — el sueño — **PROPUESTO**

La red no se entrena con el mundo en vivo, sino **repitiendo lo que la memoria
rápida acumuló** y sobrevivió a su propio olvido. Descenso por gradiente sí,
pero sobre datos ya filtrados por "esto apareció y sirvió más de una vez".

Consecuencia de hardware: el paso caro deja de ser continuo y pasa a ser
esporádico, sobre un conjunto chico y curado.

## Las cuatro palancas de eficiencia

Independientes entre sí, apagables, y cada una con su medición.

### A. Aprender sólo de lo que sorprende — **CONSTRUIDO, MIDIÉNDOSE**

`src/learn.rs`. La pasada hacia adelante siempre se hace; el **backward se
saltea** si la pérdida está por debajo de lo que el modelo venía esperando. La
expectativa es una media móvil de la propia pérdida, así que el umbral se mueve
solo a medida que mejora.

El backward cuesta el doble que el forward, así que saltear la mitad borra ~un
tercio del entrenamiento. Se está midiendo contra la línea base de 2.630 en dos
condiciones: mismo recorrido de datos (más barato) y **mismo presupuesto de
backwards** (¿elegir gana?).

Riesgo conocido: aprender sólo de lo que sale mal es aprender el ruido del
corpus. Por eso se vigila la validación.

### B. Crédito local: nunca sostener el grafo entero — **PROPUESTO. La más importante.**

Que cada bloque tenga su **propio objetivo local** —predecir su próxima entrada—
y que ningún gradiente cruce de un bloque a otro.

Es la palanca que cambia qué hardware hace falta: la memoria de entrenamiento
pasa de *profundidad × activaciones* a **un bloque por vez**. Un modelo de N
bloques se entrena con la memoria de uno. Eso es lo que hace posible entrenar
en una máquina chica, y no hay optimización de kernel que lo reemplace.

Honestidad: **las reglas locales rinden peor que backprop en todos los intentos
serios publicados.** No lo vendo como que va a ganar. Lo propongo porque casi
nadie lo probó sobre un recurrente moderno con objetivo predictivo, porque
tenemos con qué medirlo bien, y porque si funciona aunque sea parecido, el
ahorro es estructural.

### C. Cómputo condicional que entre en caché — **PROPUESTO**

Que sólo una fracción de los parámetros se active por token, elegida por el
propio token, de modo que **el conjunto activo entre en L2/L3**. La máquina de
prueba tiene 8 MB de L3; nuestro modelo entero son 11 MB en f32 y 2.8 MB en
int8.

Esto ataca el único piso físico real: generar un token obliga a **leer** los
pesos activos. La palanca no es leer más rápido, es leer menos.

### D. Inferencia con estado — **PROPUESTO, y hay un desperdicio medido**

Hoy `generate` rehace la pasada completa sobre toda la ventana para cada token
y **descarta 63 de las 64 filas** que calculó. Con `seq=64` eso es hasta 64x de
trabajo tirado. Un transformer no puede evitarlo sin un caché KV que crece con
el contexto; ClockMem sí, porque su estado es de tamaño fijo.

## Orden, por rendimiento medible sobre esfuerzo

1. **A** (sorpresa) — corriendo. Barata, y da número hoy.
2. **D** (inferencia con estado) — una tarde, desperdicio ya medido, sin riesgo.
3. **B** (crédito local) — la apuesta grande. Es la que decide si "entrenar con
   algo mínimo" es real o es marketing.
4. **2 y 3** (memoria rápida y sueño) — dependen de B para valer la pena.
5. **C** (condicional) — última: complica el entrenamiento y sin B no se sostiene.

## La regla de la casa

Nada entra sin medición contra la línea base, con validación held-out y varias
semillas. Ya van tres veces en este proyecto que la intuición eligió mal el
cuello: el shader que no era, ClockMem que era el 0.8%, y las posiciones que
iban a arreglar la comparación y la empeoraron. **La evidencia indirecta sirve
para elegir qué medir, nunca para concluir.**
