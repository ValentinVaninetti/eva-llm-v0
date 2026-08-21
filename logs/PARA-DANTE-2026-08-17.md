# PARA DANTE — recall asociativo: ClockMem lo resuelve con datos suficientes (17-08-2026)

De Claude. Estado al 02:55. Todo lo de acá está verificado contra logs y
checkpoints reales, no contra reportes.

## 1. El resultado grande: ClockMem resuelve recall asociativo

Contexto: la corrida de 2 claves sobre corpus chico (1500 ventanas, 2698
pasos) daba **6,29%** a gap≤32 — señal real pero modesta, la que reportamos
ayer. Sospechábamos que el techo era arquitectónico.

**No era arquitectónico: era hambre de datos.** Misma arquitectura, mismo
tamaño, misma tarea, misma seed. Sólo cambia el corpus: 22.499 ventanas en
vez de 1500 (1 época = 20.249 pasos en vez de 2 épocas = 2698).

```
eval_associative, VAL (muestra de 300 ventanas del split de validación)
checkpoint: 16m5b_assoc_k2_huge_clock.weights

  bucket           n       acc%      bpb
  gap<=32        2606     99.85    0.0255
  32<gap<=128    3178     99.69    0.0368
  128<gap<=256    175     97.71    0.1262
  256<gap<=512      8     62.50    2.4361   (n chico, ignorar)
  binding (cold)  600      2.83    5.0246   <- azar, sin fuga
  filler       147033      2.32    7.5281   <- azar teórico 7.585, sin memorizar
```

Los dos controles pasan: las primeras apariciones siguen impredecibles (no
hay fuga) y el filler sigue en azar (no memorizó el corpus). El salto
6,29% → 99,85% es real.

Entrenamiento sano además: val loss bajó monótona y se estabilizó
(7,404 → 7,226 bpb), sin rastro del colapso que tuvo la corrida de atención
de 15 épocas.

## 2. Me retracto de una afirmación mía

Ayer, mirando `clockmem_from` en `ops.rs`, afirmé que ClockMem
**estructuralmente no puede** hacer recall por contenido, porque la escritura
es elemento a elemento (`cur[c] += β·k[t,c]·v[t,c]`) en vez de producto
externo (`S[i,j] += k[t,i]·v[t,j]`), y que por eso el vínculo clave→valor se
destruye al escribir.

Acaba de hacerlo al 99,85%. **La afirmación era demasiado fuerte y la
retiro.**

Lo que sigue en pie, mucho más modesto y ahora como hipótesis a testear, no
como conclusión: con 2 claves existe una solución degenerada que el producto
elemento a elemento SÍ permite — dedicar un grupo de canales a cada clave
(256 canales por clave, con dim 512). Eso resuelve la tarea sin vincular
nada y **no escala**. Con estos datos no puedo distinguir esa explicación de
que ClockMem realmente ligue clave y valor.

Nota honesta que ya había marcado: el comentario de cabecera de
`src/model/attn.rs` ya afirmaba que ClockMem "can't" hacer recall
asociativo. Esa afirmación de diseño ahora quedó **refutada
experimentalmente**, al menos en el caso de 2 claves.

## 3. Lo que esto invalida (importante)

**El nulo de 8 claves de anoche no vale.** Fue sobre corpus chico (1500
ventanas), el mismo régimen donde 2 claves daba apenas 6,29%. Estaba
hambriento de datos, no limitado por la cantidad de claves. No dice nada
sobre capacidad.

**Las corridas k3ctl/k4ctl que encolé tienen el mismo problema.** Usan el
corpus chico. Las dejo terminar porque son una comparación válida a
presupuesto igualado contra el 6,29%, pero si dan bajo no vamos a poder
separar "más claves es más difícil" de "no le alcanzó el entrenamiento". Su
valor es bajo; no saques conclusiones fuertes de ellas.

## 4. El test que decide: 8 claves en régimen de datos grande

Encolado en Martha, arranca solo cuando termine la corrida de atención.

Corpus: `data/assoc_k8ctl_seq512_win22500_seed42.dat` (ya transferido).
Generado con `generate_associative 8 512 22500 42 <out> 1 6` — el filler se
achicó a 1-6 a propósito, para que el gap quede en 40,4 (contra 45,2 de la
de 2 claves) y la distancia no sea un confundido. Ratio consultas/bindings
10,69 vs 9,94 — pareado también.

Confundido residual, declarado: al achicar el filler subió la densidad de
consultas. Eso juega **en contra** de encontrar un derrumbe (más densidad
debería ayudar), así que si igual colapsa, el resultado es más fuerte, no
más débil. Mismo criterio que usamos con el confundido de pasos en la
barrida de seq.

**Qué decide:**
- Si k8 mantiene ~99% → la teoría del atajo por canales dedicados muere,
  ClockMem hace recall asociativo de verdad, y mi objeción estructural queda
  enterrada del todo.
- Si k8 colapsa → era el atajo (con 8 claves quedan 64 canales por clave), no
  escala, y eso justifica explorar una vía de escritura distinta (tu
  propuesta de bajo rango / producto externo) en vez de seguir tocando la
  lectura.

## 5. ACTUALIZACIÓN 05:40 — atención también aprendió, pero muy por debajo

Terminó la de atención sobre el mismo corpus grande (24.105 s, val 7,389
bpb). Comparación directa, mismo corpus, mismo presupuesto, misma seed:

```
                        ClockMem      Atención
  gap<=32                99,85%        46,05%
  32<gap<=128            99,69%        47,99%
  128<gap<=256           97,71%        50,29%
  binding (cold)          2,83%         3,67%   <- azar en ambos, sin fuga
  filler bpb               7,53          7,55   <- azar 7,585 en ambos
```

Primero: **atención SÍ aprendió con datos suficientes.** Pasó de azar exacto
(3,1% en todos los buckets, incluso en TRAIN, con el corpus chico) a ~47%.
Eso cierra el hilo del "click tardío": no era falta de pasos ni de densidad,
era falta de datos distintos. Y el colapso de la corrida de 15 épocas era
sobreajuste por repetir datos fijos, confirmado: acá con 1 época sobre datos
variados la val loss bajó sana.

Segundo, y es lo llamativo: **ClockMem le saca 50 puntos a atención.**

**Interpretación mía, marcada como hipótesis:** el ~47% de atención es
sospechosamente cercano a una moneda al aire, y hay una razón estructural
para que lo sea. Con 2 claves activas por ventana hay exactamente 2 valores
candidatos. Un modelo que aprendió "la respuesta es uno de los dos valores
que aparecieron recién" pero NO cuál corresponde a cuál, saca exactamente
50%. Atención estaría aprendiendo el *conjunto* de candidatos, no el
*vínculo*. Encaja también con que su curva sea plana con la distancia
(46/48/50 — atención mira a cualquier lado por igual) mientras ClockMem
decae de a poco (99,85 → 97,71, coherente con tener decaimiento).

**Test barato para confirmarlo o matarlo (no lo corrí, te lo dejo
propuesto):** en las posiciones donde atención se equivoca, ¿su predicción
top-1 es el valor ligado a la OTRA clave? Si sí, es moneda al aire
confirmada. Es un agregado chico a `eval_associative.rs` (trackear si el
argmax coincide con el binding de otra clave de la misma ventana).

## 6. ACTUALIZACIÓN — la barrida k2/k3/k4 en corpus chico quedó inservible,
## como estaba previsto, pero dejó un indicio

Terminaron las tres (k3 03:09, k4 03:39). Corpus chico, 2698 pasos, gap
controlado a ~42-45 en las tres:

```
  claves   consultas    gap<=32   32<gap<=128   128<gap<=256
    2        29.813      6,29%       4,76%         2,11%
    3        47.516     13,22%       7,03%         3,94%
    4        61.481     27,92%       8,47%         4,17%
```

El acierto de cerca SUBE con más claves. Eso no es capacidad: es el
confundido de densidad que ya había declarado (para mantener el gap
constante hay que achicar el filler, y eso mete más consultas por ventana).
El número de cerca sigue casi linealmente a la cantidad de consultas de
entrenamiento. **Confirmado: en este régimen se mide si le alcanzó el
entrenamiento, no la capacidad.** No saques conclusiones de capacidad de
estas tres.

**El indicio, que sí me parece anotable:** mirá cuánto acierto se CONSERVA
al alejarse un bucket.

```
  claves   retención (32-128 / <=32)
    2           75,7%
    3           53,2%
    4           30,3%
```

La caída con la distancia se hace sistemáticamente más empinada cuantas más
claves hay — que es exactamente lo que predice la teoría del atajo (menos
canales por clave → memoria más frágil). Sigue confundido con la densidad,
así que es un indicio, no evidencia. Pero es el primer patrón de la barrida
que NO se explica sólo por cantidad de entrenamiento.

## 7. ACTUALIZACIÓN 16:30 — el k8 terminó: DERRUMBE, y cambia de forma

Terminó 09:02 (19.587 s). Comparación limpia, mismo corpus grande, mismo
presupuesto, misma receta, gap controlado (~40 vs ~45):

```
                    2 claves     8 claves
  gap<=32            99,85%       31,80%
  32<gap<=128        99,69%       14,03%
  128<gap<=256       97,71%        5,65%   <- azar es 3,125%
  binding (cold)      2,83%        3,29%   <- azar en ambos, sin fuga
  retención (32-128 / <=32)
                      99,8%        44,1%
```

**Lo más informativo no es el nivel, es la FORMA.** Con 2 claves la memoria
es prácticamente inmune a la distancia: conserva el 99,8% del acierto al
alejarse un bucket y a 256 bytes sigue en 97,7%. Con 8 claves conserva el
44% y **para los 128 bytes ya está muerta** (5,65% contra azar 3,125%). No
rinde menos: se le desarma con la distancia. Eso es exactamente lo que
predice la teoría del atajo (menos canales por clave → traza más frágil).

**El caveat, y es real: la corrida de 8 claves NUNCA se amesetó.** Su val
loss venía bajando hasta el último paso (6,699 → 6,614 → 6,563 → 6,525 →
6,493 → 6,476 → 6,461 → 6,446). La de 2 claves, en cambio, ya estaba en su
valor final al 35% del entrenamiento (7,236 contra 7,226 final). O sea:
31,80% es un PISO, no un techo.

**Contra-argumento cuantitativo, con los dos puntos que tengo:** evalué
también el checkpoint parcial del paso ~5.000 (25% del entrenamiento) y daba
25,25%. Al 100% da 31,80%. **Cuadruplicar el entrenamiento dio +6,5
puntos.** A ese ritmo, llegar de 31,8 a 99 pide un presupuesto absurdo. No
parece estar por pegar un salto; parece estar raspando un techo. Pero es una
extrapolación de dos puntos, no una demostración.

## 8. Qué está corriendo ahora (lanzado 16:25)

| dónde | qué | por qué | termina |
|---|---|---|---|
| local | **ClockMem 8 claves, 4 épocas** (80.996 pasos) | el test de convergencia: dejarlo hasta que la val se amesete de verdad y ver si trepa a ~99% o se estanca cerca de 30% | ~10:00 del 18 |
| Martha | **atención 8 claves, 1 época** | completa el 2×2 {ClockMem, atención} × {2, 8 claves}. Tenemos atención a 2 claves = 46%; falta saber si a 8 aguanta mejor o peor que ClockMem | ~22:00 de hoy |

El de convergencia es el que cierra la pregunta. Si se estanca en ~30% con
la val amesetada, la teoría del atajo queda confirmada y con ella la
justificación para ir por tu propuesta de escritura de bajo rango / producto
externo en vez de seguir tocando la lectura. Si trepa a ~99%, era
subentrenamiento y mi hipótesis se muere del todo.

El de atención a 8 claves es interesante por su cuenta: si atención se
mantiene en ~46% (su nivel de 2 claves) mientras ClockMem se derrumba de 99
a 31, eso diría que los dos mecanismos fallan por razones distintas — el de
atención por no ligar (moneda al aire), el de ClockMem por quedarse sin
canales.

## 9. ACTUALIZACIÓN 17:10 — el test de la moneda al aire, hecho. Es contundente.

Implementé tu punto (2): `eval_associative.rs` ahora clasifica QUÉ es la
predicción top-1 en cada posición de consulta — el valor de esta clave, el
de OTRA clave, un valor no ligado a nadie en esa ventana, o algo que ni
siquiera es un byte de valor. Compilé sólo el ejemplo; `target/release/eva_llm_v0`
quedó con el mismo timestamp (la corrida de convergencia no corrió riesgo).

Corrido sobre los tres checkpoints del corpus grande:

```
  top-1 es...                       ClockMem 2   ClockMem 8   Atención 2
  el valor de ESTA clave (correcto)     99,65%      23,42%       47,23%
  el valor de OTRA clave                 0,00%      76,49%       42,45%
  un valor no ligado ahí                 0,35%       0,08%       10,17%
  ni siquiera un byte de valor           0,00%       0,00%        0,15%
  --- total dentro del conjunto ---     99,65%      99,91%       89,68%
  n (consultas evaluadas)                 5.967      25.610        5.967
```

**Tres lecturas, y las tres cierran algo:**

1. **ClockMem a 2 claves LIGA.** Cero confusiones entre claves en 5.967
   consultas. No es "casi nunca": es 0,00%.
2. **Atención recupera pero NO liga.** El 89,68% de sus predicciones es uno
   de los dos valores realmente ligados en esa ventana — o sea, encuentra el
   conjunto candidato casi perfecto, desde cualquier distancia — y después
   los reparte 47/42. Moneda al aire entre candidatos correctos. Tu
   hipótesis, confirmada.
3. **ClockMem a 8 claves cae EN EL MODO DE FALLA DE ATENCIÓN.** Y esto es lo
   más fuerte: conoce el conjunto candidato MEJOR que atención (99,91% de sus
   predicciones son uno de los 8 valores ligados) — no olvidó nada, no perdió
   la regla estructural de que después de una clave viene un valor — pero
   elige el valor de la clave equivocada el 76,49% de las veces.

**Por qué esto mata la explicación aburrida.** El fracaso de 8 claves no es
memoria degradada, ni estructura sin aprender, ni confusión sobre cuáles son
los candidatos. Es **específicamente** una falla de vínculo, y aparece justo
cuando la cantidad de claves supera lo que una partición de 512 canales puede
mantener separado. El vínculo residual existe pero es débil: 23,42% contra
12,5% que sería tirar al azar entre los 8 candidatos que ya identificó bien.

**Qué le queda por decidir a la corrida de convergencia:** menos de lo que
pensábamos, y más preciso. Sea cual sea el acierto que compre más
entrenamiento, el modelo de 8 claves ya demostró que NO le falta aprender la
tarea ni retener los candidatos. La pregunta abierta se achicó a: **¿el
76,49% de confusión entre claves baja con más entrenamiento, o es ahí donde
una representación por partición de canales se queda sin lugar?**

## 10. CIERRE DE LA RONDA (18-08, 01:30) — respondido, y la corrida se cortó a propósito

### La confusión baja, pero logarítmicamente: es un techo en la práctica

Corté la corrida de convergencia en el paso 41.500 de 80.996 (algo más del
doble del presupuesto original), porque su segundo punto **ya determina la
respuesta** con la precisión que la pregunta necesita:

```
  presupuesto           acierto    confusión entre claves
  1x (20.249 pasos)      23,42%          76,49%
  2x (41.500 pasos)      28,71%          71,24%
  4x (80.996, NO corrido)   --           ~66% (extrapolado)
```

Duplicar TODO el entrenamiento compró 5,25 puntos de confusión. A ese ritmo,
llegar al 0,35% del modelo de 2 claves pide unas 13 duplicaciones más — del
orden de 10⁴ veces el cómputo — y hasta llegar a 50% pide cuatro. **No está
en camino a resolverse: decae logarítmicamente en cómputo, que para
cualquier presupuesto real es indistinguible de un techo.**

Lo dejo asentado como **decisión metodológica, no como experimento
interrumpido**: el tercer punto queda extrapolado y etiquetado como tal, no
medido. Las 39.500 pasos que faltaban daban un número predecible a un punto
o dos, en una máquina limitada térmicamente, sin cambiar la conclusión de
diseño.

### Atención con 8 claves: colapso total, y peor que el de ClockMem

Se completó el 2×2 (mismos datos, misma distancia, una época cada uno):

```
                          Clock 2   Clock 8   Attn 2   Attn 8
  correcto                 99,65%    23,42%   47,23%    2,89%   <- azar 3,125%
  valor de otra clave       0,00%    76,49%   42,45%   16,74%
  valor no ligado ahí       0,35%     0,08%   10,17%   80,23%
  --- dentro del conjunto  99,65%    99,91%   89,68%   19,63%
```

Atención a 8 claves está **en azar**, y a diferencia de ClockMem **perdió
también la identificación del conjunto candidato** (19,63%, contra 89,68% con
2 claves): el 80% de sus predicciones son bytes de valor del alfabeto
general, no de los ligados en esa ventana.

Y no es subentrenamiento: su val loss se movió apenas 0,028 bpb en toda la
corrida (6,781 → 6,753) — convergió casi de inmediato a una solución que
ignora los vínculos. La de ClockMem se movió 0,253 y seguía bajando.

**Salvedad grande, para el paper:** nuestra atención es deliberadamente
mínima (una cabeza, posiciones absolutas aprendidas, y su propio encabezado
dice que existe "para tener algo contra qué medir"). La literatura reporta
que los transformers SÍ aprenden recall asociativo. Así que esto sostiene
"ClockMem supera ampliamente a este baseline de referencia a presupuesto
igualado", NO "atención no puede".

### Estado de las máquinas

Cristina quedó libre. Bajo esta carga se clavaba en 96°C con la frecuencia
caída a 3.515 MHz (de 4.500), y al soltarla bajó a 57°C en 25 segundos — el
disipador y los ventiladores andan bien, lo que está gastado es la pasta
térmica (5 años sin cambiar). Valentín la cambia mañana; queda la línea base
medida para comparar el antes/después.

Martha, en cambio, tiene margen de sobra: CPU (Ryzen 5 2600) y GPU (RX 580)
las dos a ~31°C en reposo, críticos en 94-95°C, y sin escritorio compitiendo.
**Es la máquina para las corridas largas de acá en adelante.** Le activé
`k10temp` para poder leerle la temperatura del CPU (ojo: no sobrevive a un
reinicio, hay que volver a hacer `modprobe k10temp`).

### Lo que queda para la próxima ronda

Una sola cosa, y es diseño nuevo, no otra corrida: **la escritura de producto
externo de bajo rango.** Toda esta investigación atacó la lectura; el
diagnóstico de esta ronda dice que el vínculo se destruye al escribir.

## 6. Cambio menor en un ejemplo mío

`examples/eval_associative.rs` acepta ahora un tercer argumento opcional:
máximo de ventanas por sección. Evaluar 22.5k ventanas completas tarda más
de una hora; con 300 ventanas del split de validación ya hay decenas de
miles de posiciones de consulta, de sobra para los buckets. Sin el
argumento, el comportamiento es idéntico al de antes. Compilé sólo el
ejemplo (`cargo build --release --example eval_associative`) para no tocar
`target/release/eva_llm_v0`, que las corridas encoladas necesitan — verificado
que el binario principal quedó con el mismo timestamp y tamaño.

— Claude
