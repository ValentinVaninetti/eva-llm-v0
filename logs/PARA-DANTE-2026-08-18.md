> **LEÉ PRIMERO `PARA-DANTE-2026-08-20.md`**, que es la puesta al día
> ordenada. Este archivo quedó como bitácora cronológica con apéndices
> apilados; sirve para el detalle, no para ponerse al día.

# PARA DANTE — #58 implementado (escritura de producto externo) + corrida k4 (18-08-2026)

**AVISO DE `src/` (regla 5):** toqué cuatro archivos. Todo lo nuevo está
detrás de `EVA_WRITE_LOWRANK`, que por defecto está apagado. Con el flag
apagado el modelo es el de siempre — verificado contra un número tuyo, no
por lectura del código (ver §3).

- `src/tensor/ops.rs` — op nuevo `clockmem_lowrank_from` (agregado al final)
- `src/tensor/autograd.rs` — arm `"clockmem_lowrank"` (insertado antes de
  `"clockmem_gated"`)
- `src/model/clock.rs` — cinco campos nuevos, el constructor y el dispatch
- `src/tests.rs` — tres tests nuevos (83 → 86)
- `examples/lowrank_probe.rs` — nuevo, no toca `src/`

---

## 1. Qué corre en Martha ahora

`dim=512, k=4, corpus grande` — PID **122674**, arrancó 01:45, termina ~07:15.

```
EVA_GPU=1 ./eva_llm_v0_bin_r2 train \
  --data data/assoc_k4ctl_seq512_win22500_seed42.dat \
  --dim 512 --ffn 1024 --blocks 5 --seq 512 --val 0.1 --seed 7 \
  --gate-beta 0.0 --log 500 --epochs 1 \
  --out 16m5b_assoc_k4ctl_huge_clock.weights
```

**Por qué esta corrida y no #58 directamente.** Toda la evidencia del cuadro
de claves varió *la cantidad de claves*. La cuenta que sostiene #58 es el
reparto de canales — y **`dim` no se tocó nunca**. Se infirió, no se
manipuló. Además es el control que #58 necesita igual: la escritura de bajo
rango agrega parámetros, y sin la curva de "parámetros solos" no se puede
decir si mejoró el producto externo o la plata extra.

Corpus generado con `generate_associative 4 512 22500 42 <out> 1 16`. El
filler se eligió para que la distancia quede pareada:

```
              gap medio   consultas/binding
  k=2            45,2          9,94
  k=4 (nueva)    39,1         11,33
  k=8            40,4         10,69
```

Queda mejor pareada con k8 (39,1 vs 40,4) que el pareo k8-vs-k2 que ya
habíamos aceptado. Self-check del generador: 0 fallas. Mismo confundido
residual declarado que en el k8 (más densidad de consultas juega EN CONTRA
de encontrar derrumbe, así que un derrumbe igual sería más fuerte).

La sigue: `dim=1024, k=8` (~22 h, 4x FLOPs) — restaura los canales por clave
al nivel de `dim=512/k=4`. La cuenta predice que esas dos den parecido. Ojo
al confundido, declarado: duplicar `dim` sube la capacidad en general, no
sólo los canales por clave; por eso importa el par, no la dirección sola.

Bajar `dim` en vez de subirlo sale 4x más barato pero **no sirve**: 23,42%
está a 11 puntos del azar entre candidatos (12,5%), así que un
empeoramiento se comprime contra el piso y queda ambiguo.

## 1b. RESULTADO: k=4 a dim 512 — el derrumbe es un ACANTILADO, no una pendiente

Terminó 07:10 (19.454 s). Evaluado con 300 ventanas, `EVA_GPU=1`:

```
                  canales/clave   correcto   otra clave   dentro del conjunto
  dim 512 / k=2       256          99,65%      0,00%          99,65%
  dim 512 / k=4       128          28,82%     70,68%          99,50%   <- NUEVO
  dim 512 / k=8        64          23,42%     76,49%          99,91%
```

**Mi predicción era que k=4 cayera en el medio y salió mal.** Cae casi hasta
el fondo: 71 puntos de 2 a 4 claves, y sólo 5 más de 4 a 8.

**Y esto rompe la versión de la cuenta que teníamos escrita.** Si el recurso
fuera "512 canales repartidos entre claves", partirlos de 256 a 128 por
clave no puede costar 71 puntos y después partirlos otra vez costar 5.

**El argumento que me parece que lo mata del todo, y es de conteo:** guardar
cuál de 32 valores le toca a una clave son 5 bits. Con 4 claves quedan 128
canales por clave; con 8 quedan 64. **Los dos son absurdamente más que
suficientes para 5 bits.** Así que quedarse sin canales NO puede ser la
explicación de la falla a k=4. Si la capacidad sobra y aun así no liga, lo
que falla es encontrar la solución, no tener lugar para guardarla — o sea
**una falla de optimización, no de capacidad**.

Ojo, eso es razonamiento, no medición. Lo que lo decide es la corrida
`dim256/k2` (ver abajo).

Anomalía marcada como duda, NO como hallazgo: la ventaja sobre el azar
*dentro de los candidatos ya identificados* no es monótona.

```
        azar entre candidatos   logrado   ventaja
  k=4          24,9%            28,82%     1,16x
  k=8          12,5%            23,42%     1,88x
```

Con 8 claves conserva más vínculo relativo que con 4. Ninguna versión de
"se queda sin canales" predice eso. Puede ser ruido de una corrida por
celda. No construir nada encima sin repetirlo.

**Convergencia:** k4 no aplanó del todo (últimos saltos de validación
−0,0041) pero está 2,5x más plano que el k8 en el mismo punto (−0,0105). Si
alguno está subentrenado es el k8. Y duplicar presupuesto ya se midió que
compra ~5 puntos, no 70.

**Qué queda por decidir, y ahora la apuesta es mucho más grande de lo que
era cuando diseñé estas corridas:**

```
  corrida        si mandan los canales/clave   si manda la cantidad de claves
  dim256 / k=2      ~28,8% (como 512/k4)          ~99,6% (como 512/k2)
  dim256 / k=4      ~23,4% (como 512/k8)          ~28,8% (como 512/k4)
```

**`dim256/k2` es la decisiva: 70 puntos entre las dos hipótesis.** La diseñé
como confirmación barata y quedó siendo la que parte el asunto al medio.

**Qué significa para #58, si se confirma que es optimización y no
capacidad:** el §4.11 del paper ya había marcado que el resultado de 2
claves "admite una solución degenerada que no escalaría" — dedicar ~256
canales a cada clave sin ligar nada. El acantilado es exactamente lo que se
ve si esa solución degenerada es TODO lo que había: anda con 2 y se cae
apenas se pide más. O sea que nunca hubo vínculo, ni siquiera con 2 claves.
Eso **refuerza** el motivo de #58 (el producto externo da un lugar donde
guardar un vínculo, que hoy no existe), pero **agrega un riesgo nuevo**: si
la falla es de optimización, darle el lugar puede no alcanzar, porque el
gradiente podría no llevarlo ahí igual. Habrá que mirar no sólo el acierto
final sino si `M` se usa (la sonda `lowrank_probe` ya existe para eso).

## 1c. RESULTADO GRANDE: achicar el modelo a la mitad DUPLICO el vinculo

`dim256/k4` termino 10:03 (10.365 s). La cuenta del reparto de canales queda
LIQUIDADA, y de la manera mas directa posible.

```
                  params  can/clave  val bpb  correcto  otra clave  dentro conj.
  dim 512 / k=2   13,4M     256        --      99,65%     0,00%       99,65%
  dim 512 / k=4   13,4M     128       6,971    28,82%    70,68%       99,50%
  dim 256 / k=4    5,4M      64       6,941  **48,56%**  43,97%       92,53%
  dim 512 / k=8   13,4M      64       6,446    23,42%    76,49%       99,91%
```

Las dos filas del medio son la misma corrida con UN numero cambiado en la
linea de comando (`--dim 512` -> `--dim 256`). Mismo corpus (md5 verificado
identico en las dos maquinas), mismo seed, mismo ffn, mismo binario.

**Doble refutacion:**
- `dim256/k4` tiene los MISMOS 64 canales por clave que `dim512/k8`, y saca
  48,56% contra 23,42%.
- `dim256/k4` tiene LA MITAD de canales por clave que `dim512/k4`, y saca
  48,56% contra 28,82%.

Los canales por clave no predicen nada. **Menos canales ligo mejor.**

**Por que, y el desglose lo dice solo:**

```
              identifica el conjunto   liga (sobre azar entre candidatos)
  dim 512            99,50%                        1,16x
  dim 256            92,53%                        2,10x
```

El modelo ancho es MEJOR en la parte degenerada de la tarea (saber cuales
son los candidatos, que se resuelve dedicando canales) y PEOR en la real.
Con lugar de sobra, el gradiente encuentra la solucion vaga y se queda ahi.
El angosto no puede pagarsela. **Es falla de optimizacion, no de capacidad**
— y ahora tiene numero, no solo el argumento de conteo de la seccion 1b.

**Dos cosas que lo refuerzan:**
- El dim 256 tiene 40% de los parametros y MEJOR val bpb (6,941 vs 6,971).
  No cambio modelado por vinculo: gano en los dos.
- Esta MENOS convergido (ultimos saltos -0,010 contra -0,004 del dim 512) y
  aun asi saca 20 puntos mas. La brecha se abriria, no se cerraria.

**Controles, todos chequeados antes de creerselo:** binding (cold) 2,75%
contra azar 3,125%; filler bpb 7,4221 contra azar 7,585 (nada memorizado);
md5 del corpus identico local y en martha.

**Para #58:** el riesgo pasa de hipotetico a documentado. Si el gradiente
prefiere la solucion vaga cuando hay lugar, darle a `M` un lugar donde
guardar vinculos puede no alcanzar. Al medirlo, `lowrank_probe` no es
opcional: hay que ver si `M` se USA o si queda de adorno.

**Siguiente propuesto (~1 h):** `dim=128, k=4`. Si "mas angosto liga mejor"
es direccion y no la casualidad de un par, tiene que pasar 48,56%. Si en
cambio se derrumba, hay un optimo intermedio, que tambien es hallazgo.

## 2. #58: la escritura de producto externo de bajo rango

```
kr[t,i] = sum_c k[t,c]*pk[c,i]      vr[t,j] = sum_c v[t,c]*pv[c,j]
qr[t,i] = sum_c q[t,c]*pq[c,i]
M[i,j]  = am[i]*M[i,j] + beta*kr[t,i]*vr[t,j]     <- acá vive el vínculo
rd[t,j] = sum_i qr[t,i]*M[i,j]
mem[t,c]= sum_j rd[t,j]*po[j,c]
out[t,c]= q[t,c]*cur[c]*g[t,c] + mem[t,c]
```

Tres decisiones de diseño, y ninguna es de gusto — las tres salen de
resultados que ya están en el repo:

1. **La vía elemento a elemento se queda.** Saca 99,65% con 2 claves;
   sacarla cambiaría dos cosas a la vez. La de bajo rango es ADITIVA.
2. **La vía nueva NO pasa por `g`.** La §4 entera es sobre una compuerta
   multiplicativa destruyendo una lectura ya correcta, y lo que funcionó
   (R1/inject) fue mandar el valor al residual SIN `q*g`. Repetir `*g` acá
   sería reconstruir exactamente la falla de la que el op quiere escapar.
   No hay compuerta sobre `mem`, ni con piso ni sin piso.
3. **`po` arranca en CERO**, así el A/B contra `--arch clock` no difiere en
   nada en el paso 0 (mismo criterio que el init delta de `wread`).

**Costo:** `4*d*r` params (65.536 con d=512, r=32) y estado `r*r` por bloque
(1024 números contra 512).

**Limitación declarada:** `M` arranca en cero cada ventana y NO se acarrea
entre ventanas, a diferencia de `cur`. Para el benchmark asociativo eso es
correcto y no apenas tolerable (los vínculos se generan frescos por ventana
de 512 y `--seq 512` coincide), pero **no es drop-in para las corridas de
estado persistente sobre texto real**. Si vas por ahí, hay que plomear el
acarreo de `M` por el mismo camino que `s0`.

**Nota sobre el arranque:** con `po=0` la vía aporta cero, así que en el
paso 1 `pk/pv/pq/log_clock_m` reciben gradiente CERO — sólo `po` se mueve.
Desde el paso 2 aprenden todos. Es un retraso de un paso, **no** la trampa
de saturación de §4.6 (el gradiente de `po` en cero es sano, no hay meseta).
Está pineado en un test para que nadie lo redescubra como bug.

## 3. Cómo lo verifiqué (regla 2: "toca X" no es "X anda, medido")

| qué | cómo | resultado |
|---|---|---|
| backward correcto | gradcheck numérico de **los 11 tensores** | pasa |
| **el gradcheck tiene dientes** | 3 mutaciones deliberadas al backward | las 3 se cachan; al restaurar vuelve a pasar |
| no perturba la base | test: con `po=0` sale **idéntico** a `clockmem_from`, salida y estado acarreado | pasa |
| flag apagado = binario tuyo | conteo de params contra el de tus logs | **13.405.701**, exacto |
| flag prendido r=32 | esperado +5*(4*512*32+32) | **13.733.541**, exacto |
| entrena | humo, modelo chico | loss 5,68 → 5,05 |
| **los params se mueven de verdad** | `lowrank_probe` sobre un checkpoint real | `po` rms 0,012 y 0,010 en los dos bloques — salió del cero |
| suite | `cargo test --release` | **86** (eran 83) |

La fila que más me importa es la penúltima. La memoria del proyecto tiene
anotado un día con tres mecanismos que "andaban" sin producir nada; `po`
arrancando en cero exacto da la lectura más limpia posible de "esta vía
está viva", y por eso el probe existe.

**Lo que NO está medido:** el sobrecosto real de cómputo (estimé ~6% de un
bloque con d=512/r=32, sin medir), y por supuesto si sirve. Nada de eso se
puede saber hasta correrlo, y las corridas comparables van a Martha en
serie, así que va después del k4 y del `dim=1024`.

## 4. Gotcha nuevo para el documento

`pkill -f PATRON` tiene **el mismo problema que `pgrep -f`**: si lo mandás
por ssh, el patrón está en la línea de comando de la shell remota y se mata
a sí misma. Me cortó la sesión. El entrenamiento no se tocó (verifiqué por
PID), pero conviene: matar por PID, o escribir el script y copiarlo con
`scp` en vez de meterlo por heredoc.

— Claude

---

# APENDICE 2026-08-19: dos cosas que hay que corregir en el paper

## A. `--seed` NUNCA vario la inicializacion. Ni una vez.

`EvaModel::new` inicializaba con la constante **fija** `Rng::new(0xE7A1)`.
`--seed` alimenta el orden de las ventanas de entrenamiento y la generacion
de muestras; la linea 387 de `train.rs` (`seed ^ 0xC0FFEE`) es para las
cabezas auxiliares del credito local, que estas corridas ni usan.

**Consecuencia sobre una afirmacion publicada.** El PAPER-DRAFT linea 49
dice: *"ClockMem wins by 2.2% bpb at seq=64 on prose, three seeds,
non-overlapping"*. Esas tres corridas **arrancaron de pesos identicos**;
variaron el orden de los datos. No invalida el numero (tres ordenes
distintos sin solapamiento sigue siendo evidencia, y los modelos entrenados
si difieren) pero **"tres semillas" le sugiere al lector robustez a la
inicializacion, que no se probo**. Hay que reescribirlo como lo que es, y
si queremos la afirmacion fuerte, ahora se puede medir.

**Arreglado (`src/model/mod.rs`):** `EVA_INIT_SEED=n` sobreescribe la
constante. Sin la variable queda `0xE7A1` exacto.

Verificado en las tres direcciones, no por lectura del codigo:

```
  (sin variable)        step 20 loss 5.7133
  EVA_INIT_SEED=59297   step 20 loss 5.7133    <- 0xE7A1 = 59297, identico
  EVA_INIT_SEED=59298   step 20 loss 5.7177    <- distinto, el flag hace algo
```

Suite: 86, sin cambios.

## B. Me equivoque al llamar "loteria" a la tabla de cinco celdas

Dije que las cinco celdas eran cinco muestras de una loteria de cuencas.
Con lo de arriba, es falso: `dim512/k2` y `dim512/k4` tienen el mismo ancho,
entonces la misma constante de init produce **exactamente los mismos pesos
iniciales**. Entre esas dos no hubo ningun sorteo — 697x contra 0,50x sale
de la TAREA (2 claves contra 4), no del azar. El sorteo de init solo se
mueve al cambiar `dim`, porque cambia la secuencia de extracciones.

La lectura corregida de la tabla:
- entre celdas del MISMO ancho, lo que difiere es la tarea;
- al cambiar el ancho se mueven init y capacidad juntos, sin separar.

## C. Que mide entonces la cola de 4 seeds que esta corriendo

Es **mas limpia** de lo que dije, no menos: con init clavado por constante y
todo lo demas igual, los seeds 8/9/10/11 de `dim256/k2` varian **una sola
cosa, el orden de los datos**. Si el orden de los datos solo alcanza para
dar vuelta el ratio CLAVE/RELLENO (0,28x de seed 7 -> 20x+), eso es loteria
de optimizacion pura. Si los cuatro quedan en ~0,3x, el resultado esta
determinado por (ancho, corpus, init), los tres fijos, y hay que buscar
mecanismo.

Lo que NO cubre y ahora si se puede correr: variar `EVA_INIT_SEED` con
`--seed` fijo, que es el sorteo complementario.

---

# 19-08, RESULTADO PRINCIPAL: es una LOTERIA DE OPTIMIZACION

## El experimento

4 seeds de `dim256/k2` (mas el seed 7 que ya existia). Verificado que las
corridas difieren en UNA cosa: 5.392.133 params las cuatro, mismo corpus de
22.499 ventanas, mismo binario, y `--seed` **no toca la inicializacion**
(ver apendice anterior), asi que **los pesos iniciales son identicos**. Lo
unico que cambia es el ORDEN en que ve las mismas ventanas.

```
  seed 7    7,303 bpb   ciego  (direccional +0,000)   ~50%  moneda
  seed 8    7,345       ciego  (+0,000)                41,81%
  seed 9    7,298       ciego  (+0,000)                43,91%
  seed 10   7,235       LEE    (+0,985)                99,77%  <- pico real
```

**Uno de cada cuatro encuentra la solucion que lee la consulta.**

## Tres consecuencias, ninguna interpretativa

1. **El objetivo SI premia leer la consulta.** El seed 10 tiene la mejor val
   bpb, por 0,063 sobre el siguiente. No es que la tarea no lo pida: el
   gradiente casi nunca llega.
2. **La suerte se decide en el primer cuarto.** Trayectorias por snapshot:

```
              seed 10          seeds 8 y 9
  paso  5000  +0,1934  32%     ~0,0000  22%
  paso 10000  +0,7922  86%     ~0,0000  40%
  paso 15000  +0,9758  99%     ~0,0005  40%
  paso 20000  +0,9852 100%
```

   Al 25% ya estan separados por tres ordenes de magnitud. No hay transicion
   tardia tipo induction-head: hay una bifurcacion temprana. Y en los ciegos
   la lectura **nunca aparece**, ni transitoriamente — asi que la hipotesis
   de "aprende y desaprende" queda descartada para esta celda.
3. **ClockMem con escritura elemento a elemento SI puede ligar**, al 99,77%
   con 2 claves y 256 canales. **Esto toca el motivo declarado de #58.** La
   frase "la escritura elemento a elemento no tiene donde guardar un
   vinculo" es falsa como esta escrita: puede. El problema es de
   OPTIMIZACION, no de representacion. Hay que reescribir la motivacion de
   #58 en el paper y aca.

## Hallazgo lateral con consecuencia practica

El checkpoint FINAL del seed 10 es PEOR que el del paso 20000:

```
  paso 20000   99,77%   direccional +0,9852   val 7,229
  FINAL        92,18%   direccional +0,8868   val 7,235
```

md5 distintos, misma medicion, y las dos metricas coinciden (la validacion
periodica tambien SUBE al final, unico caso de los tres). En ~249 pasos
perdio 7,6 puntos de vinculo por 0,006 bpb de loss: **la metrica de vinculo
es mucho mas sensible que la bpb**, porque las consultas son una fraccion
chica de los bytes.

**Consecuencia: tomamos siempre el ultimo checkpoint, y el ultimo no es el
mejor.** Todos los numeros que reportamos de corridas anteriores pueden
estar subestimando el pico. Salvedad: n=1, magnitud chica, no lo llamo
inestabilidad hasta verlo otra vez.

## Lo que esto le hace al plan

`query_swap` cambia de diagnostico a **criterio de admision**: una corrida
cuyo modelo no lee la consulta no puede contestar ninguna pregunta sobre
mecanismos de vinculo. Antes de comparar arquitecturas hay que verificar
que ambos brazos leen la consulta, o el A/B mide la loteria y no la
arquitectura.

Y hace falta una pregunta nueva, que es mejor que la que traiamos:
**#58 no deberia medirse como "mejora el acierto" sino como "cambia la
PROBABILIDAD de que el sorteo salga bien"**. Eso pide varios seeds por
brazo, no uno.

---

# 20-08: LA COMPUERTA `g` QUEDA DESCARTADA (linea cerrada)

Ablacion `EVA_NO_GATE=1` (`out = q*cur` en vez de `q*cur*g`), 12 pares
APAREADOS e intercalados con la base, protocolo corto (corte en el paso
5000, validado 4 de 4 contra el veredicto final).

```
  BASE     1 de 12 (cortas) + 1 de 5 (completas)  =  2 de 17   11,8%
  NOGATE   0 de 12
```

**Criterio fijado ANTES de correr: 5 o mas de 12. Salio 0.** Fisher p~0,34:
ni ayuda ni perjudica. Linea cerrada por regla 4.

**No contradice la seccion 4 del paper.** Alli la lectura EXISTIA y la
compuerta la tapaba. Aca la lectura NUNCA SE FORMA y sacar la compuerta no
ayuda a que se forme. Dos fenomenos distintos, y ahora esta medido.

**Verificaciones hechas antes de creer el resultado:**
- binario r3 con flags apagados reproduce r2 digito a digito
  (5.2828/5.1720/5.1604/5.1571 en los pasos 500-2000); la cola abortaba sola
  si no coincidia
- `EVA_NO_GATE=1` confirmado en `/proc/PID/environ` del proceso real, no
  supuesto desde el script
- conteo de parametros identico con y sin ablacion (5.392.133): `wg` queda
  en la lista pero sin gradiente, a proposito, para que el A/B no difiera en
  tamano de modelo
- protocolo corto validado por via independiente: corridas completas 1 de 5,
  cortas 1 de 12 -- misma tasa, distintos seeds

**Curiosidad anotada, no señal:** nogate seed 22 dio +0,0089, diez veces los
otros ciegos (todos <=0,0009) pero quince veces por debajo del umbral. Un
caso en 24. Si reaparece, mirar.

## Mapa de ejes al 20-08

```
  canales por clave    REFUTADO (dos celdas con 128 dan 50% y 29%)
  ancho                no lo explica (512 tiene el mejor Y uno de los peores)
  cantidad de claves   no lo explica
  orden de los datos   ES LOTERIA, 11,8% (2 de 17)
  compuerta g          DESCARTADA (0 de 12)
  inicializacion       NUNCA PROBADO -- imposible hasta el 19-08
```

**Lo unico estructural que queda sin tocar es la inicializacion.** Con
`EVA_INIT_SEED` variando pesos y `--seed` FIJO (orden de datos constante),
12 corridas cortas ~8h30 contestan si el sorteo lo decide el camino o el
punto de partida. Es la pregunta complementaria exacta a la que ya
contestamos.
