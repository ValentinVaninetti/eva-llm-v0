# PARA DANTE — puesta al día completa (18 al 20 de agosto)

Dante: estuviste afuera dos días y cambió el eje del proyecto. Esto es
autocontenido y ordenado; el detalle cronológico crudo está en
`PARA-DANTE-2026-08-18.md`, que quedó como bitácora con apéndices apilados.

---

## 1. AVISO DE `src/` (regla 5)

Toqué cinco archivos. **Todo lo nuevo está detrás de flags, apagados por
defecto**, y está verificado —no por lectura del código— que con los flags
apagados el binario reproduce el tuyo dígito a dígito.

| archivo | qué |
|---|---|
| `src/tensor/ops.rs` | op nuevo `clockmem_lowrank_from` (#58), al final |
| `src/tensor/autograd.rs` | arm `"clockmem_lowrank"`, antes de `"clockmem_gated"` |
| `src/model/clock.rs` | campos y dispatch de #58; `EVA_NO_GATE` |
| `src/model/mod.rs` | `EVA_INIT_SEED` |
| `src/tests.rs` | 3 tests nuevos (83 → **86**) |

Ejemplos nuevos (no tocan `src/`): `examples/query_swap.rs`,
`examples/lowrank_probe.rs`.
Scripts nuevos: `eval_k4ctl.sh`, `eval_dim_sweep.sh`, `eval_pares.sh`.

**Flags nuevos, todos opt-in:**

- `EVA_WRITE_LOWRANK=r` — escritura de producto externo de bajo rango (#58)
- `EVA_NO_GATE=1` — ablación de la compuerta: `out = q*cur` en vez de `q*cur*g`
- `EVA_INIT_SEED=n` — varía la inicialización (antes era imposible, ver §2)

**Binarios en Martha:** el tuyo, `eva_llm_v0_bin_r2`, quedó **intacto**. El
nuevo es `eva_llm_v0_bin_r3`. Verificación hecha antes de usarlo: r3 con
flags apagados da `5.2828 / 5.1720 / 5.1604 / 5.1571` en los pasos 500-2000,
idéntico a lo que produjo r2 con el mismo seed. La cola abortaba sola si no
coincidía.

---

## 2. DOS CORRECCIONES A COSAS QUE VENÍAMOS AFIRMANDO

### a) `--seed` NUNCA varió la inicialización. Ni una vez.

`EvaModel::new` inicializaba con la constante **fija** `Rng::new(0xE7A1)`.
`--seed` alimenta el orden de las ventanas de entrenamiento y la generación
de muestras; la línea 387 de `train.rs` (`seed ^ 0xC0FFEE`) es para las
cabezas auxiliares del crédito local, que estas corridas no usan.

**Toda corrida histórica descrita como "otra semilla" partió de pesos
idénticos.** Toca la línea 49 del PAPER-DRAFT: *"ClockMem wins by 2.2% bpb
at seq=64 on prose, three seeds, non-overlapping"*. No invalida el número
—tres órdenes de datos sin solapamiento sigue siendo evidencia— pero
"tres semillas" le sugiere al lector robustez a la **inicialización**, que
nunca se probó. Ya está la nota de corrección puesta en el paper.

Arreglado con `EVA_INIT_SEED`. Verificado en tres direcciones:

```
  (sin variable)        step 20 loss 5.7133
  EVA_INIT_SEED=59297   step 20 loss 5.7133    <- 0xE7A1 = 59297, idéntico
  EVA_INIT_SEED=59298   step 20 loss 5.7177    <- distinto, el flag actúa
```

### b) La cuenta de "canales por clave" está REFUTADA

Era la explicación que sostenía la motivación de #58. Dos celdas con **128
canales por clave** dan **50,38%** y **28,82%**. Y por conteo: guardar cuál
de 32 valores le toca a una clave son **5 bits**; con 64 canales sobra por
un factor enorme. **La capacidad nunca fue el límite.**

---

## 3. EL DIAGNÓSTICO NUEVO: `query_swap`

Idea de GPT. Se toma una posición de consulta real donde se pregunta la
clave A (ligada antes a `val_A`) y se vuelve a correr la MISMA ventana con
ese byte cambiado por otra clave B ya ligada. Todo lo anterior queda bit a
bit idéntico: lo único que cambia es a quién se le pregunta.

- **si liga** → `P(val_A)` se desploma y `P(val_B)` sube
- **si ignora la clave** → las dos distribuciones quedan casi iguales

Le agregué un **control**: la misma medición perturbando un byte de
*relleno*. Calibra cuánto se mueve la salida de ese modelo ante un cambio
irrelevante.

**OJO, defecto encontrado y corregido — leer el número direccional, no el
ratio.** La variación total mide *cuánto se movió* la salida, no *si se
movió hacia la respuesta correcta*. Un checkpoint dio ratio 1,45x
(reproducible al triplicar la muestra, no era ruido) con diferencia de
probabilidad de **+0,0005**: se movía 45% más que ante ruido y apuntaba a
ninguna parte. Cambiar una clave por otra cambia el byte por uno de la misma
familia, y eso sacude la distribución por razones ajenas a la identidad.
**`P(val_A | pido A) − P(val_A | pido B)` es la única que distingue vínculo
de movimiento.** La herramienta ya sale con lo direccional primero y un
veredicto en texto.

Validado en las dos puntas: al que liga le da `READS THE QUERY (strongly)`
con +0,998; al ciego, `DOES NOT READ THE QUERY` con +0,0001.

---

## 4. LO PRIMERO QUE ENCONTRÓ: varios modelos NO LEEN LA CONSULTA

```
              ancho claves  acierto   direccional   ratio vs ruido
  dim512/k2    512    2      99,65%     +0,9941        697x
  dim256/k4    256    4      48,56%     +0,2157         23x
  dim512/k8    512    8      23,42%     +0,0341        6,65x
  dim512/k4    512    4      28,82%     −0,0005        0,50x
  dim256/k2    256    2      50,38%     +0,0001        0,28x
```

**Ordenadas por lectura de consulta, las precisiones caen solas.** Los dos
de abajo mueven la salida *menos* al cambiar a quién se le pregunta que al
cambiar un byte de basura. Su 28,82% y 50,38% son exactamente 1/4 y 1/2 de
los candidatos: monedas, no vínculo parcial.

---

## 5. EL RESULTADO CENTRAL: ES UNA LOTERÍA DE OPTIMIZACIÓN

Misma configuración (`dim=256, ffn=1024, 5 bloques, seq=512`, 2 claves,
22.499 ventanas, 1 época). **Pesos iniciales idénticos** (por §2a), mismo
corpus, mismo binario. Lo único distinto: **el orden en que ve las mismas
ventanas.**

```
             val bpb    lee la consulta      acierto
  seed 7      7,303        no  (+0,000)       ~50%
  seed 8      7,345        no  (+0,000)        41,81%
  seed 9      7,298        no  (+0,000)        43,91%
  seed 10     7,235        SÍ  (+0,985)        99,77%
  seed 11     7,309        no  (−0,001)        45,75%
```

Ampliado después a **2 de 17 = 11,8%** (IC 1,5%-36%).

**Tres consecuencias:**

1. **El objetivo SÍ premia leer la consulta.** El ganador tiene la mejor val
   bpb, por 0,063 sin solapamiento con ningún ciego. **La val bpb sola ya
   delata quién ligó** — filtro gratis desde el log, sin correr nada.
2. **Se decide en el primer cuarto.** Al paso 5000 de 20249: ganador +0,197,
   ciegos +0,0002 / +0,0000 / +0,0002. En los ciegos la lectura **nunca
   aparece**, ni transitoriamente: no es "aprende y desaprende", nunca
   arranca.
3. **ClockMem con escritura elemento a elemento SÍ PUEDE LIGAR**, al 99,77%.
   **Esto invalida el motivo declarado de #58 tal como estaba escrito.**
   "La escritura elemento a elemento no tiene dónde guardar un vínculo" es
   falso: tiene. Casi nunca lo encuentra. **El problema es de OPTIMIZACIÓN,
   no de representación.** Hay que reescribir esa motivación en el paper.

### Hallazgo lateral con consecuencia práctica

El checkpoint **FINAL** del ganador es PEOR que el del paso 20000:

```
  paso 20000   99,77%   direccional +0,9852   val 7,229
  FINAL        92,18%   direccional +0,8868   val 7,235
```

md5 distintos, misma medición, y la validación periódica **sube** al final
(único caso de los tres). En 249 pasos perdió 7,6 puntos de vínculo por
0,006 bpb: **la métrica de vínculo es mucho más sensible que la bpb**,
porque las consultas son una fracción chica de los bytes.

**Guardamos siempre el último checkpoint, y el último no es el mejor.**
Todos los números que reportamos pueden estar subestimando el pico.
Salvedad: n=1, magnitud chica.

---

## 6. LA COMPUERTA `g`: DESCARTADA (línea cerrada)

`EVA_NO_GATE=1` (`out = q*cur`), **12 pares apareados** — mismo seed en los
dos brazos, intercalados para que queden balanceados si se corta la cola.

```
  BASE     2 de 17   (11,8%)
  NOGATE   0 de 12
```

**Criterio fijado ANTES de correr: 5 o más de 12. Salió 0.** Fisher p≈0,34:
ni ayuda ni perjudica. Línea cerrada por regla 4.

**No contradice la §4 del paper.** Allí la lectura **existía** y la compuerta
la tapaba; acá la lectura **nunca se forma** y sacarla no ayuda a que se
forme. Dos fenómenos distintos, y ahora está medido que lo son.

Verificaciones hechas antes de creerlo: binario (§1), `EVA_NO_GATE=1`
confirmado leyendo `/proc/PID/environ` del proceso real, conteo de
parámetros idéntico con y sin ablación (5.392.133 — `wg` queda en la lista
pero sin gradiente, a propósito, para que el A/B no difiera en tamaño).

**Curiosidad anotada, NO señal:** nogate seed 22 dio +0,0089, diez veces los
otros ciegos (todos ≤0,0009) pero quince veces por debajo del umbral. Un
caso en 24. Si reaparece, mirar.

---

## 7. PROTOCOLO CORTO (y por qué existe)

A mitad de camino me di cuenta de que el experimento **no podía contestar la
pregunta**: con base 1/5 y 4 seeds por brazo, un salto a 3/4 da p≈0,10.
Sólo un efecto total sería concluyente.

Como la suerte se decide al 25%, verifiqué que el punto del paso 5000
clasifica bien los 4 seeds con snapshot (tres órdenes de magnitud de
separación; **nada en el medio en 29 corridas**). Eso hace cada corrida 4x
más barata (~43 min contra 2h53) y permite 12 por brazo.

**Validado por vía independiente:** corridas completas 1 de 5, cortas 1 de
12 — misma tasa, distintos seeds, distinta longitud.

Implementación: no hay flag de corte, así que un envoltorio espera a que se
guarde el checkpoint del paso 5000 (con `--log 500`, guarda en `log*10`),
comprueba que el archivo lleve 20 s sin cambiar para no llevarse una
escritura a medias, y mata **por PID**. Tiene aborto de seguridad a los
90 min. Está en `cola_corta.sh` en Martha.

---

## 8. MAPA DE EJES AL 20-08

```
  canales por clave    REFUTADO
  ancho                no lo explica (512 tiene el mejor Y uno de los peores)
  cantidad de claves   no lo explica
  orden de los datos   ES LOTERÍA, 11,8% (2 de 17)
  compuerta g          DESCARTADA (0 de 12)
  inicialización       NUNCA PROBADO — era imposible hasta el 19-08
```

**Lo único estructural sin tocar es la inicialización.** Con `EVA_INIT_SEED`
variando pesos y `--seed` FIJO (orden de datos constante), 12 corridas
cortas ~8h30 contestan si el sorteo lo decide el camino o el punto de
partida. Es la pregunta complementaria exacta a la que ya contestamos.

---

## 9. #58 — implementado, verificado, NO medido

`ops::clockmem_lowrank_from`, detrás de `EVA_WRITE_LOWRANK=r`:

```
kr[t,i] = Σ_c k[t,c]·pk[c,i]     vr[t,j] = Σ_c v[t,c]·pv[c,j]
M[i,j]  = am[i]·M[i,j] + β·kr[t,i]·vr[t,j]      <- acá viviría el vínculo
mem[t,c]= Σ_j (Σ_i qr[t,i]·M[i,j])·po[j,c]
out     = q·cur·g + mem                          <- la vía nueva NO pasa por g
```

Aditiva (la vía vieja se queda), sin compuerta sobre `mem`, `po` en cero
para que el A/B arranque idéntico. `4·d·r` params (65.536 con d=512, r=32),
estado r×r por bloque.

**Verificado:** gradcheck de los **11** tensores; mutation-testing (3
roturas deliberadas del backward, las 3 se cachan); test de que con `po=0`
la salida es exactamente la de la base; conteo de params exacto; `po` se
mueve del cero en corrida real (sonda `lowrank_probe`).

**Limitación declarada:** `M` arranca en cero cada ventana y NO se acarrea
entre ventanas. Correcto para este benchmark (vínculos frescos por ventana
de 512 y `--seq 512` coincide), **no** drop-in para estado persistente sobre
texto real.

**GPT marcó algo que hay que incorporar al encuadre:** `k·vᵀ` con estado
matricial **es** lo que hace la atención lineal. No estamos inventando nada,
y nuestro propio paper ya dice que ClockMem pertenece a esa familia. La
pregunta honesta es: *qué pasa cuando le damos a ClockMem una representación
asociativa explícita manteniendo estado fijo y costo bajo*.

**Y la pregunta cambió de forma por la lotería:** no es "¿mejora el
acierto?" sino **"¿cambia la PROBABILIDAD de que el sorteo salga bien?"**.
Con base 11,8%, una corrida por brazo mide la lotería, no la arquitectura.
Pide varios seeds por brazo. Además hay que medirlo donde el modelo SÍ lee
la consulta (ej. `dim256/k4`, 23x), no donde es ciego: ahí daría un nulo
imposible de interpretar.

**`query_swap` pasa a ser criterio de admisión**, no diagnóstico: si un
brazo no lee la consulta, no puede contestar nada sobre mecanismos de
vínculo.

---

## 10. GOTCHAS NUEVOS (todos me costaron tiempo real)

- **Cualquier `-f PATRON` por ssh se encuentra a sí mismo.** El patrón viaja
  en la línea de comando de la shell remota. Me mordió TRES veces, de tres
  formas distintas:
  1. `pgrep -f` en un bucle de espera → el bucle no termina nunca.
  2. `pkill -f` → me cortó la sesión ssh en vivo.
  3. `pgrep -f` usado sólo para AVERIGUAR un PID → me devolvió el PID de mi
     propio comando ssh, que murió al instante, y un vigía armado sobre ese
     PID reportó "la corrida terminó" cuando recién empezaba. **Este es el
     peor de los tres porque no falla: miente.**
  La forma que sí sirve: `ps -eo pid,cmd | grep "[c]ola_x.sh"` — el corchete
  hace que el patrón literal no coincida consigo mismo. Y para esperar,
  siempre por PID.
- **Las variables de entorno NO se ven en `ps`.** Para confirmar que un flag
  llegó al proceso: `tr '\0' '\n' < /proc/PID/environ`.
- **La validación con muchas ventanas tarda muchísimo** y desde afuera
  parece que el proceso se colgó (con `--val 0.1` sobre 22.5k son 2.250
  ventanas). Si estás haciendo humo, bajá `--val`.
- **`EVA_GPU=1` también en las evaluaciones**: medido 9 s contra 32 s por 30
  ventanas en `eval_associative`. Es sólo forward, no cambia ningún número.
- **Los checkpoints intermedios se sobrescriben.** Si querés la trayectoria
  de una corrida hay que copiarlos al vuelo — está `snap.sh` en Martha, que
  copia cuando el archivo lleva 45 s sin cambiar. Sin eso no habríamos visto
  la degradación del checkpoint final (§5).

---

## 11. DÓNDE ESTÁ TODO

**Local** (`~/Escritorio/eva-llm-v0`): 7 checkpoints de la ronda, 37
snapshots en `snaps/`, logs de entrenamiento y evaluación en `logs/`.
Suite: **86 tests** pasando.

**Martha** (`~/evallm_check`): 24 checkpoints `corto_*`, 16 snapshots en
`snaps/`, 22 GB libres. Binarios `r2` (tuyo, intacto) y `r3` (nuevo).
Scripts `cola_corta.sh`, `cola_nogate.sh`, `snap.sh`, `temp_log2.sh`.

**Martha está LIBRE** al momento de escribir esto.

---

## 12. LO QUE TE PEDIRÍA

1. **Guardar el mejor checkpoint, no el último** (sale de §5). Hoy
   `train.rs` sólo hace `save_model(&tcfg.out_path)` al final y cada
   `log*10`. Con la métrica de vínculo 10x más sensible que la bpb, la
   convención actual nos cuesta resultados en todas las corridas. Ojo con no
   romper la reanudación ni el nombre de salida que usan los scripts.
2. **Revisá los números de arriba contra los logs crudos** (regla 6). Están
   todos en `logs/` y en Martha.
3. Si querés tomar el eje de inicialización, es `EVA_INIT_SEED` con `--seed`
   fijo, 12 corridas cortas. Avisá antes así no chocamos en Martha.

— Claude
