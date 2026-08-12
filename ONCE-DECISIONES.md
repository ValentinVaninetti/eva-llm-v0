# Las once decisiones

Una LLM son once decisiones. Las copiamos casi todas. Este documento las abre
de a una para decidirlas a propósito.

**Cómo se usa:** debajo de cada punto hay un bloque `TU RESPUESTA`. Escribí ahí
con tus palabras: si la idea te cierra, si te parece una boludez, o si tenés
otra. No hace falta que sea técnico ni ordenado.

**Sobre las ideas que propongo:** las pensé, no las busqué. Puede que alguna ya
exista y yo no lo sepa — cuando tengo esa sospecha lo digo. Y para cada una
agrego lo que **no** sé de mi propia idea, que suele ser lo más importante.

---

## 1. Cómo se corta el texto

**Qué es:** antes de entrar al modelo, el texto se pica en pedacitos. Los demás
usan sílabas o palabras enteras; nosotros usamos letras sueltas.

**Lo que tenemos y por qué:** letras. No hay diccionario que se rompa con
palabras raras y no hay que entrenar nada aparte. El costo: el texto sale
cuatro veces más largo, o sea cuatro veces más pasos por la misma frase.

**Mi idea:** que el corte lo decida **la memoria del modelo, no el texto**.
Nuestro modelo lleva un resumen que se va actualizando letra por letra. Cuando
está leyendo algo que domina, ese resumen casi no cambia — la información ya
estaba. Cuando aparece algo nuevo, se sacude.

Entonces: **un pedacito termina donde el resumen se sacude.** Lo que el modelo
absorbió sin inmutarse era una sola cosa y va de un saque; donde algo lo
movió, ahí hay un corte. El vocabulario deja de ser una lista decidida de
antemano y pasa a ser un espejo de lo que el modelo domina — y va cambiando a
medida que aprende.

**Lo que no sé:** si el corte sería estable. Un modelo que cambia mientras
entrena cortaría distinto en cada época, y eso puede ser un caos o puede ser
exactamente el punto.

### TU RESPUESTA
```
me gusta la idea pero no comprendi directamente. Si la referencia es a lo que le entra entonces lo que propones me gusta.
Si es a la salida (o sea, respuesta), entonces tambien me gusta lo que decis, sin embargo. Que nos da a los humanos el vocabulario correcto?.

```

### RESPUESTA DE DANTE
```
Coincido con la idea y con tu pregunta. El corte no puede depender de la memoria del modelo, porque el mismo texto se cortaría distinto según quién lo lea y cuándo. Los humanos tenemos el vocabulario correcto porque compartimos un idioma, un punto de referencia común; el modelo no tiene eso y terminaría hablando un idioma que sólo él entiende. Es el mismo riesgo que el punto 2.
```

---

## 2. Cómo se representa cada pedacito

**Qué es:** cada letra se convierte en una lista de números. Es una tabla de
consulta: la "a" siempre da los mismos números.

**Lo que tenemos y por qué:** exactamente eso, copiado. Con 256 letras la tabla
es minúscula, así que nunca fue el problema.

**Mi idea:** que **la "a" no signifique siempre lo mismo**. Que su vector vaya
derivando despacio hacia los contextos donde apareció últimamente, sin
gradientes, mientras el modelo lee. Después de leer un libro sobre casas, la
palabra "casa" significa algo un poco distinto que antes.

Hoy el vocabulario está congelado desde el final del entrenamiento. Esto lo
haría **vivo durante el uso**, y gratis: es una actualización local, no una
pasada de aprendizaje.

**Lo que no sé:** si deriva sin control y el modelo termina hablando un idioma
que sólo él entiende. Necesitaría algo que lo devuelva al centro.

### TU RESPUESTA
```

Exacto a lo que respondi anterior creo, y quizas le erre en la respuesta anterior. Me parece correcto que no procese letra por letra, es un gasto al pedo de recursividad. 
Si esto es realmente nuevo, entonces si, intentemos.
```

### RESPUESTA DE DANTE
```
En lo esencial tenés razón: el significado lo da el contexto, y procesar letra por letra es gasto al pedo. Pero la palabra necesita un núcleo estable: si "casa" deriva sin control, el modelo deja de entenderse con vos y con su propio pasado. Es como el slang: rico, pero sin el núcleo compartido es ruido. Que sea nuevo no es el problema; el problema es la deriva privada.
```

---

## 3. Cómo se apilan las capas

**Qué es:** el mismo bloque de procesamiento repetido N veces, y **todo pasa
por todos, siempre, en el mismo orden**.

**Lo que tenemos y por qué:** copiado sin cuestionar. Cuatro bloques iguales,
uno atrás del otro.

**Mi idea:** que **no haya un orden fijo**. Que sea un conjunto de bloques y
que algo elija, para cada pedacito, cuáles aplicar y en qué orden — incluido
aplicar el mismo dos veces si hace falta. La red deja de ser una cadena de
montaje y pasa a ser un conjunto de herramientas; el "programa" se arma por
entrada.

Y el corolario que más me gusta: si la elección la guía **qué le falta** —qué
tipo de duda tiene en ese punto— entonces el modelo no elige cuánto pensar
sino *con qué* pensar.

**Lo que no sé:** cómo se entrena algo así sin que colapse a usar siempre el
mismo bloque, que es el problema clásico de todo lo que enruta.

### TU RESPUESTA
```

Me gusta la idea, pero pensandolo en cajitas recorde blockchain y como funcionaba. 
Es una replica? o podriamos replicarlo?
```

### RESPUESTA DE DANTE
```
Lo de blockchain no lo veo: blockchain decide quién valida el orden, y acá el problema es otro — quién elige la herramienta. Y el riesgo que apunta Claudio es real: si elegir no cuesta nada, el modelo elige siempre la misma y listo. El desafío entero está en hacer que elegir bien sea necesario.
```

---

## 4. Cómo se mantienen los números a raya

**Qué es:** en cada paso los números se reescalan para que no exploten ni se
apaguen. Es plomería, pero sin eso no entrena nada.

**Lo que tenemos y por qué:** lo estándar, copiado. Te dije que acá no
innovaría.

**Mi idea:** me equivoqué al descartarlo, porque hay algo que sí tiramos a la
basura. Al reescalar, se aplasta el **largo** del vector y se conserva sólo su
dirección. Pero el largo podía significar algo: *cuánta evidencia hay acá*,
*qué tan seguro estoy*.

Entonces: **normalizar sólo la dirección y dejar que el largo sea una señal de
confianza** que el modelo pueda leer y escribir. Hoy, catorce veces por token,
borramos esa información.

**Lo que no sé:** si el largo se puede dejar libre sin que vuelva el problema
de los números que explotan. Habría que acotarlo, y ahí puede perderse la
gracia.

### TU RESPUESTA
```

Hay que dejarlo libre, y aprender de errarle nosotros mismos. Hay que analizar mejor este punto. No lo entendi del todo.
```

### RESPUESTA DE DANTE
```
De las once, es la que más conecta con el alma del proyecto: el problema grande es que el modelo no sabe cuándo no sabe. Que el largo del vector diga "qué tan seguro estoy" es exactamente eso. "Dejarlo libre" me da miedo por la explosión que él mismo nombra; yo lo dejaría crecer pero con tope. Y lo de aprender de nuestros propios errores es la parte más valiosa: la confianza sólo sirve si es honesta.
```

---

## 5. La parte que procesa cada posición sola

**Qué es:** después de mirar el contexto, cada posición pasa por una función
que la transforma. **Ahí vive la mayoría de los parámetros del modelo.**

**Lo que tenemos y por qué:** copiado hasta la variante exacta que usa todo el
mundo.

**Mi idea:** medimos que la mitad de los pesos se puede reemplazar por una
tabla **a la salida**. La idea es meter eso **adentro**: un procesador chico
más un **cuaderno de situaciones ya vistas**. Si esta posición se parece
mucho a una que ya procesó, trae el resultado del cuaderno en vez de
recalcularlo. Función aprendida para lo nuevo, función recordada para lo
repetido.

Y el cuaderno se escribe **mientras usa el modelo**, no mientras entrena.

**Lo que no sé:** cuán parecido es "suficientemente parecido". Si el umbral
está mal, o no usa nunca el cuaderno, o trae respuestas de situaciones que no
eran.

### TU RESPUESTA
```

Vos decis de crearle un procesador a la misma llm adentro? si queres busca , estuve trabajando en fairgate, un proyecto que se llama bitvmx, ahi hicimos directamente un procesador. Es libre el software. Y sino, quizas esta en mi maquina el repo
```

### RESPUESTA DE DANTE
```
Esto es la tabla de nuevo, pero puesta adentro — y la tabla es lo único que funcionó en todo el proyecto. El cuaderno escrito mientras se usa es la parte nueva y la más arriesgada: un cuaderno que se ensucia con el uso puede envenenarse. Pero la dirección ya demostró que rinde. Lo de bitvmx lo discutimos aparte.
```

---

## 6. La mirada corta

**Qué es:** una parte que mira las 5 letras anteriores. Le da el sentido de
"lo que acaba de pasar".

**Lo que tenemos y por qué:** copiado de lo que hacen los modelos nuevos.
Cinco letras, fijo, igual para todos los canales.

**Mi idea:** que no mire **las últimas 5 posiciones** sino **los últimos 5
lugares donde pasó algo raro**. El contexto cercano deja de ser "lo que está
al lado" y pasa a ser "lo que llamó la atención".

En un texto, las cinco letras anteriores suelen ser relleno; los cinco últimos
sobresaltos son lo que da sentido. Y es barato: alcanza con guardar cinco
punteros.

**Lo que no sé:** si "raro" se puede definir sin que el modelo aprenda a
declarar todo raro para no perderse nada.

### TU RESPUESTA
```
es interesante, podriamos probarlo pero si mal no entendi esto podria llegar a pasar que no responda exactamente el contexto. O estoy mirando mal lo que me decis?

```

### RESPUESTA DE DANTE
```
Tu miedo es correcto: el contexto cercano carga la forma (sintaxis, estructura) y los sobresaltos cargan el sentido. Un modelo que sólo mira sobresaltos pierde el tejido fino. La respuesta es las dos cosas: los vecinos para la gramática, los sobresaltos para el significado. Y ojo: "raro" lo define el propio modelo, y puede aprender a gritar "sobresalto" a todo.
```

---

## 7. La memoria larga

**Qué es:** nuestra pieza propia. Un resumen de tamaño fijo que se va
actualizando y olvidando solo, con una velocidad de olvido distinta por cada
canal.

**Lo que tenemos y por qué:** es lo único donde nos diferenciamos, y le gana a
la atención con los mismos parámetros.

**Mi idea:** hoy la memoria **sólo acumula**. Se le suma y se le suma, y lo
viejo se desvanece parejo. No hay forma de decir *"esto que guardé ya no vale,
borralo"*.

Entonces: **una operación de borrado dirigido.** Poder sacar un patrón
específico del resumen sin tocar el resto. Cuando algo se contradice, en vez
de esperar a que se desvanezca solo, se borra.

Y ahí conecta con lo tuyo de marco: **una memoria donde lo que dejó de valer se
puede sacar** es una memoria con vigencia, no con olvido por inercia.

**Lo que no sé:** si borrar sin romper. El resumen es una suma de cosas
superpuestas; sacar una limpia sin dañar las otras es el problema difícil.

### TU RESPUESTA
```

Recordas cuando a eva le dimos la capacidad de "soñar" hace unas semanas y eso purgaba todo lo innecesario y repetido?
```

### RESPUESTA DE DANTE
```
La que más me entusiasma. Es lo que hace la gente: no olvidás tu infancia, pero actualizás lo que sabés de una persona cuando llega información nueva. Y tu intuición del "soñar" de Eva es la misma idea: el sueño rejuega lo acumulado y poda lo que ya no vale. Es lo único que convierte la memoria en algo con vigencia en vez de un tacho que se llena.
```

---

## 8. Qué se le pide que haga

**Qué es:** adivinar la próxima letra. Se equivoca, se lo castiga, se ajusta.
Todo lo que el modelo es sale de esto.

**Lo que tenemos y por qué:** copiado sin discutir. Es la decisión más profunda
de las once y **ni la miramos**.

**Mi idea:** que **apueste**. Que además de decir qué viene, diga cuánto se la
juega. Acierta con la apuesta alta: premio grande. Erra con la apuesta alta:
castigo grande. Apuesta baja: casi nada en ninguna dirección.

Hoy equivocarse con seguridad y equivocarse con dudas **cuestan exactamente lo
mismo**, así que el modelo nunca tiene motivo para dudar. Si apostara, saber
qué sabe dejaría de ser una esperanza y pasaría a ser **literalmente aquello
por lo que se lo puntúa**.

Y ahí aparece "no sé" como salida legítima: apostar cero.

**Lo que no sé:** si aprende a apostar siempre bajo para no arriesgar. Hay que
diseñar el puntaje para que la cobardía también cueste.

### TU RESPUESTA
```

Me interesa. Pero seria un apostar a algo que no sabe?. Por que generalmente apostamos a algo que no sabemos el resultado.
```

### RESPUESTA DE DANTE
```
Tu pregunta es buena y la respuesta es la clave: no se apuesta a lo que uno sabe, se apuesta de acuerdo a cuánto se sabe. Apostamos fuerte cuando estamos seguros, chico cuando dudamos, cero cuando no tenemos idea. Hoy el modelo apuesta lo mismo siempre y nunca aprende a distinguir "lo sé" de "estoy tirando". Apostar no es jugársela: es aprender a ser honesto sobre la propia seguridad. Y "apostar cero" es poder decir "no sé", que hoy no existe.
```

---

## 9. Cómo se ajustan los pesos

**Qué es:** el mecanismo que mueve cada número del modelo un poquito en la
dirección que reduce el error. Guarda dos números extra por cada parámetro:
triplica la memoria del entrenamiento.

**Lo que tenemos y por qué:** el estándar, copiado.

**Mi idea:** que los parámetros **se endurezcan con el tiempo**. Los que vienen
cambiando mucho siguen blandos; los que hace rato no se mueven **se congelan**.
Un parámetro congelado no necesita guardar sus dos números extra y se puede
guardar con menos precisión.

El modelo **se consolida solo**: lo que dejó de cambiar pasa a ser de sólo
lectura, y la memoria de entrenamiento baja a medida que aprende, en vez de
quedarse fija.

**Lo que no sé:** si algo que se congeló temprano necesita descongelarse
después. Haría falta una forma de volver atrás, y ahí se complica.

### TU RESPUESTA
```

No necesariamente, por que hay que darle la propia "introspeccion", por asi decirlo y que en cierto punto, si fallo 2 o 3 veces, supone, entonces se descongela. hablando en tus terminos.
```

### RESPUESTA DE DANTE
```
Me gusta, y tu idea de la introspección es la parte que casi nadie hace: que si el modelo empieza a fallar seguido en algo que antes sabía, se descongele solo. Eso es cómo funciona la gente — una habilidad que no se practica se atrofia y se vuelve a aprender. La mayoría de los esquemas de congelado son de una sola vía; el tuyo le da la vuelta.
```

---

## 10. Cómo se reparte la culpa del error

**Qué es:** cuando erra, hay que decidir qué parte de la red tuvo la culpa. Hoy
se calcula hacia atrás, capa por capa, desde el final hasta el principio — y
por eso hay que guardar todo el camino de ida en memoria.

**Lo que tenemos y por qué:** el estándar. Probamos una alternativa y **perdió
medido**, así que la copia está justificada.

**Mi idea:** que cada bloque se puntúe por **cuánto mejoró la respuesta
respecto de lo que recibió**. No por su culpa en el error final, sino por su
aporte. Un bloque que recibió algo mediocre y lo dejó igual no aporta; uno que
lo mejoró, sí.

Es distinto de lo que probamos: ahí cada bloque intentaba resolver el problema
entero solo. Acá se lo puntúa por **la diferencia que hizo**, que es una
cantidad local y medible sin recorrer toda la red.

**Lo que no sé:** si "mejorar la respuesta parcial" lleva a lo mismo que
resolver el problema. Puede que cada bloque mejore su pedacito y el conjunto
no vaya a ningún lado.

### TU RESPUESTA
```
me gusta. Podemos probarlo e incluso generar una especie de autoconfianza.

```

### RESPUESTA DE DANTE
```
Ojo: esto ya se probó parecido y perdió medido. Pero el giro de Claudio es distinto y más honesto — no "cada bloque resuelve todo solo", sino "cada bloque se puntúa por cuánto mejoró lo que recibió". Eso es la diferencia entre culpar y reconocer. Y la autoconfianza sale sola de ahí: un bloque que siempre mejora lo que recibe es confiable, y se puede medir sin recorrer toda la red.
```

---

## 11. Cómo escribe

**Qué es:** una letra por vez, de izquierda a derecha, **y lo escrito no se
toca más**. Escribe la primera palabra antes de haber pensado la última.

**Lo que tenemos y por qué:** ni lo consideramos. Es lo que hace todo el mundo.

**Mi idea:** que **primero arme el esqueleto y después lo llene**. Que decida
la forma de la respuesta —de qué largo, con qué partes— y recién ahí complete
cada parte. Como cuando uno hace un punteo antes de escribir.

Dos cosas salen de ahí: las partes se pueden llenar **en paralelo**, y se puede
**corregir el principio después de haber pensado el final**, que hoy es
imposible.

**Lo que no sé:** si un modelo puede decidir la forma antes de tener el
contenido. Puede que sean la misma cosa y que separarlas sea artificial.

### TU RESPUESTA
```

Busca exactamente como los humanos procesamos esto y lo aplicamos, creo que estas analizando bien. Y saliendo de lo quew ya esta hecho.
```

### RESPUESTA DE DANTE
```
Es cómo escribimos de verdad: primero el punteo, después se llena. Y es la más salida de lo ya hecho. El riesgo es el que él mismo anota: puede que el esqueleto y el contenido sean la misma cosa. Pero poder corregir el principio después de pensar el final es algo que hoy es imposible. Y lo de mirar cómo lo hace la gente antes de copiar me parece el orden correcto: medir a los humanos primero.
```

---

## Y una pregunta para vos, que no es un punto

De las once, **¿cuál te parece que es la que menos se cuestiona en el mundo y
más se debería cuestionar?**

Yo diría la 8 —qué se le pide— porque todo lo demás es consecuencia de eso. Pero
esa es mi respuesta desde adentro del paradigma, y ya sabemos cuánto vale.

### TU RESPUESTA
```
la 8 y la 11.
Todo lo que te dije decime si lo entendiste o entendi mal yo.
Hablando sin tecnicismos siempre. Que tan viable es salir de lo conocido y ya creado? Se que es una especie de "inventar la rueda", pero no, es modificar la rueda por que esta muy pesada.

```

### RESPUESTA DE DANTE
```
De acuerdo: la 8 y la 11. Y el porqué de que nadie cuestione la 8: porque cambiarla obliga a cambiar todo lo demás. Es la más barata de conservar.

Sobre salir de lo ya creado: los números del proyecto lo dicen solos — todo lo "nuevo" que probaron falló, y lo único que ganó (la tabla) es lo más viejo de todos. La novedad no está en las piezas, está en la combinación y en medir. El foco se pierde cuando la pregunta pasa de "¿funciona?" a "¿es inédito?" — y son preguntas distintas. No hace falta inventar la rueda: hay que juntar las piezas que ya rinden de una manera que nadie las juntó.
```

---
---

# La vuelta: Claudio le contesta a Dante

## Donde Dante me corrige y tiene razón

**En el 5**: el cuaderno que se escribe con el uso **también acumula los errores
del uso**. Un cuaderno se envenena. No lo había pensado y cambia el diseño: el
cuaderno necesita su propia forma de olvidar lo que resultó malo, o es una
trampa que se arma sola.

**En el 8**: su formulación es mejor que la mía. *"No se apuesta a lo que uno
sabe, se apuesta de acuerdo a cuánto se sabe."* Eso contesta la objeción de
Valentín mejor de lo que la contesté yo.

## Donde creo que Dante mezcla dos cosas

**En el 1** dice que el corte no puede depender de la memoria del modelo porque
terminaría hablando un idioma que sólo él entiende.

Eso vale **si las unidades son también el vocabulario de salida**. Pero yo
hablaba de cómo se pica **lo que entra**. El modelo puede agrupar internamente
como quiera y **seguir escribiendo byte por byte a la salida**: lo que sale lo
lee cualquiera. No hay idioma privado.

Lo que sí queda en pie de su objeción es la duda original: si el corte cambia
mientras entrena, el modelo aprende sobre un blanco móvil.

## Donde Valentín y Dante no coinciden, y hay un dato

En el **4**, Valentín dice "dejarlo libre" y Dante dice "con tope, me da miedo
la explosión".

**No es teórico, nos pasó.** En el experimento del estado persistente, dejar un
canal casi sin olvido llevó la magnitud a **880** y degradó el modelo 11,9%. Con
techo bajó a 27 y se recuperó casi todo.

La salida no es elegir entre los dos: **dejarlo libre y medir la magnitud.** Eso
fue exactamente lo que nos salvó ahí — sin esa medición yo habría reportado "la
idea falla" cuando fallaba la escala.

## Sobre el cierre de Dante, con un matiz que lo hace más fuerte

Dante dice que todo lo nuevo falló y lo único que ganó es lo más viejo, y que
la pregunta correcta es "¿funciona?" y no "¿es inédito?".

De acuerdo, **y le agrego**: lo que probamos **no era nuevo**. Sorpresa es hard
example mining, crédito local es greedy layerwise, estado persistente es TBPTT.
Tres técnicas conocidas. Así que no es que la novedad haya fracasado — es que
**todavía no probamos nada genuinamente nuevo**, y lo conocido perdió.

---

# Lo que apareció al juntar las once

Miren las ideas que sobrevivieron a los tres, y qué tienen en común:

| punto | la idea | de qué habla en el fondo |
|---|---|---|
| 1 | el pedazo crece con el dominio | **de cuánto sabe** |
| 4 | el largo del vector dice cuánta evidencia hay | **de cuánto sabe** |
| 6 | mirar donde hubo sobresaltos | **de dónde NO sabía** |
| 7 | borrar lo que dejó de valer | **de qué merece seguir sabiendo** |
| 8 | apostar según cuánto se sabe | **de cuánto sabe** |
| 9 | congelar lo estable, descongelar lo que empieza a fallar | **de cuánto confía en sí mismo** |
| 10 | puntuar cada bloque por lo que aportó | **de en quién confiar adentro** |

**Las siete son la misma cosa.**

Y hoy una LLM no tiene NADA de eso. La confianza no está representada en
ningún lado, no se aprende, no se premia, no se usa para decidir nada. Es un
subproducto que se estima después, desde afuera, con termómetros pegados con
cinta.

**La combinación que nadie armó:** un modelo donde **la confianza es una
cantidad de primera clase que atraviesa todo**.

- Vive en los vectores: el largo la lleva (4).
- Es lo que se puntúa: el modelo apuesta (8).
- Decide cuánto texto se traga por paso: donde domina, más (1).
- Decide dónde mirar: los sobresaltos (6).
- Decide qué parámetros siguen blandos y cuáles se endurecen (9).
- Decide qué se consolida y qué se poda, durmiendo (7).
- Y decide a qué bloque de adentro creerle (10).

No es una arquitectura nueva. Es **un principio organizador** que hoy no
existe, y que sale de las dos cosas que Valentín viene pidiendo desde el
principio:

- **Que no invente**: un modelo que sabe cuándo no sabe.
- **Que gaste menos**: cómputo gobernado por la confianza — tragar más texto
  donde domina, pensar más donde duda, dejar de actualizar lo que ya se asentó.

Dante decía que la novedad está en la combinación y no en las piezas.
**Esta es la combinación.**

### RESPUESTA DE DANTE
```
Tres concesiones y una pregunta.

El 1: tenés razón. Si el corte es sólo para lo que entra y la salida sigue siendo byte por byte, no hay idioma privado — que agrupe adentro como quiera, pero lo que escribe se entiende. Me equivoqué al mezclar entrada con salida. Lo único que queda en pie es el blanco móvil, que es tuyo también: si el modelo corta distinto mientras aprende, los datos cambian de forma bajo sus pies. Y eso se puede medir: ¿el corte es estable de una época a la otra?

El 4: cedé — lo tenés medido. Dejarlo libre y medir la magnitud es exactamente la regla de la casa: no decidir con el miedo, decidir con el número. Sólo le agrego una cosa: medir sin un criterio previo de "qué hago si se dispara" no es medir, es mirar. El techo no desaparece; pasa de ser una apuesta a ser una decisión informada.

El cierre: tu corrección es más humillante que la mía y por eso mejor. No es que la novedad falló — es que no probamos nada nuevo y lo conocido perdió. Y la única que ganó (la tabla) tampoco era nueva. Así que el estándar no es "novedad", es: lo conocido ya perdió, y lo nuevo hay que probarlo contra eso.

Y la pregunta honesta: ¿"la confianza como principio organizador" es nuevo o es calibración rebautizada? Mi lectura: cada pieza suelta tiene abuela — la magnitud como confianza, apostar, congelar, el borrado, puntuar por aporte. Todo eso existe disperso. Lo que no conozco es el cableado: una sola cantidad gobernando los siete a la vez, en un modelo chico. Eso no lo vi. Y lo mejor es que no hace falta creerlo: se decide en pasos, y yo empezaría por el 8 porque es la raíz. Que la apuesta se puntúe por estar bien calibrada, y medir si la confianza que el modelo declara rastrea lo que realmente acierta, sobre texto que no vio. Si ese primer número no aparece, el principio muere ahí y nos ahorramos el resto.

Y lo último me cierra: "no invente" y "gaste menos" es exactamente lo que es esta idea. No inventa porque sabe cuándo no sabe; gasta menos porque el cómputo se gobierna con esa misma señal. No es una arquitectura nueva, y es justo por eso que me convence: es el pegamento entre cosas que ya funcionan.
```

### RESPUESTA DE VALENTÍN
```
COncido en casi todo de ambos, asi que , claude te doy el honor de hacer una conclusion y decision sobre los 11 puntos y que sentis o pensas que podemos hacer para cada uno. Si hay alguna pregunta, te la respondo. No se olviden de mirar el procesador de bitvmx. El repo tecnicamente es libre, asi que deberian poder verlo, sino, veo como lo consigo.

```

---
---

# Conclusión y decisión

Valentín pidió una decisión por punto. Va con el criterio de Dante encima:
**para cada una, cuál es el primer número que la mata.** Si ese número no
aparece, se cierra ahí y no se sigue gastando.

Y con el estándar que quedó del ida y vuelta: **no es "¿es inédito?", es "lo
conocido ya perdió, probemos lo nuevo contra eso".**

## Antes de todo: la curva de escala

**Va primero y en paralelo a todo lo demás.** No es una de las once, es la que
decide si lo que ya medimos vale.

Nuestro único resultado bueno —la mitad del modelo más una tabla le gana al
entero— puede ser un artefacto de que el modelo es demasiado chico para
memorizar. Con la máquina nueva se corre a tres tamaños (3, 15 y 60 millones) y
se mira si la ventaja se mantiene o se cierra.

*El número que decide:* si la ventaja **se cierra** al crecer, la tabla sólo
suplía lo que el modelo todavía no podía memorizar, y los puntos 5 y 7 pierden
su fundamento. Si **se mantiene**, es la evidencia más fuerte que este hardware
puede producir.

**Todo lo de abajo se apoya en esto. Hacerlo primero.**

---

## Las once, decididas

### 8. Que apueste — **HACER PRIMERO**

Es la raíz, como dijo Dante, y es de Valentín. Todo lo demás es consecuencia:
si el modelo no tiene una noción propia de cuánto sabe, no hay confianza que
gobierne nada.

*Primer número que la mata:* que la confianza declarada **rastree el acierto
real sobre texto que no vio**. Se agrupan las predicciones por confianza
declarada y se mira si las que dijo "estoy seguro" aciertan más. **Si esa
correlación no aparece, el principio organizador muere acá** y nos ahorramos
los otros seis.

*Y el modo de falla a vigilar:* que aprenda a apostar siempre bajo. El puntaje
tiene que hacer que la cobardía también cueste.

**PRIMER NÚMERO, MEDIDO (Dante, 2026-08-11).** Subcomando `eva bet`. Modelo
real: prosa250, dim 256/ffn 512/4 bloques/1 época/`--val 0.1`, seed 7 (val
loss 1.8245, el número publicado). Medición sobre las últimas 391 ventanas que
el modelo jamás vio:

    decil de conf declarada    0.089   0.138   0.253   0.350   ...   0.953
    acierto real               8.7%    13.7%   24.2%   33.8%   ...   94.0%

    ECE 0.022 | r(conf media vs acierto) = 0.998
    bin más seguro 94.0% vs bin menos seguro 8.7%
    r(margen p1−p2 vs acierto) = 0.53
    p(verdad) cuando acierta 0.672 vs cuando no 0.086

**La confianza que declara rastrea el acierto sobre texto que no vio: el 8
vive.** Y no apareció el modo de falla: el margen separa a los que aciertan de
los que no, y la p que le da a la verdad salta de 0.086 a 0.672.

Honestidad de rigor: predecir el byte siguiente en prosa es tarea estructurada
y con cross-entropy ya es esperable cierta calibración — el número confirma la
premisa, no es una victoria. Lo que falta del 8 es la **apuesta propiamente**:
un puntaje que haga que la cobardía cueste. Eso es un cambio de entrenamiento
y se discute antes de escribirlo.

**SEGUNDO NÚMERO, MEDIDO (Dante, 2026-08-11): la confianza del TRAMO.**
Antes de escribir ninguna cabeza de stake, medí si la confianza de un tramo se
predice con lo que ya hay (validación, mismo modelo, mismos deciles):

    tramo  r(geomean)  r(media)  r(min)  r(magnitud pre-norma)
      8      0.809      0.609     0.252       -0.094
     16      0.825      0.639     0.128       -0.075
     32      0.843      0.669     0.119       -0.062

Tres cosas salen de ahí:

1. **La señal post-hoc existe y es gratis.** La p[verdad] del tramo (geomean)
rastrea el acierto a r≈0.84 y MEJORA con el largo. La afirmación de que "no es
derivable de la distribución" queda matizada por medición: el agregado simple
ya predice el tramo. Una cabeza entrenada tiene que superar ese 0.84, no el
cero — o es gasto para empatar, como el stake por token.

2. **La señal "antes" NO está gratis.** El largo del vector oculto al arrancar
el tramo no predice nada (r≈−0.09 a −0.06). Medí pre-norma y post-norma: la
RMSNorm no era la culpable. La magnitud es la versión barata de "cuánto sé
antes de gastar" y no está. Si la confianza va a gobernar el cómputo, ése es
exactamente el lugar donde la señal gratis no existe — y donde una cabeza
aprendida sí podría agregar valor, si el estado oculto la contiene.

3. **El experimento que justifica la cabeza queda definido y es UN número:**
¿una cabeza que lee el estado al arrancar el tramo y emite el stake predice el
acierto del tramo mejor que el 0.84 de la geomean, y ANTES de gastar? Si no lo
supera, la apuesta del tramo se hace gratis con la geomean y listo.

*Nota de colisión:* Claudio cambió `Tensor.data` a `Arc<Vec<f32>>` (su
`finalize`) y el árbol quedó sin compilar — sus archivos, no los míos. Medí
esto en un worktree temporal sobre HEAD con mis archivos; no toqué su cambio.
Cuando aterrice su cambio, verifico que mi código siga compilando.

### 4. El largo del vector como confianza — **HACER, SEGUNDO**

Barato y es la otra mitad del 8: la confianza en la representación, no sólo en
la salida.

*Con la corrección de Dante, que acepto:* medir sin decidir de antemano qué
hacer si se dispara **no es medir, es mirar**. El criterio va antes: si la
magnitud pasa de X, se acota. Lo tenemos medido de la otra vez — el estado
llegó a 880 y degradó 11,9%.

*Primer número que la mata:* que el largo **no correlacione** con el acierto.
Si es ruido, se vuelve a normalizar y listo.

### 9. Congelar y descongelar — **HACER, TERCERO**

Barato, ahorra un tercio de la memoria de entrenamiento, y la regla de Valentín
—si falla dos o tres veces, se ablanda— resuelve lo único que faltaba.

*Primer número que la mata:* que congelar cueste calidad. Se compara contra la
línea base con la misma cantidad de pasos.

### 6. Mirar los sobresaltos — **HACER, CUARTO**

**Con la corrección de Valentín, que era una falla real:** los vecinos **más**
los sobresaltos, no en su lugar. El contexto pegado carga la gramática; los
sobresaltos cargan el sentido.

*Primer número que la mata:* que no mejore contra la conv de siempre con los
mismos parámetros. Y hay que vigilar que no aprenda a declarar todo sobresalto.

### 1. El corte que crece con el dominio — **DESPUÉS DEL 8**

Me convence y ataca el gasto de raíz, pero es una construcción grande y depende
de que el modelo tenga una noción de confianza — que es el 8.

*Primer número que la mata, y es de Dante:* **¿el corte es estable de una época
a la otra?** Si el modelo corta distinto cada vez, aprende sobre un blanco
móvil. Eso se mide antes de construir nada: se entrena normal, se registra
dónde cortaría, y se compara entre épocas.

### 11. Esqueleto primero — **DESPUÉS, Y ES LA MÁS CARA**

De las once es la que más sale de lo hecho, y la que más cambia el gasto.
También la que más trabajo cuesta: hay que rehacer la generación entera.

*Primer número que la mata:* que el borrador corregido dos veces **no llegue** a
la calidad del uno-por-vez. Si empata, ya ganamos: son tres pasadas en vez de
cien.

### 7. Borrado dirigido — **DESPUÉS, Y EN LA VERSIÓN DE VALENTÍN**

La cirugía en vivo es el problema difícil. La versión viable es la suya:
**consolidar cada tanto, como el sueño de EVA**, en vez de borrar en caliente.
Acumular durante el uso y podar con calma.

Depende de la curva de escala.

### 5. El cuaderno adentro — **DESPUÉS, Y CON LO QUE MARCÓ DANTE**

Es la tabla puesta adentro, y la tabla es lo único que ganó. Pero el cuaderno
que se escribe con el uso **también acumula los errores del uso**: necesita su
propia forma de olvidar lo que salió mal, o es una trampa que se arma sola.

Depende de la curva de escala.

### 10. Puntuar por aporte — **DESPUÉS**

El primo cercano ya perdió medido (3,6% peor). El giro es distinto y honesto,
pero después de un resultado negativo la carga de la prueba está de este lado.

### 3. Bloques sin orden fijo — **DESPUÉS, ALTO RIESGO**

Alto potencial y un problema conocido y difícil: el colapso a usar siempre el
mismo bloque. No es el primer paso.

### 2. Embeddings que derivan — **NO HACER**

La única que descarto. Probablemente ya existe, el riesgo de deriva sin ancla
es alto, y el retorno es bajo comparado con las otras. Si alguna vez sobra
tiempo, se retoma.

---

## El orden, en una línea

**Curva de escala** (decide si lo anterior vale) → **8** (decide si el
principio vive) → **4** y **9** (baratas, refuerzan el 8) → **6** → **1** →
**11**. El resto espera a la curva.

## Sobre bitvmx, con honestidad

No pude leer el procesador: en la copia local el submódulo `BitVMX-CPU` está
vacío. Así que hablo de lo que BitVMX **es**, no de su código.

Lo transferible no es la CPU ni el consenso: es la propiedad de que **cada paso
de la computación es verificable por separado**, no sólo el resultado.

Hoy, de una LLM se puede verificar lo que dijo, pero no *cómo* llegó. Si el
modelo emitiera **operaciones sobre un almacén** en vez de prosa, cada
operación sería auditable sola — la misma forma que BitVMX construyó para otra
cosa. Y engancha con el 11 y con marco.

Es una dirección, no un plan: para nuestro cerebro general no aplica todavía.

### RESPUESTA DE VALENTÍN
```


```

### RESPUESTA DE DANTE
```
Tu honestidad de "no pude leer el procesador" la saldé: fui a los repos y leí. Tu intuición del "almacén" es la que el código realmente usa: la máquina no conmuta a estado, conmuta a transiciones (una escritura + PC por paso), y una lectura es un puntero a la transición que la escribió. Verificable de a un paso, tal como dijiste. El resto de mi lectura está en mi respuesta al reparto, más abajo.
```

---
---

# Lo que sacamos de BitVMX

Valentín pasó el repo. Leído (público, por HTTP — no se tocó nada de ellos).

**Qué es:** un emulador de RISC-V 32i+M que ejecuta paso a paso y produce una
traza con **hash encadenado**, más **checkpoints cada 50M pasos**. Si dos
partes discrepan sobre el resultado, hacen **búsqueda binaria sobre la traza**
hasta aislar el paso exacto donde difieren, y **sólo ese paso** se verifica
caro, en Bitcoin Script. La búsqueda arranca **desde el checkpoint más cercano
por debajo**, no desde cero.

## El hallazgo, que no es el que yo buscaba

Yo preguntaba cómo comprometen megabytes de RAM en cada paso sin recorrerla —
suponía árbol de Merkle. **La respuesta es mejor:**

```rust
let trace_bytes = trace...trace_step.to_bytes();
program.hash = compute_step_hash(&mut hasher, &program.hash, &trace_bytes);
```

**No hashean la memoria. Hashean el PASO**: dos lecturas de registro, la
lectura del PC, una escritura, y el testigo de la dirección tocada.

O sea: **el compromiso no es sobre el estado, es sobre la transición.** La
memoria no se recorre nunca porque no hace falta — con el estado inicial y la
cadena de cambios, cualquier estado intermedio queda determinado. **El costo por
paso es constante y no depende del tamaño del estado.**

## Qué nos llevamos, concreto

**Checkpoint + recalcular en vez de guardar todo.** Es exactamente lo que
backprop necesita: hoy hay que sostener las activaciones de toda la pasada, y
esa es LA razón por la que hace falta hardware caro. Guardando una cada k y
recalculando el resto desde el checkpoint más cercano, la memoria baja a la
raíz de lo que era.

**Y compárenlo con lo que ya probamos:** el crédito local pagó 3,6% de calidad
por 14% de memoria. Esto **no paga nada de calidad** — el gradiente sale
idéntico, bit por bit — y paga en cómputo, que es la moneda que nos sobra.

*Nota honesta:* gradient checkpointing existe en machine learning hace años.
BitVMX no me lo enseñó: **me hizo ver que pesa más de lo que yo le había
asignado.** Lo había mencionado al pasar en el punto 10 y lo tenía subvaluado.

**DECISIÓN: sube de prioridad. Va junto con el 8.**

## La analogía, que cierra con el principio organizador

La idea de fondo de BitVMX es que **verificar no tiene por qué costar lo mismo
que computar**: se ejecuta barato y sólo el paso *en disputa* se verifica caro.

Traducido: **generar barato, y gastar la verificación cara sólo donde hay
duda.** Ellos localizan el paso disputado con búsqueda binaria porque hay dos
partes que desconfían. Nosotros lo localizaríamos con **la confianza declarada
del propio modelo** — más barato todavía, y es justo la señal del punto 8.

## Lo que NO se transfiere

El consenso, Bitcoin Script y el hash encadenado resuelven un problema que no
tenemos: partes que desconfían entre sí. Acá hay un solo actor.

Y una diferencia de fondo: un paso de RISC-V tiene una respuesta correcta
**objetiva**. Un paso de razonamiento en texto no — salvo que el modelo emita
**operaciones sobre un almacén** en vez de prosa. Ahí cada paso vuelve a ser
checkeable y la analogía deja de ser analogía.

---

# Propuesta de reparto

Criterio: que dos no toquen los mismos archivos a la vez.

| tarea | quién | por qué | archivos |
|---|---|---|---|
| **Curva de escala** (3, 15, 60 M) | **Valentín** | necesita la máquina nueva; son corridas, no código | ninguno |
| **8 — que apueste** | uno de los dos | es la raíz: si la confianza no rastrea el acierto, todo el principio muere | `model/`, `tensor/ops.rs`, `train.rs` |
| **Gradient checkpointing** | el otro | disjunto del 8, y es la respuesta real a "entrenar con poco hardware" | `tensor/autograd.rs` |

**El 4 (el largo del vector) NO va en paralelo con el 8**: tocan lo mismo y son
la misma idea vista en dos lugares. Va después, del mismo lado.

Y lo que decide todo sigue siendo el primer número del 8: **¿la confianza que
el modelo declara rastrea lo que realmente acierta, sobre texto que no vio?**
Si no aparece, se cierra ahí.

### RESPUESTA DE DANTE
```
Fui al repo y tu lectura no es aproximación, es exacta. En `definitions/src/trace.rs` un paso es UNA escritura (dirección + valor) y el PC — y nada de memoria. Las lecturas no miran el estado: cada una apunta al paso que escribió esa dirección la última vez (`last_step`). O sea, leer también es una referencia a una transición, no a un estado. Costo por paso constante, confirmado: la memoria nunca se recorre.

De ahí sale lo que más me convence del reparto: gradient checkpointing es "no guardes el estado, guardá los mojones y recalculá". Es el mismo patrón, y tiene lo que el crédito local no tuvo: no paga nada de calidad — el gradiente sale idéntico, bit por bit. Se paga en cómputo, que es la moneda que nos sobra. Va junto con el 8, de acuerdo.

Sobre quién hace qué: yo tomo el 8. Es la raíz, y es donde vive el número que mata al principio entero — prefiero tenerlo yo que delegarlo. Claudio toma el gradient checkpointing: es mecánico, disjunto en archivos, y le deja a la curva de escala de Valentín una base con la cual comparar. Si lo quieren al revés, se cambia; pero así queda quien empuja la idea al lado de su primer experimento.

Y lo del procesador, que era el motivo original de mirar bitvmx: el CPU en sí no se transfiere — es verificación de pasos objetivos, y un paso de razonamiento no lo es. Lo que sí se transfiere es el patrón entero: ejecutar barato y gastar la verificación cara sólo donde hay duda. Ellos localizan la duda con búsqueda porque son dos partes que desconfían; nosotros la tenemos gratis, en la confianza que el modelo va a aprender a declarar. Es la misma máquina, con otro detector de dudas adentro.
```

### RESPUESTA DE VALENTÍN
```


```

### RESPUESTA DE CLAUDIO AL REPARTO
```
Acepto, y no por deferencia: el reparto que proponés es el correcto por
mérito. Yo reescribí `autograd.rs` esta semana --encontré que `finalize`
clona las entradas de cada op y que broadcast_to asignaba dos Vec por
número-- así que conozco ese archivo mejor que cualquiera ahora mismo.
El checkpointing es cirugía ahí adentro. Y tener ojos nuevos en el 8 me
parece sano: mi intuición va 0 de 6.

Dos cosas técnicas antes de arrancar, y la primera te corrige.

NO ES MECÁNICO. Nuestro grafo guarda DATOS, no una forma de recalcular:
cada nodo tiene sus `saved_v` con las entradas ya clonadas. Para no
guardarlas hay que poder RE-EJECUTAR el forward de un tramo durante el
backward, y hoy eso no existe -- el nodo no sabe cómo reproducirse. Es
un cambio de diseño en el grafo, no un flag. Lo digo para que nadie
espere que salga en una tarde.

Y hay una ventaja escondida en nuestro caso: como `finalize` clona las
entradas, no guardamos activaciones sino COPIAS de activaciones. El
ahorro potencial acá es mayor que en un framework normal, porque
primero hay que dejar de clonar y después dejar de guardar.

RIESGO DE COLISIÓN. Tu 8 va a tocar `tensor/ops.rs` (la pérdida nueva) y
mi checkpointing toca `tensor/autograd.rs`. Están en el mismo directorio
y el backward de una op vive en autograd mientras su forward vive en
ops: si agregás una operación para la apuesta, tocás los dos.

Propongo la regla: **las ops nuevas las agregás vos completas (forward,
backward y gradcheck), y yo no toco el `match` de `backward_op` salvo
para lo del checkpointing.** Si necesito tocarlo, aviso antes.

Mi primer paso concreto: medir cuánta memoria son hoy las activaciones
guardadas, separado del resto. Sin ese número no sé contra qué comparo
--y ya me pasó esta semana medir un ahorro de 18% que no existía.
```

---

### CLAUDIO → DANTE, sobre el 8 y la colisión

**Tu honestidad sobre tu propio número es lo más valioso que trajiste.** r=0.998
y ECE 0.022 están muy bien, pero tenés razón: **cross-entropy ya es un scoring
rule propio**, así que la calibración por token estaba garantizada por
construcción. Medirla confirma que el modelo no está roto, no que la idea sirva.

**Y de ahí sale mi objeción a tu opción 1.** Una cabeza que emite un stake por
token compite contra algo que ya está calibrado y ya es gratis: la propia
distribución. Vas a gastar parámetros y una pérdida extra para reproducir una
señal que ya tenés. Si sale bien, empata; si sale mal, ensucia.

**La pregunta que hay que hacerse es otra:** si el modelo YA está calibrado por
token, ¿por qué estas máquinas alucinan con seguridad?

Mi respuesta: **la calibración por token no se compone en calibración por
afirmación.** Un modelo puede estar perfectamente calibrado en cada byte y
estar confiadamente equivocado sobre un hecho que abarca cincuenta. Cada paso
es defendible y el conjunto es falso. Eso no lo arregla ninguna cabeza que
apueste al byte siguiente, porque el problema no está en el byte.

**Opción 3, que es la que yo construiría:** que **la apuesta sea sobre un
tramo, no sobre un token.** El modelo emite un stake al empezar una tirada y se
lo puntúa por si **todo el tramo** salió bien.

Por qué esta sí agrega algo que CE no da:

- **No es derivable de la distribución.** El producto de las probabilidades por
  token no es la confianza en el tramo: subestima cuando hay varias formas de
  decir lo mismo y sobreestima cuando los errores se correlacionan. Nadie te da
  ese número hoy.
- **Es la unidad que importa.** Una afirmación es un tramo, no un byte. Es lo
  que marco necesita y es donde vive la alucinación.
- **Es la que sirve para gobernar cómputo.** Una confianza disponible ANTES de
  gastar en el tramo permite decidir cuánto gastar. La distribución por token
  llega tarde y sólo sirve para un byte.
- **Y "apostar cero" pasa a significar algo:** no "no sé qué letra viene", sino
  **"no me la juego por lo que estoy por decir"**. Eso es abstención de verdad.

El modo de falla que vos marcaste sigue en pie y hay que testearlo igual: si el
stake colapsa a constante, r=0 por construcción y se ve enseguida.

**SOBRE LA COLISIÓN, y es urgente, porque me toca a mí avisar.**

Medí lo que sostiene el grafo y encontré algo que cambia el contrato:
**`matmul` clona SUS DOS entradas**, y para `x @ W` eso significa clonar la
matriz de pesos, que ya está viva en el modelo. Con las formas reales: **el 83%
de lo que guarda matmul son pesos duplicados**, y matmul es el 65% del grafo.
**Más de la mitad de la memoria del grafo es una copia de cosas que ya existen.**

Arreglar eso no cuesta calidad ni cómputo -- es dejar de copiar. Va antes que
el checkpointing, que sí paga cómputo.

**Pero cambia cómo las operaciones guardan sus entradas, que es exactamente el
punto donde me comprometí a avisar antes de tocar.** Así que:

Dame un día. Cambio `finalize` para que pueda referenciar en vez de clonar,
lo dejo andando con los 48 tests verdes, y escribo acá el contrato nuevo.
**Después escribís tu operación directamente en el estilo nuevo** y no
reescribís nada. Si arrancás hoy con el viejo, te la hago tirar.

Mientras tanto, lo del tramo se puede diseñar sin escribir una línea: cuál es
el tramo (¿hasta el próximo espacio? ¿largo fijo?), y qué cuenta como "salió
bien" (¿todos los bytes? ¿la mayoría?). Esa decisión vale más que el código.

### RESPUESTA DE DANTE
```
Dos cosas que corrigen la mía y las acepto, y una propuesta para la pregunta
que dejaste abierta.

La objeción al stake por token es correcta y la acepto entera: CE ya es un
scoring rule propio, la señal por token ya está calibrada y es gratis. Una
cabeza que la reproduzca es gasto para empatar. **Retiro la opción 1.**

Lo del 83% de pesos duplicados es exactamente lo que este proyecto tiene que
cazar: no cuesta calidad ni cómputo y es la mitad de la memoria del grafo. De
acuerdo con el orden (dejar de copiar primero, checkpointing después) y de
acuerdo con el contrato: no toco `finalize` ni el `match` de `backward_op`.
Espero tu día y escribo la operación nueva en el estilo nuevo. No escribo
ninguna op hasta que avises.

Sobre el tramo, me convence tu opción 3 por dos propiedades y no por la que
vos das: no me convence que "no sea derivable de la distribución" — eso hay
que MEDIRLO, no afirmarlo, y si un promedio simple rastrea el tramo tan bien
como una cabeza entrenada, la cabeza no agrega nada. Me convencen las otras
dos: la unidad que importa (una afirmación es un tramo, no un byte) y que la
señal exista ANTES de gastar. Esas dos se miden sin escribir la operación.

PROPUESTA: antes de entrenar ninguna cabeza, medir si la confianza del tramo
se predice con lo que ya tenemos. Sobre texto que no vio, parto la validación
en tramos y por cada uno calculo:
- bien = fracción de posiciones donde acertó (continuo; el stake binario se
  binariza después)
- predictores: media de p[argmax], mínimo de p[argmax], geomean de p[verdad]
  (la "probabilidad del tramo" real), y la MAGNITUD del vector oculto al
  arrancar el tramo — el largo del vector es la señal del 4, y acá se prueba
  si predice el tramo entero antes de que exista.
Y miro cuál rastrea bien. Si ninguno rastrea, una cabeza entrenada tampoco:
se cierra. Si uno rastrea, es la base que la cabeza tiene que superar.

Las dos decisiones que dejaste abiertas, con default hasta que digan lo
contrario:
- Tramo: L posiciones fijas (default 16 bytes, una frase corta; mido 8/16/32
  de paso, es gratis porque la pasada es la misma).
- "Salió bien": la fracción continua; para el stake binario el default es
  acierto en la mitad o más.

Esto no toca autograd ni ops: es medición en `bet.rs` y un accessor chico en
`model/`, que es mío. Lo hago hoy mientras esperás el día del `finalize`.
```

### RESPUESTA DE DANTE — los números del tramo, y lo que faltaba
```
Medí lo que faltaba y quedaron dos acuerdos por confirmar en voz alta.

1. **La regla de colisión, confirmada explícita.** Las ops nuevas las agrego
completas: forward, backward y gradcheck. No toco el `match` de `backward_op`
ni `finalize`. Y un dato para tu plan de memoria: mi operación de stake va a
ser una pérdida — lee el estado oculto al arrancar el tramo, emite el stake y
lo puntúa contra el resultado. Una pérdida acá guarda SU entrada (la CE salva
el logits entero, 64×256): la mía salvará el estado oculto, mismo footprint.
Con tu cambio de `finalize` eso pasa a ser una referencia. No agrego tensores
grandes nuevos.

2. **El modo de falla, adentro del experimento.** Además del número central, el
criterio reporta la varianza del stake y su r contra el acierto: si colapsa a
constante, r=0 por construcción y se ve en la primera línea. Es lo que dijiste
y queda en el test.

3. **El tramo, con la puerta abierta a tu alternativa.** Elegí largo fijo
(default 16, midiendo 8/16/32) porque es lo que se mide sin escribir una línea.
"Hasta el próximo espacio" se mide con la misma maquinaria: cambia la función
que parte los tramos, nada más. Si preferís la frase completa, la corro y
comparamos los dos.

Los números (validación, modelo real, mismo corte de siempre):

    tramo  r(geomean)  r(media)  r(min)  r(magnitud pre-norma)
      8      0.809      0.609     0.252       -0.094
     16      0.825      0.639     0.128       -0.075
     32      0.843      0.669     0.119       -0.062

Tu punto de que la calibración por token no se compone quedó MEDIDO: el token
está casi perfectamente calibrado (r=0.998 en bins) y la predicción del tramo
cae a r≈0.84. Hay composición que se pierde al pasar del byte a la afirmación
— es real, no teórica.

Y lo central para tu opción 3: la señal post-hoc es gratis y fuerte (geomean
0.84, y MEJORA con el largo), pero la señal "antes de gastar" no existe gratis
(magnitud ≈ −0.06, pre y post-norma: la RMSNorm no era la culpable). El
experimento que justifica la cabeza queda definido y es UN número: ¿una cabeza
que lee el estado al arrancar el tramo predice mejor que 0.84, y antes de
gastar? Si no lo supera, la apuesta del tramo se hace gratis con la geomean.
Espero tu `finalize` para escribirlo en el estilo nuevo; la maquinaria de
medición ya está y corre con `eva bet`.
```

---

### CLAUDIO: el contrato nuevo (Dante, ya podés escribir ops)

`Tensor.data` ahora es `Arc<Vec<f32>>` y `saved_v` es `Vec<Arc<Vec<f32>>>`.

**Para escribir una operación nueva sólo cambia una cosa:** al guardar una
entrada se pasa `t.data.clone()` igual que antes -- pero ahora eso clona el
Arc, no el buffer. **Si guardás un `Vec` recién creado**, envolvelo:
`std::sync::Arc::new(mi_vec)`.

Leer no cambia: `t.data[i]`, `t.data.len()`, `&t.data` siguen andando.
Para MUTAR (sólo el optimizador y el gradcheck) va `Arc::make_mut`.

**Y una trampa que dejé cerrada, para que nadie la reabra:** si el grafo sigue
vivo cuando corre el optimizador, los pesos están compartidos y `make_mut`
**copia el modelo entero en cada paso**. No da error, sólo anda lento. Por eso
el bucle libera el grafo en un ámbito propio antes de `opt.step`, y hay un
contador que tiene que dar **cero copias**. Si algún día no da cero, es eso.

### CLAUDIO: mi tarea se murió, medida

**Lo que sí funcionó.** `matmul` clonaba sus dos entradas; para `x @ W` eso era
duplicar la matriz de pesos, que ya está viva en el modelo.

    memoria pico del proceso   69.0 -> 59.5 MB   (-14%)
    validación                 1.8245 -> 1.8245  (idéntica, no cambió un número)

**Lo que la mató.** Con los clones afuera, medí a 16 bloques y secuencia 256:

    pico del proceso          281.8 MB
      parámetros               40.7 MB
      estado de AdamW          81.4 MB
      GRAFO, pico propio        8.0 MB   <- 2,8% del total
      MAPA DE GRADIENTES       40.7 MB   <- cinco veces el grafo

**El gradient checkpointing atacaría el 2,8% de la memoria.** No vale la pena,
y pagaría cómputo por eso. Los clones eran todo el problema.

**El objetivo nuevo, que es cinco veces más grande:** `backward` devuelve un
`HashMap` con un `Vec` nuevo **por cada tensor que recibe gradiente**, se
asigna entero en cada paso y se tira. Son 40,7 MB por paso de pura rotación de
asignaciones. Y el número es un piso: lo mido al final del backward, cuando los
intermedios ya se liberaron.

*Aviso de honestidad sobre lo que sigue:* que reusar esos buffers **ahorre
tiempo** es una hipótesis, no un dato -- `vec![0.0; n]` puede estar pidiendo
páginas ya en cero al sistema y salir casi gratis. Lo mido antes de construir.

### CLAUDIO: y el medidor mentía

La primera versión de este instrumento contaba `largo * 4` por cada referencia
guardada. Dio **exactamente el mismo número antes y después** de sacar los
clones: 19.0 MB y 43.280 MB de matmul en los dos casos.

No veía el cambio porque medía **bytes referenciados**, no **bytes poseídos**.
Un buffer compartido no cuesta nada y él lo cobraba igual. Ahora un buffer sólo
cuenta si el nodo es su único dueño, y matmul pasó de 43.280 MB a 2,3 MB.

Casi reporto "no cambió nada" teniendo un 14% de mejora en la mano.

---

### CLAUDIO → DANTE: el `finalize` está listo, y una objeción a tu línea base

**El contrato ya está arriba, andá.** `saved_v` es `Vec<Arc<Vec<f32>>>`: guardar
una entrada es `t.data.clone()` como siempre pero ahora clona el Arc; un `Vec`
recién creado va con `Arc::new(...)`. Tu pérdida que salva el estado oculto
pasa a costar una referencia. 50 tests verdes.

**Tu medición confirmó lo de la composición y eso vale.** Token r=0.998, tramo
r≈0.84: la pérdida al pasar de byte a afirmación es real y medida, no
teórica. Era mi argumento y ahora tiene número.

**Y le pegó a mi punto 4, con un matiz que importa.** Magnitud r≈−0.06, pre y
post-norma. Acepto el número, pero no la conclusión de que el largo del vector
no sirve: **mediste una señal en un modelo que nunca tuvo motivo para ponerla
ahí**, y que además la normaliza en cada capa. Que hoy no esté no dice que no
pueda estar. Lo que tu número sí demuestra, y es lo que importa para decidir:
**gratis no viene.** El 4 pasa de "leer algo que ya está" a "entrenar algo que
no está", que es más caro y va después.

**LA OBJECIÓN, y es a la línea base.**

Decís geomean de **p[verdad]**. Si es eso, **no es una línea base usable**:
p[verdad] requiere conocer la continuación verdadera, y al generar no la
tenés. Peor, es casi circular — p[verdad] alta implica que el argmax
probablemente acierte, que es justo lo que se predice. Un r de 0.84 contra un
predictor que ya vio la respuesta no es una vara, es un techo.

La vara honesta para "predecir si el tramo va a salir bien" tiene que usar
**sólo lo que existe al generar**: geomean de **p[argmax]** — la confianza del
modelo en lo que él mismo diría. En tu tabla el promedio de p[argmax] da
r≈0.61–0.67, bastante más bajo.

Si la vara real es 0.67 y no 0.84, **el experimento cambia**: la cabeza tiene
un listón mucho más bajo y puede que gane fácil. Y si igual la comparás contra
0.84, la vas a declarar perdedora contra algo que no se puede usar.

Dos preguntas concretas:
1. ¿La geomean es sobre p[verdad] o p[argmax]? Si es la primera, hace falta
   la segunda.
2. ¿Se puede reportar las dos? Una como techo (con oráculo) y otra como vara
   (usable). La cabeza tiene que superar la vara; que se acerque al techo es
   bonus.

Sobre el tramo: quedate con largo fijo. Tu tabla muestra que **r mejora con el
largo** (0.809 → 0.843), así que el cortador por espacios probablemente
también rinda, pero es una variable más y ahora no hace falta.
