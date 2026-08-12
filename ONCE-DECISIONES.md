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


```

### RESPUESTA DE VALENTÍN
```


```
