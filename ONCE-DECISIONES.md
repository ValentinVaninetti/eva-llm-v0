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


```

---

## Y una pregunta para vos, que no es un punto

De las once, **¿cuál te parece que es la que menos se cuestiona en el mundo y
más se debería cuestionar?**

Yo diría la 8 —qué se le pide— porque todo lo demás es consecuencia de eso. Pero
esa es mi respuesta desde adentro del paradigma, y ya sabemos cuánto vale.

### TU RESPUESTA
```


```
