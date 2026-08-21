# Propuesta experimental: un LLM que aprende cambiando su propia física

Quiero plantearte una arquitectura bastante más extrema que "Transformer + memoria" o "Transformer + plasticidad".

La hipótesis central es esta:

> No quiero entrenar una red que tenga pesos y aprenda.
> Quiero entrenar una red que aprenda qué significa para ella aprender.

La diferencia parece semántica, pero técnicamente cambia todo.

La arquitectura no tendría una función de actualización de pesos fija.

Tampoco tendría simplemente un optimizador externo tipo Adam.

La red tendría una dinámica interna de adaptación y, durante el entrenamiento meta, aprenderíamos esa dinámica.

Pero quiero llevarlo todavía más lejos:

## 1. El peso no es un número

Una conexión no sería:

    w_ij = 0.17

Sería un pequeño estado dinámico:

    S_ij(t) = {
        magnitude,
        velocity,
        age,
        pressure,
        fatigue,
        affinity,
        volatility
    }

No necesariamente usaría exactamente estas variables. Son conceptuales.

La idea fundamental es que una conexión tenga una "historia física".

Dos conexiones con exactamente el mismo peso podrían comportarse diferente porque llegaron a ese peso de maneras diferentes.

Por ejemplo:

    conexión A:
    0.4 → 0.41 → 0.42 → 0.43

    conexión B:
    0.9 → 0.2 → 0.8 → 0.43

Ambas terminan en:

    w = 0.43

Pero para la red no son equivalentes.

La primera es estable.

La segunda es traumática.

El estado interno de la conexión conserva esa diferencia.

---

# 2. El modelo no tiene learning rate

Quiero eliminar conceptualmente el learning rate global.

No:

    W ← W - η∇W

Sino:

    ΔS_ij = Φ(S_ij, x_i, y_j, contexto, error, tiempo)

Donde Φ es una función aprendida.

La red aprende una ley de plasticidad.

Esto tiene precedentes: existe investigación sobre meta-aprendizaje de reglas de plasticidad, incluso usando redes para parametrizar reglas de modificación sináptica. Por eso esta parte por sí sola no es la novedad que estoy buscando.

La diferencia sería que quiero hacer que Φ no solamente controle cuánto cambia una conexión.

Quiero que controle:

    si cambia
    cuándo cambia
    cuánto dura el cambio
    si el cambio se consolida
    si la conexión pierde importancia
    si debe ser reemplazada
    si debe generar otra conexión
    y qué dinámica temporal debería adoptar

Es decir:

    Φ = física de aprendizaje

---

# 3. Una conexión puede morir

La arquitectura tendría plasticidad estructural.

Una conexión podría pasar:

    ALIVE
      ↓
    WEAK
      ↓
    DORMANT
      ↓
    DEAD

Pero también podría aparecer una nueva conexión.

La topología dejaría de ser completamente estática.

Esto tampoco es una idea completamente nueva en sí misma: existen redes con topología dinámica y trabajos de plasticidad estructural.

Lo raro sería utilizarlo como parte fundamental de un modelo lingüístico autoregresivo y, especialmente, hacer que la propia dinámica aprendida decida cuándo una conexión debe existir.

No quiero:

    arquitectura fija
    +
    pesos variables

Quiero:

    arquitectura parcialmente viva
    +
    pesos dinámicos
    +
    reglas de plasticidad dinámicas

---

# 4. Pero hay una cosa todavía más importante

## La red puede cambiar la velocidad a la que aprende.

Cada conexión tiene una escala temporal propia.

Una conexión podría reaccionar en:

    1 token

otra en:

    50 tokens

otra en:

    10.000 tokens

otra prácticamente nunca.

Pero no quiero asignar esos tiempos manualmente.

Quiero que emerjan.

Entonces:

    τ_ij(t)

también es parte del estado dinámico.

La red puede aprender:

    "esto debe ser extremadamente plástico"

o:

    "esto debe volverse casi cristalino"

Eso produce una jerarquía temporal sin tener que construir explícitamente:

    memoria corta
    memoria media
    memoria larga

La temporalidad sería una propiedad emergente de la propia red.

---

# 5. Ahora viene la parte realmente extraña

## La representación no sería solamente el estado de activación.

La representación de un concepto sería:

    actividad neuronal
    +
    configuración de conexiones
    +
    velocidades de cambio
    +
    historia reciente

Es decir:

    representación(t)
    ≠
    hidden_state(t)

Sería algo más parecido a:

    representation(t)
      =
    F(
        activity(t),
        topology(t),
        synaptic_state(t),
        temporal_state(t)
    )

Esto significa que dos ejecuciones con exactamente el mismo input podrían producir estados internos diferentes si la red llegó a ellos mediante historias diferentes.

La red tendría una especie de "memoria física".

No una memoria externa.

No un vector recuperado de una base de datos.

La propia materia computacional del modelo sería la memoria.

---

# 6. Quiero eliminar otra cosa: el concepto de "error" único

En un LLM convencional tenemos:

    target token
    ↓
    prediction
    ↓
    loss

Quiero dividir el error en diferentes escalas.

Por ejemplo:

    error inmediato
    error contextual
    error temporal
    error estructural
    error de estabilidad

Un token puede ser incorrecto pero producir una representación interna excelente para los próximos 100 tokens.

Otro token puede ser correcto pero deformar el estado de la red de forma peligrosa.

Entonces:

    token_correctness

no necesariamente sería igual a:

    learning_value

La red debería poder aprender que:

> "Este error local me permitió construir una estructura mejor."

Eso introduce una señal que podríamos llamar:

    FUTURE LEARNING VALUE

---

# 7. La función objetivo más loca

En vez de optimizar únicamente:

    minimizar perplexity

quiero probar algo como:

    L =
        L_token
        +
        λ L_dynamics
        +
        μ L_stability
        +
        γ L_future_learning

Donde:

    L_token
        = calidad inmediata

    L_dynamics
        = comportamiento de la dinámica interna

    L_stability
        = evitar que el sistema destruya lo aprendido

    L_future_learning
        = cuánto cuesta aprender la próxima tarea

Y esto último es lo verdaderamente importante.

---

# 8. El modelo sería evaluado por cómo aprende después

Supongamos:

    TASK A
    TASK B
    TASK C

Después de aprender A, B debería resultar más fácil.

Después de A+B, C debería resultar más fácil.

Entonces no mediríamos solamente:

    performance(A)
    performance(B)
    performance(C)

Mediríamos:

    LearningCost(B | A)

y:

    LearningCost(C | A,B)

La pregunta sería:

> ¿El aprendizaje actual está construyendo una red que hace que el aprendizaje futuro sea más barato?

Eso podría hacer que el modelo descubra representaciones útiles no porque sean óptimas para el dataset actual, sino porque son buenas "infraestructuras" para adquirir conocimiento posterior.

---

# 9. Y ahora quiero romper incluso eso

## La red puede aprender a cambiar sus propias reglas de plasticidad.

Tenemos:

    Nivel 0
    actividad

    Nivel 1
    conexiones

    Nivel 2
    plasticidad

    Nivel 3
    meta-plasticidad

Pero quiero permitir:

    Nivel 4
    plasticidad de la plasticidad

No necesariamente como otra red gigante.

Puede ser una pequeña función que controle parámetros de la dinámica:

    Φ_t

y produzca:

    Φ_(t+1)

Conceptualmente:

    experiencia
        ↓
    cambio sináptico
        ↓
    cambio de plasticidad
        ↓
    cambio de cómo cambia la plasticidad

La red no solamente aprende.

Aprende cómo debería aprender.

Y posteriormente aprende cuándo debería dejar de aprender de esa manera.

---

# 10. Esto genera un concepto interesante: "metabolismo cognitivo"

Quiero introducir una señal global que no represente información lingüística directamente.

Algo como:

    cognitive_pressure

No sería atención.

No sería reward.

No sería loss.

Sería una variable interna que intenta representar:

    "¿Cuánto está costando mantener estable el sistema mientras aprende?"

Si la presión aumenta:

    la plasticidad puede disminuir

o:

    puede aumentar

dependiendo de la dinámica que la propia red haya aprendido.

Entonces podrían emerger estados equivalentes a:

    exploración
    consolidación
    reparación
    adaptación
    estabilización
    olvido

Pero no serían modos programados.

Serían atractores dinámicos.

---

# 11. Y quiero introducir una regla todavía más rara

## Las conexiones no deberían competir solamente por activación.

También deberían competir por existencia.

Si dos grupos de conexiones realizan aproximadamente la misma función, la red podría descubrir que mantener ambos tiene un costo.

Entonces:

    redundancia
        ↓
    competencia
        ↓
    eliminación

Pero una conexión eliminada no necesariamente desaparece para siempre.

Puede quedar:

    DORMANT

y regresar si el entorno cambia.

Entonces el modelo desarrolla algo parecido a:

    "memoria latente"

sin almacenar explícitamente el conocimiento en un buffer.

---

# 12. Esto permite una idea brutal

## El modelo podría aprender que algo que olvidó era importante.

Imaginemos:

    conocimiento A

se vuelve inútil durante millones de tokens.

La red lo comprime:

    A → dormant

Después aparece nuevamente información relacionada.

La dinámica puede reactivar las conexiones.

No necesitamos:

    recuperar documento A
    buscar embedding A
    insertar memoria A

La propia topología contiene la posibilidad de recuperación.

El aprendizaje sería:

    aprender
    →
    comprimir
    →
    dormir
    →
    reactivar

No:

    aprender
    →
    almacenar eternamente

---

# 13. Quiero ir todavía más lejos

## La red no debería tener una representación semántica fija.

Quiero que los conceptos sean "órbitas".

Un concepto no sería:

    perro = vector X

Sería:

    perro =
        trayectoria dinámica
        dentro del espacio neuronal

Por ejemplo:

    DOG
      ↓
    animal
      ↓
    domestic
      ↓
    four-legged
      ↓
    pet
      ↓
    bark
      ↓
    specific-context

El concepto sería una trayectoria que atraviesa diferentes regiones del estado interno.

Así:

    meaning ≈ trajectory

y no:

    meaning ≈ point

Esto también abre una posibilidad muy extraña:

dos conceptos podrían tener representaciones muy diferentes instantáneamente pero compartir la misma dinámica.

Es decir:

    semantic similarity

podría ser:

    similarity of trajectories

en lugar de:

    cosine(vector_A, vector_B)

---

# 14. La consecuencia más importante

Si esto funciona, el modelo ya no sería exactamente un Transformer.

Sería algo más parecido a:

    DYNAMIC LANGUAGE ORGANISM

con:

    parámetros iniciales
    +
    estado dinámico
    +
    plasticidad
    +
    meta-plasticidad
    +
    topología adaptable
    +
    memoria temporal distribuida

Y el "modelo" en un momento dado sería:

    M(t) = {W(t), G(t), P(t), T(t)}

donde:

    W = magnitudes sinápticas
    G = topología
    P = reglas/estado de plasticidad
    T = escalas temporales

El modelo sería diferente después de leer 1 millón de tokens.

No porque haya actualizado un checkpoint.

Porque literalmente sería otra máquina.

---

# 15. El experimento que propongo

No empezar con 9B parámetros.

De hecho, sería un error.

Construiría algo ridículamente pequeño:

    vocab: 128–512 tokens
    embedding: 32–64
    hidden: 128–256
    2–4 bloques
    contexto: 128–512

Y compararía cuatro sistemas:

    A:
    Transformer + Adam

    B:
    Transformer + plasticidad fija

    C:
    Transformer + plasticidad aprendida

    D:
    Dynamic Plastic Language Network

        pesos dinámicos
        +
        escalas temporales
        +
        plasticidad aprendida
        +
        topología parcialmente mutable

Después introduciría tareas secuenciales.

Ejemplo:

    Grammar A
    ↓
    Grammar B
    ↓
    Grammar C
    ↓
    Grammar A modificada
    ↓
    Grammar D

Y mediría:

    perplexity
    adaptación
    tokens necesarios para aprender una nueva gramática
    recuperación después del cambio de distribución
    catastrophic forgetting
    número de conexiones activas
    estabilidad de la topología
    coste computacional
    learning cost futuro

La métrica que más me interesa sería:

    FUTURE LEARNING COST

---

# 16. La prueba definitiva

Entrenamos dos redes hasta que tengan aproximadamente la misma perplexity.

Después les damos una distribución completamente nueva.

Si la red convencional necesita:

    100.000 tokens

y la dinámica plástica necesita:

    20.000

tenemos algo.

Pero hay una prueba mucho más interesante.

Después de miles de tareas:

    A
    B
    C
    D
    E
    ...

le damos nuevamente A.

Quiero medir si el modelo plástico puede recuperar competencia rápidamente sin tener almacenado explícitamente A.

Eso demostraría que la red desarrolló:

    latent structural memory

en su propia dinámica.

---

# 17. La pregunta filosófica/técnica que quiero probar

La arquitectura tradicional asume:

    aprendizaje = modificar parámetros

Esta propuesta plantea:

    aprendizaje =
        modificar el sistema
        que modifica los parámetros

Y después:

    inteligencia =
        tener una dinámica
        capaz de modificarse
        de manera útil bajo experiencia

Por lo tanto, quizá no deberíamos buscar "los mejores pesos".

Deberíamos buscar:

> **la mejor dinámica capaz de producir buenos pesos continuamente.**

Eso es bastante diferente.

---

# 18. Nombre provisional

No lo llamaría simplemente "plastic Transformer".

Algo más propio:

    DYNAMICAL PLASTIC LANGUAGE MODEL

o:

    DPLM

Pero me gusta más el concepto:

    SELF-ALTERING LANGUAGE NETWORK

    SALN

Porque el objetivo final sería literalmente:

> una red lingüística que puede modificar parcialmente su propia estructura y su propia velocidad de aprendizaje como consecuencia de la experiencia.

---

# 19. Una última locura que quiero dejar abierta

No asumiría que las neuronas tienen que producir:

    scalar activation

Podríamos experimentar posteriormente con una unidad cuya salida sea:

    estado(t)

y no:

    activation

Es decir:

    neuron(t+1)
      =
    F(
        neuron(t),
        input(t),
        local_state(t)
    )

Una neurona sería un pequeño sistema dinámico.

Entonces cada unidad podría tener:

    memoria interna
    energía
    adaptación
    recuperación
    oscilación
    umbral

y la red entera sería un sistema dinámico acoplado.

Ahí desaparece progresivamente la separación tradicional entre:

    arquitectura
    memoria
    activación
    aprendizaje
    optimizador

Todo se convierte en una única cosa:

    DINÁMICA.

Y ése sería el experimento realmente interesante:

> ¿Podemos construir un modelo de lenguaje donde aprender no sea una operación externa aplicada sobre una red, sino una propiedad física de la red misma?

Si la respuesta es sí, entonces no habríamos hecho simplemente otro LLM.

Habríamos diseñado una máquina cuya arquitectura fundamental es:

    "cambiar de manera útil".

Ese sería el punto donde yo empezaría a experimentar.

IMPORTANTE: no afirmaría que cada componente individual de esto es inédito. Plasticidad aprendida, plasticidad estructural, redes con topología dinámica y meta-aprendizaje ya existen como líneas de investigación. La apuesta potencialmente novedosa está en combinarlos alrededor de una arquitectura lingüística donde el objeto aprendido no sea solamente W, sino la dinámica completa de adaptación y su costo futuro. Ésa es la parte que habría que validar mediante una búsqueda bibliográfica mucho más exhaustiva y, sobre todo, mediante experimentos.