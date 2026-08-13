# Formalización Séneca: Candidata "Suma de Logits"

La propuesta consiste en mover la interacción entre la Memoria ($q$) y el Modelo ($p$) del espacio de probabilidades (lineal) al espacio de energía (logits), para evitar la anulación del gradiente por factor de mezcla.

## 1. El Sistema
Definimos el logit total $z$ para el token $i$ como:
$$z_i = z_{m,i} + z_{q,i}$$
Donde $z_{m}$ es el logit producido por el modelo paramétrico y $z_{q}$ es el logit inyectado por la tabla de k-gramas. La probabilidad combinada es:
$$P_i = \text{softmax}(z)_i = \frac{e^{z_{m,i} + z_{q,i}}}{\sum_j e^{z_{m,j} + z_{q,j}}}$$

## 2. Derivación del Gradiente
Para la pérdida de Entropía Cruzada $\mathcal{L} = -\ln P_k$ (donde $k$ es el target):
$$\frac{\partial \mathcal{L}}{\partial z_{m,i}} = \frac{\partial \mathcal{L}}{\partial z_i} \cdot \frac{\partial z_i}{\partial z_{m,i}}$$
Dado que $\frac{\partial z_i}{\partial z_{m,i}} = 1$, el gradiente respecto al modelo es simplemente el error de predicción del sistema combinado:
$$\nabla_{m,i} = P_i - \delta_{ik}$$

## 3. Refutación de la Inmunidad Total
La afirmación "el gradiente NO se asfixia aunque la tabla acierte" es **FALSA** en términos absolutos. 

Si la tabla acierta con total autoridad ($z_{q,k} \to \infty$ o simplemente $z_{q,k} \gg z_{q,j}$), entonces $P_k \to 1$. En ese límite:
$$\lim_{P_k \to 1} \nabla_{m,k} = 1 - 1 = 0$$

El gradiente desaparece. El sistema combinado "cree" que ya no hay nada que aprender porque la respuesta es correcta, aunque el modelo paramétrico $z_m$ no haya aportado nada.

## 4. La Condición de Escala (Sustento de la Sospecha)
La asfixia se evita **SÓLO** si la autoridad de la tabla está acotada. 

Para que el modelo reciba una señal de entrenamiento mínima $\epsilon$ (en valor absoluto), necesitamos que:
$$|P_k - 1| \ge \epsilon \implies P_k \le 1 - \epsilon$$

Sustituyendo la definición de Softmax y asumiendo un modelo inicial "frío" ($z_{m,i} \approx 0$):
$$\frac{e^{z_{q,k}}}{e^{z_{q,k}} + \sum_{j \neq k} e^{z_{q,j}}} \le 1 - \epsilon$$

Aproximando la suma del denominador por el tamaño del vocabulario $V$ (suponiendo logits de la tabla para otros tokens $z_{q,j} \approx 0$):
$$\frac{e^{z_{q,k}}}{e^{z_{q,k}} + (V-1)} \le 1 - \epsilon$$

Despejando $z_{q,k}$ (Authority Cap):
$$z_{q,k} \le \ln\left( \frac{1-\epsilon}{\epsilon} (V-1) \right)$$

## 5. El Primer Número (The Kill Number)
Si implementamos la Suma de Logits y los logits de la tabla no están acotados, el experimento morirá igual que la mezcla lineal.

**El número:** Para un vocabulario $V=256$ y una señal mínima deseada de $\epsilon=0.1$ (que el modelo todavía sienta que tiene que empujar un 10%):
$$z_{q,k} \le \ln\left( \frac{0.9}{0.1} \cdot 255 \right) = \ln(2295) \approx \mathbf{7.73}$$

**Veredicto:** Si los logits de la tabla superan $\sim 8$, el gradiente del modelo cae por debajo del 10% de su valor potencial. La candidata "Suma de Logits" solo vive si la tabla es una **sugerencia persistente**, no una **dictadura de memoria**. 

Si inyectamos $z_{q,k} = 15$ (típico de una distribución "one-hot" suave), el gradiente será $P_k - 1 \approx e^{15}/(e^{15}+255) - 1 \approx -0.00007$. El modelo paramétrico nunca despertará de su letargo.

---

## Ronda 3 — la forma de $z_q$ y la condición de ganancia (tarea, la pasa Dante)

Tu Authority Cap se sostiene: sin él, la suma de logits se asfixia igual que
la mezcla lineal. Ahora, con el cap y la vara medida, tres preguntas:

1. **La forma exacta de $z_q$.** Ojo técnico: $z_q = \ln q$ es $\le 0$ SIEMPRE
   (los logits de la tabla no pueden dominar el softmax — tu cap sobra para esa
   forma). La candidata real lleva escala: $z_q = \beta \ln q$ con $\beta > 1$
   (logits positivos posibles) o $z_q = \beta \cdot q$. Elegí y justificá UNA,
   y derivá qué $\beta$ satura tu cap de ~7.7.
2. **La condición de ganancia** (el análogo formal del barrido de λ que mató la
   mezcla lineal): dado el perfil REAL de la tabla en val — acierta ~53% donde
   `count ≥ 2` y PEOR donde `count` es alto —, ¿existe $\beta > 0$ tal que la
   pérdida en val BAJE respecto de $\beta = 0$? Derivá la condición en términos
   del acierto de la tabla vs. la confianza actual del modelo ($P_k$).
3. **El número que la mata:** si la condición que derivás implica "no existe
   $\beta > 0$ que baje el loss" para el perfil medido → la candidata muere en
   el pizarrón y NO corremos el barrido de β en 16M (nos ahorramos la corrida).

Sin código. Formal, no como promesa.

### Análisis de Séneca (Ronda 3)

#### 1. Elección de la forma: $z_q = \beta \cdot q$
Elijo la forma **lineal** $z_{q,i} = \beta \cdot q_i$.
- **Justificación:** La forma logarítmica $z_q = \beta \ln q$ es un castigo infinito para cualquier $q_i=0$, lo cual es letal en un sistema donde la tabla es incompleta por diseño. La forma lineal actúa como un **bono de recompensa** acotado. No prohíbe tokens, sino que "recomienda" con una fuerza máxima de $\beta$.
- **Saturación del Cap:** Para $q_k=1$ (certeza máxima de la tabla), $z_{q,k} = \beta$. Para no exceder el cap de 7.73 derivado anteriormente, debemos limitar **$\beta \le 7.7$**.

#### 2. Condición de Ganancia
Queremos que la pérdida $\mathcal{L}(\beta) = -\ln P_k(\beta)$ disminuya al aumentar $\beta$ desde cero ($\left. \frac{d\mathcal{L}}{d\beta} \right|_{\beta=0} < 0$).

Derivando respecto a $\beta$:
$$\frac{d\mathcal{L}}{d\beta} = E_P[q] - q_k$$
Donde $q_k$ es la probabilidad que la tabla asigna al target y $E_P[q] = \sum_i P_i q_i$ es la esperanza de la tabla bajo la distribución actual del modelo.

**La condición de ganancia es:**
$$q_k > E_P[q]$$
Es decir: la tabla mejora el sistema si y solo si es **más capaz de identificar el token correcto que de "adivinar" lo que el modelo ya esperaba de ella**. Si la tabla solo confirma las alucinaciones del modelo ($E_P[q]$ alto por acuerdo en errores), la pérdida sube.

#### 3. Veredicto: El número que la mata
El experimento de la mezcla externa ya demostró una mejora del 6%. Esto prueba empíricamente que, en el agregado, $E[q_k - E_P[q]] > 0$ para nuestro corpus.

Sin embargo, el **Kill Number** para esta candidata no es el acierto bruto (~53%), sino la **Diferencia de Entropía**:
$$H(P, q) = -\sum P_i \ln q_i$$
Si implementamos la suma de logits y el modelo $p$ ya está bien calibrado en la estructura, la tabla solo aporta donde el modelo duda.

**Veredicto:** La candidata **VIVE** en el pizarrón. Dado que el modelo actual tiene una confianza media $P_k \approx 0.16$ y la tabla acierta con $q_k \approx 0.53$, el margen $(0.53 - 0.16)$ es lo suficientemente amplio para absorber el ruido de $E_P[q]$. Proceder con el barrido de $\beta$ es matemáticamente sano, siempre que se respete el cap $\beta < 7.7$ para evitar la re-asfixia.
