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
