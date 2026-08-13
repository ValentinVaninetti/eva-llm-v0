# Orfeo — Auditoría de factibilidad

**Fecha:** 2026-08-13  
**Tarea:** Auditar factibilidad de dos candidatas para integrar la tabla de k-gramas en el entrenamiento.

---

## 1. ¿stake_loss se puede reusar tal cual para la cabecita auxiliar?

**Respuesta: Casi sí, pero requiere un pequeño ajuste.**

En `src/stake.rs` (y `src/tensor/ops.rs:stake_loss`), la operación espera:
- `hidden`: tensor de tamaño (S, D) — estado oculto
- `w`: vector de tamaño (D,) — proyección lineal
- `b`: escalar — bias
- `bien`: slice de tamaño K — fracción de aciertos por tramo
- `span_len`: longitud del tramo

**Problema:** La salida de `stake_loss` es un **escalar** (el loss promedio sobre K tramos), no un tensor por posición.

Para usarlo como cabecita auxiliar que se suma al loss principal, necesitamos **un valor por posición** (o por tramo), no un único escalar. La `stake_loss` actual se usa como pérdida extra que se suma al loss principal, pero el modelo recibe el gradiente **promediado sobre todos los tramos**.

**Qué falta tocar:**
- `src/tensor/ops.rs:stake_loss` (líneas ~328-380): cambiar para que devuelva un tensor de tamaño (K,) en vez de un escalar
- O crear una nueva operación `stake_per_tramo` que devuelva el loss por tramo (no promediado)

**Conclusión:** `stake_loss` **no se puede reusar tal cual**. Requiere cambiar la operación para que devuelva un tensor por tramo, no un escalar.

---

## 2. ¿Existe forward_hidden/un punto donde meter una sonda detached?

**Respuesta: Sí, hay dos opciones viables.**

### Opción A: `EvaModel::forward_hidden` (`src/model/mod.rs:110-124`)

Esta función ya existe y devuelve:
- `logits`: (S, V) — salida del modelo
- `hidden`: (S, D) — estado oculto ANTES de la RMSNorm final

**Uso para sonda detached:**
```rust
let (logits, hidden) = model.forward_hidden(ids);
// hidden entra detached a la sonda
let stake = sonda.forward(&hidden.detach());
```

**Ventaja:** Ya existe, no toca el grafo del modelo.  
**Desventaja:** Requiere correr el modelo dos veces si queremos logits + stake.

### Opción B: `EvaModel::forward_carrying` (`src/model/mod.rs:145-164`)

Esta función es para generación con estado persistente, pero también tiene acceso al `hidden`:

```rust
let x = self.norm_out.forward(&x);
let logits = ops::matmul(&x, &self.head_w);
```

Podríamos agregar una versión `forward_carrying_hidden` que devuelva el estado antes de la norm_out.

**Conclusión:** **Sí hay un punto viable** — `forward_hidden` es suficiente. Solo hace falta agregar la sonda detached sobre `hidden`.

---

## 3. ¿Qué hay en el grafo para sumar logits y softmax separable?

### 3a. Suma de logits

**Respuesta: `ops::add` sirve perfectamente.**

En `src/tensor/ops.rs:17-24`:
```rust
pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    let out_shape = broadcast_shape(&a.shape, &b.shape);
    let out = add_arrays(&a.data, &a.shape, &b.data, &b.shape, &out_shape);
    ...
}
```

✅ Sirve para sumar logits del modelo + logits de la tabla (broadcasting maneja (S,V) + (V,)).

### 3b. Softmax separable (no fundido con CE)

**Respuesta: `softmax_causal` existe, pero es para atención. Necesitamos softmax por fila.**

En `src/tensor/ops.rs:43-63`:
```rust
pub fn softmax_causal(x: &Tensor) -> Tensor {
    assert_eq!(x.shape.len(), 2, "softmax_causal expects (S,S)");
    ...
}
```

**Problema:** `softmax_causal` asume matriz cuadrada (S,S) y hace masking causal. Para logits (S,V) necesitamos **softmax por fila sin mask**.

**Qué falta:**
- Operación `softmax_rows(x: &Tensor) -> Tensor` que haga softmax sobre la dimensión 1 (columnas) de (S,V).

**Alternativa viable:** Crear un nuevo op `logit_softmax` que:
1. Haga softmax por fila sobre logits (S,V) → probabilidades (S,V)
2. Guarde las probabilidades en `saved_v` para el backward
3. Devuelva las probabilidades

**Conclusión:** `ops::add` **sí sirve**, pero falta un **softmax separable por fila**. No es una operación grande, pero hay que crearla (es una versión simplificada de `softmax_causal` sin el masking).

---

## Resumen final

| Candidata | Factibilidad | Mínimo que falta |
|-----------|--------------|------------------|
| **Stake auxiliar (span_len=1)** | ❌ **No tal cual** | Cambiar `stake_loss` para devolver (K,) en vez de escalar |
| **Sonda detached sobre hidden** | ✅ **Sí** | `forward_hidden` ya existe, solo agregar la sonda |
| **Suma de logits + softmax separable** | ⚠️ **Parcial** | `ops::add` sí sirve; falta `softmax_rows` (operación nueva pequeña) |

**Veredicto:** **Una sí y otra no**.

- La **sonda detached sobre `hidden`** es factible sin cambios mayores.
- La **stake_loss como cabecita auxiliar** requiere modificar la operación para que devuelva por tramo, no promediado.
- La **suma de logits** requiere crear un nuevo op `softmax_rows` (poco trabajo, ~30 líneas).

**Mínimo para implementar la candidata de Claudio (sonda sobre hidden + stake por tramo):**
1. Crear `softmax_rows` en `ops.rs` (~20 líneas, test incluido)
2. Modificar `stake_loss` para devolver loss por tramo (~10 líneas)
3. Agregar sonda detached en `train.rs` o `cli.rs` (flag `--stake-aux`)
4. Sumar `logits + lambda * table_logits` y aplicar softmax
5. Gradcheck en `tests.rs` (2-3 tests)

**Tiempo estimado:** 2-3 horas de desarrollo + 1 hora de tests.
