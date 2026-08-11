//! **De qué aprende el modelo.** Reglas intercambiables, no un `if` en el bucle.
//!
//! La convención de hoy es que todo ejemplo merece el mismo esfuerzo: para cada
//! token se hace la pasada hacia adelante, la de atrás y el paso del
//! optimizador, cueste lo que cueste y aporte lo que aporte. Pero la mayoría de
//! un corpus es trivialmente predecible --espacios, la segunda mitad de una
//! palabra, "de la"-- y el backward cuesta el doble que el forward. Se está
//! pagando el precio completo por lo que ya se sabe.
//!
//! Un animal no funciona así: aprende de lo que lo sorprende. Acá la sorpresa
//! tiene una definición operativa y barata -- **la pérdida de este ejemplo
//! contra lo que el modelo viene esperando** -- y se puede medir si sirve.
//!
//! Cada regla decide UNA cosa: si este ejemplo se aprende o se saltea. Lo que
//! se saltea se saltea entero: backward y optimizador.

/// Decide si vale la pena aprender de un ejemplo, sabiendo cuánto costó
/// predecirlo.
pub trait LearnGate {
    /// `loss` ya está calculada: la pasada hacia adelante siempre se hace,
    /// porque sin ella no hay con qué decidir (ni predicción que dar).
    fn should_learn(&mut self, loss: f32) -> bool;

    /// Para que la corrida diga con qué regla se entrenó.
    fn name(&self) -> String;
}

/// La convención actual: de todo se aprende. Es la línea base contra la que se
/// mide cualquier otra cosa.
pub struct Always;

impl LearnGate for Always {
    fn should_learn(&mut self, _loss: f32) -> bool {
        true
    }
    fn name(&self) -> String {
        "todo".into()
    }
}

/// Aprender sólo de lo que sorprende.
///
/// La expectativa es una media móvil de la propia pérdida reciente, así que el
/// umbral **se mueve solo** a medida que el modelo mejora: no hay un número
/// mágico que envejezca mal. Con `k = 1.0` se saltea lo que salió mejor que el
/// promedio; con `k = 0.8`, sólo lo bastante más fácil que el promedio.
///
/// EL RIESGO CONOCIDO, que hay que vigilar y no esconder: si sólo se aprende de
/// lo que sale mal, se termina aprendiendo **el ruido del corpus**, que es
/// justamente lo que siempre sale mal. Por eso la corrida reporta validación:
/// si esto pasa, la pérdida de validación se despega de la de entrenamiento y
/// se ve.
pub struct Surprise {
    k: f32,
    /// Cuánto pesa lo nuevo en la media móvil.
    rate: f32,
    ema: Option<f32>,
    /// Ejemplos al principio que se aprenden sí o sí. Sin esto, la primera
    /// media se calcula con un modelo aleatorio y no significa nada.
    warmup: usize,
    seen: usize,
}

impl Surprise {
    pub fn new(k: f32, warmup: usize) -> Self {
        Surprise { k, rate: 0.02, ema: None, warmup, seen: 0 }
    }
}

impl LearnGate for Surprise {
    fn should_learn(&mut self, loss: f32) -> bool {
        self.seen += 1;
        let expected = self.ema.unwrap_or(loss);
        // La media se actualiza SIEMPRE, se aprenda o no: es la expectativa
        // sobre el mundo, no sobre lo que se estudió. Si sólo se actualizara
        // con los ejemplos difíciles, subiría sola y terminaría salteando todo.
        self.ema = Some(match self.ema {
            None => loss,
            Some(e) => e + self.rate * (loss - e),
        });
        self.seen <= self.warmup || loss >= self.k * expected
    }

    fn name(&self) -> String {
        format!("sorpresa k={:.2}", self.k)
    }
}

/// Arma la regla desde la línea de comandos. `k <= 0` es la línea base.
pub fn gate(k: f32, warmup: usize) -> Box<dyn LearnGate> {
    if k <= 0.0 {
        Box::new(Always)
    } else {
        Box::new(Surprise::new(k, warmup))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn always_learns_everything() {
        let mut g = Always;
        assert!(g.should_learn(0.0));
        assert!(g.should_learn(99.0));
    }

    #[test]
    fn surprise_skips_the_easy_and_keeps_the_hard() {
        let mut g = Surprise::new(1.0, 0);
        // Se estabiliza alrededor de 2.0.
        for _ in 0..500 {
            g.should_learn(2.0);
        }
        assert!(!g.should_learn(1.0), "algo mucho más fácil que el promedio debería saltearse");
        assert!(g.should_learn(4.0), "algo mucho más difícil debería aprenderse");
    }

    #[test]
    fn warmup_learns_unconditionally() {
        // La primera media se calcularía con un modelo aleatorio: no significa
        // nada y decidiría mal justo cuando más importa.
        let mut g = Surprise::new(1.0, 10);
        for i in 0..10 {
            assert!(g.should_learn(0.0001), "el ejemplo {i} del calentamiento se salteó");
        }
    }

    #[test]
    fn the_expectation_tracks_everything_not_only_what_was_studied() {
        // Si la media se actualizara sólo con lo aprendido (lo difícil),
        // subiría sola hasta saltear todo. Acá tiene que bajar cuando el mundo
        // se vuelve fácil, y volver a aceptar lo difícil.
        let mut g = Surprise::new(1.0, 0);
        for _ in 0..500 {
            g.should_learn(5.0);
        }
        for _ in 0..500 {
            g.should_learn(0.5);
        }
        assert!(g.should_learn(1.5), "la expectativa no siguió al mundo hacia abajo");
    }
}
