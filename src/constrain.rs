//! Qué puede venir después. Piezas intercambiables.
//!
//! DOS COSAS DISTINTAS, y sólo la primera existía antes acá:
//!
//! 1. **Que no elija mal.** Una máscara sobre los logits ya lo lograba.
//! 2. **Que no gaste.** Eso la máscara NO lo lograba, porque se aplicaba
//!    *después* de calcular los 256 logits: el modelo hacía todo el trabajo y
//!    después se tiraba lo ilegal.
//!
//! La diferencia está en CUÁNDO se pregunta. Si la restricción se consulta
//! **antes** del paso, puede decir "acá sólo puede venir esto" y entonces:
//!
//! - Con **una sola** continuación posible no hay nada que decidir: se emite y
//!   no se calcula la proyección de salida.
//! - Con **k** continuaciones se calculan k columnas de la cabeza en vez de
//!   todo el vocabulario.
//!
//! CUÁNTO VALE ESO depende del vocabulario, y conviene decirlo: en este modelo
//! byte-level la cabeza es `dim × 256`, apenas ~2,5% del trabajo por token, así
//! que el ahorro se ve poco. Con un vocabulario real de 32 mil, la cabeza pasa
//! a ser `dim × 32000` y **domina** el costo por token. Ahí saltearla no es un
//! detalle.
//!
//! LA ADVERTENCIA QUE VIENE MEDIDA de la otra punta del proyecto: una
//! restricción **floja** salió peor que ninguna -- con gramática suelta
//! aparecieron 8 citas basura donde sin gramática había cero. Media
//! restricción no es media protección: es la protección apagada más la ilusión
//! de tenerla. Si una restricción no puede decir con certeza qué es legal, que
//! devuelva `Any` y no una lista a medias.

/// Lo que la estructura permite en esta posición.
pub enum Allowed {
    /// Cualquier token: el modelo decide libre.
    Any,
    /// Sólo estos, en orden. **Con uno solo el modelo no se consulta.**
    Only(Vec<usize>),
}

impl Allowed {
    /// Si hay una sola opción, no hay decisión que tomar.
    pub fn forced(&self) -> Option<usize> {
        match self {
            Allowed::Only(v) if v.len() == 1 => Some(v[0]),
            _ => None,
        }
    }
}

pub trait Constraint {
    /// Se consulta ANTES de correr el modelo. `so_far` es todo lo emitido.
    fn allowed(&mut self, so_far: &[usize]) -> Allowed;

    /// Para que la corrida diga bajo qué restricción se generó.
    fn name(&self) -> String;
}

/// Sin restricción: el modelo decide todo. Es la línea base.
pub struct Free;

impl Constraint for Free {
    fn allowed(&mut self, _so_far: &[usize]) -> Allowed {
        Allowed::Any
    }
    fn name(&self) -> String {
        "libre".into()
    }
}

/// Un molde con huecos: texto fijo y tramos donde el modelo decide.
///
/// Es la forma que tiene la salida útil de un sistema real -- marco emite
/// `{"decir": "...", "citando": [2]}` y de esos ~30 caracteres el modelo sólo
/// elige los del hueco. Todo lo demás es sintaxis que se sabe de antemano y
/// que hoy se le hace producir token por token, pagando el modelo entero por
/// cada llave y cada coma.
pub struct Skeleton {
    /// Tramos alternados: fijo, hueco, fijo, hueco... El hueco lleva su largo
    /// máximo y el byte que lo cierra.
    fixed: Vec<Vec<usize>>,
    slots: Vec<(usize, usize)>,
    /// Posición dentro del molde.
    at: usize,
    /// Cuántos bytes lleva emitidos el hueco actual.
    in_slot: usize,
}

impl Skeleton {
    /// `partes` alterna texto fijo y huecos: el primer y último tramo son
    /// fijos. Un hueco es `(largo máximo, byte que lo cierra)`.
    ///
    /// OJO CON EL BYTE QUE CIERRA: lo emite el hueco, no el tramo siguiente.
    /// Para `{"di":"hola"}` el molde es `["{\"di\":\"", "}"]` con el hueco
    /// cerrando en `"`. Poner la comilla también al principio de la cola la
    /// pediría dos veces -- me pasó escribiendo el test.
    pub fn new(partes: &[&str], slots: &[(usize, u8)]) -> Self {
        Skeleton {
            fixed: partes.iter().map(|p| p.bytes().map(|b| b as usize).collect()).collect(),
            slots: slots.iter().map(|&(n, b)| (n, b as usize)).collect(),
            at: 0,
            in_slot: 0,
        }
    }

    /// Cuántos bytes del molde son fijos: el techo de lo que se puede saltear.
    pub fn fixed_bytes(&self) -> usize {
        self.fixed.iter().map(|f| f.len()).sum()
    }
}

impl Constraint for Skeleton {
    fn allowed(&mut self, so_far: &[usize]) -> Allowed {
        // Reconstruye dónde está mirando lo ya emitido. Sin estado propio que
        // se pueda desincronizar del texto: el molde se relee cada vez.
        let mut i = 0usize;
        let mut fi = 0usize;
        let mut si = 0usize;
        loop {
            // Tramo fijo.
            if fi < self.fixed.len() {
                let f = &self.fixed[fi];
                let ya = so_far.len() - i;
                if ya < f.len() {
                    return Allowed::Only(vec![f[ya]]);
                }
                i += f.len();
                fi += 1;
            }
            // Hueco.
            if si < self.slots.len() {
                let (max, cierra) = self.slots[si];
                let mut n = 0;
                while i + n < so_far.len() && so_far[i + n] != cierra && n < max {
                    n += 1;
                }
                if i + n >= so_far.len() {
                    // Adentro del hueco: decide el modelo, salvo que se pase
                    // del largo, donde lo único legal es cerrar.
                    return if n >= max { Allowed::Only(vec![cierra]) } else { Allowed::Any };
                }
                i += n + 1; // el byte que cierra ya está
                si += 1;
            } else if fi >= self.fixed.len() {
                return Allowed::Any;
            }
        }
    }

    fn name(&self) -> String {
        format!("molde ({} tramos fijos, {} huecos)", self.fixed.len(), self.slots.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(s: &str) -> Vec<usize> {
        s.bytes().map(|b| b as usize).collect()
    }

    fn como_texto(a: &Allowed) -> String {
        match a {
            Allowed::Any => "libre".into(),
            Allowed::Only(v) => v.iter().map(|&b| b as u8 as char).collect(),
        }
    }

    #[test]
    fn the_fixed_part_leaves_nothing_to_decide() {
        let mut k = Skeleton::new(&[r#"{"di":""#, "}"], &[(20, b'"')]);
        // Byte por byte del prefijo: siempre una sola opción.
        for (i, esperado) in r#"{"di":""#.bytes().enumerate() {
            let hasta = bytes(&r#"{"di":""#[..i]);
            let a = k.allowed(&hasta);
            assert_eq!(Some(esperado as usize), a.forced(), "en la posición {i}");
        }
    }

    #[test]
    fn inside_the_slot_the_model_decides() {
        let mut k = Skeleton::new(&[r#"{"di":""#, "}"], &[(20, b'"')]);
        let a = k.allowed(&bytes(r#"{"di":"ho"#));
        assert!(a.forced().is_none(), "adentro del hueco tiene que decidir el modelo");
    }

    #[test]
    fn a_slot_that_ran_out_can_only_close() {
        // El techo de largo es parte de la restricción: sin él, el hueco es un
        // grado de libertad que se llena con basura -- medido en el otro
        // proyecto, la lista de citas sin techo citaba todo.
        let mut k = Skeleton::new(&[r#"{"di":""#, "}"], &[(3, b'"')]);
        let a = k.allowed(&bytes(r#"{"di":"abc"#));
        assert_eq!(Some(b'"' as usize), a.forced(), "pasado el techo sólo puede cerrar");
    }

    #[test]
    fn after_the_slot_the_tail_is_forced_again() {
        let mut k = Skeleton::new(&[r#"{"di":""#, "}"], &[(20, b'"')]);
        assert_eq!(Some(b'}' as usize), k.allowed(&bytes(r#"{"di":"hola""#)).forced());
    }

    #[test]
    fn free_decides_nothing() {
        assert!(Free.allowed(&[]).forced().is_none());
    }
}
