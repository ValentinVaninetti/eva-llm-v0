# GEMINI.md — Eva LLM v0

Contexto de proyecto para Gemini CLI.

## Qué es

Proyecto de investigación en Rust: una LLM chica entrenada desde cero, con
arquitectura propia `ClockMem` y atención causal como baseline. Leé `README.md`
y `DISENO.md` para el diseño; `PARA-OPENCODE.md` y `ONCE-DECISIONES.md` para el
historial y las decisiones.

## Reglas anti-fuga (obligatorias)

Estás corriendo en el tier gratuito de la Gemini API: lo que se envía a Google
puede usarse para mejorar productos y queda fuera de nuestro control. Por eso:

1. **Nunca** leas ni incluyas en contexto los archivos de pesos (`*.weights`,
   p. ej. `16m5b_seed7.weights`): son binarios de decenas de MB y no aportan a
   ninguna tarea. Referíte a ellos por nombre y metadata, no por contenido.
2. **Nunca** subas la carpeta `/data` ni el corpus salvo que la tarea lo exija
   explícitamente y de forma acotada.
3. **Nunca** repitas, imprimas ni guardes claves o secretos (`GEMINI_API_KEY`,
   tokens, etc.) que veas en el entorno. Si un comando los muestra, no los
   reproduzcas en la conversación.
4. No adjuntes el repo entero ni archivos grandes (`target/`, binarios) como
   contexto: incluí sólo lo que la tarea necesita.
5. Antes de ejecutar cualquier comando que envíe datos hacia afuera (curl,
   git push, web fetch de contenido local), confirmalo con el usuario.
6. No filtres datos personales ni contenido privado del usuario en respuestas
   ni en llamadas a herramientas.

## Uso

- Modelo por defecto: `gemini-2.5-pro` (`.gemini/settings.json`).
- Comandos útiles: `cargo test --release`, `cargo run --release -- ...` (ver
  `README.md`).
