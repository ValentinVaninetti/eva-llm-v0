# scratch/ — código muerto preservado

`mix.rs`: el "Cuaderno" (mezclar la distribución `q` de la tabla EN EL TARGET,
dentro del grafo). **Experiment meassured and closed — negativo monotóno.**
Veredicto completo en `TRABAJO-2026-08-12.md` (sección "CUADERNO ADENTRO" y
"Veredicto de Dante sobre el gate por count"). Se quitó del build el 2026-08-13
porque referenciaba ops (`mixed_ce`/`mixed_ce_count`) que ya no existen, y
porque ese mecanismo no puede enseñar (la mezcla en el target suaviza el
objetivo donde el modelo está mal y la tabla bien).

NO volver a compilar. Si algo lo necesita de referencia (la lógica del
contexto `+1` y de construir `q` por posición es útil para la suma de logits),
copiar la parte que sirva, no re-importar el módulo.
