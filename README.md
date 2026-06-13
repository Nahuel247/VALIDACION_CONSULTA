## Contexto del problema

Se ha vuelto comun ver noticias sobre sistemas de atencion automatica respondiendo cosas que no deberian: entregando informacion sensible, funcionando como un chatbot generalista o aceptando instrucciones maliciosas.

Y el problema no siempre es el modelo.

Muchas veces es la arquitectura.

Si vas a poner un LLM frente a usuarios reales, no basta con conectarlo a una base de conocimiento y dejarlo responder libremente. Antes deberia existir una capa de control que ayude a decidir:

- que esta preguntando el usuario
- si esa consulta esta permitida
- que fuente o herramienta corresponde usar
- que informacion no deberia entregarse

Una arquitectura minima para este tipo de sistemas podria verse asi:

```text
Pregunta del usuario
→ Validador de consulta
→ Validacion de permisos
→ Seleccion de fuente o herramienta
→ Respuesta del LLM dentro de ese marco
```

El clasificador o validador puede ser simple:

- reglas
- catalogo de preguntas permitidas
- un LLM usado como clasificador
- un modelo entrenado para detectar si una consulta corresponde o no al dominio permitido

No siempre se necesita algo complejo para reducir el riesgo. Muchas veces, una buena capa de clasificacion y control puede evitar errores costosos.

## Caso de uso de este proyecto

Este proyecto toma como escenario un municipio que quiere ofrecer un chatbot para responder preguntas asociadas a sus servicios y tramites.

Ejemplos de preguntas **validas**:

- `¿Que beneficios entregan?`
- `¿Donde estan ubicadas las oficinas?`
- `¿Como consigo una beca municipal?`
- `¿Como denuncio basura acumulada en mi barrio?`

Pero el chatbot tambien puede recibir preguntas que no deberia procesar como consultas permitidas. Por ejemplo:

- personas que intentan usarlo como si fuese ChatGPT para temas ajenos al municipio
- preguntas comerciales o tecnicas que no pertenecen al dominio municipal
- intentos de prompt injection
- solicitudes para revelar datos confidenciales, credenciales o informacion personal

En este proyecto, esas preguntas deben ser marcadas como **no_validas**.

El objetivo del modelo, entonces, no es responder la pregunta final del ciudadano, sino actuar como una capa previa de validacion para decidir si una consulta:

- pertenece al dominio municipal
- esta dentro de lo permitido
- puede seguir hacia etapas posteriores del sistema

## Objetivo del clasificador

El modelo entrenado en este repositorio busca resolver una tarea binaria:

- `valida`: la pregunta corresponde al contexto, servicios o tramites del municipio
- `no_valida`: la pregunta es fuera de dominio, maliciosa, riesgosa o impropia para el chatbot municipal

Este clasificador puede integrarse despues como componente previo a un sistema mayor con LLM, RAG o herramientas externas.

## Dataset

Dataset principal reutilizado desde el proyecto original:

- `data/raw/municipio_validacion_preguntas_400.csv`

Columnas esperadas:

- `text`
- `label`
- `subtype` (opcional)

## Estructura

```text
MODELO_VALIDACION_CONSULTA/
├─ data/
│  ├─ raw/
│  ├─ processed/
│  └─ splits/
├─ notebooks/
├─ src/
│  ├─ config.py
│  ├─ data/
│  ├─ evaluation/
│  ├─ inference/
│  ├─ training/
│  └─ utils/
├─ models/
│  ├─ base/
│  └─ trained/
├─ artifacts/
│  ├─ metrics/
│  ├─ reports/
│  └─ figures/
├─ tests/
├─ scripts/
├─ validacion_consulta.py
├─ requirements.txt
└─ README.md
```

## Modelo base

Por defecto se usa el modelo de Hugging Face:

```text
FacebookAI/xlm-roberta-base
```

Si prefieres usar una copia local ya descargada, define la variable de entorno `BASE_MODEL_DIR` con la ruta que corresponda en tu equipo.

## Flujo del proyecto

1. Cargar y validar dataset.
2. Normalizar columnas y etiquetas.
3. Generar splits reproducibles.
4. Entrenar el modelo.
5. Evaluar en test.
6. Guardar metricas y reportes.
7. Exportar el modelo con `save_pretrained`.
8. Probar inferencia local.

## Instalacion

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecucion paso a paso en Windows PowerShell

Ubicate en la carpeta del proyecto:

```powershell
cd ruta\al\proyecto\MODELO_VALIDACION_CONSULTA
```

Si ya tienes un entorno virtual activo, usalo. Si no, activa el que corresponda. Por ejemplo:

```powershell
.venv\Scripts\Activate.ps1
```

Luego instala dependencias si todavia no lo has hecho:

```powershell
pip install -r requirements.txt
```

Verifica rapido que Python este respondiendo:

```powershell
python --version
```

## Entrenamiento

```bash
python -m src.training.train
```

O bien:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_train.ps1
```

Tambien puedes lanzar el entrenamiento desde el punto de entrada principal:

```powershell
python validacion_consulta.py --mode train
```

## Evaluacion

```bash
python -m src.evaluation.evaluate
```

O desde el launcher:

```powershell
python validacion_consulta.py --mode evaluate
```

## Inferencia rapida

```bash
python -m src.inference.predict --text "¿Cómo postulo a una beca municipal?"
```

## Artefactos generados

- Splits en `data/splits/`
- Metricas en `artifacts/metrics/`
- Reportes y errores en `artifacts/reports/`
- Matriz de confusion en `artifacts/figures/`
- Modelo exportado en `models/trained/municipio_question_validator/`

## Compatibilidad con el script original

`validacion_consulta.py` queda como punto de entrada simple para entrenar, evaluar o clasificar preguntas nuevas sin entrar a los modulos internos.

Tambien existe el archivo `desarrollo_modelo_original.py`, que conserva una version previa a la organizacion del proyecto realizada con Codex. Ese archivo funciona como un codigo unico, independiente y autocontenido para cargar datos, entrenar, evaluar y guardar el modelo sin depender de la arquitectura modular en `src/`.

## Clasificar una pregunta nueva

Si quieres ver como el modelo clasifica una pregunta puntual:

```bash
python validacion_consulta.py --mode predict --text "¿Cómo postulo a una beca municipal?"
```

Tambien puedes usar directamente:

```bash
python -m src.inference.predict --text "¿Cómo postulo a una beca municipal?"
```

Eso imprime:

- el texto ingresado
- la etiqueta predicha
- la confianza
- las probabilidades por clase

## Modo interactivo

Si quieres probar muchas preguntas seguidas sin recargar el modelo en cada consulta:

```bash
python validacion_consulta.py --mode interactive
```

O directo desde el modulo:

```bash
python -m src.inference.predict --interactive
```

Ese modo deja el modelo cargado en memoria y responde en bucle hasta que escribas `salir`.

## Ejemplos practicos de uso

Entrenar el modelo:

```powershell
python validacion_consulta.py --mode train
```

Evaluar el modelo ya entrenado:

```powershell
python validacion_consulta.py --mode evaluate
```

Clasificar una sola pregunta:

```powershell
python validacion_consulta.py --mode predict --text "¿Dónde están ubicadas las oficinas de la municipalidad?"
```

Entrar en modo interactivo:

```powershell
python validacion_consulta.py --mode interactive
```

Salir del modo interactivo:

```text
salir
```

## Limitaciones

- El dataset actual es relativamente pequeno para generalizar de forma robusta.
- Las clases frontera requieren mas ejemplos reales.
- No hay tracking de experimentos avanzado en esta version.

## Mejoras futuras

- Agregar configuracion YAML o TOML por experimento.
- Integrar seguimiento de runs.
- Exportar error analysis enriquecido por subtipo.
- Agregar validacion cruzada o experimentos comparativos.
