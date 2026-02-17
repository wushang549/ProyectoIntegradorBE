# ProyectoIntegradorBE

Backend en FastAPI para analisis `granulate`.

## Ejecutar local

1. (Opcional) Crear y activar entorno virtual.
2. Instalar dependencias:

```bash
pip install -r requirements.txt
```

3. Levantar servidor:

```bash
uvicorn main:app --reload
```

El API queda disponible en:
- `http://127.0.0.1:8000`
- Swagger UI: `http://127.0.0.1:8000/docs`

## Ollama local (labels de clusters y jerarquia)

El pipeline de analisis usa un LLM local por Ollama para etiquetas cortas.

1. Instalar Ollama: `https://ollama.com/download`
2. Levantar servicio local (default `http://localhost:11434`)
3. Descargar modelo:

```bash
ollama pull gemma3:1b
```

Si Ollama no esta disponible, el backend cae automaticamente a etiquetas TF-IDF.

## Nuevo API de analisis completo

Base: `/v1/analysis`

- `POST /v1/analysis` crea una corrida (texto o CSV) y responde inmediato con `analysis_id`.
- `GET /v1/analysis/{analysis_id}/status` para polling.
- `GET /v1/analysis/{analysis_id}/overview`
- `GET /v1/analysis/{analysis_id}/map`
- `GET /v1/analysis/{analysis_id}/clusters`
- `GET /v1/analysis/{analysis_id}/granulate?include_items=true|false`
- `GET /v1/analysis/{analysis_id}/hierarchy`

### Input CSV soportado

Reglas de validacion:
- extension `.csv`
- archivo no vacio
- tamano maximo `50MB`
- columna de texto aceptada: `text` o `transcript` (tambien soporta alias comunes como `message`/`content`)
- al menos una fila con `text` no vacio

### Ejemplo rapido (CSV)

```bash
curl -X POST "http://127.0.0.1:8000/v1/analysis" \
  -F "input_type=csv" \
  -F "file=@sample.csv" \
  -F "options={\"k_clusters\":8,\"granulate\":true,\"granulate_max_rows\":200,\"granulate_return_items\":false}"
```

### Ejemplo rapido (texto)

```bash
curl -X POST "http://127.0.0.1:8000/v1/analysis" \
  -F "input_type=text" \
  -F "text=The app is fast but support response was slow and pricing is high." \
  -F "options={\"k_clusters\":4,\"granulate\":true}"
```

Luego consultar estado:

```bash
curl "http://127.0.0.1:8000/v1/analysis/{analysis_id}/status"
```

Los artefactos se persisten en `uploads/{analysis_id}/`.
