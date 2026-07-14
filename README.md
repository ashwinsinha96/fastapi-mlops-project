# Iris Prediction API

A simple FastAPI service that serves a trained ML model to predict Iris flower species from four input features.

## Features

- REST API built with **FastAPI**
- Request validation using **Pydantic**
- Serves predictions from a pre-trained `model.pkl` (scikit-learn)
- Returns both numeric class ID and human-readable species name

## Project Structure

```
.
├── main.py          # FastAPI application
├── model.pkl         # Trained classification model (joblib dump)
├── requirements.txt
└── README.md
```

## Requirements

```
fastapi
uvicorn
joblib
numpy
scikit-learn
pydantic
```

Install with:

```bash
pip install -r requirements.txt
```

## Running the API

```bash
uvicorn main:app --reload
```

The API will be available at:

```
http://127.0.0.1:8000
```

Interactive docs (Swagger UI):

```
http://127.0.0.1:8000/docs
```

## Endpoints

### `GET /`

Health check.

**Response**

```json
{
  "message": "Iris Prediction API Running"
}
```

### `POST /predict`

Predict the Iris species from 4 input features: `[sepal_length, sepal_width, petal_length, petal_width]`.

**Request Body**

```json
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

**Response**

```json
{
  "prediction": 0,
  "species": "setosa"
}
```

**Class Mapping**

| ID | Species    |
|----|------------|
| 0  | setosa     |
| 1  | versicolor |
| 2  | virginica  |

## Error Handling

Invalid input (e.g. wrong number of features, non-numeric values) returns a `400` status code with an error detail message.

## Example (cURL)

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

## Notes

- `model.pkl` must be a scikit-learn classifier trained on the standard Iris dataset (4 features, 3 classes) and saved via `joblib.dump()`.
- For production, consider adding a `/health` endpoint, model versioning, and logging.

