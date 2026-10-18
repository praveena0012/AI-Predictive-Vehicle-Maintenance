# BusPredict demo

A small working prototype based on the proposal: Python/scikit-learn model + FastAPI prediction service + browser dashboard. It uses a clearly labelled sample dataset for demonstration; replace it with a validated public bus dataset before reporting results.

## Run
1. Install Python 3.10+.
2. From this folder: `pip install -r requirements.txt`
3. Train: `python backend/train_model.py`
4. Start API: `uvicorn backend.main:app --reload`
5. Open `frontend/index.html` in a browser.

The browser calls `http://127.0.0.1:8000/predict`. The API also exposes `/docs` for interactive testing.

## Next integration steps
- Replace sample data and document its licence, provenance, target definition, and class balance.
- Add MySQL tables for vehicles, inspections, predictions, users, and alerts.
- Add authentication and role-based access.
- Consume the API from JavaFX, or package the prediction service behind a controlled integration layer.
- Evaluate with stratified cross-validation and prioritize recall for Critical Issue.


## Project Features

BusPredict is a demonstration project that explores how artificial intelligence can support predictive maintenance for buses.

The prototype includes:

- **Machine Learning:** Uses a scikit-learn model to generate maintenance predictions.
- **Prediction API:** Uses FastAPI to provide an endpoint for prediction requests.
- **Web Dashboard:** Provides a browser-based interface for interacting with the prediction service.
- **Sample Dataset:** Uses sample data for demonstration purposes.
- **Future Improvements:** Identifies opportunities to integrate validated datasets, a database, authentication, and further model evaluation.

## Project Goal

The goal of BusPredict is to help identify potential vehicle maintenance issues and support maintenance planning. The prototype requires further validation with suitable real-world data before its predictions can be relied upon in operational settings.