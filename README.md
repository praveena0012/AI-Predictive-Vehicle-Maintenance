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
