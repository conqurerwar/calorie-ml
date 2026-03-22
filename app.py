"""
CalorieML — Flask Backend
Run: python app.py
Then open: http://localhost:5000
"""

from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np
import os

app = Flask(__name__, static_folder='static')

# ── Load all models at startup ──────────────────────────────
BASE = os.path.join(os.path.dirname(__file__), 'models')

print("Loading models...")
SCALER = joblib.load(os.path.join(BASE, 'scaler.pkl'))
MODELS = {
    'Linear Regression':  joblib.load(os.path.join(BASE, 'linear_regression.pkl')),
    'Ridge Regression':   joblib.load(os.path.join(BASE, 'ridge_regression.pkl')),
    'Random Forest':      joblib.load(os.path.join(BASE, 'random_forest.pkl')),
    'Gradient Boosting':  joblib.load(os.path.join(BASE, 'gradient_boosting.pkl')),
}
print("✅ All 4 models loaded!")

FEATURES = ['protein_g', 'carbs_g', 'fat_g', 'fiber_g', 'sugar_g']

MODEL_META = {
    'Linear Regression':  {'MAE': 23.23, 'RMSE': 28.97, 'R2': 0.9318, 'icon': '📏', 'color': '#38bdf8'},
    'Ridge Regression':   {'MAE': 23.23, 'RMSE': 28.97, 'R2': 0.9318, 'icon': '🎯', 'color': '#6ee7b7'},
    'Random Forest':      {'MAE': 25.40, 'RMSE': 32.26, 'R2': 0.9154, 'icon': '🌲', 'color': '#f59e0b'},
    'Gradient Boosting':  {'MAE': 24.57, 'RMSE': 30.64, 'R2': 0.9237, 'icon': '🚀', 'color': '#a78bfa'},
}
BEST_MODEL = 'Ridge Regression'


def classify_calories(cal):
    if cal < 400:
        return 'low'
    elif cal < 600:
        return 'medium'
    else:
        return 'high'


def predict_single(model_name, x_raw):
    """Predict calories using the specified model."""
    model = MODELS[model_name]
    x = np.array([x_raw])

    # Linear models use scaler (wrapped in Pipeline)
    if model_name in ('Linear Regression', 'Ridge Regression'):
        x_scaled = SCALER.transform(
            np.array([[x_raw[i] for i in range(len(FEATURES))]],
                     dtype=float)
        )
        pred = model.named_steps['model'].predict(x_scaled)[0]
    else:
        # Tree-based models use raw features
        import pandas as pd
        x_df = pd.DataFrame([x_raw], columns=FEATURES)
        pred = model.predict(x_df)[0]

    return max(0.0, float(pred))


# ── Routes ─────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    POST /api/predict
    Body: { protein_g, carbs_g, fat_g, fiber_g, sugar_g, model }
    Returns: { calories, calorie_class, model, all_predictions, macro_breakdown }
    """
    try:
        data = request.get_json()

        # Validate inputs
        errors = {}
        x_raw = []
        for feat in FEATURES:
            val = data.get(feat)
            if val is None:
                errors[feat] = 'Missing value'
            else:
                try:
                    val = float(val)
                    if val < 0:
                        errors[feat] = 'Must be >= 0'
                    else:
                        x_raw.append(val)
                except (TypeError, ValueError):
                    errors[feat] = 'Must be a number'

        if errors:
            return jsonify({'error': 'Validation failed', 'details': errors}), 400

        model_name = data.get('model', BEST_MODEL)
        if model_name not in MODELS:
            return jsonify({'error': f'Unknown model: {model_name}'}), 400

        # Predict with selected model
        calories = predict_single(model_name, x_raw)
        calorie_class = classify_calories(calories)

        # Predict with ALL models for comparison
        all_preds = {}
        for name in MODELS:
            try:
                pred = predict_single(name, x_raw)
                all_preds[name] = {
                    'calories': round(pred, 2),
                    'calorie_class': classify_calories(pred),
                    **MODEL_META[name],
                    'is_best': name == BEST_MODEL,
                }
            except Exception as e:
                all_preds[name] = {'error': str(e)}

        # Macro calorie breakdown (4 cal/g protein+carb, 9 cal/g fat)
        macro_breakdown = {
            'protein_kcal': round(x_raw[0] * 4, 1),
            'carbs_kcal':   round(x_raw[1] * 4, 1),
            'fat_kcal':     round(x_raw[2] * 9, 1),
        }

        return jsonify({
            'model':          model_name,
            'calories':       round(calories, 2),
            'calorie_class':  calorie_class,
            'all_predictions': all_preds,
            'macro_breakdown': macro_breakdown,
            'inputs': dict(zip(FEATURES, x_raw)),
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/models', methods=['GET'])
def get_models():
    """Return model metadata."""
    return jsonify({
        'models': MODEL_META,
        'best_model': BEST_MODEL,
        'features': FEATURES,
    })


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'models_loaded': list(MODELS.keys())})


if __name__ == '__main__':
    print("\n🚀 CalorieML Flask Server")
    print("   Open http://localhost:5000 in your browser\n")
    app.run(debug=True, port=5000)
