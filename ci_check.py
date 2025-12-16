"""
CI smoke test: ensure models can be deserialized with current requirements.
Exits non-zero if loading fails so pushes fail early instead of breaking prod.
"""
import sys
import traceback

try:
    from models.heart_disease_prediction_optimized import OptimizedHeartDiseasePredictor
    p = OptimizedHeartDiseasePredictor()
    ok = p.load_models()
    assert ok and getattr(p, 'best_model_instance', None) is not None, "Model artifacts failed to load"
    print({
        'loaded': ok,
        'best': getattr(p, 'best_model', None),
        'has_instance': p.best_model_instance is not None,
    })
except Exception as e:
    traceback.print_exc()
    print(f"CI model load failed: {e}")
    sys.exit(1)
