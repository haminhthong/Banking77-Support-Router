"""Artifact contract: chỉ một layout canonical, không manifest/hash alias."""

from src.banking_router.config import ARTIFACTS_DIR


def test_canonical_artifacts_exist():
    required = {"intent_model.joblib", "metadata.json", "taxonomy.json", "routing_policy.json"}
    assert {path.name for path in ARTIFACTS_DIR.iterdir() if path.is_file()} >= required
    assert not any((ARTIFACTS_DIR / name).exists() for name in ("manifest.json", "model_config.json", "policy.json"))
