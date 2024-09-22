import pytest
from unittest import mock
from app import load_models_async, model, ctr_transformer
from src.entities.feature_params import FeatureParams
from src.entities.split_params import SplittingParams
from src.entities.train_params import TrainingParams
from src.entities.train_pipeline_params import TrainingPipelineParams


#@pytest.fixture(autouse=True)
#def reset_model():
#    global model, ctr_transformer
#    model = None
#    ctr_transformer = None


@pytest.mark.asyncio
async def test_load_models_async():
    #global model, ctr_transformer
    #model = None
    #ctr_transformer = None

    # arrange
    # Instantiating the required parameters for the training pipeline
    splitting_params = SplittingParams(val_size=0.2, random_state=42)
    feature_params = FeatureParams(
        count_features=["feature1", "feature2"],
        ctr_features=["category1", "category2"],
        target_col="target"
    )
    training_params = TrainingParams(model_type="CatBoost", learning_rate=0.1, n_estimators=100)

    # Now, instantiate the TrainingPipelineParams
    training_pipeline_params = TrainingPipelineParams(
        output_model_path="../models/model.pkl",
        output_transformer_path="../models/transformer.pkl",
        output_ctr_transformer_path="../models/ctr_transformer.pkl",
        metric_path="../metrics/metrics.json",
        splitting_params=splitting_params,
        feature_params=feature_params,
        train_params=training_params
    )

    # Mock joblib.load to avoid loading actual files
    with mock.patch("app.joblib.load") as mocked_joblib_load:
        # Mock return values for the model and transformer loading
        mocked_joblib_load.side_effect = ["mocked_model", "mocked_transformer"]

        # act
        loaded_model, loaded_transformer = await load_models_async(training_pipeline_params, force_reload=True)

        # assert
        # Check that joblib.load was called twice (once for the model, once for the transformer)
        assert mocked_joblib_load.call_count == 2

        # Check that the model and transformer were loaded correctly
        assert loaded_model == "mocked_model"
        assert loaded_transformer == "mocked_transformer"
