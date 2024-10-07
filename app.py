import asyncio
import sys
import logging
import os
import threading
import time
from typing import List
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import joblib
from catboost import CatBoostClassifier
from fastapi.concurrency import run_in_threadpool

from src.entities.train_pipeline_params import (
    TrainingPipelineParams,
    read_training_pipeline_params,
)
from src.features.CtrTransformer import CtrTransformer
from src.models.model_fit_predict import predict_model

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


class AdOpportunity(BaseModel):
    data: list
    features: list


class ClickResponse(BaseModel):
    device_ip: str
    click_proba: float


app = FastAPI()

# asyncio Lock for loading models
model_lock = asyncio.Lock()
# global vars
model = None
ctr_transformer = None
is_model_loading = False
is_transformer_loading = False


@app.get("/")
def main():
    return "it is entry point of our predictor"


async def load_models_async(training_pipeline_params: TrainingPipelineParams, force_reload=False):
    """Load model and transformer asynchronously under the protection of the asyncio mutex."""

    global model, ctr_transformer, is_model_loading, is_transformer_loading
    # print("app/load_models_async call")

    # Ensure that only one coroutine can load models at a time
    async with model_lock:
        # print("app/load_models_async lock: ", model, is_model_loading, model is None and not is_model_loading)

        # return models if they are already loaded
        if (model is None and not is_model_loading) or force_reload:
            is_model_loading = True
            logger.info("app/load_models_async fetch model")

            # test purpose
            # await asyncio.sleep(300)
            # print("app/load_models_async fetch model")

            model = await asyncio.to_thread(
                joblib.load,
                training_pipeline_params.output_model_path
            )
            is_model_loading = False

        if (ctr_transformer is None and not is_transformer_loading) or force_reload:
            is_transformer_loading = True
            logger.info("app/load_models_async fetch ctr_transformer")
            ctr_transformer = await asyncio.to_thread(
                joblib.load,
                training_pipeline_params.output_ctr_transformer_path
            )
            is_transformer_loading = False

    return model, ctr_transformer


def check_models(model, ctr_transformer):
    """Check if models are loaded, load asynchronously if not."""
    #model, ctr_transformer = load_models_async(training_pipeline_params)
    if model is None or ctr_transformer is None:
        logger.error("app/check_models models are None")
        raise HTTPException(status_code=400, detail="Models are unavailable")


def check_schema(features: list, training_pipeline_params: TrainingPipelineParams):
    if not set(training_pipeline_params.feature_params.ctr_features).issubset(
        set(features)
    ):
        logger.error("app/check_schema missing columns")
        raise HTTPException(
            status_code=400, detail=f"Missing features in schema {features}"
        )


def make_predict(
    data: list,
    features: list,
    model: CatBoostClassifier,
    ctr_transformer: CtrTransformer,
    training_pipeline_params: TrainingPipelineParams,
) -> List[ClickResponse]:

    # Log the current thread ID
    current_thread = threading.current_thread()
    logger.info(f"app/predict: thread {current_thread.name} with ID {current_thread.ident}")

    # test purpose
    #time.sleep(300)
    #print("app/make_predict fetch model")

    # Check data schema
    check_schema(features, training_pipeline_params)

    # Ensure the model and transformer are loaded
    check_models(model, ctr_transformer)
    logger.debug("app/predict check_models passed")

    df = pd.DataFrame(data, columns=features)

    features = ctr_transformer.transform(df)
    predicted_proba, _ = predict_model(model, features)

    logger.debug("df.device_ip: ", df["device_ip"].values[0])
    logger.debug("predicted_proba", predicted_proba, predicted_proba[0, 1])

    return [
        ClickResponse(
            device_ip=df["device_ip"].values[0],
            click_proba=round(predicted_proba[0, 1], 4),
        )
    ]


# Use dependency injection to ensure model is loaded before proceeding with the prediction
# Dependency function responsible for loading the models
async def predict_dependencies():

    if is_model_loading or is_transformer_loading:
        raise HTTPException(status_code=503, detail="Model loading in progress")

    logger.debug("app/predict run")

    config_path = "configs/train_config.yaml"
    training_pipeline_params: TrainingPipelineParams = read_training_pipeline_params(
        config_path
    )
    logger.debug(f"app/predict training_pipeline_params: {training_pipeline_params}")

    # Load the models asynchronously (this will be fast if they are already loaded)
    model, ctr_transformer = await load_models_async(training_pipeline_params)

    return model, ctr_transformer, training_pipeline_params


@app.post("/predict/", response_model=List[ClickResponse])
async def predict(request: AdOpportunity, deps: tuple = Depends(predict_dependencies)):
    # The 'deps' argument is populated with the return value of predict_dependencies
    model, ctr_transformer, training_pipeline_params = deps

    return await run_in_threadpool(
        make_predict,
        request.data, request.features, model, ctr_transformer, training_pipeline_params
    )


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))
