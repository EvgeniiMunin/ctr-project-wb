# ctr-project-wb


## Sem3. Pytest
В юнит тестах проверяем функционал отдельных модулей
- `test_make_dataset`: тестируем чтение датасета и сплит
- `test_ctr_transformer`: тестируем CTR и time transformer'ы
- `test_transformer`: тестируем трансформы девайса и пользователя
- `test_model_fetch`: тестируем асинхронную загрузку CatBoost предиктора и пайплайна с трансформерами
- `test_train_model`: тестируем обучение пайплайна

В интеграционных тестах проверяем сервис end-to-end. Отправляем запрос на mock сервера и проверяем статусы ответов и их содержимое.
Для этого создаем тест-клиент FastAPI `client = TestClient(app)`
- `test_integration_inference`: отправляем запросы на эндпоинты `/` и `/predict` с валидными и не валидными данными
и проверяем `response.json()` и статус код

### Зависимости
- Для тестирования воспользуемся фрейморком [pytest](https://docs.pytest.org/en/stable/).
- Для генерация синтетических данных для обучения пайплайна воспользуемся библиотекой [faker](https://faker.readthedocs.io/en/master/).
- Для проверки покрытия кода тестами воспользуемся [pytest-cov](https://pytest-cov.readthedocs.io/en/latest/index.html)

Для уставновки зависимостей предварительно внесем изменения в `pyproject.toml`

```toml
[tool.poetry.dev-dependencies]
pytest = "8.3.3"
pytest-asyncio = "^0.24.0"
pytest-cov = "5.0.0"
faker = "^30.3.0"

# omit scripts from coverage report
[tool.coverage.run]
omit = [
    "src/inference/make_request.py",
    "src/inference/make_concurrent_request.py"
]
```

и обновим `poetry.lock`

```sh
poetry lock 
poetry install
```

### Commands
```sh
# execute project from root
export PYTHONPATH=$PYTHONPATH:$(pwd) 

# run pytest
poetry run pytest tests/

# check code coverage 
poetry run pytest --cov=. --cov-report=term-missing
```

В случае успешного прохождения тестов также должен отобразится репорт по покрытию каждого модуля:

```
---------- coverage: platform darwin, python 3.12.4-final-0 ----------
Name                                         Stmts   Miss  Cover   Missing
--------------------------------------------------------------------------
app.py                                          84      4    95%   96-97, 154, 182
src/__init__.py                                  0      0   100%
src/data/__init__.py                             0      0   100%
src/data/make_dataset.py                         9      0   100%
src/entities/__init__.py                         0      0   100%
src/entities/feature_params.py                   7      0   100%
src/entities/split_params.py                     5      0   100%
src/entities/train_params.py                     9      0   100%
src/entities/train_pipeline_params.py           35      1    97%   51
src/features/CtrTransformer.py                  38      0   100%
src/features/DeviceCountTransformer.py          20      0   100%
src/features/UserCountTransformer.py            26      0   100%
src/features/__init__.py                         0      0   100%
src/features/build_transformer.py               32      0   100%
src/inference/__init__.py                        0      0   100%
src/models/__init__.py                           0      0   100%
src/models/model_fit_predict.py                 22      1    95%   49
tests/conftest.py                               48      0   100%
tests/data/test_make_dataset.py                 22      0   100%
tests/features/test_ctr_transformer.py          23      0   100%
tests/features/test_transformer.py              46      0   100%
tests/models/test_integration_inference.py      31      0   100%
tests/models/test_model_fetch.py                19      0   100%
tests/models/test_train_model.py                48      0   100%
--------------------------------------------------------------------------
TOTAL                                          524      6    99%

```