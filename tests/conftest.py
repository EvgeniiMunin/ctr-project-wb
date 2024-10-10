import os
import logging
import random
import sys

import pandas as pd
import pytest
from faker import Faker

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

fake = Faker()

CATEGORIES = ["A", "B", "C", "D"]


@pytest.fixture()
def generate_synthetic_data():
    def _generate_synthetic_data(num_rows=50):
        data = {
            "id": [random.randint(1_000_000_000, 9_999_999_999_999) for _ in range(num_rows)],
            "click": [random.randint(0, 1) for _ in range(num_rows)],
            "hour": [fake.date_time_this_year().strftime("%y%m%d%H") for _ in range(num_rows)],
            "C1": [random.randint(1000, 9999) for _ in range(num_rows)],
            "banner_pos": [random.randint(0, 7) for _ in range(num_rows)],
            "site_id": [fake.uuid4()[:8] for _ in range(num_rows)],
            "site_domain": [fake.uuid4()[:8] for _ in range(num_rows)],
            "site_category": [fake.uuid4()[:8] for _ in range(num_rows)],
            "app_id": [fake.uuid4()[:8] for _ in range(num_rows)],
            "app_domain": [fake.uuid4()[:8] for _ in range(num_rows)],
            "app_category": [fake.uuid4()[:8] for _ in range(num_rows)],
            "device_id": [fake.uuid4()[:8] for _ in range(num_rows)],
            "device_ip": [fake.ipv4() for _ in range(num_rows)],
            "device_model": [fake.uuid4()[:8] for _ in range(num_rows)],
            "device_type": [random.randint(0, 4) for _ in range(num_rows)],
            "device_conn_type": [random.randint(0, 3) for _ in range(num_rows)],
            "C14": [random.randint(10000, 30000) for _ in range(num_rows)],
            "C15": [random.randint(300, 600) for _ in range(num_rows)],
            "C16": [random.randint(30, 80) for _ in range(num_rows)],
            "C17": [random.randint(1000, 5000) for _ in range(num_rows)],
            "C18": [random.randint(0, 5) for _ in range(num_rows)],
            "C19": [random.randint(0, 5000) for _ in range(num_rows)],
            "C20": [random.randint(-1, 5000) for _ in range(num_rows)],
            "C21": [random.randint(0, 100) for _ in range(num_rows)],
        }

        df = pd.DataFrame(data)
        return df

    return _generate_synthetic_data


@pytest.fixture()
def generate_synthetic_processed_data(generate_synthetic_data):
    raw_df = generate_synthetic_data(100)

    def _generate_synthetic_processed_data():
        num_rows = raw_df.shape[0]
        processed_data = {
            "device_id_count": [random.randint(0, 5000) for _ in range(num_rows)],
            "device_ip_count": [random.randint(0, 5000) for _ in range(num_rows)],
            "hour_of_day": [random.randint(0, 23) for _ in range(num_rows)],
            "day_of_week": [fake.day_of_week() for _ in range(num_rows)],
            "hourly_user_count": [random.randint(1, 200000) for _ in range(num_rows)],
        }

        processed_df = pd.DataFrame(processed_data)
        return pd.concat([raw_df, processed_df], axis=1)

    return _generate_synthetic_processed_data


@pytest.fixture()
def processed_dataset_path(generate_synthetic_processed_data):
    curdir = os.path.dirname(__file__)
    logger.info(curdir)

    df = generate_synthetic_processed_data()
    file_path = curdir + "/synthetic_processed.csv"

    df.to_csv(file_path, index=False)
    return str(file_path)


@pytest.fixture()
def dataset_path(generate_synthetic_data):
    curdir = os.path.dirname(__file__)
    logger.info(curdir)

    df = generate_synthetic_data(100)
    file_path = curdir + "/synthetic_train.csv"

    df.to_csv(file_path, index=False)
    return str(file_path)


@pytest.fixture()
def target_col():
    return "click"
