import pandas as pd
from sklearn.model_selection import train_test_split
import os

from common_utils import load_data

DATA_PATH = os.path.abspath(os.path.join(os.getcwd(), '..', 'data', 'heart_failure_clinical_records.csv'))
TRAIN_VAL_PATH = os.path.abspath(os.path.join(os.getcwd(), '..', 'data', 'train_val_data.csv'))
TEST_PATH = os.path.abspath(os.path.join(os.getcwd(), '..', 'data', 'test_data.csv'))


def main():
    dead_df = load_data(DATA_PATH, True)

    # Split into training/validation and test
    train_val_df, test_df = train_test_split(dead_df, test_size=0.15, random_state=42)

    train_val_df.to_csv(TRAIN_VAL_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print(f"✅ Saved train/val data to: {TRAIN_VAL_PATH}")
    print(f"✅ Saved test data to: {TEST_PATH}")


if __name__ == '__main__':
    main()
