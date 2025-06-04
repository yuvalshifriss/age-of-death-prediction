import pandas as pd


def load_data(file_path: str, only_dead: bool = False) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    if only_dead:
        df = df[df['DEATH_EVENT'] == 1]
    return df


