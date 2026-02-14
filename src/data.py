import pandas as pd

def load_adult_data(source="uci", local_path="data/adult.csv"):
    """
    Load Adult Income dataset.
    source: "uci" (default) or "local"
    local_path: path to local CSV if source="local"
    Returns: DataFrame
    """
    if source == "uci":
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        columns = [
            "age","workclass","fnlwgt","education","education-num","marital-status",
            "occupation","relationship","race","sex","capital-gain","capital-loss",
            "hours-per-week","native-country","income"
        ]
        df = pd.read_csv(url, header=None, names=columns,
                         na_values=" ?", skipinitialspace=True)
    else:
        df = pd.read_csv(local_path)
    return df