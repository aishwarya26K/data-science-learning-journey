import pickle
import warnings
import sys

warnings.filterwarnings('ignore')

file_path = "d:/lrnbay/data-science-learning-journey/Projects/EDA Projects/Netflix Stock/stock_price_model.pkl"

with open(file_path, 'rb') as f:
    model_data = pickle.load(f)

with open('model_info.txt', 'w') as out:
    out.write(f"Keys in dictionary: {list(model_data.keys())}\n")
    for key, value in model_data.items():
        out.write(f"\nKey: {key}\n")
        out.write(f"Type: {type(value)}\n")
        if hasattr(value, 'feature_names_in_'):
            out.write(f"Features in {key}: {list(value.feature_names_in_)}\n")
        if hasattr(value, 'get_params'):
            out.write(f"Params: {value.get_params()}\n")
