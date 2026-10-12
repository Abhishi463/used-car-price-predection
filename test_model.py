import pickle
import pandas as pd

with open('Auto_Price_Pred_Model', 'rb') as f:
    model = pickle.load(f)

test_cases = [
    {'make_model': 'Audi A3', 'hp_kW': 120, 'age': 1, 'km': 15000, 'Gears': 6, 'Gearing_Type': 'Manual'},
    {'make_model': 'Renault Clio', 'hp_kW': 80, 'age': 2, 'km': 30000, 'Gears': 5, 'Gearing_Type': 'Automatic'},
    {'make_model': 'Opel Corsa', 'hp_kW': 75, 'age': 0, 'km': 5000, 'Gears': 6, 'Gearing_Type': 'Semi-automatic'},
]
for tc in test_cases:
    df = pd.DataFrame([tc])
    pred = model.predict(df)
    print(tc['make_model'] + ': EUR ' + str(int(pred[0])))

print('All predictions successful!')
