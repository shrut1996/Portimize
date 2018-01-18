from keras.models import model_from_json
import pandas as pd

def lstm(portfolio_prices):
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    portfolio_prices = pd.read_csv('imio.csv')
    return loaded_model.predict(portfolio_prices)

