from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

def train_model(x_train, y_train):
    #inicijalizacija
    model = LinearRegression()

    #treniranje
    model.fit(x_train,y_train)

    joblib.dump(model,'trained_model.pkl')
    return model

def evaluate_model(model,x_test, y_test):
    
    y_pred = model.predict(x_test)

    #mse
    mse = mean_squared_error(y_test,y_pred)
    print(f'Mean Squared Error: {mse}')
    return mse
