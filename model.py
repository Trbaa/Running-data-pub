from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

def train_model_lr(x_train, y_train):
      #inicijalizacija
    lr_model = LinearRegression()

    #treniranje
    lr_model.fit(x_train,y_train)

    joblib.dump(lr_model,'model/trained_model_lr.pkl')
    return lr_model

def train_model_rf(x_train,y_train):
    rf_model = RandomForestRegressor(n_estimators=100,random_state=42)
    rf_model.fit(x_train,y_train)
    joblib.dump(rf_model,'model/trained_model_rf.pkl')

    return rf_model


def evaluate_model(model,x_test, y_test,model_name='Model'):
    
    y_pred = model.predict(x_test)

    #mse
    mse = mean_squared_error(y_test,y_pred)
    print(f'{model_name} - Mean Squared Error: {mse}')
    return mse
