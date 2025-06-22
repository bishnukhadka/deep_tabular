## HAS ERRORS. NOT ABLE TO USE NOW. 

import optuna
from benchmarking_utils.utils import suggest_from_space
from benchmarking_utils.opt_space.mambular_optspace import MambularOptSpace
from mambular.models.mambular import MambularRegressor

def objective(trial):
    space = MambularOptSpace()
    params = suggest_from_space(trial, space)
    
    # Example: you need to define your MLP and training loop here.
    # For illustration, we use a dummy accuracy based on the params.
    model = MambularRegressor(n_layers=params['n_layers'], dropout=params['dropout'])
    model.fit(
        X_train[:10], 
        y_train[:10], 
        max_epochs=5, 
        lr=params['lr'], 
        )
    
    result = model.evaluate(X_test[:10], y_test[:10])
    print(result)
    return result['Mean Squared Error']

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)