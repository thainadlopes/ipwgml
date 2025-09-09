import numpy as np
import torch
import xgboost as xgb
import optuna
import random
from ipwgml.input import GMI
from ipwgml.target import TargetConfig
from ipwgml.pytorch.datasets import SPRTabular
from torch.utils.data import DataLoader
from ipwgml.evaluation import Evaluator
from ipwgml.pytorch import PytorchRetrieval

import threading
from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error 

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

inputs = [GMI(normalize="minmax", nan=-1.5, include_angles=False)]
target_config = TargetConfig(min_rqi=0.5)
geometry = "on_swath"
batch_size = 1024
ipwgml_path = "/storage/ipwgml"


def load_limited(loader):
    X_list, y_list = [], []
    for x, y in loader:
        x = x.numpy()
        y = y.numpy()
        mask = np.isfinite(y)
        x, y = x[mask], y[mask]
        X_list.append(x)
        y_list.append(y)
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y


print("Loading training data...")
training_data = SPRTabular(
    reference_sensor="gmi",
    geometry=geometry,
    split="training",
    retrieval_input=inputs,
    batch_size=batch_size,
    target_config=target_config,
    stack=True,
    ipwgml_path=ipwgml_path,
    download=False
)
training_loader = DataLoader(training_data, shuffle=True, batch_size=None, num_workers=4)

validation_data = SPRTabular(
    reference_sensor="gmi",
    geometry="on_swath",
    split="validation",
    retrieval_input=inputs,
    batch_size=batch_size,
    target_config=target_config,
    stack=True,
    ipwgml_path=ipwgml_path,
    download=False,
    shuffle=False
)

validation_loader = DataLoader(validation_data, shuffle=True, batch_size=None)
training_loader = DataLoader(training_data, batch_size=None, num_workers=4)
validation_loader = DataLoader(validation_data, shuffle=True, batch_size=None, num_workers=4)
X_train, y_train_full = load_limited(training_loader)
X_val, y_val_full = load_limited(validation_loader)

y_precip = y_train_full
y_precip_mask = (y_train_full > 1e-3).astype(int)
y_heavy_mask = (y_train_full > 10).astype(int)
y_val_precip = y_val_full
y_val_precip_mask = (y_val_full > 1e-3).astype(int)
y_val_heavy_mask = (y_val_full > 10).astype(int)

GPU_IDS = [0, 2]
TRIAL_COUNT = 0
TRIAL_LOCK = threading.Lock()

def get_gpu_id():
    global TRIAL_COUNT
    with TRIAL_LOCK:
        gpu_id = GPU_IDS[TRIAL_COUNT % len(GPU_IDS)]
        TRIAL_COUNT += 1
    return gpu_id

# Intervalo de hiperparametros 
def get_regressor_params(trial):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 2000, 3500, log=True),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.01, log=True),
        'max_depth': trial.suggest_int('max_depth', 8, 14),
        'gamma': trial.suggest_float('gamma', 1e-8, 8.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.2, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 5.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }

# Busca de hiperparametros 
def objective_reg(trial):
    gpu_id = get_gpu_id()
    params = { **get_regressor_params(trial), "tree_method": "gpu_hist", "gpu_id": gpu_id, "random_state": 42 }
    model = xgb.XGBRegressor(**params, objective="reg:squarederror")
    model.fit(X_train, y_precip, eval_set=[(X_val, y_val_precip)], eval_metric='rmse', early_stopping_rounds=50, verbose=False)
    preds = model.predict(X_val)
    mse = mean_squared_error(y_val_precip, preds)
    return mse

# N_TRIALS = 50
# study_reg = optuna.create_study(direction="minimize")
# study_reg.optimize(objective_reg, n_trials=N_TRIALS, n_jobs=3)

def train_model(model_class, params, gpu_id, X_train, y_train, X_val, y_val):
    print(f"Iniciando treinamento do modelo na GPU {gpu_id}")
    model_params = params.copy()
    model_params["gpu_id"] = gpu_id
    model = model_class(**model_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
    print(f"Treinamento na GPU {gpu_id} finalizado.")
    return model

best_params_reg = {
'n_estimators': 2428, 'learning_rate': 0.00874720109940644, 'max_depth': 11, 'gamma': 0.1302993987998777, 
'subsample': 0.7792092554134096, 'colsample_bytree': 0.8891924297661156, 'reg_alpha': 0.0006835672023367944, 
'reg_lambda': 9.536400865394835, "tree_method": "gpu_hist", "objective": "reg:squarederror", "random_state": 42
}

training_jobs = [
    {"model_class": xgb.XGBRegressor, "params": best_params_reg, "y_train": y_precip, "y_val": y_val_precip},
    {"model_class": xgb.XGBRegressor, "params": best_params_reg, "y_train": y_precip_mask, "y_val": y_val_precip_mask},
    {"model_class": xgb.XGBRegressor, "params": best_params_reg, "y_train": y_heavy_mask, "y_val": y_val_heavy_mask}
]

trained_models = Parallel(n_jobs=3)( 
    delayed(train_model)(
        model_class=job["model_class"], params=job["params"], gpu_id=gpu_id,
        X_train=X_train, y_train=job["y_train"], X_val=X_val, y_val=job["y_val"]
    ) for gpu_id, job in enumerate(training_jobs)
)
model_precip, model_prob, model_prob_heavy = trained_models


class XGBoostOutput(torch.nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = models

    def forward(self, x):
        x_np = x.detach().cpu().numpy()
        precip = self.models["surface_precip"].predict(x_np)
        
        prob_raw = self.models["probability_of_precip"].predict(x_np)
        prob = np.clip(prob_raw, 0, 1) 
        
        prob_heavy_raw = self.models["probability_of_heavy_precip"].predict(x_np)
        prob_heavy = np.clip(prob_heavy_raw, 0, 1) 

        out = {
            "surface_precip": torch.from_numpy(precip).unsqueeze(-1),
            "probability_of_precip": torch.from_numpy(prob).unsqueeze(-1),
            "probability_of_heavy_precip": torch.from_numpy(prob_heavy).unsqueeze(-1),
        }
        return out

xgb_multi = XGBoostOutput({
    "surface_precip": model_precip,
    "probability_of_precip": model_prob,
    "probability_of_heavy_precip": model_prob_heavy
})

wrapped = PytorchRetrieval(
    model=xgb_multi, retrieval_input=inputs, stack=True, device=torch.device("cuda:0"), logits=False
)

evaluator = Evaluator(
    reference_sensor="gmi", geometry=geometry, retrieval_input=inputs, ipwgml_path=ipwgml_path, download=False
)
print("\nRunning evaluation...")
evaluator.evaluate(retrieval_fn=wrapped, input_data_format="tabular", batch_size=4048, n_processes=1)

print("\nPrecipitation quantification")
print(evaluator.get_precip_quantification_results(name="XGBOOST (GMI)").T.to_string())
print("\nPrecipitation detection")
print(evaluator.get_precip_detection_results(name="XGBOOST (GMI)").T.to_string())
print("\nHeavy precipitation detection")
print(evaluator.get_heavy_precip_detection_results(name="XGBOOST (GMI)").T.to_string())