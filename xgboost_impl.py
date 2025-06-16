import numpy as np
import torch
import xgboost as xgb
import optuna
from ipwgml.input import GMI
from ipwgml.target import TargetConfig
from ipwgml.pytorch.datasets import SPRTabular
from torch.utils.data import DataLoader
from ipwgml.evaluation import Evaluator
from ipwgml.pytorch import PytorchRetrieval

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

validation_data = SPRTabular(
    reference_sensor="gmi",
    geometry=geometry,
    split="validation",
    retrieval_input=inputs,
    batch_size=batch_size,
    target_config=target_config,
    stack=True,
    ipwgml_path=ipwgml_path,
    download=False,
    shuffle=False
)

training_loader = DataLoader(training_data, batch_size=None, num_workers=4)
validation_loader = DataLoader(validation_data, batch_size=None, num_workers=4)

X_train, y_train_full = load_limited(training_loader)
X_val, y_val_full = load_limited(validation_loader)


y_precip = y_train_full
y_precip_mask = (y_train_full > 1e-3).astype(float)
y_heavy_mask = (y_train_full > 10).astype(float)

y_val_precip = y_val_full
y_val_precip_mask = (y_val_full > 1e-3).astype(float)
y_val_heavy_mask = (y_val_full > 10).astype(float)

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 500, 3000),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 1.0, log=True),
        "gamma": trial.suggest_float("gamma", 0, 2),
        "tree_method": "gpu_hist",
        "random_state": 42,
        "n_jobs": -1
    }

    reg_model = xgb.XGBRegressor(**params, objective="reg:squarederror")
    reg_model.fit(X_train, y_precip, eval_set=[(X_val, y_val_precip)], early_stopping_rounds=50, verbose=False)

    y_pred = reg_model.predict(X_val)
    mse = np.mean((y_pred - y_val_precip) ** 2)
    return mse


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

best_params = {
    **study.best_params,
    "tree_method": "gpu_hist",
    "objective": "reg:squarederror",
    "n_jobs": -1,
    "random_state": 42
}

params_clf = {
    "n_estimators": 1500,
    "max_depth": 8,
    "learning_rate": 0.02,
    "tree_method": "gpu_hist",
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "objective": "binary:logistic",
    "eval_metric": "logloss"
}

model_precip = xgb.XGBRegressor(**best_params)
model_prob = xgb.XGBClassifier(**params_clf)
model_prob_heavy = xgb.XGBClassifier(**params_clf)

model_precip.fit(X_train, y_precip, eval_set=[(X_val, y_val_precip)], early_stopping_rounds=50, verbose=False)
model_prob.fit(X_train, y_precip_mask, eval_set=[(X_val, y_val_precip_mask)], early_stopping_rounds=50, verbose=False)
model_prob_heavy.fit(X_train, y_heavy_mask, eval_set=[(X_val, y_val_heavy_mask)], early_stopping_rounds=50, verbose=False)


class XGBMultiOutput(torch.nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = models

    def forward(self, x):
        x_np = x.detach().cpu().numpy()
        
        precip = self.models["surface_precip"].predict(x_np)
        prob = self.models["probability_of_precip"].predict_proba(x_np)[:, 1]
        prob_heavy = self.models["probability_of_heavy_precip"].predict_proba(x_np)[:, 1]

        out = {
            "surface_precip": torch.from_numpy(precip).unsqueeze(-1),
            "probability_of_precip": torch.from_numpy(prob).unsqueeze(-1),
            "probability_of_heavy_precip": torch.from_numpy(prob_heavy).unsqueeze(-1),
        }
        return out

xgb_multi = XGBMultiOutput({
    "surface_precip": model_precip,
    "probability_of_precip": model_prob,
    "probability_of_heavy_precip": model_prob_heavy
})

wrapped = PytorchRetrieval(
    model=xgb_multi,
    retrieval_input=inputs,
    stack=True,
    device=torch.device("cuda")
)

evaluator = Evaluator(
    reference_sensor="gmi",
    geometry=geometry,
    retrieval_input=inputs,
    ipwgml_path=ipwgml_path,
    download=False
)

print("Running evaluation...")
evaluator.evaluate(retrieval_fn=wrapped, input_data_format="tabular", batch_size=4048, n_processes=1)


print("\n--- Evaluation Results ---")
print("Precipitation quantification")
precipitation_quantification = evaluator.get_precip_quantification_results(name="XGBOOST (GMI)").T
print(precipitation_quantification.to_string()) 
sc = evaluator.precip_quantification_metrics[-1].compute()
