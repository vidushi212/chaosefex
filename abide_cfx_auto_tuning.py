import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from abide_chaosnet_lts import ChaosNetLTS


def chaosnet_cross_validate(X, y, param_grid, n_splits=5):

    best_acc = 0.0
    best_params = None

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    for n_neurons in param_grid["n_neurons"]:
        for lr in param_grid["learning_rate"]:
            for epochs in param_grid["max_epochs"]:

                accs = []

                for train_idx, test_idx in skf.split(X, y):

                    model = ChaosNetLTS(
                        n_neurons=n_neurons,
                        learning_rate=lr,
                        max_epochs=epochs,
                    )

                    model.fit(X[train_idx], y[train_idx])
                    preds = model.predict(X[test_idx])

                    accs.append(accuracy_score(y[test_idx], preds))

                mean_acc = np.mean(accs)

                if mean_acc > best_acc:
                    best_acc = mean_acc
                    best_params = {
                        "n_neurons": n_neurons,
                        "learning_rate": lr,
                        "max_epochs": epochs,
                    }

    return best_params, best_acc
