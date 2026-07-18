import numpy as np

def focal_loss_lgb(y_pred, dataset, alpha=0.25, gamma=2.0):
    y_true = dataset.get_label()
    
    # sigmoid
    p = 1.0 / (1.0 + np.exp(-y_pred))
    
    # avoid numerical issues
    eps = 1e-9
    p = np.clip(p, eps, 1 - eps)
    
    # pt
    pt = np.where(y_true == 1, p, 1 - p)
    
    # gradient
    grad = (
        alpha * (y_true - p) *
        ((1 - pt) ** gamma) *
        (gamma * pt * np.log(pt) + pt - 1)
    )
    
    # hessian (approximation, commonly used)
    hess = (
        alpha * ((1 - pt) ** gamma) *
        p * (1 - p)
    )
    
    return grad, hess