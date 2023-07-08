import importlib
import os

registered_metrics = dict()

def register_metrics(name):
    def wrapper(metrics_fn):
        registered_metrics[name] = metrics_fn
        return metrics_fn
    return wrapper

metrics_dir = os.path.dirname(__file__)
for f in os.listdir(metrics_dir):
    fpath = os.path.join(metrics_dir,f)
    if not f.startswith('.') and (f.endswith('.py')):
        fname = f[:f.find('.py')]
        module = importlib.import_module(f'.{fname}','trainer.Metrics')
for key,fn in registered_metrics.items():
    globals()[key] = fn
