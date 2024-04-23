from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator

from vdb.chem.fp.base import BaseFPFunc


class EmbeddedModel(Pipeline):
    def __init__(self, fp_func: BaseFPFunc, model: BaseEstimator, scaler: BaseEstimator = None,
                 memory: bool = False, verbose: bool = False):

        _steps = [("fp_func", fp_func)]
        if scaler is not None:
            _steps += [("scaler", scaler)]
        _steps += [("model", model)]

        super().__init__(_steps, memory=memory, verbose=verbose)

    def extract_model(self):
        return self.steps[-1][1]  # model is always the last step in the pipeline

    def get_fp_func(self):
        return self.steps[0][1]  # fp_func is always the first step in the pipeline
