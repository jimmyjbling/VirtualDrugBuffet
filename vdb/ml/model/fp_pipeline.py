import abc
from typing import Literal

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier, VotingRegressor

import numpy.typing as npt

from vdb.ml.scaler import BaseScaler
from vdb.curate import CurationWorkflow
from vdb.data import SmilesLoaderIO
from vdb.chem.fp import BaseFPFunc


# TODO add a feature_selection step and y_preperation_pipeline step
class MoleculeModelPipeline(Pipeline):
    def __init__(self, fp_func: BaseFPFunc, model: BaseEstimator, data_loader: SmilesLoaderIO = None,
                 curation_workflow: CurationWorkflow = None, scaler: BaseScaler = None):
        super().__init__(steps=None)


# class FpModel(Pipeline, UnlearnedEmbeddingModel):
#     def __init__(self, steps: list[tuple[str, BaseEstimator]], memory: str = None, verbose: bool = False):
#
#         super().__init__(steps, memory=memory, verbose=verbose)
#
#     def extract_model(self):
#         return self.steps[-1][1]  # model is always the last step in the pipeline
#
#     def get_fp_func(self):
#         return self.steps[0][1]  # fp_func is always the first step in the pipeline
#
#
# class EnsembleFpPipelineModel(abc.ABC):
#
#     @abc.abstractmethod
#     def extract_model(self):
#         raise NotImplemented
#
#     def get_fp_funcs(self):
#         return [e.get_fp_func() for e in self.estimators_]
#
#
# class VotingClassifierFpPipelineModel(VotingClassifier, UnlearnedEmbeddingModel):
#     def __init__(self, estimators: list[tuple[str, FpPipelineModel]], voting: Literal["soft", "hard"] = "hard",
#                  weights: npt.ArrayLike = None, n_jobs: int = None):
#         self._voting = voting
#         self._weights = weights
#         self._n_jobs = n_jobs
#         super().__init__(estimators, voting=voting, weights=weights, n_jobs=n_jobs)
#
#     def extract_model(self):
#         return VotingClassifier(estimators=[e.extract_model() for e in self.estimators_],
#                                 voting=self.voting, weights=self.weights, n_jobs=self._n_jobs)
#
#
# class VotingRegressorFpPipelineModel(VotingRegressor, UnlearnedEmbeddingModel):
#     def __init__(self, estimators: list[tuple[str, FpPipelineModel]], weights: npt.ArrayLike = None, n_jobs: int = None):
#         self._weights = weights
#         self._n_jobs = n_jobs
#         super().__init__(estimators, weights=weights, n_jobs=n_jobs)
#
#     def extract_model(self):
#         return VotingRegressor(estimators=[e.extract_model() for e in self.estimators],
#                                weights=self.weights, n_jobs=self._n_jobs)
