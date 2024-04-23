import random
import warnings
from typing import Callable

from functools import partial

import torch
from sklearn.base import BaseEstimator
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset

import numpy as np
from sklearn.metrics import average_precision_score

from rdkit import Chem

from tqdm import tqdm

from vdb.chem.fp.mol_graph import MolGraph, MolGraphFunc, ATOM_FDIM, BOND_FDIM
from vdb.utils import isnan


def _index_select_ND(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Selects the message features from source corresponding to the atom or bond indices in index.

    :param source: A tensor of shape (num_bonds, hidden_size) containing message features.
    :param index: A tensor of shape (num_atoms/num_bonds, max_num_bonds) containing the atom or bond
    indices to select from source.
    :return: A tensor of shape (num_atoms/num_bonds, max_num_bonds, hidden_size) containing the message
    features corresponding to the atoms/bonds specified in index.
    """
    index_size = index.size()  # (num_atoms/num_bonds, max_num_bonds)
    suffix_dim = source.size()[1:]  # (hidden_size,)
    final_size = index_size + suffix_dim  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    target = source.index_select(dim=0, index=index.view(-1))  # (num_atoms/num_bonds * max_num_bonds, hidden_size)
    target = target.view(final_size)  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    return target


def negative_log_likelihood(pred_targets, pred_var, targets):
    clamped_var = torch.clamp(pred_var, min=0.00001)
    return torch.log(2 * np.pi * clamped_var) / 2 + (pred_targets - targets) ** 2 / (2 * clamped_var)


def dirichlet_loss(y, alphas, lam=1):
    """
    Use Evidential Learning Dirichlet loss from Sensoy et al
    :y: labels to predict
    :alphas: predicted parameters for Dirichlet
    :lambda: coefficient to weight KL term

    :return: Loss
    """

    def KL(alpha):
        """
        Compute KL for Dirichlet defined by alpha to uniform dirichlet
        :alpha: parameters for Dirichlet

        :return: KL
        """
        beta = torch.ones_like(alpha)
        S_alpha = torch.sum(alpha, dim=-1, keepdim=True)
        S_beta = torch.sum(beta, dim=-1, keepdim=True)

        ln_alpha = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=-1, keepdim=True)
        ln_beta = torch.sum(torch.lgamma(beta), dim=-1, keepdim=True) - torch.lgamma(S_beta)

        # digamma terms
        dg_alpha = torch.digamma(alpha)
        dg_S_alpha = torch.digamma(S_alpha)

        # KL
        kl = ln_alpha + ln_beta + torch.sum((alpha - beta) * (dg_alpha - dg_S_alpha), dim=-1, keepdim=True)
        return kl

    # Hard code to 2 classes per task, since this assumption is already made
    # for the existing chemprop classification tasks
    num_classes = 2
    num_tasks = y.shape[1]

    y_one_hot = torch.eye(num_classes)[y.long()]
    if y.is_cuda:
        y_one_hot = y_one_hot.cuda()

    alphas = torch.reshape(alphas, (alphas.shape[0], num_tasks, num_classes))

    # SOS term
    S = torch.sum(alphas, dim=-1, keepdim=True)
    p = alphas / S
    A = torch.sum(torch.pow((y_one_hot - p), 2), dim=-1, keepdim=True)
    B = torch.sum((p * (1 - p)) / (S + 1), dim=-1, keepdim=True)
    SOS = A + B

    # KL
    alpha_hat = y_one_hot + (1 - y_one_hot) * alphas
    KL = lam * KL(alpha_hat)

    # loss = torch.mean(SOS + KL)
    loss = SOS + KL
    loss = torch.mean(loss, dim=-1)
    return loss


# updated evidential regression loss
def evidential_loss(mu, v, alpha, beta, targets, lam=1, epsilon=1e-4):
    """
    Use Deep Evidential Regression negative log likelihood loss + evidential
        regularizer

    :mu: pred mean parameter for NIG
    :v: pred lam parameter for NIG
    :alpha: predicted parameter for NIG
    :beta: Predicted parmaeter for NIG
    :targets: Outputs to predict

    :return: Loss
    """
    # Calculate NLL loss
    twoBlambda = 2 * beta * (1 + v)
    nll = 0.5 * torch.log(np.pi / v) \
          - alpha * torch.log(twoBlambda) \
          + (alpha + 0.5) * torch.log(v * (targets - mu) ** 2 + twoBlambda) \
          + torch.lgamma(alpha) \
          - torch.lgamma(alpha + 0.5)

    L_NLL = nll

    # Calculate regularizer based on absolute error of prediction
    error = torch.abs((targets - mu))
    reg = error * (2 * v + alpha)
    L_REG = reg  # torch.mean(reg, dim=-1)

    loss = L_NLL + lam * (L_REG - epsilon)

    return loss


class MoleculeDatapoint:
    """A MoleculeDatapoint contains a single molecule and its associated features and targets."""

    def __init__(self, smiles: str, endpoint: float or int, name: str = None):
        """
        Initializes a MoleculeDatapoint, which contains a single molecule.
        """

        self.compound_name = name
        self.smiles = smiles  # str
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.targets = endpoint
        self.features = None


class MoleculeDataset(Dataset):
    """A MoleculeDataset contains a list of molecules and their associated features and targets."""

    def __init__(self, data: list[MoleculeDatapoint]):
        """
        Initializes a MoleculeDataset, which contains a list of MoleculeDatapoints (i.e. a list of molecules).

        :param data: A list of MoleculeDatapoints.
        """
        self.data = data
        self.scaler = None

    def compound_names(self) -> list[str] or None:
        """
        Returns the compound names associated with the molecule (if they exist).

        :return: A list of compound names or None if the dataset does not contain compound names.
        """
        if len(self.data) == 0 or self.data[0].compound_name is None:
            return None

        return [d.compound_name for d in self.data]

    def smiles(self) -> list[str]:
        """
        Returns the smi strings associated with the molecules.

        :return: A list of smi strings.
        """
        return [d.smiles for d in self.data]

    def mols(self) -> list[Chem.Mol]:
        """
        Returns the RDKit molecules associated with the molecules.

        :return: A list of RDKit Mols.
        """
        return [d.mol for d in self.data]

    def targets(self) -> list[float or int]:
        """
        Returns the targets associated with each molecule.

        :return: A list of numerics containing the targets.
        """
        return [d.targets for d in self.data]

    def sample(self, sample_size: int):
        """
        Samples a random subset of the dataset.

        :param sample_size: The size of the sample to produce.
        """
        self.data = random.sample(self.data, sample_size)

    def shuffle(self, seed: int = None):
        """
        Shuffles the dataset.

        :param seed: Optional random seed.
        """
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.data)

    def sort(self, key: Callable):
        """
        Sorts the dataset using the provided key.

        :param key: A function on a MoleculeDatapoint to determine the sorting order.
        """
        self.data.sort(key=key)

    def sample_inds(self, inds: list[int]):
        """
        Samples the dataset according to specified indicies and returns new
        dataset.

        :param inds: A list of desired inds of the dataset to keep.
        """
        data = [self.data[i] for i in inds]
        return MoleculeDataset(data)

    def __len__(self) -> int:
        """
        Returns the length of the dataset (i.e. the number of molecules).

        :return: The length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, item) -> MoleculeDatapoint or list[MoleculeDatapoint]:
        """
        Gets one or more MoleculeDatapoints via an index or slice.

        :param item: An index (int) or a slice object.
        :return: A MoleculeDatapoint if an int is provided or a list of MoleculeDatapoints if a slice is provided.
        """
        return self.data[item]


class BatchMolGraph:
    """
    A BatchMolGraph represents the graph structure and featurization of a batch of molecules.

    A BatchMolGraph contains the attributes of a MolGraph plus:
    - smiles_batch: A list of smi strings.
    - n_mols: The number of molecules in the batch.
    - atom_fdim: The dimensionality of the atom features.
    - bond_fdim: The dimensionality of the bond features (technically the combined atom/bond features).
    - a_scope: A list of tuples indicating the start and end atom indices for each molecule.
    - b_scope: A list of tuples indicating the start and end bond indices for each molecule.
    - max_num_bonds: The maximum number of bonds neighboring an atom in this batch.
    - b2b: (Optional) A mapping from a bond index to incoming bond indices.
    - a2a: (Optional): A mapping from an atom index to neighboring atom indices.
    """

    def __init__(self, mol_graphs: list[MolGraph]):
        self.smiles_batch = [mol_graph.smiles for mol_graph in mol_graphs]
        self.n_mols = len(self.smiles_batch)

        self.atom_fdim = ATOM_FDIM
        self.bond_fdim = BOND_FDIM + ATOM_FDIM

        # Start n_atoms and n_bonds at 1 b/c zero padding
        self.n_atoms = 1  # number of atoms (start at 1 b/c need index 0 as padding)
        self.n_bonds = 1  # number of bonds (start at 1 b/c need index 0 as padding)
        self.a_scope = []  # list of tuples indicating (start_atom_index, num_atoms) for each molecule
        self.b_scope = []  # list of tuples indicating (start_bond_index, num_bonds) for each molecule

        # All start with zero padding so that indexing with zero padding returns zeros
        f_atoms = [[0] * self.atom_fdim]  # atom features
        f_bonds = [[0] * self.bond_fdim]  # combined atom/bond features
        a2b = [[]]  # mapping from atom index to incoming bond indices
        b2a = [0]  # mapping from bond index to the index of the atom the bond is coming from
        b2revb = [0]  # mapping from bond index to the index of the reverse bond
        for mol_graph in mol_graphs:
            f_atoms.extend(mol_graph.f_atoms)
            f_bonds.extend(mol_graph.f_bonds)

            for a in range(mol_graph.n_atoms):
                a2b.append([b + self.n_bonds for b in mol_graph.a2b[a]])

            for b in range(mol_graph.n_bonds):
                b2a.append(self.n_atoms + mol_graph.b2a[b])
                b2revb.append(self.n_bonds + mol_graph.b2revb[b])

            self.a_scope.append((self.n_atoms, mol_graph.n_atoms))
            self.b_scope.append((self.n_bonds, mol_graph.n_bonds))
            self.n_atoms += mol_graph.n_atoms
            self.n_bonds += mol_graph.n_bonds

        self.max_num_bonds = max(len(in_bonds) for in_bonds in a2b)

        self.f_atoms = torch.FloatTensor(f_atoms)
        self.f_bonds = torch.FloatTensor(f_bonds)
        self.a2b = torch.LongTensor([a2b[a] + [0] * (self.max_num_bonds - len(a2b[a])) for a in range(self.n_atoms)])
        self.b2a = torch.LongTensor(b2a)
        self.b2revb = torch.LongTensor(b2revb)
        self.b2b = None  # try to avoid computing b2b b/c O(n_atoms^3)
        self.a2a = None  # only needed if using atom messages

    def get_components(self) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor, torch.LongTensor,
                                      torch.LongTensor, list[tuple[int, int]], list[tuple[int, int]]]:
        """
        Returns the components of the BatchMolGraph.

        :return: A tuple containing PyTorch tensors with the atom features, bond features, and graph structure
        and two lists indicating the scope of the atoms and bonds (i.e. which molecules they belong to).
        """
        return self.f_atoms, self.f_bonds, self.a2b, self.b2a, self.b2revb, self.a_scope, self.b_scope

    def get_b2b(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each bond index to all the incoming bond indices.

        :return: A PyTorch tensor containing the mapping from each bond index to all the incoming bond indices.
        """

        if self.b2b is None:
            b2b = self.a2b[self.b2a]  # num_bonds x max_num_bonds
            # b2b includes reverse edge for each bond so need to mask out
            revmask = (b2b != self.b2revb.unsqueeze(1).repeat(1, b2b.size(1))).long()  # num_bonds x max_num_bonds
            self.b2b = b2b * revmask

        return self.b2b

    def get_a2a(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each atom index to all neighboring atom indices.

        :return: A PyTorch tensor containing the mapping from each bond index to all the incodming bond indices.
        """
        if self.a2a is None:
            # b = a1 --> a2
            # a2b maps a2 to all incoming bonds b
            # b2a maps each bond b to the atom it comes from a1
            # thus b2a[a2b] maps atom a2 to neighboring atoms a1
            self.a2a = self.b2a[self.a2b]  # num_atoms x max_num_bonds

        return self.a2a


def mol2graph(smiles_batch: list[str]) -> BatchMolGraph:
    """
    Converts a list of SMILES strings to a BatchMolGraph containing the batch of molecular graphs.

    :param smiles_batch: A list of SMILES strings.
    :return: A BatchMolGraph containing the combined molecular graph for the molecules
    """
    mol_graphs = []
    for smiles in smiles_batch:
        mol_graphs.append(MolGraph(smiles))

    return BatchMolGraph(mol_graphs)


class NoamLR(_LRScheduler):
    """
    Noam learning rate scheduler with piecewise linear increase and exponential decay.

    The learning rate increases linearly from init_lr to max_lr over the course of
    the first warmup_steps (where warmup_steps = warmup_epochs * steps_per_epoch).
    Then the learning rate decreases exponentially from max_lr to final_lr over the
    course of the remaining total_steps - warmup_steps (where total_steps =
    total_epochs * steps_per_epoch). This is roughly based on the learning rate
    schedule from Attention is All You Need, section 5.3 (https://arxiv.org/abs/1706.03762).
    """

    def __init__(self,
                 optimizer,
                 warmup_epochs: list[float or int],
                 total_epochs: list[int],
                 steps_per_epoch: int,
                 init_lr: list[float],
                 max_lr: list[float],
                 final_lr: list[float]):
        """
        Initializes the learning rate scheduler.

        :param optimizer: A PyTorch optimizer.
        :param warmup_epochs: The number of epochs during which to linearly increase the learning rate.
        :param total_epochs: The total number of epochs.
        :param init_lr: The initial learning rate.
        :param max_lr: The maximum learning rate (achieved after warmup_epochs).
        :param final_lr: The final learning rate (achieved after total_epochs).
        """
        assert len(optimizer.param_groups) == len(warmup_epochs) == len(total_epochs) == len(init_lr) == \
               len(max_lr) == len(final_lr)

        self.num_lrs = len(optimizer.param_groups)

        self.optimizer = optimizer
        self.warmup_epochs = np.array(warmup_epochs)
        self.total_epochs = np.array(total_epochs)
        self.steps_per_epoch = steps_per_epoch
        self.init_lr = np.array(init_lr)
        self.max_lr = np.array(max_lr)
        self.final_lr = np.array(final_lr)

        self.current_step = 0
        self.lr = init_lr
        self.warmup_steps = (self.warmup_epochs * self.steps_per_epoch).astype(int)
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.linear_increment = (self.max_lr - self.init_lr) / self.warmup_steps
        self.exponential_gamma = (self.final_lr / self.max_lr) ** (1 / (self.total_steps - self.warmup_steps))

        super(NoamLR, self).__init__(optimizer)

    def get_lr(self) -> list[float]:
        """Gets a list of the current learning rates."""
        if self.warmup_steps is None:
            raise RuntimeError('must call `set_steps_per_epoch()` before using scheduler')
        return list(self.lr)

    def step(self, current_step: int = None):
        """
        Updates the learning rate by taking a step.

        :param current_step: Optionally specify what step to set the learning rate to.
        If None, current_step = self.current_step + 1.
        """
        if self.warmup_steps is None:
            raise RuntimeError('must call `set_steps_per_epoch()` before using scheduler')
        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step += 1

        for i in range(self.num_lrs):
            if self.current_step <= self.warmup_steps[i]:
                self.lr[i] = self.init_lr[i] + self.current_step * self.linear_increment[i]
            elif self.current_step <= self.total_steps[i]:
                self.lr[i] = self.max_lr[i] * (self.exponential_gamma[i] ** (self.current_step - self.warmup_steps[i]))
            else:  # theoretically this case should never be reached since training should stop at total_steps
                self.lr[i] = self.final_lr[i]

            self.optimizer.param_groups[i]['lr'] = self.lr[i]


class MPNEncoder(nn.Module):
    """A message passing neural network for encoding a molecule."""

    def __init__(self,
                 atom_fdim: int = ATOM_FDIM,
                 bond_fdim: int = BOND_FDIM,
                 hidden_size: int = 300,
                 bias: bool = True,
                 depth: int = 3,
                 dropout: float = 0.0,
                 undirected: bool = False,
                 use_cuda: bool = True
                 ):
        """Initializes the MPNEncoder.

        :param atom_fdim: Atom features dimension.
        :param bond_fdim: Bond features dimension.
        """
        super(MPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_size = hidden_size
        self.bias = bias
        self.depth = depth
        self.dropout = dropout
        self.layers_per_message = 1
        self.undirected = undirected
        self.use_cuda = use_cuda

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = torch.nn.ReLU()

        # Cached zeros
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)

        # Input
        input_dim = self.bond_fdim
        self.W_i = nn.Linear(input_dim, self.hidden_size, bias=self.bias)

        # Shared weight matrix across depths (default)
        self.W_h = nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)
        self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)

    def forward(self, mol_graph: BatchMolGraph):
        """
        Encodes a batch of molecular graphs.

        :param mol_graph: A BatchMolGraph representing a batch of molecular graphs.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """

        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components()

        if self.use_cuda or next(self.parameters()).is_cuda:
            f_atoms, f_bonds, a2b, b2a, b2revb = f_atoms.cuda(), f_bonds.cuda(), a2b.cuda(), b2a.cuda(), b2revb.cuda()

        _input = self.W_i(f_bonds)  # num_bonds x hidden_size
        message = self.act_func(_input)  # num_bonds x hidden_size

        # Message passing
        for depth in range(self.depth - 1):
            if self.undirected:
                message = (message + message[b2revb]) / 2

            # m(a1 -> a2) = [sum_{a0 \in nei(a1)} m(a0 -> a1)] - m(a2 -> a1)
            # message      a_message = sum(nei_a_message)      rev_message
            nei_a_message = _index_select_ND(message, a2b)  # num_atoms x max_num_bonds x hidden
            a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
            rev_message = message[b2revb]  # num_bonds x hidden
            message = a_message[b2a] - rev_message  # num_bonds x hidden

            message = self.W_h(message)
            message = self.act_func(_input + message)  # num_bonds x hidden_size
            message = self.dropout_layer(message)  # num_bonds x hidden

        nei_a_message = _index_select_ND(message, a2b)  # num_atoms x max_num_bonds x hidden
        a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
        a_input = torch.cat([f_atoms, a_message], dim=1)  # num_atoms x (atom_fdim + hidden)
        atom_hiddens = self.act_func(self.W_o(a_input))  # num_atoms x hidden
        atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden

        # Readout
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
                mol_vec = cur_hiddens  # (num_atoms, hidden_size)

                mol_vec = mol_vec.sum(dim=0) / a_size
                mol_vecs.append(mol_vec)

        return torch.stack(mol_vecs, dim=0)  # num_molecules x hidden


class MPN(nn.Module):
    """A message passing neural network for encoding a molecule."""

    def __init__(
            self,
            atom_fdim: int = ATOM_FDIM,
            bond_fdim: int = BOND_FDIM,
            hidden_size: int = 300,
            bias: bool = True,
            depth: int = 3,
            dropout: float = 0.0,
            undirected: bool = False,
            use_cuda: bool = True,
            graph_input: bool = False):
        """
        Initializes the MPN.

        :param atom_fdim: Atom features dimension.
        :param bond_fdim: Bond features dimension.
        :param graph_input: If true, expect BatchMolGraph as input_.
                            Otherwise, expect a list of smi strings as input_.
        """
        super(MPN, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.graph_input = graph_input
        self.encoder = MPNEncoder(
            atom_fdim=self.atom_fdim,
            bond_fdim=self.bond_fdim,
            hidden_size=hidden_size,
            bias=bias,
            depth=depth,
            dropout=dropout,
            undirected=undirected,
            use_cuda=use_cuda
        )

    def forward(self, batch: list[str] or BatchMolGraph) -> torch.Tensor:
        """
        Encodes a batch of molecular SMILES strings.

        :param batch: A list of SMILES strings or a BatchMolGraph (if self.graph_input is True).
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) contains the encoding of each molecule.
        """
        if not self.graph_input:  # if features only, batch won't even be used
            batch = mol2graph(batch)

        output = self.encoder.forward(batch)

        return output


class MoleculeModel(nn.Module):
    """A MoleculeModel is a model that contains a message passing network following by feed-forward layers."""

    def __init__(self, classification: bool, confidence: bool = False):
        """
        Initializes the MoleculeModel.

        :param classification: Whether the model is a classification model.
        :param confidence: Whether confidence values should be predicted.
        """
        super(MoleculeModel, self).__init__()

        self.classification = classification
        # NOTE: A confidence flag is only set if the model must handle returning
        # confidence internally and for evidential learning.
        self.confidence = confidence

        self.args = None
        self.encoder = None
        self.ffn = None
        self.ffn_args = None  # save the ffn args for logging

        self.dropout = None

        if self.classification:
            self.final_activation = nn.Identity()

    def create_encoder(self, **kwargs):
        """
        Creates the message passing encoder for the model.
        """
        self.encoder = MPN(**kwargs)

    def create_ffn(self, ff_hidden_size: int = 300, last_hidden_size: int = 300,
                   dropout: float = 0.0, n_layers: int = 2):
        """
        Creates the feed-forward network for the model.
        """
        first_linear_dim = self.encoder.encoder.hidden_size

        self.dropout = nn.Dropout(dropout)

        activation = nn.ReLU()

        output_size = 1

        if self.confidence:  # if confidence should be learned
            if self.classification:  # dirichlet
                # For each task, output both the positive and negative
                # evidence for that task
                output_size *= 2
            else:  # normal inverse gamma
                # For each task, output the parameters of the NIG
                # distribution (gamma, lambda, alpha, beta)
                output_size *= 4

        # Create FFN layers
        if n_layers == 1:
            ffn = [
                self.dropout,
                nn.Linear(first_linear_dim, output_size)
            ]
        else:
            ffn = [
                self.dropout,
                nn.Linear(first_linear_dim, ff_hidden_size)
            ]
            for _ in range(n_layers - 3):
                ffn.extend([
                    activation,
                    self.dropout,
                    nn.Linear(ff_hidden_size, ff_hidden_size),
                ])

            ffn.extend([
                activation,
                self.dropout,
                nn.Linear(ff_hidden_size, last_hidden_size),
            ])

            ffn.extend([
                activation,
                self.dropout,
                nn.Linear(last_hidden_size, output_size),
            ])

        # Create FFN model
        self.ffn = nn.Sequential(*ffn)
        self.ffn_args = {
            'ff_hidden_size': ff_hidden_size,
            'last_hidden_size': last_hidden_size,
            'dropout': dropout,
            'n_layers': n_layers,
            'output_size': output_size
        }

    def forward(self, *input_):
        """
        Runs the MoleculeModel on input_.

        :param input_: Input.
        :return: The output of the MoleculeModel.
        """
        output = self.ffn(self.encoder(*input_))

        if self.confidence:
            if self.classification:
                # Convert the outputs into the parameters of a Dirichlet
                # distribution (alpha).
                output = nn.functional.softplus(output) + 1

            else:
                min_val = 1e-6
                # Split the outputs into the four distribution parameters
                means, loglambdas, logalphas, logbetas = torch.split(output, output.shape[1] // 4, dim=1)
                lambdas = torch.nn.Softplus()(loglambdas) + min_val
                alphas = torch.nn.Softplus()(
                    logalphas) + min_val + 1  # add 1 for numerical contraints of Gamma function
                betas = torch.nn.Softplus()(logbetas) + min_val

                # Return these parameters as the output of the model
                output = torch.stack((means, lambdas, alphas, betas),
                                     dim=2).view(output.size())

        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if self.classification and not self.training:
            output = self.final_activation(output)

        return output


class DirectedMPNNClassifier(BaseEstimator):
    def __init__(
            self,
            confidence: bool = False,
            atom_fdim: int = ATOM_FDIM,
            bond_fdim: int = BOND_FDIM,
            hidden_size: int = 300,
            bias: bool = True,
            depth: int = 3,
            ff_hidden_size: int = 300,
            last_hidden_size: int = 300,
            n_layers: int = 2,
            dropout: float = 0.0,
            regularizer_coeff: float = 1.0,
            use_cuda: bool = True
    ):
        super().__init__(fp_func=MolGraphFunc())

        self.model_object = MoleculeModel(classification=True, confidence=confidence)
        self.model_object.create_encoder(
            atom_fdim=atom_fdim,
            bond_fdim=atom_fdim + bond_fdim,
            hidden_size=hidden_size,
            bias=bias,
            depth=depth,
            dropout=dropout,
            undirected=False,
            use_cuda=use_cuda,
            graph_input=True
        )
        self.model_object.create_ffn(
            ff_hidden_size=ff_hidden_size,
            last_hidden_size=last_hidden_size,
            n_layers=n_layers,
            dropout=dropout
        )

        if confidence:
            self.loss_func = partial(dirichlet_loss, lam=regularizer_coeff)
        else:
            self.loss_func = nn.BCEWithLogitsLoss(reduction='none')

        self.confidence = confidence
        self.cuda = use_cuda

    def fit(self, x, y,
            num_epochs: int = 30,
            batch_size: int = 50,
            warmup_epochs: int = 2,
            init_lr: float = 1e-4,
            max_lr: float = 1e-3,
            final_lr: float = 1e-6,
            early_stopping_patience: int = 5,
            use_tqdm: bool = False):

        optimizer = torch.optim.Adam([{'params': self.model_object.parameters(),
                                       'lr': init_lr, 'weight_decay': 0}])

        y = self._check_label(y)

        # check for nan
        not_nans = ~isnan(x)
        if sum(not_nans) != len(not_nans):
            warnings.warn("detected bad data; silently removing for training")
            x = x[not_nans]
            y = y[not_nans]

        splitter = ScaffoldSplit(n_fold=1, train_size=0.9, test_size=0.0)
        splitter.fit(smiles=[_[0].smiles for _ in x], y=y)

        train_idx, val_idx = next(splitter.training_splits())

        train_x = x[train_idx]
        train_y = y[train_idx]
        val_x = x[val_idx]
        val_y = y[val_idx]

        scheduler = NoamLR(
            optimizer=optimizer,
            warmup_epochs=[warmup_epochs],
            total_epochs=[num_epochs],
            steps_per_epoch=(len(train_x)//batch_size)+1,
            init_lr=[init_lr],
            max_lr=[max_lr],
            final_lr=[final_lr]
        )

        best_score = 0
        best_epoch = 0
        patience = 0
        pbar = tqdm(total=num_epochs*((len(train_x)//batch_size)+1), disable=not use_tqdm, desc="training MPNN")
        for epoch in range(num_epochs):
            # shuffle training data
            total_loss = 0
            p = np.random.permutation(len(train_x))
            _x = train_x[p]
            _y = train_y[p]
            for batch_idx in range(0, len(_x), batch_size):
                batch_x = BatchMolGraph(_x[batch_idx:batch_idx + batch_size].flatten().tolist())
                batch_y = torch.tensor(_y[batch_idx:batch_idx + batch_size])

                if self.cuda:
                    batch_y = batch_y.cuda()

                self.model_object.zero_grad()
                preds = self.model_object(batch_x)
                if self.confidence:
                    loss = self.loss_func(batch_y, alphas=preds)
                else:
                    loss = self.loss_func(preds, batch_y.reshape((-1, 1)).float())

                loss = loss.sum() / batch_size

                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                scheduler.step()
                pbar.update(1)
            val_preds = self.predict(val_x, batch_size=batch_size)
            val_metric = average_precision_score(val_y, val_preds)
            if val_metric > best_score:
                best_score = val_metric
                best_epoch = epoch
                patience = 0
            else:
                patience += 1
            epoch_loss = total_loss
            print(epoch, epoch_loss, val_metric, best_score, best_epoch)

            if early_stopping_patience and (patience >= early_stopping_patience):
                # TODO add logging here
                break

    def predict_proba(self, x, batch_size: int = 50):
        self.model_object.eval()
        preds = []

        for batch_idx in range(0, len(x), batch_size):
            batch = x[batch_idx:batch_idx + batch_size]
            not_nans = ~isnan(batch)
            batch_x = BatchMolGraph(batch[not_nans].tolist())
            batch_preds = np.full((len(batch), 1), fill_value=np.nan, dtype=np.float32)
            with torch.no_grad():
                _batch_preds = self.model_object(batch_x).data.cpu().numpy().flatten()
                batch_preds[not_nans] = _batch_preds
            preds.extend(batch_preds.flatten().tolist())

        if self.confidence:
            p = []
            c = []
            for i in range(len(preds)):
                alphas = np.array(preds[i])
                p.append((alphas / np.sum(alphas))[1])
                c.append(2 / np.sum(alphas))
        else:
            p = preds
            c = np.ones(len(p))

        # someday allow this to return c, or incorporate c into the prediction
        return np.array(p) * np.array(c)

    def predict(self, x, batch_size: int = 50):
        pred_probas = self.predict_proba(x, batch_size=batch_size)
        return (pred_probas > 0.5).astype(int)
