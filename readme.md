# VirtualDrugBuffet
This VirtualDrugBuffet (VDB) is a python package meant to serve as the main code base for the 
[Molecular Modeling Lab (MML) @ UNC](https://molecularmodelinglab.github.io/).
Students in the lab can use this as a jumping off point to learn how to deal with chemicals in python.
VDB includes a bunch of simple calls to common steps or events that occur in cheminformatic pipelines (e.g. clustering).
The goal is to abstract and generalize the functions as much as possible so that a newcomer to the filed can quickly
implement desired workflows.
Think of it like KNIME-lite with out a GUI.
It also has the benefit of being easier to read and understand any workflows just by looking at them.

### Philosophy 
One of the core philosophies in the lab (and science) is reproducibility. With that in mind, VDB is built in a way that 
attempts to track as record as much as what happens as possible.

To do this, VDB takes a DAG (Direct Acyclic Graph) like workflow approach.
In this way, all commands in VDB are viewed as "steps".
These steps need to be pre-initialized with the setting to define their behavior.
These steps are then assembled into a pipeline (DAG) in a specific order based on what behavior is desired.
The steps are then all executed at the same time by simply calling the workflow
In this way, it is easy to save exactly what the workflow did, and save the whole workflow itself. 
Additional use of the lab MLFlow server (TODO need to set this up still) can register these workflows and the datasets
they were run/trained on. 

### SciKit Learn API
VDB uses the SciKit-Learn API for all of its steps. This allows them to be easily linked using some like the Pipeline
object from scikit-learn.
It also provides a consistent API for us
if/when we choose to build our own Pipeline-like objects since we know exactly how to interface with each object

### Model Framework
One of the steps that Josh and I identified when we were liking about issue the lab has was lack of clear documentation
of how a dataset was pre-processed for modeling.
This was even more challenging because we all use custom scripts for this. 
To help get around that we re-framed our thought of what a "model" is into a 3 step process: 
1. Curation
2. Embedding
3. Fitting

All models follow this simple paradigm, even those with learning embeddings
   (a GCN first need to convert a SMILES into a graph, a SmilesTransformer need to token SMILES).
Thus no longer are our traditional models detached from the steps that were used to create the data in the right format.
This will be especially helpful for exchanging models between students, since I know exactly what preprocessing needs to
occur for me to use the model (since it is physically part of is).
Thus, in VDB all models are actually a linear pipeline of 3 "step" classes: `CurationWorkflow`, `FPFunc`, `BaseModel`
There are different options for each, but they all inherit from these base classes

## Steps in VDB
VDB has several different types of "steps", which are formally Python classes.
They are broken up into families,
such that each family member has the same desired output but uses different algorithms to get there.
All Steps are `BaseEstimator` objects from SKLearn.
Most are also `TransformerMixIns` (meaning they alter the input).
Below is some brief overview of these:

### DataIO
DataIO "steps" are used to load and write datasets to and from files.
It has three subfamilies: `DataLoader`, `Dataset` and `DataWriter`

#### DataLoader

#### Dataset

#### DataWriter
TODO

### FPFunc
`FPFunc` is a family of steps that are used to convert a SMILES (or rdkit.Mol) into some "featurization" space. This is
generally done so that the chemicals can be input into a model, but could also be for other tasks (like clustering based on features).
While traditionally we think of fingerprinting as a process to convert a chemical into a vector of numerics, in VDB it
is just any function that converts SMILES/Mols into a different computer format.
So converting Mols into the graphical vector input for a GCN is an "FPFunc",
and so is tokenizing SMILES for a Transformer.

FPFunc all are Children of the `BaseFPFunc` class. They are all so all Children of ONE of the following classes:
- `BinaryFPFUnc`
- `ContinousFPFunc`
- `DiscreteFPFunc`
- `ObjectFPFunc`

Intuitively, these are used to give some notion of what type of output the FPFunc will have.
For example, any FPFunc that is a child of `ObjectFPFunc` will return some kind of non-array object
(like a NetworkX Graph object).

All `FPFunc` objects define a `_func` function
that specifies how the FPFunc will convert a single SMILES into the featurization.
As a user, you will never call this function yourself,
instead you interface with it using `generate_fps()`, `transfrom` or simply using the class object as a callable:

```python
from vdb.chem.fp import ECFP4
my_fingerprinter = ECFP4()

smis = ["CCCC", "CCCO", "c1ccccc1"]
fps1 = my_fingerprinter.generate_fps(smis)
fps2 = my_fingerprinter(smis)
fps3 = my_fingerprinter.transform(smis)
```

FPFuncs have the `transform` (and `fit_transform`) function because they are SciKit-Learn TransformerMixins.
That makes them able to be used in any SciKit-Learn Pipelines
so that you can pass SMILES right into the model like so

```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from vdb.chem.fp import ECFP4

smis = ["CCCC", "CCCO", "c1ccccc1"]
labels = [1,0,1]

clf = RandomForestClassifier(n_estimators=5)
my_pipeline = Pipeline(steps=[("my_fingerprinter", ECFP4()), ("my_model", clf)])
my_pipeline.predict_proba(smis)
```

FPFunc will always return numpy 2-D arrays of the resulting outputs.
So something like ECFP4 with 2048 bits will return a array of shape (num_smiles, 2048).
Even if you have an `ObjectFpFunc` or one that returns a single value for the smiles, it will still be 2D, in shape
(num_smiles, 1).
Any SMILES that fail to be fingerprinted will have `np.nan` values populating the whole row for that SMILES

Most FPFunc use RDkit to generate the FP.
The `FPFunc` class will take care of converting from the RDKit FP object classes into numpy arrays for you.
However, there are some cases where you might want the RDKit FP object (BulkTanimotoSimilarity is a good example).
Therefore, `FPFuncs`
that can be converted into RDKit FP object will also inherit from the `RDKitFPFunc` class,
which gives access to the `generate_fps_as_rdkit_objects()` function.
This function will return a list of the respective RDKit FP objects rather than the numpy array of the FPs.

Lastly, all steps in VDB must be pickle-able (so they can be easily saved).
RDKit or other PythonBoost functions are not natively capable of this.
But fear not VDB has you covered by using a clever wrapping trick to get around this, so they can still be pickled :)

There are many of these, a list of current ones can be found below (with their parent classes)

- ECFP4 | `DiscreteFPFunc`, `RDKitFPFunc`
- ECFP6 | `DiscreteFPFunc`, `RDKitFPFunc`
- FCFP4 | `DiscreteFPFunc`, `RDKitFPFunc`
- FCFP6 | `DiscreteFPFunc`, `RDKitFPFunc`
- BinaryECFP4 | `BinaryFPFunc`, `RDKitFPFunc`
- BinaryECFP6 | `BinaryFPFunc`, `RDKitFPFunc`
- BinaryFCFP4 | `BinaryFPFunc`, `RDKitFPFunc`
- BinaryFCFP6 | `BinaryFPFunc`, `RDKitFPFunc`
- AtomPair | `DiscreteFPFunc`, `RDKitFPFunc`
- TopTor | `DiscreteFPFunc`, `RDKitFPFunc`
- Avalon | `DiscreteFPFunc`, `RDKitFPFunc`
- BinaryAtomPair | `BinaryFPFunc`, `RDKitFPFunc`
- BinaryTopTor | `BinaryFPFunc`, `RDKitFPFunc`
- BinaryAvalon | `BinaryFPFunc`, `RDKitFPFunc`
- RDK | `BinaryFPFunc`, `RDKitFPFunc`
- MACCS | `BinaryFPFunc`, `RDKitFPFunc`

### Curation
The curation module of VDB is broken into 2 steps, defining CurationSteps and defining a CurationWorkflow made up of 
`CurationSteps`.

#### CurationSteps
`CurationSteps` are single curation actions that are detect a specific issues with any passed chemicals. For example `CurateInorganic`, flags compounds that are inorganic as bad.
While most CurationFuncs only flag compounds that are bad,
some can modify compounds to stnadarize them, like `CurateFlatten` to remove stereochemistry.
`CurationsSteps` are all children of the `CurationStep` class, which makes them callable objects.
When called, a `CurationStep` uses its stored `_func` callable to execute the curation.
If you add any new CurationSteps you need to provide this function.
To call a CurationStep you must pass the X (SMILES) and optional y (Labels).
The CurationStep will return three things, the X (with updates if the curation step modified the X), the y
(with updates if the curation step modified the y or `None` if no y was passed)
and a boolean mask defining which rows in X passed curation. 

```python
from vdb.curate.steps import CurateFlatten

smis = ["CCCC", "CCCO", "c1ccccc1"]
my_new_smiles, _, mask = CurateFlatten()(smis)

# you can also pass a y value
labels = [1,0,1]
my_new_smiles, my_new_labels, mask = CurateFlatten()(smis, labels)
```

Because `CurationSteps` return a boolean mask rather than just removing the bad rows,
they do NOT qualify as Transformers in the SciKit-Learn API.
Also, we often want to execute many steps in a row,
retaining information like which compounds failed whihc steps and their original row in the dataset they came from.
This can be alot, so to help with that there are `CurationWorkflows`

#### CurationWorkflows
CurationWorkflows, unlike CurationSteps, are fully SciKit-Learn API compliant Transformers.
They are created by passing the list of CurationSteps you want to execute (in the order you want to execute them).
Then, you can call the workflow with `run_workflow()`or `transform()` and all the steps will be excuted in order on your X and y input

```python
from vdb.curate.steps import CurateValid, CurateFlatten, CurateCanonicalize
from vdb.curate import CurationWorkflow

my_steps = [CurateValid(), CurateFlatten(), CurateCanonicalize()]
my_workflow = CurationWorkflow(steps=my_steps)

smis = ["CCCC", "CCCO", "c1ccccc1"]
labels = [1,0,1]

curated_X, curated_y, mask = my_workflow.run_workflow(smis, labels)
```

Unlike a CurationStep, the CurationWorkflow will only return the X and y rows that passed curation.
It will also return the boolean mask that is the union of all CurationSteps.
It should be noted the `transform()` will not return the mask (to make it Transformer API compliant)

The real beauty of all this is that under the hood, `CurationWorkflow`
is store a bunch of information about the curation process, like how many compounds failed each step.
When you call `run_workflow` you can tell the CurationWorkflow to print out this report to the console.
If you want to save it as a file, during initalization of the workflow you can pass `report_path=\PATH\TO\REPORT.txt`
which tell the workflow to save the curation report to that file.
It will also do some logging that you can access in the vdb.log file (if you set `do_logging=True`). 

Just like all steps, it can be saved and used in a SKLearn Pipeline
```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from vdb.chem.fp import ECFP4
from vdb.curate.steps import CurateValid, CurateFlatten, CurateCanonicalize
from vdb.curate import CurationWorkflow

my_steps = [CurateValid(), CurateFlatten(), CurateCanonicalize()]
my_workflow = CurationWorkflow(steps=my_steps)
smis = ["CCCC", "CCCO", "c1ccccc1"]
labels = [1,0,1]

clf = RandomForestClassifier(n_estimators=5)
my_pipeline = Pipeline(steps=[("my_fingerprinter", ECFP4()), ("my_curation", my_workflow), ("my_model", clf)])
my_pipeline.predict_proba(smis)
```
