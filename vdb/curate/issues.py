import enum


class CurationIssue(enum.Enum):
    mixture = "compound is a mixture"
    inorganic = "compound is inorganic"
    boron = "compound has boron"
    rdkit_failed = "rdkit failed to read molecule"
    failed_adding_Hs = "rdkit failed to add H to molecule"
    failed_gen_random_3d = "rdkit failed to generated a random 3d pose for molecule"
    flatten_failed = "compound failed to be flattened"
    sanitize_failed = "compound failed to be sanitized"
    neutralize_failed = "compound failed to be neutralized"
    canonical_failed = "compound failed to be canonicalized"
    duplicate = "compound is duplicate"
    disagreeing_duplicate = "compound has duplicate with disagreeing label"
    missing_label = "label value is missing"
    non_numeric_label = "label value is not numeric"
    wrong_mw = "molecule weight too big or small"
