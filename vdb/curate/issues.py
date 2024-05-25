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


CURATION_NAME_TO_ISSUE_ENUM = {
    "mixture": CurationIssue.mixture,
    "inorganic": CurationIssue.inorganic,
    "boron": CurationIssue.boron,
    "rdkit": CurationIssue.rdkit_failed,
    "add_h": CurationIssue.failed_adding_Hs,
    "add_3d": CurationIssue.failed_gen_random_3d,
    "flatten": CurationIssue.flatten_failed,
    "sanitize": CurationIssue.sanitize_failed,
    "neutralize": CurationIssue.neutralize_failed,
    "canonicalize": CurationIssue.canonical_failed,
    "remove_duplicates": CurationIssue.duplicate,
    "remove_disagreeing_duplicates": CurationIssue.disagreeing_duplicate,
    "remove_duplicates_greedy": CurationIssue.duplicate,
    "missing_label": CurationIssue.missing_label,
    "numeric_label": CurationIssue.non_numeric_label
}
