import enum


class CurationNote(enum.Enum):
    flattened = "compound flattened"
    sanitized = "compound sanitized"
    neutralized = "compound neutralized"
    canonical = "compound made canonical"
    added_Hs = "explict H added to compound"
    demixed = "seperated out largest mixture component"
    added_random_3d_conformer = "3d conformer added to compound"
    label_made_numeric = "label made numeric"
    digitized_label = "digitized label"
    binarize_label = "binarize label"
