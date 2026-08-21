import pickle
from collections import Counter
import numpy as np
from io import StringIO
from Bio.PDB import PDBIO, PDBParser


def count_amino_acids(pdb_text):
    residues = set()
    for line in pdb_text.splitlines():
        if line.startswith("ATOM"):
            chain = line[21].strip()
            resi = line[22:26].strip()
            icode = line[26].strip()
            residues.add((chain, resi, icode))
    return len(residues)


def load_trb(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_motif_hal_idx0(trb):
    # 0-indexed positions in the output chain that RFdiffusion held fixed as the motif
    return np.asarray(sorted(int(idx) for idx in trb["con_hal_idx0"]))


def contiguous_runs(sorted_indices):
    runs = []
    start = prev = None
    for idx in sorted_indices:
        idx = int(idx)
        if start is None:
            start = prev = idx
        elif idx == prev + 1:
            prev = idx
        else:
            runs.append((start, prev))
            start = prev = idx
    if start is not None:
        runs.append((start, prev))
    return runs


def load_reversed_trajectory(path, hold_final_frames=8):
    if not path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {path}")
    trajectory_text = path.read_text(encoding="utf-8")
    frames = [frame.strip() for frame in trajectory_text.split("ENDMDL") if frame.strip()]
    reversed_frames = list(reversed(frames))
    # Pause before looping by repeating the final structure
    reversed_frames = reversed_frames + [reversed_frames[-1]] * hold_final_frames
    reversed_text = "\nENDMDL\n".join(reversed_frames) + "\nENDMDL\n"
    return reversed_text, len(reversed_frames)


def get_trajectory_ca_stack(pdb_text):
    # CA coordinates per frame, in the file's native model order:
    # frame 0 is the final x0 (t=0), the last frame is the noisiest (t=T)
    frames = [frame.strip() for frame in pdb_text.split("ENDMDL") if frame.strip()]
    stack = [
        [
            (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            for line in frame.splitlines()
            if line.startswith("ATOM") and line[12:16].strip() == "CA"
        ]
        for frame in frames
    ]
    return np.asarray(stack, dtype=float)


def get_residue_plddt(residue, type="ca"):
    # CA or all-atom mean pLDDT
    if type == "ca" and "CA" in residue:
        return float(residue["CA"].bfactor)
    elif type == "mean":
        bf = [atom.bfactor for atom in residue.get_atoms()]
        return float(np.mean(bf))


def get_pdb_plddt(pdb_file, type="ca"):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("model", pdb_file)
    plddt_res = []
    for model in structure:
        for chain in model:
            for res in chain:
                if res.id[0] != " ":  # skip hetero/water
                    continue
                plddt_res.append(get_residue_plddt(res, type)*100 )  # convert to 0-100 scale
    return plddt_res


PLDDT_CONFIDENCE_RANGES = [
    {"lower": 0, "upper": 50, "color": "#ff7d45", "label": "Very low (<50)"},
    {"lower": 50, "upper": 70, "color": "#ffdb13", "label": "Low (50-70)"},
    {"lower": 70, "upper": 90, "color": "#65cbf3", "label": "High (70-90)"},
    {"lower": 90, "upper": 100, "color": "#0053d6", "label": "Very high (>90)"},
]

def get_plddt_ranges(plddt_values):
    ranges = []
    for plddt_range in PLDDT_CONFIDENCE_RANGES:
        lower = plddt_range["lower"]
        upper = plddt_range["upper"]
        if upper == 100:
            residue_mask = (plddt_values >= lower) & (plddt_values <= upper)
        else:
            residue_mask = (plddt_values >= lower) & (plddt_values < upper)
        ranges.append((residue_mask, plddt_range["color"]))
    return ranges


def transform_coordinate(coordinate, center1, center2, rotation):
    centered = [coordinate[i] - center1[i] for i in range(3)]
    return np.asarray([
        sum(rotation[i][j] * centered[j] for j in range(3)) + center2[i]
        for i in range(3)
    ])


def align_structures(esmfold_pdb_text, rf_pdb_text):
    parser = PDBParser(QUIET=True)
    rf_structure = parser.get_structure("rfdiffusion", StringIO(rf_pdb_text))
    esmfold_structure = parser.get_structure("esmfold", StringIO(esmfold_pdb_text))
    rf_ca_atoms = [atom for atom in rf_structure.get_atoms() if atom.get_name() == "CA"]
    esmfold_ca_atoms = [atom for atom in esmfold_structure.get_atoms() if atom.get_name() == "CA"]
    if len(rf_ca_atoms) != len(esmfold_ca_atoms):
        raise ValueError(f"C-alpha counts differ: RFdiffusion {len(rf_ca_atoms)}, ESMFold {len(esmfold_ca_atoms)}")

    rf_ca = np.asarray([atom.coord for atom in rf_ca_atoms], dtype=float)
    esmfold_ca = np.asarray([atom.coord for atom in esmfold_ca_atoms], dtype=float)
    rf_center = np.mean(rf_ca, axis=0)
    esmfold_center = np.mean(esmfold_ca, axis=0)
    reference = rf_ca - rf_center
    mobile = esmfold_ca - esmfold_center

    # Horn quaternion fit, using scalar arithmetic to avoid NumPy's BLAS backend.
    s = [[sum(mobile[k, i] * reference[k, j] for k in range(len(mobile))) for j in range(3)] for i in range(3)]
    sxx, sxy, sxz = s[0]
    syx, syy, syz = s[1]
    szx, szy, szz = s[2]
    horn = [
        [sxx + syy + szz, syz - szy, szx - sxz, sxy - syx],
        [syz - szy, sxx - syy - szz, sxy + syx, szx + sxz],
        [szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy],
        [sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz],
    ]
    shift = max(sum(abs(value) for value in row) for row in horn)
    for i in range(4):
        horn[i][i] += shift
    quaternion = [1.0, 0.0, 0.0, 0.0]
    for _ in range(2000):
        updated = [sum(horn[i][j] * quaternion[j] for j in range(4)) for i in range(4)]
        norm = sum(value * value for value in updated) ** 0.5
        quaternion = [value / norm for value in updated]
    w, x, y, z = quaternion
    rotation = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]

    for atom in esmfold_structure.get_atoms():
        atom.coord = transform_coordinate(atom.coord, esmfold_center, rf_center, rotation)

    esmfold_ca_aligned = np.asarray([atom.coord for atom in esmfold_ca_atoms])
    ca_distances = np.sqrt(np.sum((esmfold_ca_aligned - rf_ca) ** 2, axis=1))
    ca_rmsd = np.sqrt(np.mean(ca_distances ** 2))
    length = len(rf_ca_atoms)
    d0 = max(0.5, 1.24 * np.cbrt(max(length - 15, 1)) - 1.8)
    tm_score = np.mean(1.0 / (1.0 + (ca_distances / d0) ** 2))

    aligned_esmfold_buffer = StringIO()
    pdb_writer = PDBIO()
    pdb_writer.set_structure(esmfold_structure)
    pdb_writer.save(aligned_esmfold_buffer)

    return {
        "aligned_esmfold_pdb": aligned_esmfold_buffer.getvalue(),
        "ca_distances": ca_distances,
        "ca_rmsd": ca_rmsd,
        "tm_score": tm_score,
        "length": length,
    }
