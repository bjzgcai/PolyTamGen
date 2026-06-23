import contextlib
import logging
import os
import os.path as op
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple, List
import pickle as pkl
from collections import defaultdict
import hashlib
import math
from multiprocessing import Process, Manager

# External Dependencies
import AutoDockTools
from meeko import MoleculePreparation
from openbabel import pybel
from rdkit import Chem
from rdkit.Chem import AllChem
from vina import Vina
import gemmi

# --- Setup Logging ---
# 只配置当前模块的logger，不影响全局
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --- Utility Functions ---

@contextlib.contextmanager
def timing(msg: str):
    """A context manager to log the duration of a code block."""
    logger.info("Started: %s", msg)
    tic = time.time()
    yield
    toc = time.time()
    logger.info("Finished: %s in %.3f seconds", msg, toc - tic)

def supress_stdout(func):
    """A decorator to suppress standard output from a function."""
    def wrapper(*args, **kwargs):
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull):
                return func(*args, **kwargs)
    return wrapper

# --- Core Classes and Functions ---

class PrepProt:
    """
    Handles the preparation of a protein receptor for docking.
    This class encapsulates steps like file format conversion, water removal,
    hydrogen addition, and PDBQT file generation.
    """
    def __init__(self, prot_file: str):
        if not op.exists(prot_file):
            raise FileNotFoundError(f"Protein file not found: {prot_file}")
        self.original_path = prot_file
        self.file_format = Path(prot_file).suffix.lower()

    def prepare_for_docking(self, workdir: Path) -> Path:
        """
        Runs the full preparation pipeline for the protein.
        1. Converts CIF to PDB (if needed), as downstream tools work best with PDB.
        2. Removes water molecules.
        3. Adds hydrogens using pdb2pqr.
        4. Generates the final PDBQT file.
        Returns the path to the prepared PDBQT file.
        """
        # 1. Standardize to PDB format using Gemmi
        pdb_path = workdir / f"{Path(self.original_path).stem}.pdb"
        try:
            st = gemmi.read_structure(self.original_path)
            st.remove_waters()
            st.write_pdb(str(pdb_path))
        except (RuntimeError, ValueError) as e:
            raise RuntimeError(f"Gemmi failed to parse or write {self.original_path}: {e}") from e

        # 2. Add hydrogens using pdb2pqr
        pqr_path = workdir / f"{Path(pdb_path).stem}.pqr"
        try:
            # Use subprocess.run for better error handling and timeout control
            subprocess.run(
                ["pdb2pqr30", "--ff=AMBER", str(pdb_path), str(pqr_path)],
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                check=True,  # Raises CalledProcessError on non-zero exit codes
                timeout=120  # 2-minute timeout
            )
            receptor_for_pdbqt = pqr_path
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"pdb2pqr failed for {pdb_path}: {e}. Proceeding without pdb2pqr.")
            # Fallback: use the water-removed PDB file directly
            receptor_for_pdbqt = pdb_path

        # 3. Generate PDBQT using prepare_receptor4.py
        prot_pdbqt_path = workdir / f"{Path(self.original_path).stem}.pdbqt"
        prepare_receptor_script = op.join(AutoDockTools.__path__[0], "Utilities24/prepare_receptor4.py")
        try:
            subprocess.run(
                [sys.executable, prepare_receptor_script, "-r", str(receptor_for_pdbqt), "-o", str(prot_pdbqt_path)],
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                check=True,
                timeout=120
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
            if isinstance(e, subprocess.CalledProcessError) and hasattr(e, 'stderr') and e.stderr is not None:
                stderr = e.stderr.decode()
            else:
                stderr = str(e)
            raise RuntimeError(f"prepare_receptor4.py failed for {receptor_for_pdbqt}: {stderr}") from e

        return prot_pdbqt_path


class PrepLig:
    """Handles the preparation of a ligand molecule for docking."""
    def __init__(self, input_mol: str, mol_format: str):
        try:
            if mol_format == "smi":
                self.ob_mol = pybel.readstring("smi", input_mol)
            elif mol_format == "sdf":
                self.ob_mol = next(pybel.readfile(mol_format, input_mol))
            else:
                raise ValueError(f"mol_format {mol_format} not supported")
        except Exception as e:
            raise ValueError(f"Failed to read ligand: {input_mol}") from e

    def addH(self, polaronly: bool = False, correctforph: bool = True, PH: float = 7.4):
        """Adds hydrogens to the ligand."""
        self.ob_mol.OBMol.AddHydrogens(polaronly, correctforph, PH)

    def gen_conf(self, seed: int = 0):
        """Generates a 3D conformation for the ligand using RDKit."""
        sdf_block = self.ob_mol.write("sdf")
        rdkit_mol = Chem.MolFromMolBlock(sdf_block, removeHs=False)
        if not rdkit_mol:
            raise ValueError("Could not create RDKit molecule from OpenBabel input.")
        
        # Use ETKDGv3 for better conformer generation
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        conf_id = AllChem.EmbedMolecule(rdkit_mol, params)
        
        if conf_id == -1:
            logger.warning("Could not embed 3D conformer, falling back to 2D.")
            AllChem.Compute2DCoords(rdkit_mol)
        else:
            try:
                AllChem.MMFFOptimizeMolecule(rdkit_mol, maxIters=500, confId=conf_id)
            except Exception as e:
                logger.warning(f"MMFF optimization failed: {e}. Using unoptimized conformer.")

        self.ob_mol = pybel.readstring("sdf", Chem.MolToMolBlock(rdkit_mol))

    @supress_stdout
    def get_pdbqt(self, lig_pdbqt_path: Optional[str] = None):
        """Generates a PDBQT file or string for the ligand using Meeko."""
        preparator = MoleculePreparation()
        preparator.prepare(self.ob_mol.OBMol)
        if lig_pdbqt_path:
            preparator.write_pdbqt_file(lig_pdbqt_path)
            return None
        return preparator.write_pdbqt_string()


def get_box_from_structure(structure_file: str, buffer: float = 5.0) -> Tuple[List[float], List[float]]:
    """
    Calculates a bounding box from all heavy atoms in a structure file.
    Uses Gemmi for robust parsing of PDB and CIF files.
    
    Args:
        structure_file: Path to the PDB or CIF file.
        buffer: A buffer in angstroms to add to each dimension of the box.

    Returns:
        A tuple containing the box center (list of 3 floats) and box size (list of 3 floats).
    """
    try:
        st = gemmi.read_structure(structure_file)
        st.remove_hydrogens()
        st.remove_waters()
    except (RuntimeError, ValueError) as e:
        raise ValueError(f"Failed to parse structure file {structure_file}: {e}") from e
        
    all_coords = [atom.pos.tolist() for model in st for chain in model for residue in chain for atom in residue]
    if not all_coords:
        raise ValueError("No atomic coordinates found in the structure file.")

    # Transpose list of lists to get lists of x, y, z coordinates
    coords_by_axis = list(zip(*all_coords))
    min_coords = [min(c) for c in coords_by_axis]
    max_coords = [max(c) for c in coords_by_axis]

    center = [(max_c + min_c) / 2 for max_c, min_c in zip(max_coords, min_coords)]
    box_size = [(max_c - min_c) + buffer for max_c, min_c in zip(max_coords, min_coords)]

    return center, box_size


def _prepare_receptor_cached(
    receptor_path: str,
    cache_dir: str,
    buffer: float = 5.0,
) -> Tuple[str, List[float], List[float]]:
    """
    Prepare receptor once and cache the heavy results (PDBQT + docking box).

    This function is internal to this module and does NOT change the public
    dock() API. It is safe to call from multiple processes.

    Returns:
        prepared_receptor_pdbqt_path, center, box_size
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Use receptor path and mtime to build a stable cache key
    try:
        mtime = os.path.getmtime(receptor_path)
    except OSError:
        mtime = 0.0
    key_str = f"{receptor_path}|{mtime}|{buffer}"
    key_hash = hashlib.md5(key_str.encode()).hexdigest()
    receptor_cache_dir = os.path.join(cache_dir, f"receptor_{key_hash}")
    os.makedirs(receptor_cache_dir, exist_ok=True)

    pdbqt_path = os.path.join(
        receptor_cache_dir, f"{Path(receptor_path).stem}.pdbqt"
    )
    meta_path = os.path.join(receptor_cache_dir, "box_meta.pkl")

    center: List[float]
    box_size: List[float]

    # Fast path: both PDBQT and box information exist
    if os.path.exists(pdbqt_path) and os.path.exists(meta_path):
        try:
            with open(meta_path, "rb") as f:
                data = pkl.load(f)
                center = data["center"]
                box_size = data["box_size"]
            return pdbqt_path, center, box_size
        except Exception:
            # Fallback to recompute if cache is corrupted
            logger.warning(
                "Receptor cache seems corrupted for %s. Recomputing.", receptor_path
            )

    # Recompute receptor preparation and box
    center, box_size = get_box_from_structure(receptor_path, buffer=buffer)
    prep = PrepProt(receptor_path)
    prepared_pdbqt = prep.prepare_for_docking(Path(receptor_cache_dir))

    # Ensure we use the computed path (may differ in name)
    pdbqt_path = str(prepared_pdbqt)

    # Save metadata
    try:
        with open(meta_path, "wb") as f:
            pkl.dump({"center": center, "box_size": box_size}, f)
    except Exception as e:
        logger.warning("Failed to write receptor cache meta for %s: %s", receptor_path, e)

    return pdbqt_path, center, box_size


def _dock_worker(
    receptor_path: str,
    ligand_smiles: str,
    center: List[float],
    box_size: List[float],
    output_complex_path: Optional[str],
    num_ligand_confs: int,
    tmpdir_str: str,
    result_dict,
    exhaustiveness: int,
    seed: int,
):
    """
    The actual docking workflow, designed to be run in a separate process.
    This encapsulates protein/ligand prep and the Vina docking loop.
    """
    try:
        tmpdir = Path(tmpdir_str)
        
        # 1. Prepare Protein
        # If receptor is already a prepared PDBQT file, reuse it directly.
        # Otherwise, run the full preparation pipeline.
        with timing("protein preparation"):
            receptor_path_obj = Path(receptor_path)
            if receptor_path_obj.suffix.lower() == ".pdbqt":
                prot_pdbqt = receptor_path_obj
            else:
                prot = PrepProt(receptor_path)
                prot_pdbqt = prot.prepare_for_docking(tmpdir)

        # 2. Prepare Ligand Conformations
        with timing(f"generating {num_ligand_confs} ligand conformer(s)"):
            ligand_pdbqt_paths = []
            for conf_seed in range(num_ligand_confs):
                lig = PrepLig(ligand_smiles, "smi")
                lig.addH()
                lig.gen_conf(seed=conf_seed)
                lig_pdbqt_path = tmpdir / f"ligand_{conf_seed}.pdbqt"
                lig.get_pdbqt(str(lig_pdbqt_path))
                ligand_pdbqt_paths.append(lig_pdbqt_path)

        # 3. Docking
        # Vina maps are computed once, then each ligand conformer is docked.
        with timing("Vina docking"):
            vina = Vina(sf_name="vina", seed=seed, verbosity=0)
            vina.set_receptor(str(prot_pdbqt))
            vina.compute_vina_maps(center=center, box_size=box_size)

            best_affinity = float('inf')
            best_pose_pdbqt_str = None

            for lig_pdbqt in ligand_pdbqt_paths:
                vina.set_ligand_from_file(str(lig_pdbqt))
                vina.dock(exhaustiveness=exhaustiveness, n_poses=1)
                energy = vina.energies(n_poses=1)[0][0]
                if energy < best_affinity:
                    best_affinity = energy
                    if output_complex_path:
                        best_pose_pdbqt_str = vina.poses(n_poses=1)

            if output_complex_path and best_pose_pdbqt_str:
                with open(output_complex_path, "w") as f:
                    f.write(str(best_pose_pdbqt_str))
            
            # If no conformer docked successfully, return a high (bad) score.
            result_dict["result"] = best_affinity if best_affinity != float('inf') else 500.0

    except Exception as e:
        import traceback
        result_dict["error"] = f"{e}\n{traceback.format_exc()}"


def dock(
    receptor_path: str,
    ligand_identifier: str,
    center: Optional[Tuple[float, float, float]] = None,
    box_size: Optional[Tuple[float, float, float]] = None,
    output_complex_path: Optional[str] = None,
    num_ligand_confs: int = 1,
    cache_dir: Optional[str] = None,
    timeout: int = 300,
    exhaustiveness: int = 32,
    seed: int = 1234,
):
    """
    Main function to dock a ligand to a receptor. Manages caching, timeouts,
    and orchestrates the preparation and docking workflow.

    Args:
        receptor_path: Path to the receptor file (PDB or CIF).
        ligand_identifier: SMILES string of the ligand.
        center: Optional box center coordinates. If None, it's calculated from the receptor.
        box_size: Optional box size. If None, it's calculated from the receptor.
        output_complex_path: Path to save the best docked pose in PDBQT format.
        num_ligand_confs: Number of ligand conformations to generate and test.
        cache_dir: Directory to save/load cached docking results.
        timeout: Total time in seconds allowed for the entire docking job.
        exhaustiveness: Vina exhaustiveness parameter.
        seed: Vina random seed.

    Returns:
        The best docking affinity (float) in kcal/mol.
    """
    # Validate ligand SMILES
    try:
        ligand_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(ligand_identifier))
    except Exception:
        raise ValueError(f"Invalid ligand SMILES string provided: {ligand_identifier}")
        
    # --- Caching ---
    cache_file_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        receptor_name = Path(receptor_path).stem
        # Hash key includes key parameters to avoid stale caches
        unique_key = f"{receptor_path}|{ligand_smiles}|{num_ligand_confs}|{exhaustiveness}"
        unique_hash = hashlib.md5(unique_key.encode()).hexdigest()
        cache_file_path = op.join(cache_dir, f"{receptor_name}_{unique_hash}.pkl")
        
        # If only a score is needed, a cached affinity is sufficient.
        # If a pose file is explicitly requested and does not exist yet,
        # we must re-run docking to materialize the pose.
        if op.exists(cache_file_path) and (
            output_complex_path is None or op.exists(output_complex_path)
        ):
            logger.info(f"Loading cached result from {cache_file_path}")
            with open(cache_file_path, "rb") as f:
                return pkl.load(f)

    # --- Prepare receptor and box ---
    receptor_for_worker = receptor_path
    if center is None or box_size is None:
        if cache_dir:
            # Reuse heavy receptor preparation across ligands and runs
            receptor_for_worker, center, box_size = _prepare_receptor_cached(
                receptor_path, cache_dir, buffer=5.0
            )
        else:
            logger.info("Calculating docking box from receptor structure.")
            center, box_size = get_box_from_structure(receptor_path)
            logger.info(f"Calculated Box Center: {center}, Size: {box_size}")
    else:
        # Center/box explicitly provided; we may still reuse prepared receptor if cache_dir is set
        if cache_dir:
            receptor_for_worker, _, _ = _prepare_receptor_cached(
                receptor_path, cache_dir, buffer=5.0
            )
        else:
            receptor_for_worker = receptor_path
    
    # --- Run Docking in a Separate Process for Timeout Control ---
    manager = Manager()
    result_dict = manager.dict()
    # Define specific values for different failure modes (NaN to avoid poisoning)
    timeout_value_to_cache = float("nan")
    generic_failure_value = float("nan")

    with tempfile.TemporaryDirectory() as tmpdir:
        process = Process(
            target=_dock_worker,
            args=(
                receptor_for_worker, ligand_smiles, list(center), list(box_size),
                output_complex_path, num_ligand_confs, tmpdir, result_dict,
                exhaustiveness, seed
            )
        )
        process.start()
        process.join(timeout)

        if process.is_alive():
            process.terminate()
            process.join()
            error_message = f"Docking process timed out after {timeout} seconds."
            logger.error(error_message)
            # Cache timeout as NaN to avoid biasing scores
            if cache_dir and cache_file_path:  # Cache the timeout result
                with open(cache_file_path, "wb") as f:
                    pkl.dump(timeout_value_to_cache, f)
            raise TimeoutError(error_message)
        
        if "error" in result_dict:
            raise RuntimeError(f"Error during docking process: {result_dict['error']}")

        affinity = result_dict.get("result", generic_failure_value)

    # --- Save to Cache and Return ---
    if cache_dir and cache_file_path:
        # Only cache finite numeric values; otherwise cache NaN for consistency
        to_cache = affinity if isinstance(affinity, (int, float)) else float("nan")
        if isinstance(to_cache, float) and (math.isnan(to_cache) or math.isinf(to_cache)):
            to_cache = float("nan")
        with open(cache_file_path, "wb") as f:
            pkl.dump(to_cache, f)
        logger.info(f"Saved result to cache: {cache_file_path}")

    return affinity
