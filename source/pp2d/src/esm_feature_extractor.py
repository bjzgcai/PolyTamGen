import os
from typing import Optional, Tuple, Union, List

import torch


class ESMFeatureExtractor:
    """
    Extract ESM features for a protein pocket from a local PDB file.

    This implementation uses the Forge ESM C inference client via the esm SDK.
    It constructs an ESMProtein from a local PDB, runs the model, and returns
    features based on user-specified pooling and layer selection.

    Requirements:
        - `esm` package with SDK (esm>=3) available in the environment
        - `biotite` package for reading/handling PDBs

    Token handling:
        - By default, the Forge token is read from environment variable `ESM_FORGE_TOKEN`.
          You may also pass a token string directly to the constructor.
    """

    def __init__(
        self,
        model_name: str = "esmc-300m-2024-12",
        forge_url: str = "https://forge.evolutionaryscale.ai",
        token: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        # Lazy import to avoid hard dependency if unused
        from esm.sdk import client as esm_client

        self.model_name = model_name
        self.forge_url = forge_url
        self.token = token or os.getenv("ESM_FORGE_TOKEN")
        if not self.token:
            raise RuntimeError(
                "Forge token not provided. Set ESM_FORGE_TOKEN env var or pass token=..."
            )

        # Initialize client
        self.client = esm_client(model=self.model_name, url=self.forge_url, token=self.token)

        # device is maintained for API compatibility; Forge runs remotely
        self.device = torch.device(device) if device is not None else torch.device("cpu")

    def _handle_api_error(self, error) -> None:
        """Helper to raise a more informative error for 403 Forbidden."""
        error_msg = str(error)
        if "403" in error_msg or "Forbidden" in error_msg:
            raise RuntimeError(
                f"Forge API returned 403 Forbidden. This usually means:\n"
                f"  1. Your token is invalid or expired\n"
                f"  2. Your token doesn't have permission to access the model '{self.model_name}'\n"
                f"  3. API access is restricted\n"
                f"Please check your ESM_FORGE_TOKEN and ensure it has the correct permissions.\n"
                f"Original error: {error_msg}"
            )
        else:
            raise RuntimeError(f"Forge API error: {error_msg}")

    @staticmethod
    def _load_protein_from_pdb(
        pdb_path: str, chain_id: Optional[str] = None, protein_id: Optional[str] = None
    ):
        """
        Load an ESMProtein from a local PDB file using biotite's ProteinChain helper.

        If chain_id is None and multiple chains exist, the first chain alphabetically
        will be selected.
        """
        from esm.sdk.api import ESMProtein
        from esm.utils.structure.protein_chain import ProteinChain
        from biotite.structure.io.pdb import PDBFile, get_structure

        if not os.path.isfile(pdb_path):
            raise FileNotFoundError(f"PDB file not found: {pdb_path}")

        # Determine default protein id from filename if not provided
        protein_id = protein_id or os.path.splitext(os.path.basename(pdb_path))[0]

        # If user provided chain_id, let ProteinChain parse it directly
        if chain_id is not None:
            protein_chain = ProteinChain.from_pdb(pdb_path, chain_id=chain_id, id=protein_id)
            return ESMProtein.from_protein_chain(protein_chain)

        # Otherwise, detect available chains and pick the first alphabetically
        pdb = PDBFile.read(pdb_path)
        structure = get_structure(pdb, model=1)
        
        # Get unique chain IDs from the structure
        chains: List[str] = sorted(set(structure.chain_id))
        if not chains:
            # Fallback: let ProteinChain try without specifying a chain
            protein_chain = ProteinChain.from_pdb(pdb_path, id=protein_id)
            return ESMProtein.from_protein_chain(protein_chain)

        selected_chain = chains[0]
        protein_chain = ProteinChain.from_pdb(pdb_path, chain_id=selected_chain, id=protein_id)
        return ESMProtein.from_protein_chain(protein_chain)

    def extract(
        self,
        pdb_path: str,
        chain_id: Optional[str] = None,
        pool: bool = True,
        layer_index: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Extracts features from a PDB, with control over pooling and layer selection.

        Args:
            pdb_path: Path to the .pdb file.
            chain_id: Specific chain ID to use. If None, auto-selects.
            pool: If True, returns mean-pooled features. If False, returns per-token features.
            layer_index:
                - If None: Returns features for *all* layers.
                - If int (e.g., -1): Returns features for *only* that specific layer.
                  (Note: SDK uses -1 to mean 'all layers', so we pass non-None values directly)

        Returns:
            A torch.Tensor. The shape depends on the arguments:
            - pool=True,  layer_index=None: [num_layers, hidden_size]
            - pool=True,  layer_index=N:    [hidden_size]
            - pool=False, layer_index=None: [num_layers, num_tokens, hidden_size]
            - pool=False, layer_index=N:    [num_tokens, hidden_size]
        """
        from esm.sdk.api import LogitsConfig, ESMProtein, ESMProteinError

        # 1. Load Protein
        protein: ESMProtein = self._load_protein_from_pdb(pdb_path, chain_id)

        # 2. Encode (API call)
        try:
            protein_tensor = self.client.encode(protein)
            if isinstance(protein_tensor, ESMProteinError):
                self._handle_api_error(protein_tensor)
        except Exception as e:
            # Catch other potential exceptions (e.g., network issues)
            if "403" in str(e) or "Forbidden" in str(e):
                self._handle_api_error(ESMProteinError(str(e)))
            raise RuntimeError(f"Forge API error during encode: {e}") from e

        # 3. Build LogitsConfig based on user request
        # SDK: ith_hidden_layer=-1 means all layers.
        if self.model_name == "esmc-6b-2024-12":
            # esmc-6b-2024-12 only supports single layer output
            if layer_index is None or layer_index == -1:
                raise ValueError("layer_index must be specified for esmc-6b-2024-12")
            cfg = LogitsConfig(
                return_hidden_states=True,
                return_mean_hidden_states=False,
                ith_hidden_layer=layer_index
            )
        else:
            cfg = LogitsConfig(
                return_hidden_states=True,
                return_mean_hidden_states=False,
                ith_hidden_layer=-1
            )

        # 4. Get Logits (API call)
        try:
            output = self.client.logits(protein_tensor, cfg)
        except Exception as e:
            # Wrap low-level errors with actionable message
            if "403" in str(e) or "Forbidden" in str(e):
                self._handle_api_error(e)
            raise RuntimeError(f"Forge API error during logits: {e}") from e
        # Some SDKs may return an error object instead of raising
        has_hidden_states = hasattr(output, "hidden_states")
        if not has_hidden_states:
            try:
                from esm.sdk.api import ESMProteinError  # type: ignore
                if isinstance(output, ESMProteinError):
                    self._handle_api_error(output)
            except Exception:
                # ignore import issues, fall through to generic error
                pass
            raise RuntimeError(
                "Forge logits response missing hidden_states. "
                "This usually indicates the API returned an error payload. "
                "Please confirm your Forge token has access to the model "
                f"'{self.model_name}'. Raw response: {output}"
            )

        # 5. Process the output
        # Select the correct attribute based on whether pooling was requested
        if pool:
            tensor_data = output.hidden_states.mean(dim=-2)
        else:
            tensor_data = output.hidden_states
            
        if not isinstance(tensor_data, torch.Tensor):
            tensor = torch.tensor(tensor_data, dtype=torch.float32)
        else:
            tensor = tensor_data
        
        # 6. Squeeze batch dimension
        if tensor.ndim > 0 and tensor.shape[1] == 1:
            tensor = tensor.squeeze(1)
        
        if self.model_name != "esmc-6b-2024-12" and layer_index is not None:
            tensor = tensor[layer_index]

        return tensor.to(self.device)


def extract_esm_features(
    pdb_path: str,
    chain_id: Optional[str] = None,
    pool: bool = True,
    layer_index: Optional[int] = None,
    model_name: str = "esmc-300m-2024-12",
    forge_url: str = "https://forge.evolutionaryscale.ai",
    token: Optional[str] = None,
) -> torch.Tensor:
    """
    Functional wrapper for quick ESM feature extraction.

    Args:
        pdb_path: Path to the .pdb file.
        chain_id: Specific chain ID to use. If None, auto-selects.
        pool: If True, returns mean-pooled features. If False, returns per-token features.
        layer_index:
            - If None: Returns features for *all* layers.
            - If int (e.g., -1): Returns features for *only* that specific layer.

    Returns:
        A torch.Tensor. The shape depends on the arguments:
        - pool=True,  layer_index=None: [num_layers, hidden_size]
        - pool=True,  layer_index=N:    [hidden_size]
        - pool=False, layer_index=None: [num_layers, num_tokens, hidden_size]
        - pool=False, layer_index=N:    [num_tokens, hidden_size]
    """
    extractor = ESMFeatureExtractor(
        model_name=model_name, forge_url=forge_url, token=token
    )
    return extractor.extract(
        pdb_path=pdb_path,
        chain_id=chain_id,
        pool=pool,
        layer_index=layer_index
    )
    