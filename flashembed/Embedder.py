import logging
from huggingface_hub import snapshot_download
from tokenizers import AddedToken, Tokenizer
import onnxruntime as ort
import numpy as np
import json
from pathlib import Path
from typing import List

class Embedder:
    """ Class for embedding multi-lingual text using a pre-trained retriever model.

    Attributes:
        cache_dir (Path): Path to the cache directory where models are stored.
        model_dir (Path): Path to the directory of the specific model being used.
        session (ort.InferenceSession): The ONNX runtime session for making inferences.
        tokenizer (Tokenizer): The tokenizer for text processing.
    """

    def __init__(self, model_id: str, cache_dir: str = "/tmp", max_length: int = 512, log_level: str = "INFO"):
        """ Initializes the Ranker class with specified model and cache settings.

        Args:
            model_id (str): The name of the model to be used.
            cache_dir (str): The directory where models are cached.
            max_length (int): The maximum length of the tokens.
            log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        """
        
        # Setting up logging
        logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))
        self.logger = logging.getLogger(__name__)

        self.cache_dir: Path = Path(cache_dir)
        if not self.cache_dir.exists():
            self.logger.debug(f"Cache directory {self.cache_dir} not found. Creating it..")
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        model_id_noprefix = model_id[model_id.find("/") + 1:]
        self.model_dir: Path = self.cache_dir / model_id_noprefix
        if not self.model_dir.exists():
            self.logger.info(f"Downloading {model_id}...")
            self._download_model_files(model_id)
        
        model_file = f"{model_id_noprefix}.onnx"    
        qualified_model_file = f"{str(self.model_dir)}/onnx/{model_file}"
        self.session = ort.InferenceSession(qualified_model_file)
        self.tokenizer: Tokenizer = self._get_tokenizer(max_length)

    def _download_model_files(self, model_id: str) -> str:
        """ Downloads Model from Huggingface Hub

        Args:
            model_id (str): The HF id of the model.
        """
        
        local_dir = str(self.model_dir)
        snapshot_download(repo_id=model_id, local_dir=local_dir,
                        local_dir_use_symlinks=False, revision="main")
        return local_dir
    
    def _get_tokenizer(self, max_length: int = 512) -> Tokenizer:
        """ Initializes and configures the tokenizer with padding and truncation.

        Args:
            max_length (int): The maximum token length for truncation.

        Returns:
            Tokenizer: Configured tokenizer for text processing.
        """

        config = json.load(open(str(self.model_dir / "config.json")))
        tokenizer_config = json.load(open(str(self.model_dir / "tokenizer_config.json")))
        tokens_map = json.load(open(str(self.model_dir / "special_tokens_map.json")))
        tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))

        tokenizer.enable_truncation(max_length=min(tokenizer_config["model_max_length"], max_length))
        tokenizer.enable_padding(pad_id=config["pad_token_id"], pad_token=tokenizer_config["pad_token"])

        for token in tokens_map.values():
            if isinstance(token, str):
                tokenizer.add_special_tokens([token])
            elif isinstance(token, dict):
                tokenizer.add_special_tokens([AddedToken(**token)])

        return tokenizer

    def encode(self, passages: List[str]) -> List[np.ndarray]:
        """ Embeds a list of passages using a pre-trained model.

        Args:
            passages: The request containing passages to embed.

        Returns:
            List[np.ndarray]: The embeddings of the input list of passages.
        """

        self.logger.debug("Running embedder..")

        input_text = self.tokenizer.encode_batch(passages)
        max_length = max(len(e.ids) for e in input_text)

        def pad_or_truncate(sequence, max_length):
            if len(sequence) > max_length:
                return sequence[:max_length]
            elif len(sequence) < max_length:
                return sequence + [0] * (max_length - len(sequence))
            return sequence

        input_ids = [pad_or_truncate(e.ids, max_length) for e in input_text]
        token_type_ids = [pad_or_truncate(e.type_ids, max_length) for e in input_text]
        attention_mask = [pad_or_truncate(e.attention_mask, max_length) for e in input_text]

        onnx_input = {
            "input_ids": np.array(input_ids, dtype=np.int64),
            "attention_mask": np.array(attention_mask, dtype=np.int64)
        }

        use_token_type_ids = token_type_ids is not None and not np.all(token_type_ids == 0)
        if use_token_type_ids:
            onnx_input["token_type_ids"] = np.array(token_type_ids, dtype=np.int64)

        outputs = self.session.run(None, onnx_input)
        return outputs[0]
