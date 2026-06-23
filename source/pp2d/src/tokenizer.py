import itertools
import re
from dataclasses import dataclass

import numpy as np
import torch
from transformers import LlamaTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from .science_tokens import SCIENCE_TAG_TOKENS, SCIENCE_TOKENS


@dataclass
class TokenizerResult:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


class NatureLM1BTokenizer(PreTrainedTokenizerFast):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tag_re = re.compile(f'{"|".join(SCIENCE_TAG_TOKENS)}')
        self.smiles_re = re.compile(
            "(\\[[^\\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\\(|\\)|\\.|=|#|-|\\+|\\\\|\\/|:|~|@|\\?|>|\\*|\\$|\\%[0-9]{2}|[0-9])"
        )

        self.add_special_tokens(
            {
                "pad_token": "[PAD]",
                "unk_token": "<unk>",
            },
        )

        self.add_tokens(SCIENCE_TAG_TOKENS)
        self.add_tokens(SCIENCE_TOKENS)
        extra_tokens = []
        # protein
        for i in range(26):
            extra_tokens.append(f"<a>{chr(65 + i)}")

        # DNA, RNA, including ambiguous bases - CRITICAL: Only add these for multi-domain models
        # For molecule-focused models, we should avoid adding RNA/DNA tokens that confuse generation
        for c in "ACTGURYSWKMBDHVN":
            extra_tokens.append(f"<d>{c}")
            extra_tokens.append(f"<r>{c}")

        # materials, non-elements
        for c in "0123456789()+-":
            extra_tokens.append(f"<i>{c}")
        for i in range(26):
            extra_tokens.append(f"<i>{chr(65 + i)}")
            extra_tokens.append(f"<i>{chr(97 + i)}")
        # DNA six
        self.is_dna_six = kwargs.get("is_dna_six", False)
        if self.is_dna_six:
            nucleotides = ["A", "T", "C", "G"]
            all_kmers = ["".join(p) for p in itertools.product(nucleotides, repeat=6)]
            self.dna_six_encode_dict = {}
            for i in range(len(all_kmers)):
                extra_tokens.append(f"<d>{all_kmers[i]}")
                self.dna_six_encode_dict[all_kmers[i]] = True
        self.add_tokens(extra_tokens)
        self.split_special_tokens = True  # Ensure _tokenize() can access special tokens

    def _tokenize_entity(self, text: str, prefix: str, tok: str):
        if tok == "smiles":
            tokens = self.smiles_re.findall(text)
        elif tok == "space":
            tokens = text.split(" ")
        else:
            if self.is_dna_six and prefix == "d":
                i = 0
                tokens = []
                while i < len(text):
                    if text[i : i + 6] in self.dna_six_encode_dict:
                        tokens.append(text[i : i + 6])
                        i += 6
                    else:
                        tokens.append(text[i])
                        i += 1
            else:
                tokens = list(text)

        ret = []
        for t in tokens:
            if t == "":
                continue

            if t.startswith("<sg") and t.endswith(">"):
                # No <i> tag for subgroups
                ret.append(t)
            else:
                ret.append(f"<{prefix}>{t}")
        return ret

    def _tokenize_by_tag(self, span, tag, **kwargs):
        if tag in ["mol", "product", "reactants", "fragA", "fragB"]:
            tokens = self._tokenize_entity(span, "m", tok="smiles")
        elif tag in ["protein", "antibody"]:
            tokens = self._tokenize_entity(span, "a", tok="list")
        elif tag == "material":
            tokens = self._tokenize_entity(span, "i", tok="space")
        elif tag == "dna":
            tokens = self._tokenize_entity(span, "d", tok="list")
        elif tag == "rna":
            tokens = self._tokenize_entity(span, "r", tok="list")
        else:
            tokens = super().tokenize(span, **kwargs)

        return tokens

    def _tokenize(self, text, **kwargs):
        result = []
        cur_tag = None
        last_idx = 0

        known_tags = [
            "mol",
            "product",
            "reactants",
            "protein",
            "antibody",
            "material",
            "dna",
            "fragA",
            "fragB",
            "rna",
        ]

        for match in self.tag_re.finditer(text):
            start, end = match.span()
            match_str = match.group()

            if match_str.startswith("</"):
                tag = match_str[2:-1]
                if tag not in known_tags:
                    continue

                if tag != cur_tag:
                    raise ValueError(f"Tag mismatch: {tag} != {cur_tag} in '{text}'")

                span = text[last_idx:start].strip()
                tokens = self._tokenize_by_tag(span, tag, **kwargs)

                result.extend([t for t in tokens if t] + [f"</{tag}>"])
                cur_tag = None
            else:
                tag = match_str[1:-1]
                if tag not in known_tags:
                    continue

                if cur_tag is not None:
                    raise ValueError(f"Nested tag: {tag} in '{text}'")

                cur_tag = tag
                span = text[last_idx:start].strip()
                tokens = super().tokenize(span, **kwargs)

                result.extend([t for t in tokens if t] + [f"<{tag}>"])

            last_idx = end

        if last_idx < len(text):
            span = text[last_idx:].strip()
            tokens = self._tokenize_by_tag(span, cur_tag, **kwargs)
            result.extend(tokens)

        return result

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        for i in range(len(tokens)):
            # CRITICAL FIX: Remove ALL domain-specific prefixes, especially <r> and <d> for RNA/DNA
            for tag in ["<m>", "<a>", "<i>", "<r>", "<d>"]:
                tokens[i] = tokens[i].replace(tag, "")

        return super().convert_tokens_to_string(tokens)

    def _add_special_tokens(self, token_ids):
        return [self.bos_token_id] + token_ids + [self.eos_token_id]

    def _prepend_bos_tokens(self, token_ids):
        return [self.bos_token_id] + token_ids

    def _append_eos_tokens(self, token_ids):
        return token_ids + [self.eos_token_id]

    def encode(self, text, **kwargs):
        token_ids = self.convert_tokens_to_ids(self._tokenize(text))
        add_special_tokens = kwargs.get("add_special_tokens", False)
        prepend_bos = kwargs.get("prepend_bos", True)
        if "add_special_tokens" in kwargs and not add_special_tokens:
            prepend_bos = False
        append_eos = kwargs.get("append_eos", False)
        if add_special_tokens:
            token_ids = self._add_special_tokens(token_ids)
        if prepend_bos and not add_special_tokens:
            token_ids = self._prepend_bos_tokens(token_ids)
        if append_eos and not add_special_tokens:
            token_ids = self._append_eos_tokens(token_ids)
        return token_ids

    def __call__(self, text, **kwargs):
        add_special_tokens = kwargs.get("add_special_tokens", False)
        token_ids = self.convert_tokens_to_ids(self._tokenize(text))
        prepend_bos = kwargs.get("prepend_bos", True)
        if "add_special_tokens" in kwargs and not add_special_tokens:
            prepend_bos = False
        append_eos = kwargs.get("append_eos", False)
        return_tensors = kwargs.get("return_tensors", None)

        if add_special_tokens:
            token_ids = self._add_special_tokens(token_ids)
        if prepend_bos and not add_special_tokens:
            token_ids = self._prepend_bos_tokens(token_ids)
        if append_eos and not add_special_tokens:
            token_ids = self._append_eos_tokens(token_ids)

        attention_mask = [1] * len(token_ids)

        if return_tensors == "np":
            token_ids = np.array(token_ids)[np.newaxis, :]
            attention_mask = np.array(attention_mask)[np.newaxis, :]
        elif return_tensors == "pt":
            token_ids = torch.tensor(token_ids).unsqueeze(0)
            attention_mask = torch.tensor(attention_mask).unsqueeze(0)
        elif return_tensors is not None:
            raise ValueError(f"Unsupported tensor type: {return_tensors}")

        if return_tensors is not None:
            return TokenizerResult(input_ids=token_ids, attention_mask=attention_mask)
        else:
            result = {"input_ids": token_ids, "attention_mask": attention_mask}
            return result


class NatureLM8x7BTokenizer(LlamaTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tag_re = re.compile(f'{"|".join(SCIENCE_TAG_TOKENS)}')
        self.smiles_re = re.compile(
            "(\\[[^\\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\\(|\\)|\\.|=|#|-|\\+|\\\\|\\/|:|~|@|\\?|>|\\*|\\$|\\%[0-9]{2}|[0-9])"
        )

        self.add_special_tokens(
            {
                "pad_token": "[PAD]",
            },
        )

        self.add_tokens(SCIENCE_TAG_TOKENS)
        self.add_tokens(SCIENCE_TOKENS)
        extra_tokens = []
        # protein
        for i in range(26):
            extra_tokens.append(f"<a>{chr(65 + i)}")

        # DNA, RNA, including ambiguous bases
        for c in "ACTGURYSWKMBDHVN":
            extra_tokens.append(f"<d>{c}")
            extra_tokens.append(f"<r>{c}")

        # materials, non-elements
        for c in "0123456789()+-":
            extra_tokens.append(f"<i>{c}")
        for i in range(26):
            extra_tokens.append(f"<i>{chr(65 + i)}")
            extra_tokens.append(f"<i>{chr(97 + i)}")

        nucleotides = ["A", "T", "C", "G"]
        all_kmers = ["".join(p) for p in itertools.product(nucleotides, repeat=6)]
        self.dna_six_encode_dict = {}
        for i in range(len(all_kmers)):
            extra_tokens.append(f"<d>{all_kmers[i]}")
            self.dna_six_encode_dict[all_kmers[i]] = True

        self.add_tokens(extra_tokens)
        self.split_special_tokens = True  # Ensure _tokenize() can access special tokens

    def _tokenize_smiles(self, text):
        return self.smiles_re.findall(text)

    def _tokenize_by_space(self, text):
        return text.split(" ")

    def _tokeinze_as_list(self, text):
        return list(text)

    def _tokenize_dna6mer(self, text):
        i = 0
        tokens = []
        while i < len(text):
            if text[i : i + 6] in self.dna_six_encode_dict:
                tokens.append(text[i : i + 6])
                i += 6
            else:
                tokens.append(text[i])
                i += 1
        return tokens

    def _tokenize_entity(
        self, text: str, prefix: str, tok: str, use_text_num: bool = False
    ):
        if tok == "smiles":
            tokens = self._tokenize_smiles(text)
        elif tok == "space":
            tokens = self._tokenize_by_space(text)
        elif tok == "list":
            tokens = self._tokeinze_as_list(text)
        elif tok == "dna6mer":
            tokens = self._tokenize_dna6mer(text)
        else:
            raise Exception(f"unknown token type {tok}")

        ret = []
        for t in tokens:
            if t == "":
                continue

            should_add_prefix = True
            if t in ["<sg>", "<coord>", "<cs>"]:
                should_add_prefix = False
            elif t.startswith("<sg") and t.endswith(">"):
                # No <i> tag for spacegroups
                should_add_prefix = False
            elif use_text_num and t in "01234567890.+-":
                should_add_prefix = False

            if should_add_prefix:
                ret.append(f"<{prefix}>{t}")
            else:
                ret.append(t)
        return ret

    def _tokenize_by_tag(self, span, tag, **kwargs):
        span = span.strip(" ")

        if tag in ["mol", "product", "reactants", "fragA", "fragB", "reagent"]:
            tokens = self._tokenize_entity(span, "m", tok="smiles")
        elif tag in ["protein", "antibody"]:
            tokens = self._tokenize_entity(span, "a", tok="list")
        elif tag in ["material", "fcf"]:
            tokens = self._tokenize_entity(span, "i", tok="space", use_text_num=True)
        elif tag in ["cf1", "cf2"]:
            tokens = self._tokenize_entity(span, "i", tok="space", use_text_num=True)
        elif tag == "dna":
            tokens = self._tokenize_entity(span, "d", tok="list")
        elif tag == "dna6mer":
            tokens = self._tokenize_entity(span, "d", tok="dna6mer")
        elif tag == "rna":
            tokens = self._tokenize_entity(span, "r", tok="list")
        else:  # text
            tokens = super()._tokenize(span, **kwargs)

        return tokens

    def _tokenize(self, text, **kwargs):
        result = []
        cur_tag = None
        last_idx = 0

        known_tags = [
            "mol",
            "product",
            "reactants",
            "protein",
            "antibody",
            "material",
            "dna",
            "fragA",
            "fragB",
            "rna",
            "reagent",
            "cf1",
            "cf2",
            "fcf",
            "dna6mer",
        ]

        for match in self.tag_re.finditer(text):
            start, end = match.span()
            match_str = match.group()

            if match_str.startswith("</"):
                tag = match_str[2:-1]
                if tag not in known_tags:
                    continue

                if tag != cur_tag and cur_tag is not None:
                    raise ValueError(f"Tag mismatch: {tag} != {cur_tag} in '{text}'")

                span = text[last_idx:start]
                tokens = self._tokenize_by_tag(span, tag, **kwargs)

                result.extend([t for t in tokens if t])
                if cur_tag is not None:
                    result.append(f"</{cur_tag}>")
                cur_tag = None
            else:
                tag = match_str[1:-1]
                if tag not in known_tags:
                    continue

                if cur_tag is not None:
                    raise ValueError(f"Nested tag: {tag} in '{text}'")

                cur_tag = tag
                span = text[last_idx:start]
                tokens = self._tokenize_by_tag(span, None, **kwargs)

                result.extend([t for t in tokens if t] + [f"<{tag}>"])

            last_idx = end

        if last_idx < len(text):
            span = text[last_idx:]
            tokens = self._tokenize_by_tag(span, cur_tag, **kwargs)
            result.extend(tokens)

        return result

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        for i in range(len(tokens)):
            # CRITICAL FIX: Remove ALL domain-specific prefixes, especially <r> and <d> for RNA/DNA
            for tag in ["<m>", "<a>", "<i>", "<r>", "<d>"]:
                tokens[i] = tokens[i].replace(tag, "")

        return super().convert_tokens_to_string(tokens)
