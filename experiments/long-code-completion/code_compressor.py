import torch
import numpy as np
from typing import List, Union, Tuple, Dict, Optional
import re
import math
import zlib
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from tqdm import tqdm
import copy
import bisect
import json
from loguru import logger

class EntropyChunking:
    def __init__(self, model_name="Qwen/Qwen2.5-Coder-0.5B-Instruct"):
        """Entropy-based text chunking implementation"""
        logger.debug(f"Loading Entropy chunking model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.debug(f"Entropy chunking model loaded on device: {self.device}")

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences, inserting empty lines for double newlines"""
        # First replace double newlines with a special marker
        text_with_markers = text.replace('\n\n', '\n__EMPTY_LINE__\n')
        
        # Split by single newlines
        lines = text_with_markers.split('\n')
        
        # Process lines: replace markers with empty strings, keep original lines
        sentences = []
        for line in lines:
            if line == '__EMPTY_LINE__':
                sentences.append(' ')  # Empty line for double newline breaks
            else:
                sentences.append(line)  # Keep original line with indentation
        
        return sentences

    def calculate_sentence_ppl(self, sentences: List[str]) -> List[float]:
        """Calculate perplexity for each sentence based on preceding context"""
        ppls = []
        
        for i, sentence in enumerate(sentences):
            if i == 0:
                context = ""
                target = sentence
            else:
                context = "\n".join(sentences[:i])
                target = sentence
            
            ppl = self._compute_ppl(context, target)
            ppls.append(ppl)
        
        return ppls

    def _compute_ppl(self, context: str, target: str) -> float:
        """Compute perplexity of target text given context"""
        # Handle empty target lines
        if not target:
            return 0.0  # Assign zero perplexity to empty lines
            
        if context:
            full_text = context + "\n" + target
            context_tokens = self.tokenizer(context + "\n", return_tensors="pt", add_special_tokens=True)
            context_length = context_tokens.input_ids.shape[1]
        else:
            full_text = target
            context_length = 0
        
        inputs = self.tokenizer(full_text, return_tensors="pt", add_special_tokens=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        if context_length > 0:
            target_logits = logits[0, context_length-1:-1]
            target_labels = inputs.input_ids[0, context_length:]
        else:
            target_logits = logits[0, :-1]
            target_labels = inputs.input_ids[0, 1:]
        
        if len(target_labels) > 0:
            log_probs = torch.log_softmax(target_logits, dim=-1)
            token_log_probs = log_probs[torch.arange(len(target_labels)), target_labels]
            avg_log_prob = token_log_probs.mean().item()
            ppl = math.exp(-avg_log_prob)
        else:
            ppl = float('inf')

        # take log2 of ppl
        ppl = math.log2(ppl)
        
        return ppl

    def calculate_adaptive_thresholds(self, ppls: List[float], k: float = 0.2) -> dict:
        """Calculate adaptive thresholds using different statistical methods"""
        # Filter out infinite and NaN values
        valid_ppls = [p for p in ppls if not math.isinf(p) and not math.isnan(p) and p > 0]
        
        if len(valid_ppls) < 3:
            # Fallback to fixed threshold if not enough valid data
            return {
                'std': 0.5,
                'robust_std': 0.5,
                'iqr': 0.5,
                'mad': 0.5
            }
        
        valid_ppls = np.array(valid_ppls)
        
        # Method 1: Standard deviation based
        mean_ppl = np.mean(valid_ppls)
        std_ppl = np.std(valid_ppls)
        threshold_std = mean_ppl + k * std_ppl
        
        # Method 2: Robust standard deviation (using median and MAD)
        median_ppl = np.median(valid_ppls)
        mad = np.median(np.abs(valid_ppls - median_ppl))
        robust_std = mad * 1.4826  # Convert MAD to robust std estimate
        threshold_robust_std = median_ppl + k * robust_std
        
        # Method 3: IQR based (Interquartile Range)
        q25 = np.percentile(valid_ppls, 25)
        q75 = np.percentile(valid_ppls, 75)
        iqr = q75 - q25
        threshold_iqr = q75 + k * iqr
        
        # Method 4: MAD based (Median Absolute Deviation)
        threshold_mad = median_ppl + k * mad
        
        return {
            'std': threshold_std,
            'robust_std': threshold_robust_std,
            'iqr': threshold_iqr,
            'mad': threshold_mad
        }

    def find_ppl_spikes_adaptive(self, values: List[float], method: str = 'std', k: float = 0.2) -> tuple:
        """Find PPL spikes using adaptive threshold based on statistical method"""
        thresholds = self.calculate_adaptive_thresholds(values, k)
        threshold = thresholds[method]
        
        spike_indices = []
        
        for i in range(1, len(values) - 1):
            current = values[i]
            left = values[i - 1]
            right = values[i + 1]
            
            # Skip infinite or NaN values
            if math.isinf(current) or math.isnan(current):
                continue
            if math.isinf(left) or math.isnan(left):
                left = current
            if math.isinf(right) or math.isnan(right):
                right = current
            
            # Check if current PPL is significantly higher than both neighbors
            left_diff = current - left
            right_diff = current - right
            
            # Condition: Current PPL is higher than both neighbors with adaptive threshold
            if (left_diff >= threshold or right_diff >= threshold) and (left_diff >= 0 and right_diff >= 0):
                spike_indices.append(i)
        
        return spike_indices, threshold

    def chunk_text_adaptive(self, text: str, method: str = 'std', k: float = 0.2) -> tuple:
        """Perform PPL-based text chunking using adaptive spike detection"""
        sentences = self.split_into_sentences(text)
        ppls = self.calculate_sentence_ppl(sentences)
        spike_indices, threshold = self.find_ppl_spikes_adaptive(ppls, method, k)
        
        chunks = []
        # Split at spike points (after the spike line)
        split_points = [0] + [idx + 1 for idx in spike_indices] + [len(sentences)]
        
        for i in range(len(split_points) - 1):
            start = split_points[i]
            end = split_points[i + 1]
            chunk_sentences = sentences[start:end]
            chunk_text = "\n".join(chunk_sentences)
            chunks.append(chunk_text)
        
        return chunks, sentences, ppls, spike_indices

class CodeCompressor:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct-GPTQ-Int4",
        device_map: str = "cuda",
        model_config: dict = {},
    ):
        """
        Initialize the CodeCompressor with a language model for compression.
        
        Args:
            model_name: The name of the model to load from HuggingFace
            device_map: Device to load the model on
            model_config: Additional configuration for the model
        """
        self.model_name = model_name
        self.device = device_map
        self.model_config = model_config
        self.load_model(model_name, device_map, model_config)
        
        # Initialize Entropy chunking with smaller model
        logger.debug("Initializing Entropy chunking...")
        self.entropy_chunking = EntropyChunking()
        
        # Add caching system for model outputs and token information
        self.cache = {
            "token_length": {},      # Cache for token length by text
            "encodings": {},         # Cache for tokenizer encodings
            "perplexity": {},        # Cache for perplexity calculations
            "conditional_ppl": {},   # Cache for conditional perplexity
            "context_rankings": {},  # Cache for context rankings
        }
        self.max_cache_size = 1000   # Limit cache size to prevent memory issues
        
        # set up the max position embeddings and cache bos num
        self.max_position_embeddings = getattr(self.model.config, "max_position_embeddings", 4096)
        self.cache_bos_num = 10
        self.prefix_bos_num = 100
        self.context_idxs = []
    
    def load_model(
        self, model_name: str, device_map: str = "cuda", model_config: dict = {}
    ):
        """
        Load the language model and tokenizer.
        
        Args:
            model_name: The name of the model to load
            device_map: Device to load the model on
            model_config: Additional configuration for the model
        """
        logger.debug(f"Loading model {model_name} on {device_map}")
        torch_dtype = torch.bfloat16 if "torch_dtype" not in model_config else model_config["torch_dtype"]
        # model_kwargs = {"device_map": device_map, "torch_dtype": torch_dtype, "trust_remote_code": True}
        model_kwargs = {"device_map": device_map, "torch_dtype": torch_dtype, "trust_remote_code": True}
        
        for k, v in model_config.items():
            model_kwargs[k] = v
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        logger.debug("Model and tokenizer loaded successfully")
        
    def _manage_cache_size(self, cache_type):
        """
        Manage cache size by removing oldest entries when cache exceeds max size.
        
        Args:
            cache_type: The type of cache to manage
        """
        if len(self.cache[cache_type]) > self.max_cache_size:
            # Remove 20% of the oldest entries
            remove_count = int(self.max_cache_size * 0.2)
            keys_to_remove = list(self.cache[cache_type].keys())[:remove_count]
            for key in keys_to_remove:
                del self.cache[cache_type][key]
        
    def get_token_length(
        self,
        text: str,
        add_special_tokens: bool = True,
    ):
        """
        Get the number of tokens in the given text.
        
        Args:
            text: The text to tokenize
            add_special_tokens: Whether to count special tokens
            
        Returns:
            The number of tokens
        """
        # Create a cache key based on text and parameters
        cache_key = f"{text}_{add_special_tokens}"
        
        # Check if result is in cache
        if cache_key in self.cache["token_length"]:
            return self.cache["token_length"][cache_key]
        
        # Calculate token length if not in cache
        token_length = len(self.tokenizer.encode(text, add_special_tokens=add_special_tokens))
        
        # Store in cache
        self.cache["token_length"][cache_key] = token_length
        self._manage_cache_size("token_length")
        
        return token_length
    
    def get_ppl(
        self,
        text: str,
        granularity: str = "line",
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        return_kv=False,
        end=None,
        condition_mode: str = "none",
        condition_pos_id: int = 0,
    ):
        """
        Calculate perplexity for the given text at line level.
        
        Args:
            text: The text to calculate perplexity for
            granularity: The granularity of perplexity calculation (line, token, chunk)
            input_ids, attention_mask, past_key_values: Optional pre-processed inputs
            return_kv: Whether to return key-values
            end: End position for calculation
            condition_mode: Mode for conditional perplexity (none, prefix)
            condition_pos_id: Position ID for condition
            
        Returns:
            A dictionary with perplexity scores and processing information
        """
        # Create a cache key for this specific perplexity calculation
        cache_key = f"{text}_{granularity}_{condition_mode}_{condition_pos_id}"
        if past_key_values is None and not return_kv and cache_key in self.cache["perplexity"]:
            return self.cache["perplexity"][cache_key]
        
        # Initialize input processing
        if input_ids is None:
            encoding_key = text
            if encoding_key in self.cache["encodings"]:
                cached_encoding = self.cache["encodings"][encoding_key]
                input_ids = cached_encoding["input_ids"].to(self.model.device)
                attention_mask = cached_encoding["attention_mask"].to(self.model.device)
            else:
                encoding = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True
                )
                input_ids = encoding["input_ids"].to(self.model.device)
                attention_mask = encoding["attention_mask"].to(self.model.device)

                # Cache CPU copies; .to(device) on retrieval to avoid GPU residency
                self.cache["encodings"][encoding_key] = {
                    "input_ids": encoding["input_ids"],
                    "attention_mask": encoding["attention_mask"]
                }
                self._manage_cache_size("encodings")
        
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
        else:
            past_length = 0
            
        if end is None:
            end = input_ids.shape[1]
        end = min(end, past_length + self.max_position_embeddings)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids[:, past_length:end],
                attention_mask=attention_mask[:, :end],
                past_key_values=past_key_values,
                return_dict=True,
                output_hidden_states=False,
                use_cache=True,
            )
        
        # Get logits and shift
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., past_length+1:end].contiguous()
        
        # Flatten tokens for loss calculation
        active = (attention_mask[:, past_length:end] == 1)[..., :-1].view(-1)
        active_logits = shift_logits.view(-1, shift_logits.size(-1))[active]
        active_labels = shift_labels.view(-1)[active]
        
        # Calculate loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(active_logits, active_labels)
        
        # Apply condition filtering if required
        if condition_mode == "prefix":
            loss = loss[condition_pos_id:]
            
        segments = [text] if text else []
        lines_info = []
            
        # Calculate mean perplexity
        mean_loss = loss.mean() if len(loss) > 0 else torch.tensor(0.0)
        ppl = torch.exp(mean_loss).item() if mean_loss.item() != float('inf') else float('inf')
        
        result = {
            "loss": loss,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "lines_info": lines_info,
            "segments": segments,
            "ppl": ppl,
        }
        
        if return_kv:
            result["past_key_values"] = outputs.past_key_values
        else:
            # Cache the result if we're not returning KV cache.
            # Detach tensors to CPU to avoid pinning GPU memory in the cache.
            cached_result = {
                k: (v.detach().cpu() if torch.is_tensor(v) else v)
                for k, v in result.items()
            }
            self.cache["perplexity"][cache_key] = cached_result
            self._manage_cache_size("perplexity")

        return result
    
    def __get_lines_info(self, lines, input_ids, loss):
        """
        Get information about each line including start/end positions and importance.
        
        Args:
            lines: List of lines in the text
            input_ids: Token IDs for the entire text
            loss: Per-token loss values
            
        Returns:
            List of dictionaries with line information
        """
        line_info = []
        cumulative_tokens = 0
        
        input_ids_list = input_ids.cpu().tolist()
        
        for i, line in enumerate(lines):
            if not line.strip():
                continue
                
            # Encode each line to find its token length
            line_tokens = self.tokenizer.encode(line, add_special_tokens=False)
            line_length = len(line_tokens)
            
            # Find position in the tokenized text
            start_pos = cumulative_tokens
            end_pos = start_pos + line_length
            
            # Calculate mean loss (importance) for this line
            # Loss might be shorter than the token IDs due to shifting
            if isinstance(loss, torch.Tensor) and start_pos < len(loss) and end_pos <= len(loss):
                line_loss = loss[start_pos:end_pos].mean().item()
            else:
                # Handle edge cases
                line_loss = float("inf")
            
            line_info.append({
                "line": line,
                "start": start_pos,
                "end": end_pos,
                "importance": line_loss,
                "tokens": line_length
            })
            
            cumulative_tokens += line_length
            
        return line_info
    
    def get_prefix_length(self, prefix: str, text: str):
        """
        Calculate the length of a prefix in tokens when concatenated with a text.
        
        Args:
            prefix: The prefix text
            text: The main text
            
        Returns:
            Length of the prefix in tokens
        """
        possible_prefix_token = max(self.get_token_length(prefix, False) - 3, 1)
        full_input_ids = self.tokenizer(prefix + text[:100], add_special_tokens=False).input_ids
        
        for i in range(possible_prefix_token, len(full_input_ids)):
            cur_prefix = self.tokenizer.decode(full_input_ids[:i])
            if cur_prefix == prefix:
                break
                
        return i
    
    def get_condition_ppl(
        self,
        text: str,
        question: str,
        condition_in_question: str = "none",
        granularity: str = "line",
    ):
        """
        Calculate perplexity change of a question when given context text.
        A positive change means the context helps reduce question perplexity.
        
        Args:
            text: The context text
            question: The question to evaluate
            condition_in_question: Conditioning mode (none, prefix)
            granularity: Granularity for perplexity calculation
            
        Returns:
            Perplexity change for the question with/without context
        """
        # Create a cache key for this conditional perplexity calculation
        cache_key = f"{text}_{question}_{condition_in_question}_{granularity}"
        
        if cache_key in self.cache["conditional_ppl"]:
            return self.cache["conditional_ppl"][cache_key]
        
        if condition_in_question == "none":
            # Just return the perplexity of the text
            result = self.get_ppl(
                text=text, granularity=granularity, condition_mode="none"
            )
            ppl_value = result["ppl"]
        else:
            # First calculate question perplexity without context
            question_ppl_without_context = self.get_ppl(
                text=question, 
                granularity=granularity
            )["ppl"]
            
            # Then calculate question perplexity with context
            question_ppl_with_context = self.get_ppl(
                text=text + "\n\n" + question, 
                granularity=granularity,
                condition_mode="prefix",
                condition_pos_id=self.get_token_length(text + "\n\n", add_special_tokens=True)
            )["ppl"]
            
            # Calculate the change (positive means context helps)
            ppl_value = question_ppl_without_context - question_ppl_with_context
        
        # Cache the result
        self.cache["conditional_ppl"][cache_key] = ppl_value
        self._manage_cache_size("conditional_ppl")
        
        return ppl_value
       
    def control_context_budget(
        self,
        context_list: List[str],
        target_token: float,
        question: str = "",
        reorder_context: str = "original",
        condition_in_question: str = "none",
        force_context_ids: List[int] = None,
        force_context_number: int = None,
        context_budget: str = "+100",
        dynamic_context_compression_ratio: float = 0.0,
    ):
        """
        Control token budget for contexts based on relevance ranking, following LongLLMLingua.
        
        Args:
            context_list: List of contexts
            target_token: Target number of tokens
            question: Question for relevance ranking
            reorder_context: How to reorder contexts ("original", "importance", "two_stage")
            condition_in_question: Mode for conditional ranking
            force_context_ids: List of context IDs to always include
            force_context_number: Number of contexts to forcibly include
            context_budget: String expression to modify target token budget
            dynamic_context_compression_ratio: Ratio for dynamic compression (0.0-1.0)
            
        Returns:
            Selected contexts, their indices, and dynamic ratios
        """
        logger.debug(f"Controlling context budget with target_token={target_token}")
        start_time = time.time()
        
        if not context_list:
            return [], [], []
        
        # Get token counts for each context
        logger.debug("Calculating token lengths for contexts")
        context_tokens_length = [self.get_token_length(context) for context in context_list]
        
        # If total tokens already fit within budget, return all contexts
        total_tokens = sum(context_tokens_length)
        if total_tokens <= target_token:
            logger.debug(f"All contexts fit within budget ({total_tokens} <= {target_token})")
            end_time = time.time()
            logger.debug(f"Context budget control completed in {end_time - start_time:.2f} seconds")
            return context_list, list(range(len(context_list))), [0.0] * len(context_list)
        
        # Rank contexts by relevance if question is provided
        logger.debug("Ranking contexts by relevance")
        if question:
            # Get perplexity change for each context with the question
            context_ppl_changes = []
            for d, dl in zip(context_list, context_tokens_length):
                # Calculate how much this context reduces question perplexity
                ppl_change = self.get_condition_ppl(
                    d,
                    question,
                    condition_in_question,
                )
                # Apply length adjustment factor similar to before
                context_ppl_changes.append(ppl_change - dl * 2 / 250 * 0)
            
            # Sort by perplexity change - higher is better (more reduction in question perplexity)
            demonstrations_sort = sorted(enumerate(context_ppl_changes), key=lambda x: -x[1])
        else:
            # Without question, use default ordering
            demonstrations_sort = [(i, 0) for i in range(len(context_list))]
        
        # Extract ranking for later use
        self.context_idxs.append([x for idx, (x, _) in enumerate(demonstrations_sort)])
        
        # Calculate the target token budget with context_budget expression
        if target_token < 0:
            target_token = 100
        target_token = eval("target_token" + context_budget)
        
        # Initialize selected context tracking
        used = force_context_ids if force_context_ids is not None else []
        
        # Select contexts until we reach the token budget
        for idx, _ in demonstrations_sort:
            if idx >= len(context_tokens_length):
                continue
            target_token -= context_tokens_length[idx]
            if idx not in used:
                used.append(idx)
            if target_token < 0 or (
                force_context_number is not None and len(used) >= force_context_number
            ):
                break
        
        # Store original selection order
        original_used = used.copy()
        
        # Reorder contexts if requested
        if reorder_context == "original":
            used = sorted(used)
        elif reorder_context == "two_stage":
            l, r = [_ for idx, _ in enumerate(used) if idx % 2 == 0], [
                _ for idx, _ in enumerate(used) if idx % 2 == 1
            ]
            used = l + r[::-1]
        
        # Calculate dynamic compression ratios if requested
        if dynamic_context_compression_ratio > 0:
            N = len(used)
            dynamic_ratio = [
                i * (abs(dynamic_context_compression_ratio) / (N - 1)) if N > 1 else 0
                for i in range(-(N - 1), N, 2)
            ][::-1]
            dynamic_ratio_map = {i: j for i, j in zip(original_used, dynamic_ratio)}
            dynamic_ratio = [dynamic_ratio_map[i] for i in used]
        else:
            dynamic_ratio = [0.0] * len(used)
        
        # Build list of selected contexts
        selected_contexts = [context_list[idx] for idx in used if idx < len(context_list)]
        
        end_time = time.time()
        logger.debug(f"Selected {len(selected_contexts)} contexts out of {len(context_list)}")
        logger.debug(f"Context budget control completed in {end_time - start_time:.2f} seconds")
        
        return selected_contexts, used, dynamic_ratio, demonstrations_sort
    
    def compress_code_file(
        self,
        code: str,
        query: str = "",
        instruction: str = "",
        rate: float = 0.5,
        target_token: float = -1,
        language: str = "python",
        use_iterative_compression: bool = True,
        iterative_size: int = 200,
        dynamic_compression_ratio: float = 0.2,
        context_budget: str = "+100",
        rank_only: bool = False,
        fine_ratio: float = None,
        fine_grained_importance_method: str = "conditional_ppl",
        min_lines_for_fine_grained: int = 5,
        importance_beta: float = 0.5,
        use_knapsack: bool = True,
    ):
        """
        Compress a code file by first splitting it into function-based chunks and then compressing.
        Functions are prioritized based on query relevance, similar to LongLLMLingua.
        
        Args:
            code: The code to compress
            query: Query to prioritize relevant functions
            instruction: Additional instruction to guide compression
            rate: Compression rate for coarse-grained (function level) compression (0.0-1.0)
            target_token: Target number of tokens (alternative to rate)
            language: Programming language of the code
            use_iterative_compression: Whether to use iterative compression
            iterative_size: Size of each iteration for iterative compression
            dynamic_compression_ratio: Ratio for dynamic compression
            context_budget: String expression to modify token budget
            rank_only: If True, just rank and select contexts without fine-grained compression
            fine_ratio: Ratio for fine-grained line selection (0.0-1.0). If None, uses `rate`.
            fine_grained_importance_method: Method for scoring line importance ('contrastive_perplexity' or 'conditional_ppl'). Defaults to 'conditional_ppl'.
            min_lines_for_fine_grained: Minimum number of lines a function must have to undergo fine-grained compression (otherwise kept fully).
            importance_beta: Controls how much function importance affects its individual compression rate during fine-grained compression.
            use_knapsack: Whether to use knapsack algorithm for block selection (True) or greedy line-by-line approach (False).
            
        Returns:
            Compressed code and statistics with the following structure:
            {
                "original_code": Original uncompressed code,
                "compressed_code": Compressed code,
                "compressed_prompt": Complete compressed prompt with instruction and query,
                "original_tokens": Number of tokens in original code,
                "compressed_tokens": Number of tokens in compressed code,
                "final_compressed_tokens": Number of tokens in final compressed prompt,
                "compression_ratio": Ratio of compressed to original tokens,
                "function_compressions": Details about compression for each function,
                "selected_functions": Indices of selected functions,
                "demonstrations_sort": Ranking of functions by importance,
                "compressed_chunks": List of compressed code chunks
                "fine_grained_method_used": The method used for fine-grained importance scoring.
            }
        """
        logger.debug(f"Starting code file compression with rate={rate}, target_token={target_token}, language={language}")
        start_time = time.time()
        
        # Split code into function-based chunks
        logger.debug("Splitting code into function-based chunks")
        code_chunks = self.split_code_by_functions(code, language=language)
        logger.debug(f"Split code into {len(code_chunks)} chunks")
        
        # Calculate total tokens
        logger.debug("Calculating total tokens")
        total_tokens = sum(self.get_token_length(chunk) for chunk in code_chunks)
        logger.debug(f"Total tokens: {total_tokens}")

        # Determine target_token based on rate if not specified
        original_target_token = target_token # Store original value if provided
        if target_token <= 0:
            if rate <= 0:
                 # Default target if both rate and target_token are invalid
                target_token = int(total_tokens * 0.5)
                logger.warning(f"Rate and target_token invalid, defaulting target_token to {target_token}")
            else:
                target_token = int(total_tokens * rate)
        logger.debug(f"Coarse Target tokens: {target_token}")
        
        # Use context budget control to select important functions
        logger.debug("Selecting important functions using context budget control")
        selected_contexts, selected_indices, dynamic_ratios, demonstrations_sort = self.control_context_budget(
            code_chunks,
            target_token=target_token,
            question=query,
            reorder_context="original",  # Keep original order to maintain code structure
            condition_in_question="prefix",
            context_budget=context_budget,
            dynamic_context_compression_ratio=dynamic_compression_ratio,
        )
        
        # If rank_only is True, just use the selected contexts without further compression
        logger.debug("Using rank-only mode: selecting top functions without fine-grained compression")
        compressed_chunks = []
        compressed_tokens = 0
        function_compressions = {}
        
        # Just keep the selected contexts as is
        for i, chunk in enumerate(code_chunks):
            if i in selected_indices:
                compressed_chunks.append(chunk)
                chunk_tokens = self.get_token_length(chunk)
                compressed_tokens += chunk_tokens
                
                # Store compression info - no actual compression in this mode
                function_compressions[i] = {
                    "original_tokens": chunk_tokens,
                    "compressed_tokens": chunk_tokens,
                    "compression_ratio": 1.0,
                }
            else:
                # Skip this function completely
                comment_marker = "#" if language.lower() in ["python", "typescript", "rust"] else "//"
                omission_text = f"{comment_marker} ... "
                compressed_chunks.append(omission_text)
                compressed_tokens += self.get_token_length(omission_text)
        
        # Combine compressed chunks
        compressed_code = "\n\n".join(compressed_chunks)

        # --- Post-join cleanup for consecutive omission markers ---
        logger.debug("Cleaning up consecutive omission markers after joining...")
        lines = compressed_code.split("\n")
        cleaned_lines = []
        last_non_empty_line_was_omission = False
        comment_marker = "#" if language.lower() in ["python", "typescript", "rust"] else "//"
        omission_marker_content = f"{comment_marker} ...".strip() # Content to check against

        for line in lines:
            stripped_line = line.strip()
            if not stripped_line:
                # Keep empty lines
                cleaned_lines.append(line)
                # Don't reset the flag here, wait for a non-empty line
            elif stripped_line == omission_marker_content:
                if last_non_empty_line_was_omission:
                    # Skip this consecutive omission marker line
                    logger.debug(f"Skipping line: '{line}' (consecutive omission)")
                    continue
                else:
                    # Keep the first omission marker line
                    cleaned_lines.append(line)
                    last_non_empty_line_was_omission = True
            else:
                # Regular code line
                cleaned_lines.append(line)
                last_non_empty_line_was_omission = False

        compressed_code = "\n".join(cleaned_lines)
        logger.debug("Cleanup finished.")
        # --- End post-join cleanup ---


        output = f"{instruction}\n\n{compressed_code}\n\n{query}\n{instruction}"
        
        # Calculate actual compressed tokens
        final_compressed_tokens = self.get_token_length(output)
        
        end_time = time.time()
        logger.debug(f"Code file compression completed in {end_time - start_time:.2f} seconds")
        logger.debug(f"Compression ratio: {compressed_tokens / total_tokens if total_tokens > 0 else 1.0:.2f}")
        
        if rank_only:
            return {
                "original_code": code,
                "compressed_code": compressed_code,
                "compressed_prompt": output,
                "original_tokens": total_tokens,
                "compressed_tokens": compressed_tokens,
                "final_compressed_tokens": final_compressed_tokens,
                "compression_ratio": compressed_tokens / total_tokens if total_tokens > 0 else 1.0,
                "function_compressions": function_compressions,
                "selected_functions": selected_indices,
                "demonstrations_sort": demonstrations_sort,
                "compressed_chunks": compressed_chunks,
                "fine_grained_method_used": None,
            }
        else:
            # enter fine-grained compression
            logger.debug(f"Starting fine-grained compression on selected functions using method: {fine_grained_importance_method}")

            # --- Dynamic Fine-grained Rate Allocation ---
            logger.debug("Calculating dynamic fine-grained compression rates...")

            # 1. Collect data for selected functions
            selected_functions_data = []
            importance_map = {idx: score for idx, score in demonstrations_sort} # Map index to score
            total_lines_selected = 0
            for i in selected_indices:
                if i < len(code_chunks):
                    chunk = code_chunks[i]
                    # Use simple line splitting for allocation efficiency
                    lines = chunk.split("\n")
                    line_count = len(lines)
                    score = importance_map.get(i, 0.0) # Default score 0 if not found
                    selected_functions_data.append({
                        "index": i,
                        "lines": lines,
                        "line_count": line_count,
                        "score": score
                    })
                    total_lines_selected += line_count
                else:
                     logger.warning(f"Selected index {i} is out of bounds for code_chunks (length {len(code_chunks)})")


            # 2. Calculate overall fine-grained target lines
            current_fine_ratio = fine_ratio if fine_ratio is not None else rate # Use rate if fine_ratio not set
            if original_target_token > 0: # If target_token was set explicitly, derive ratio from it for fine-grained stage
                 # Estimate target lines based on the ratio of selected tokens to total tokens, then apply fine_ratio
                 selected_tokens = sum(self.get_token_length(code_chunks[d['index']]) for d in selected_functions_data)
                 effective_coarse_rate = selected_tokens / total_tokens if total_tokens > 0 else 1.0
                 # Use the user-provided fine_ratio, or fall back to rate/coarse target estimate
                 fine_target_rate = current_fine_ratio
                 logger.debug(f"Using fine_ratio={fine_target_rate} for fine-grained target calculation.")
                 target_total_lines = int(total_lines_selected * fine_target_rate)

            else: # Calculate target based on fine_ratio/rate directly applied to selected lines
                 target_total_lines = int(total_lines_selected * current_fine_ratio)
                 logger.debug(f"Using current_fine_ratio={current_fine_ratio} for fine-grained target calculation.")

            logger.debug(f"Total lines in selected functions: {total_lines_selected}")
            logger.debug(f"Target total lines after fine-grained compression: {target_total_lines}")

            # 3. Separate small and large functions
            small_functions = []
            large_functions = []
            lines_in_small_functions = 0
            lines_in_large_functions = 0

            for data in selected_functions_data:
                if data["line_count"] < min_lines_for_fine_grained:
                    small_functions.append(data)
                    lines_in_small_functions += data["line_count"]
                else:
                    large_functions.append(data)
                    lines_in_large_functions += data["line_count"]

            logger.debug(f"Found {len(small_functions)} small functions (< {min_lines_for_fine_grained} lines) with {lines_in_small_functions} total lines (will be kept).")
            logger.debug(f"Found {len(large_functions)} large functions (>= {min_lines_for_fine_grained} lines) with {lines_in_large_functions} total lines.")

            # 4. Calculate target lines for large functions
            target_lines_for_large = max(0, target_total_lines - lines_in_small_functions)
            logger.debug(f"Target lines to keep from large functions: {target_lines_for_large}")

            # 5. Allocate rates for large functions
            function_fine_ratios = {} # Map: index -> individual_fine_ratio

            if not large_functions or lines_in_large_functions == 0:
                 logger.debug("No large functions to compress further or zero lines in large functions.")
                 global_rate_for_large = 1.0 if lines_in_large_functions > 0 else 0.0 # Should be 0 if lines_in_large_functions is 0
            elif target_lines_for_large <= 0:
                 logger.debug("Target lines for large functions is <= 0. Setting rates to 0.")
                 global_rate_for_large = 0.0
            elif target_lines_for_large >= lines_in_large_functions:
                 logger.debug("Target lines for large functions >= total lines. Setting rates to 1.0.")
                 global_rate_for_large = 1.0
            else:
                global_rate_for_large = target_lines_for_large / lines_in_large_functions
                logger.debug(f"Global target rate for large functions: {global_rate_for_large:.4f}")

                # Normalize scores for weighting (MinMax scaling)
                scores = [d["score"] for d in large_functions]
                valid_scores = [s for s in scores if not math.isinf(s) and not math.isnan(s)]

                if not valid_scores or max(valid_scores) == min(valid_scores):
                    logger.debug("Scores are uniform or invalid, using global rate for all large functions.")
                    for data in large_functions:
                        function_fine_ratios[data["index"]] = global_rate_for_large
                else:
                    min_score = min(valid_scores)
                    max_score = max(valid_scores)
                    normalized_scores = [(s - min_score) / (max_score - min_score) if not math.isinf(s) and not math.isnan(s) else 0.0 for s in scores] # Normalize to [0, 1], default 0 for invalid

                    # Calculate initial biased rates
                    initial_rates = []
                    for norm_score in normalized_scores:
                        # Bias rate: higher score -> higher rate (closer to 1)
                        # Beta controls sensitivity. beta=0 -> uniform rate. beta=1 -> max sensitivity.
                        biased_rate = global_rate_for_large * (1 + importance_beta * (norm_score - 0.5) * 2) # Scale norm_score diff to [-beta, beta]
                        clamped_rate = max(0.0, min(1.0, biased_rate)) # Clamp to [0, 1]
                        initial_rates.append(clamped_rate)

                    # Calculate actual lines kept with initial rates
                    actual_lines_kept = sum(initial_rates[i] * large_functions[i]["line_count"] for i in range(len(large_functions)))
                    logger.debug(f"Initial biased rates calculated. Estimated lines kept: {actual_lines_kept:.1f}")

                    # Adjust rates proportionally to meet target
                    if actual_lines_kept > 0 and abs(actual_lines_kept - target_lines_for_large) > 1: # Adjust if difference is significant
                        adjustment_factor = target_lines_for_large / actual_lines_kept
                        logger.debug(f"Adjusting rates by factor: {adjustment_factor:.4f}")
                        final_rates = [max(0.0, min(1.0, r * adjustment_factor)) for r in initial_rates] # Adjust and clamp again
                    else:
                        logger.debug("Initial rates are close enough or actual_lines_kept is 0, no adjustment needed.")
                        final_rates = initial_rates

                    for i, data in enumerate(large_functions):
                        function_fine_ratios[data["index"]] = final_rates[i]

            # Set rate 1.0 for small functions
            for data in small_functions:
                function_fine_ratios[data["index"]] = 1.0

            # --- End Dynamic Allocation ---


            # Apply fine-grained compression to each selected function
            fine_compressed_chunks = []
            compressed_tokens = 0
            function_compressions = {}

            # Define a smoothing window size for moving average
            smoothing_window = 5
            # fine_ratio = fine_ratio if fine_ratio is not None else rate # Use the same ratio by default if fine_ratio not specified # Removed, using individual ratios now

            # Process each chunk in the original order
            # Use tqdm.auto for compatibility
            fine_grained_pbar = tqdm(enumerate(code_chunks), total=len(code_chunks), desc="Fine-Grained Compression", leave=False)
            for i, chunk in fine_grained_pbar:
            # for i, chunk in enumerate(code_chunks):
                if i in selected_indices:
                    # This function was selected during coarse-grained compression
                    individual_fine_ratio = function_fine_ratios.get(i) # Get dynamically assigned ratio
                    if individual_fine_ratio is None:
                         logger.error(f"Missing fine-grained ratio for selected function index {i}. Skipping fine-grained compression for this chunk.")
                         individual_fine_ratio = 1.0 # Fallback to keep the chunk

                    # Use Entropy chunking for fine-grained compression instead of simple line splitting
                    chunks, sentences, ppls, spike_indices = self.entropy_chunking.chunk_text_adaptive(
                        code_chunks[i], method='std', k=0.2
                    )
                    # Use chunks as lines, but preserve all chunks including empty ones to maintain formatting
                    chunk_lines = chunks  # Keep all chunks to preserve \n\n and formatting
                    chunk_line_count = len([chunk for chunk in chunk_lines if chunk.strip()])  # Count only non-empty for logic
                    chunk_score = importance_map.get(i, float('nan')) # Get score

                    logger.debug(f"Processing Func {i}: Entropy Chunks={len(chunk_lines)}, Non-empty={chunk_line_count}, Score={chunk_score:.4f}, Assigned FineRatio={individual_fine_ratio:.4f}")


                    # Skip fine-grained compression if ratio is 1.0 (or close) or function is small
                    if individual_fine_ratio >= 0.999 or chunk_line_count < min_lines_for_fine_grained:
                        note = "Kept (Ratio=1.0)" if individual_fine_ratio >= 0.999 else f"Kept (Small Func < {min_lines_for_fine_grained} lines)"
                        logger.debug(f"  - {note}")
                        fine_compressed_chunks.append(chunk)
                        chunk_tokens = self.get_token_length(chunk)
                        compressed_tokens += chunk_tokens
                        function_compressions[i] = {
                            "original_tokens": chunk_tokens,
                            "compressed_tokens": chunk_tokens,
                            "compression_ratio": 1.0,
                            "individual_fine_ratio": individual_fine_ratio,
                            "note": note,
                            "importance_method": None # No line importance calculation needed
                        }
                        continue # Move to next chunk


                    # Apply fine-grained compression only if the function is large enough
                    # and we're not in rank-only mode (already checked) and ratio < 1.0
                    if chunk_line_count >= min_lines_for_fine_grained and individual_fine_ratio < 0.999:
                        logger.debug(f"  - Applying fine-grained compression with ratio {individual_fine_ratio:.4f}")
                        fine_grained_pbar.set_description(f"Fine-Grained Compressing Func {i}")
                        
                        # Calculate target tokens for this function
                        original_func_tokens = self.get_token_length(chunk)
                        target_func_tokens = int(original_func_tokens * individual_fine_ratio)
                        
                        # Calculate importance for each block based on the chosen method
                        block_importances = []
                        importance_calculation_start = time.time()

                        if fine_grained_importance_method == "conditional_ppl":
                            # Calculate conditional PPL importance for each block
                            if not query or not query.strip():
                                logger.warning(f"Query is empty for func {i}, cannot calculate conditional PPL. Assigning 0 importance.")
                                block_importances = [0.0] * len(chunk_lines)
                            else:
                                query_ppl_result = self.get_ppl(query, granularity="line")
                                query_ppl_without_context = query_ppl_result["ppl"]

                                if math.isinf(query_ppl_without_context):
                                    logger.warning(f"Base query PPL is infinite for func {i}. Assigning 0 importance to blocks.")
                                    block_importances = [0.0] * len(chunk_lines)
                                else:
                                    pbar_cond = tqdm(enumerate(chunk_lines), total=len(chunk_lines), desc=f"Func {i} Block CondPPL", leave=False)
                                    for block_idx, block in pbar_cond:
                                        if not block.strip():
                                            block_importances.append(-float('inf'))  # Low score for empty blocks
                                            continue

                                        conditional_text = block + "\n\n" + query
                                        prefix_len_text = block + "\n\n"
                                        prefix_len = self.get_token_length(prefix_len_text, add_special_tokens=True)

                                        cond_ppl_result = self.get_ppl(
                                            text=conditional_text,
                                            granularity="line",
                                            condition_mode="prefix",
                                            condition_pos_id=prefix_len - 1
                                        )
                                        ppl_with_context = cond_ppl_result["ppl"]

                                        if math.isinf(ppl_with_context):
                                            ppl_change = -float('inf')
                                        else:
                                            ppl_change = query_ppl_without_context - ppl_with_context

                                        block_importances.append(ppl_change)
                                        pbar_cond.set_description(f"Func {i} Block CondPPL (B{block_idx}: {ppl_change:.2f})")

                        elif fine_grained_importance_method == "contrastive_perplexity":
                            # Calculate contrastive PPL importance for each block
                            fine_grained_pbar.set_description(f"Fine-Grained ContrastivePPL Func {i}")
                            
                            with torch.no_grad():
                                pbar = tqdm(enumerate(chunk_lines), total=len(chunk_lines), desc="Block Contrastive PPL", leave=False)
                                for block_idx, block in pbar:
                                    if not block.strip():
                                        block_importances.append(-float('inf'))
                                        continue

                                    # Build context from previous blocks
                                    prev_context = "\n\n".join(chunk_lines[:block_idx]) if block_idx > 0 else ""
                                    
                                    # 1. PPL(Block | prev_blocks)
                                    regular_ppl_condition = prev_context + "\n\n" if prev_context else None
                                    regular_ppl = self._calculate_perplexity_for_contrastive(block, condition_text=regular_ppl_condition)

                                    # 2. PPL(Block | query, prev_blocks)
                                    question_context_parts = [query]
                                    if prev_context:
                                        question_context_parts.append(prev_context)
                                    question_context = "\n\n".join(filter(None, question_context_parts))
                                    cond_ppl_condition = question_context + "\n\n"
                                    cond_ppl = self._calculate_perplexity_for_contrastive(block, condition_text=cond_ppl_condition)

                                    # 3. Importance = PPL(Block|prev) - PPL(Block|Q,prev)
                                    if math.isinf(regular_ppl) or math.isinf(cond_ppl):
                                        importance = -float('inf')
                                    else:
                                        importance = regular_ppl - cond_ppl

                                    block_importances.append(importance)
                                    pbar.set_description(f"Block {block_idx}: {importance:.2f}")

                        else:
                            raise ValueError(f"Unsupported fine_grained_importance_method: {fine_grained_importance_method}")

                        importance_calculation_end = time.time()
                        logger.debug(f"  - Block importance calculation took {importance_calculation_end - importance_calculation_start:.2f}s")

                        # Identify preserved blocks (function signature, comments, returns)
                        preserved_block_indices = set()
                        comment_marker = "#" if language.lower() in ["python", "typescript", "rust"] else "//"
                        
                        # Find blocks containing function signature
                        for block_idx, block in enumerate(chunk_lines):
                            block_lines = block.split('\n')
                            for line in block_lines:
                                if line.strip():
                                    # Check for function/class definitions
                                    if any(keyword in line for keyword in ['def ', 'class ', 'function ', 'fn ', 'func ']):
                                        preserved_block_indices.add(block_idx)
                                        break
                                    # Check for function-level comments
                                    if line.strip().startswith(comment_marker):
                                        preserved_block_indices.add(block_idx)
                                        break
                                    # Check for return statements
                                    if 'return ' in line:
                                        preserved_block_indices.add(block_idx)
                                        break
                                    break  # Only check first non-empty line of each block

                        # Choose selection method based on use_knapsack parameter
                        processing_start = time.time()
                        
                        if use_knapsack:
                            # Use knapsack algorithm to select blocks
                            logger.debug(f"  - Using knapsack algorithm for block selection")
                            selected_block_indices, selection_info = self._knapsack_block_selection(
                                blocks=chunk_lines,
                                block_importances=block_importances,
                                target_tokens=target_func_tokens,
                                preserved_block_indices=preserved_block_indices,
                                language=language
                            )
                            
                            # Build compressed chunk from selected blocks
                            compressed_blocks = []
                            
                            # Determine base indentation for omission markers
                            base_indentation = ""
                            for block in chunk_lines:
                                for line in block.split('\n'):
                                    if line.strip():
                                        match = re.match(r"^(\s*)", line)
                                        if match:
                                            base_indentation = match.group(1)
                                        break
                                if base_indentation:
                                    break
                            
                            omission_marker = f"{base_indentation}{comment_marker} ... "
                            
                            # Build output with omission markers for gaps
                            last_selected_idx = -1
                            for block_idx in sorted(selected_block_indices):
                                # Add omission marker if there's a gap
                                if last_selected_idx != -1 and block_idx > last_selected_idx + 1:
                                    if not compressed_blocks or compressed_blocks[-1] != omission_marker:
                                        compressed_blocks.append(omission_marker)
                                
                                compressed_blocks.append(chunk_lines[block_idx])
                                last_selected_idx = block_idx

                            # Handle trailing omission if needed
                            if last_selected_idx != -1 and last_selected_idx < len(chunk_lines) - 1:
                                if not compressed_blocks or compressed_blocks[-1] != omission_marker:
                                    compressed_blocks.append(omission_marker)

                            # Join blocks with double newlines to preserve Entropy chunk structure
                            compressed_chunk = "\n\n".join(compressed_blocks)
                            
                        else:
                            # Use original greedy line-by-line approach with smoothing
                            logger.debug(f"  - Using original greedy line-by-line approach")
                            
                            # Convert block importances to line importances for compatibility
                            lines = []
                            line_importances = []
                            line_indices = []
                            
                            for block_idx, (block, block_importance) in enumerate(zip(chunk_lines, block_importances)):
                                block_lines = block.split('\n')
                                for line_idx_in_block, line in enumerate(block_lines):
                                    global_line_idx = len(lines)
                                    lines.append(line)
                                    line_importances.append(block_importance)  # Use block importance for all lines in block
                                    line_indices.append(global_line_idx)
                            
                            # Apply original processing logic with smoothing
                            full_line_scores = [float('nan')] * len(lines)
                            for score_idx, original_line_idx in enumerate(line_indices):
                                if score_idx < len(line_importances):
                                    full_line_scores[original_line_idx] = line_importances[score_idx]

                            # Replace NaN/Inf with min valid score for consistent processing
                            valid_scores = [s for s in full_line_scores if not math.isnan(s) and not math.isinf(s)]
                            if valid_scores:
                                min_valid_score = min(valid_scores)
                                if min_valid_score == float('inf') or min_valid_score == -float('inf') or math.isnan(min_valid_score):
                                    min_replacement_score = 0.0
                                else:
                                    min_replacement_score = min_valid_score

                                processed_line_scores = []
                                for s in full_line_scores:
                                    if math.isnan(s) or s == -float('inf'):
                                        processed_line_scores.append(min_replacement_score)
                                    elif s == float('inf'):
                                        processed_line_scores.append(min_replacement_score)
                                    else:
                                        processed_line_scores.append(s)
                            else:
                                processed_line_scores = [0.0] * len(lines)

                            # Apply smoothing using moving average
                            smoothing_window = 5
                            smoothed_importances = processed_line_scores.copy()
                            num_processed_scores = len(processed_line_scores)
                            for j in range(num_processed_scores):
                                window_start = max(0, j - smoothing_window // 2)
                                window_end = min(num_processed_scores, j + smoothing_window // 2 + 1)
                                window = processed_line_scores[window_start:window_end]
                                valid_window_scores = [s for s in window if not math.isnan(s) and not math.isinf(s)]
                                if valid_window_scores:
                                    smoothed_importances[j] = sum(valid_window_scores) / len(valid_window_scores)

                            # Find preserved lines (convert block indices to line indices)
                            preserved_line_indices = set()
                            line_offset = 0
                            for block_idx, block in enumerate(chunk_lines):
                                block_lines = block.split('\n')
                                if block_idx in preserved_block_indices:
                                    for line_idx_in_block in range(len(block_lines)):
                                        preserved_line_indices.add(line_offset + line_idx_in_block)
                                line_offset += len(block_lines)

                            # Sort remaining lines by importance
                            sortable_lines = []
                            for idx in range(len(lines)):
                                if idx not in preserved_line_indices:
                                    if idx < len(line_indices) and idx < len(line_importances):
                                        original_score = line_importances[idx]
                                        if not math.isnan(original_score) and not math.isinf(original_score):
                                            smoothed_score = smoothed_importances[idx]
                                            sortable_lines.append((idx, smoothed_score))

                            # Sort descending by score
                            sorted_line_indices = sorted(sortable_lines, key=lambda x: -x[1])

                            # Calculate target number of lines to keep
                            total_lines = len(lines)
                            preserved_count = len(preserved_line_indices)
                            target_lines = max(preserved_count, int(total_lines * individual_fine_ratio))

                            # Select top lines by importance up to target
                            selected_lines = set(preserved_line_indices)
                            for idx, score in sorted_line_indices:
                                if len(selected_lines) >= target_lines:
                                    break
                                selected_lines.add(idx)

                            # Build compressed chunk from selected lines
                            compressed_chunks = []
                            base_indentation = ""
                            if lines:
                                for line in lines:
                                    if line.strip():
                                        match = re.match(r"^(\s*)", line)
                                        if match:
                                            base_indentation = match.group(1)
                                        break

                            omission_marker_line = f"{base_indentation}{comment_marker} ... "
                            
                            last_added_line_idx = -1
                            for j in range(len(lines)):
                                if j in selected_lines:
                                    if last_added_line_idx != -1 and j > last_added_line_idx + 1:
                                        if not compressed_chunks or compressed_chunks[-1] != omission_marker_line:
                                            compressed_chunks.append(omission_marker_line)
                                    compressed_chunks.append(lines[j])
                                    last_added_line_idx = j

                            if last_added_line_idx != -1 and last_added_line_idx < len(lines) - 1:
                                if not compressed_chunks or compressed_chunks[-1] != omission_marker_line:
                                    compressed_chunks.append(omission_marker_line)

                            compressed_chunk = "\n".join(compressed_chunks)
                            
                            # Create selection info for compatibility
                            selection_info = {
                                "method": "greedy_line_by_line",
                                "preserved_lines": len(preserved_line_indices),
                                "selected_lines": len(selected_lines),
                                "total_lines": len(lines),
                                "smoothing_applied": True
                            }
                            selected_block_indices = preserved_block_indices  # For compatibility

                        processing_end = time.time()
                        method_name = "knapsack" if use_knapsack else "greedy"
                        logger.debug(f"  - {method_name} selection took {processing_end - processing_start:.2f}s")
                        
                        if use_knapsack:
                            logger.debug(f"  - Selected {len(selected_block_indices)}/{len(chunk_lines)} blocks")
                        else:
                            logger.debug(f"  - Selected {len(selected_lines)}/{len(lines)} lines")

                        # Update token count and store compression info
                        fine_compressed_chunks.append(compressed_chunk)
                        compressed_chunk_tokens = self.get_token_length(compressed_chunk)
                        compressed_tokens += compressed_chunk_tokens

                        # Store compression info
                        actual_compression_ratio = compressed_chunk_tokens / original_func_tokens if original_func_tokens > 0 else 1.0
                        function_compressions[i] = {
                            "original_tokens": original_func_tokens,
                            "compressed_tokens": compressed_chunk_tokens,
                            "compression_ratio": actual_compression_ratio,
                            "individual_fine_ratio": individual_fine_ratio,
                            "preserved_blocks": list(preserved_block_indices),
                            "selected_blocks": list(selected_block_indices),
                            "selection_info": selection_info,
                            "importance_method": fine_grained_importance_method,
                            "selection_method": "knapsack" if use_knapsack else "greedy_line_by_line"
                        }
                        logger.debug(f"  - Compressed func {i}: {original_func_tokens} -> {compressed_chunk_tokens} tokens (Ratio: {actual_compression_ratio:.3f})")
                    else:
                         # This case should now be handled by the check at the beginning of the loop
                         logger.warning(f"Reached unexpected state for func {i}. Keeping chunk as is.")
                         fine_compressed_chunks.append(chunk)
                         chunk_tokens = self.get_token_length(chunk)
                         compressed_tokens += chunk_tokens
                         function_compressions[i] = {
                            "original_tokens": chunk_tokens,
                            "compressed_tokens": chunk_tokens,
                            "compression_ratio": 1.0,
                            "individual_fine_ratio": individual_fine_ratio,
                            "note": "Unexpected state, kept function.",
                            "importance_method": None
                         }

                else:
                    # This function was not selected during coarse-grained compression
                    # Add a placeholder
                    comment_marker = "#" if language.lower() in ["python", "typescript", "rust"] else "//"
                    omission_text = f"{comment_marker} ... "
                    fine_compressed_chunks.append(omission_text)
                    compressed_tokens += self.get_token_length(omission_text)
                    # Log skipped chunk
                    # logger.debug(f"Skipped Func {i} (not selected in coarse stage)")


            # Combine fine-grained compressed chunks
            compressed_code = "\n\n".join(fine_compressed_chunks)

            # --- Post-join cleanup for consecutive omission markers ---
            logger.debug("Cleaning up consecutive omission markers after joining...")
            lines = compressed_code.split("\n")
            cleaned_lines = []
            last_non_empty_line_was_omission = False
            comment_marker = "#" if language.lower() in ["python", "typescript", "rust"] else "//"
            omission_marker_content = f"{comment_marker} ...".strip() # Content to check against

            for line in lines:
                stripped_line = line.strip()
                if not stripped_line:
                    # Keep empty lines
                    cleaned_lines.append(line)
                    # Don't reset the flag here, wait for a non-empty line
                elif stripped_line == omission_marker_content:
                    if last_non_empty_line_was_omission:
                        # Skip this consecutive omission marker line
                        logger.debug(f"Skipping line: '{line}' (consecutive omission)")
                        continue
                    else:
                        # Keep the first omission marker line
                        cleaned_lines.append(line)
                        last_non_empty_line_was_omission = True
                else:
                    # Regular code line
                    cleaned_lines.append(line)
                    last_non_empty_line_was_omission = False

            compressed_code = "\n".join(cleaned_lines)
            logger.debug("Cleanup finished.")
            # --- End post-join cleanup ---


            # Ensure instruction/query parts are handled correctly, maybe use a template
            prompt_parts = []
            if instruction and instruction.strip():
                prompt_parts.append(instruction.strip())
            if compressed_code.strip():
                prompt_parts.append(compressed_code) # Already has newlines handled
            if query and query.strip():
                 # Add query, potentially repeating instruction based on original logic
                 prompt_parts.append(query.strip())
                 # Decide if instruction should be repeated after query based on original implementation's needs
                 # if instruction and instruction.strip(): # Repeat instruction if needed
                 #     prompt_parts.append(instruction.strip())

            output = "\n\n".join(prompt_parts) # Use double newline separation

            # Calculate final compressed tokens
            final_compressed_tokens = self.get_token_length(output)

            end_time = time.time()
            logger.debug(f"Fine-grained compression processing completed in {end_time - start_time:.2f} seconds")
            final_compression_ratio = compressed_tokens / total_tokens if total_tokens > 0 else 1.0
            logger.debug(f"Final Compression ratio (fine-grained tokens / total original tokens): {final_compression_ratio:.4f}")


            return {
                "original_code": code,
                "compressed_code": compressed_code,
                "compressed_prompt": output,
                "original_tokens": total_tokens,
                "compressed_tokens": compressed_tokens,
                "final_compressed_tokens": final_compressed_tokens,
                "compression_ratio": final_compression_ratio,
                "function_compressions": function_compressions,
                "selected_functions": selected_indices,
                "demonstrations_sort": demonstrations_sort,
                "compressed_chunks": fine_compressed_chunks,
                "fine_grained_method_used": fine_grained_importance_method,
            }
    
    def split_code_by_functions(self, code: str, language: str = "python", custom_separator: str = "# --CHUNK_SEPARATOR-- #") -> List[str]:
        """
        Split code into chunks based on function and class definitions for various languages.
        Also splits on custom separator if provided.
        
        Args:
            code: The code to split
            language: Programming language of the code (python, cpp, java, typescript, rust, go)
            custom_separator: Optional custom separator string to also split on
            
        Returns:
            List of code chunks, each containing a function, class, or class method
        """
        logger.debug(f"Splitting code by functions and classes for language: {language}")
        start_time = time.time()
        
        # Define regex patterns for different languages
        patterns = {
            # Python: Simplified to match 'def' or 'class' followed by content until the next def/class or end
            "python": r'(^|\n)(\s*)(def|class)\s+[^\n]+(\n(?!\s*(?:def|class)\s)[^\n]*)*',
            # C++: Improved to better handle multi-line declarations
            "cpp": r'(^|\n)(\s*)(?:class\s+[a-zA-Z_][a-zA-Z0-9_]*(?:\s*:\s*[^{]*)?|(?:[a-zA-Z_][a-zA-Z0-9_<>:,\s]*\s+)?[a-zA-Z_][a-zA-Z0-9_]*\s*\([^{;]*\)(?:\s*[^{;]*)?)\s*(?:{[^}]*}|[^;]*;)?',
            # Java: Improved for multi-line method declarations
            "java": r'(^|\n)(\s*)(?:(?:public|private|protected|static|final|abstract|synchronized)\s+)*(?:class\s+[a-zA-Z_][a-zA-Z0-9_]*(?:\s+extends\s+[a-zA-Z_][a-zA-Z0-9_]*)?(?:\s+implements\s+[^{]*)?|(?:<.*>)?(?:[a-zA-Z_][a-zA-Z0-9_<>:,\s]*)\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\([^{;]*\)(?:\s*throws\s+[^{;]*)?)\s*(?:{[^}]*}|[^;]*;)?',
            # TypeScript: Enhanced to handle multi-line methods and arrow functions
            "typescript": r'(^|\n)(\s*)(?:(?:public|private|protected|static|abstract)\s+)*(?:class\s+[a-zA-Z_][a-zA-Z0-9_]*(?:\s+extends\s+[a-zA-Z_][a-zA-Z0-9_]*)?(?:\s+implements\s+[^{]*)?|(?:(?:public|private|protected|static|async)\s+)*(?:function\s+)?(?:[a-zA-Z_][a-zA-Z0-9_]*)\s*(?:<.*>)?\s*\([^{;]*\)\s*(?::\s*[^{;]*\s*)?(?:=>)?)\s*(?:{[^}]*}|[^;]*;)?',
            # Rust: Improved for multi-line function declarations
            "rust": r'(^|\n)(\s*)(?:pub\s+)?(?:struct\s+[a-zA-Z_][a-zA-Z0-9_]*|impl(?:\s+[a-zA-Z_][a-zA-Z0-9_]*)?(?:\s+for\s+[a-zA-Z_][a-zA-Z0-9_]*)?|(?:async\s+)?fn\s+[a-zA-Z_][a-zA-Z0-9_]*\s*(?:<.*>)?\s*\([^{;]*\)(?:\s*->\s*[^{;]*\s*)?)\s*(?:{[^}]*}|[^;]*;)?',
            # Go: Improved for multi-line function declarations
            "go": r'(^|\n)(\s*)(?:type\s+[a-zA-Z_][a-zA-Z0-9_]*\s+struct|func\s+(?:\([^)]*\)\s*)?[a-zA-Z_][a-zA-Z0-9_]*\s*\([^{;]*\)(?:\s*[^{;]*\s*)?)\s*(?:{[^}]*}|[^;]*;)?',
        }
        
        # Use default Python pattern if language not supported
        if language.lower() not in patterns:
            language = "python"
        
        # First check if we need to split by custom separator
        separator_chunks = []
        if custom_separator and custom_separator in code:
            logger.debug(f"Custom separator '{custom_separator}' found, first splitting by separator")
            separator_chunks = [chunk for chunk in code.split(custom_separator) if chunk.strip()]
        else:
            separator_chunks = [code]  # Just one chunk - the entire code

        # Function to split a single chunk by functions/classes
        def split_chunk_by_pattern(chunk_code):
            function_pattern = re.compile(patterns[language.lower()], re.MULTILINE)
            matches = list(function_pattern.finditer(chunk_code))
            
            if not matches:
                return [chunk_code]  # No matches, return whole chunk
                
            result_chunks = []
            
            # Add code before first match
            if matches[0].start() > 0:
                result_chunks.append(chunk_code[:matches[0].start()])
            
            # Process each match
            for i, match in enumerate(matches):
                start = match.start()
                
                # End is either start of next match or end of code
                if i < len(matches) - 1:
                    end = matches[i + 1].start()
                else:
                    end = len(chunk_code)
                
                result_chunks.append(chunk_code[start:end])
            
            return result_chunks
        
        # Now apply function/class splitting to each separator chunk
        final_chunks = []
        for chunk in separator_chunks:
            function_chunks = split_chunk_by_pattern(chunk)
            final_chunks.extend(function_chunks)
        
        end_time = time.time()
        logger.debug(f"Code splitting completed in {end_time - start_time:.2f} seconds")
        logger.debug(f"Split code into {len(final_chunks)} chunks (using both separator and patterns)")
        
        return final_chunks

    def _calculate_perplexity_for_contrastive(self, text, condition_text=None):
        """Helper to calculate perplexity of text, optionally conditioned on condition_text"""
        if condition_text:
            full_text = condition_text + text
            inputs = self.tokenizer(full_text, return_tensors="pt", add_special_tokens=True).to(self.device) # Use add_special_tokens=True for consistency
            
            condition_input_ids = self.tokenizer(condition_text, return_tensors="pt", add_special_tokens=True).input_ids
            condition_length = condition_input_ids.size(1)

            # Handle potential edge case where condition length might exceed max length or input length
            if condition_length >= inputs.input_ids.size(1):
                 logger.warning(f"Condition length ({condition_length}) >= input length ({inputs.input_ids.size(1)}). Cannot calculate conditional PPL.")
                 return float('inf')

            with torch.no_grad():
                outputs = self.model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask) # Pass attention_mask

            # Logits for the 'text' part, labels are the 'text' part shifted
            logits = outputs.logits[0, condition_length-1:-1]
            labels = inputs.input_ids[0, condition_length:]

            if logits.size(0) == 0 or labels.size(0) == 0 or logits.size(0) != labels.size(0):
                logger.warning(f"Logits/Labels shape mismatch or empty in _calculate_perplexity_for_contrastive (cond). Logits: {logits.shape}, Labels: {labels.shape}. Returning inf.")
                return float('inf') # Return inf if shapes mismatch or empty

            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            mean_loss = loss.mean().item()
            perplexity = math.exp(mean_loss) if not math.isnan(mean_loss) and not math.isinf(mean_loss) else float('inf')

        else:
            # Calculate unconditional perplexity
            inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=True).to(self.device) # Use add_special_tokens=True
            with torch.no_grad():
                outputs = self.model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask) # Pass attention_mask

            # Logits for all tokens except last, labels are all tokens except first
            logits = outputs.logits[0, :-1]
            labels = inputs.input_ids[0, 1:]

            if logits.size(0) == 0 or labels.size(0) == 0 or logits.size(0) != labels.size(0):
                logger.warning(f"Logits/Labels shape mismatch or empty in _calculate_perplexity_for_contrastive (uncond). Logits: {logits.shape}, Labels: {labels.shape}. Returning inf.")
                return float('inf') # Return inf if shapes mismatch or empty

            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            mean_loss = loss.mean().item()
            perplexity = math.exp(mean_loss) if not math.isnan(mean_loss) and not math.isinf(mean_loss) else float('inf')

        return perplexity

    def _calculate_contrastive_perplexity(self, code_lines: List[str], question: str):
        """
        Calculate contrastive perplexity-based importance for each line of code.
        s_i = perplexity(x_i | x_{<i}) - perplexity(x_i | x^{que}, x_{<i})
        Higher score means the question helps predict the line more.

        Args:
            code_lines: List of code lines to analyze
            question: The query/question text

        Returns:
            Tuple of (line_scores, scored_indices)
        """
        logger.debug("Calculating contrastive perplexity-based line importance...")
        line_scores = []
        scored_indices = []

        with torch.no_grad():
            # Use tqdm.auto for better compatibility
            pbar = tqdm(enumerate(code_lines), total=len(code_lines), desc="Contrastive PPL", leave=False)
            for i, line in pbar:
                if not line.strip():
                    continue  # Skip empty lines

                # Ensure line has content before proceeding
                if not line:
                    logger.debug(f"Skipping empty line {i}")
                    continue

                # 1. PPL(L_i | L_<i)
                prev_context = "\n".join(code_lines[:i])
                # Add newline only if previous context exists
                regular_ppl_condition = prev_context + "\n" if prev_context else None
                regular_ppl = self._calculate_perplexity_for_contrastive(line, condition_text=regular_ppl_condition)


                # 2. PPL(L_i | Q, L_<i)
                # Combine question and previous context carefully
                question_context_parts = [question]
                if prev_context:
                    question_context_parts.append(prev_context)
                # Join with double newline between Q and prev_context if both exist
                question_context = "\n\n".join(filter(None, question_context_parts))
                # Add trailing newline before the target line
                cond_ppl_condition = question_context + "\n"
                cond_ppl = self._calculate_perplexity_for_contrastive(line, condition_text=cond_ppl_condition)

                # 3. Importance = PPL(L|prev) - PPL(L|Q,prev)
                if math.isinf(regular_ppl) or math.isinf(cond_ppl):
                    # If either is infinite, the difference isn't well-defined for ranking.
                    # Assign a very low score, potentially based on which one is inf.
                    # If regular_ppl is inf, question might still help (cond_ppl could be finite).
                    # If cond_ppl is inf, question made it worse or impossible to predict.
                    # Let's assign -inf for simplicity, meaning "least important".
                    importance = -float('inf')
                    logger.debug(f"Line {i}: Inf PPL detected. Regular: {regular_ppl}, Conditional: {cond_ppl}. Importance set to -inf")
                else:
                    importance = regular_ppl - cond_ppl
                    logger.debug(f"Line {i}: PPL(L|prev)={regular_ppl:.4f}, PPL(L|Q,prev)={cond_ppl:.4f}, Importance={importance:.4f}")

                line_scores.append(importance)
                scored_indices.append(i)
                # Update tqdm description if needed, e.g., with last score
                # pbar.set_description(f"Contrastive PPL (L{i}: {importance:.2f})")

        logger.debug(f"Finished calculating contrastive PPL for {len(line_scores)} lines.")
        return line_scores, scored_indices

    def _knapsack_block_selection(
        self,
        blocks: List[str],
        block_importances: List[float],
        target_tokens: int,
        preserved_block_indices: set = None,
        language: str = "python"
    ) -> Tuple[set, Dict]:
        """
        Use knapsack algorithm to select blocks that maximize total importance within token budget.
        
        Args:
            blocks: List of code blocks (Entropy chunks)
            block_importances: Importance scores for each block
            target_tokens: Target number of tokens to keep
            preserved_block_indices: Set of block indices that must be preserved
            language: Programming language for omission markers
            
        Returns:
            Tuple of (selected_block_indices, selection_info)
        """
        logger.debug(f"Running knapsack block selection with target_tokens={target_tokens}")
        
        if not blocks:
            return set(), {}
        
        # Calculate token weights for each block
        block_weights = [self.get_token_length(block) for block in blocks]
        
        # Handle preserved blocks
        if preserved_block_indices is None:
            preserved_block_indices = set()
        
        # Calculate tokens already used by preserved blocks
        preserved_tokens = sum(block_weights[i] for i in preserved_block_indices)
        remaining_budget = max(0, target_tokens - preserved_tokens)
        
        logger.debug(f"Preserved blocks: {len(preserved_block_indices)}, tokens: {preserved_tokens}")
        logger.debug(f"Remaining budget for knapsack: {remaining_budget}")
        
        # If no remaining budget, just return preserved blocks
        if remaining_budget <= 0:
            return preserved_block_indices, {
                "method": "knapsack",
                "preserved_only": True,
                "total_value": sum(block_importances[i] for i in preserved_block_indices),
                "total_weight": preserved_tokens
            }
        
        # Prepare items for knapsack (excluding preserved blocks)
        knapsack_items = []
        for i, (weight, value) in enumerate(zip(block_weights, block_importances)):
            if i not in preserved_block_indices:
                # Handle invalid importance scores
                if math.isnan(value) or math.isinf(value):
                    value = 0.0
                knapsack_items.append((i, weight, value))
        
        # Sort by value-to-weight ratio for efficiency (greedy approximation first)
        knapsack_items.sort(key=lambda x: x[2] / max(x[1], 1), reverse=True)
        
        # Use dynamic programming for exact knapsack solution
        # For efficiency, limit to reasonable problem size
        if len(knapsack_items) <= 100 and remaining_budget <= 2000:
            selected_indices = self._solve_knapsack_dp(knapsack_items, remaining_budget)
        else:
            # Use greedy approximation for large problems
            logger.debug("Using greedy approximation for large knapsack problem")
            selected_indices = self._solve_knapsack_greedy(knapsack_items, remaining_budget)
        
        # Combine with preserved blocks
        final_selection = preserved_block_indices.union(selected_indices)
        
        # Calculate selection statistics
        total_value = sum(block_importances[i] for i in final_selection)
        total_weight = sum(block_weights[i] for i in final_selection)
        
        selection_info = {
            "method": "knapsack",
            "preserved_blocks": len(preserved_block_indices),
            "selected_blocks": len(selected_indices),
            "total_blocks": len(final_selection),
            "total_value": total_value,
            "total_weight": total_weight,
            "target_weight": target_tokens,
            "efficiency": total_value / max(total_weight, 1)
        }
        
        logger.debug(f"Knapsack selection: {len(final_selection)}/{len(blocks)} blocks, "
                    f"value={total_value:.2f}, weight={total_weight}/{target_tokens}")
        
        return final_selection, selection_info
    
    def _solve_knapsack_dp(self, items: List[Tuple[int, int, float]], capacity: int) -> set:
        """
        Solve knapsack problem using dynamic programming.
        
        Args:
            items: List of (index, weight, value) tuples
            capacity: Maximum weight capacity
            
        Returns:
            Set of selected item indices
        """
        n = len(items)
        if n == 0 or capacity <= 0:
            return set()
        
        # DP table: dp[i][w] = maximum value using first i items with weight limit w
        dp = [[0.0 for _ in range(capacity + 1)] for _ in range(n + 1)]
        
        # Fill DP table
        for i in range(1, n + 1):
            idx, weight, value = items[i - 1]
            for w in range(capacity + 1):
                # Don't take item i
                dp[i][w] = dp[i - 1][w]
                
                # Take item i if it fits
                if weight <= w:
                    dp[i][w] = max(dp[i][w], dp[i - 1][w - weight] + value)
        
        # Backtrack to find selected items
        selected = set()
        w = capacity
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i - 1][w]:
                idx, weight, value = items[i - 1]
                selected.add(idx)
                w -= weight
        
        return selected
    
    def _solve_knapsack_greedy(self, items: List[Tuple[int, int, float]], capacity: int) -> set:
        """
        Solve knapsack problem using greedy approximation (by value/weight ratio).
        
        Args:
            items: List of (index, weight, value) tuples (should be pre-sorted by ratio)
            capacity: Maximum weight capacity
            
        Returns:
            Set of selected item indices
        """
        selected = set()
        current_weight = 0
        
        for idx, weight, value in items:
            if current_weight + weight <= capacity:
                selected.add(idx)
                current_weight += weight
        
        return selected

if __name__ == "__main__":
    # Load real examples from the dataset
    # with open("exp-cur50lines-bg5000tokens/results/deepseek-coder-6.7b-instruct/method_code_compressor_t2048_rankonly/deepseek-ai_slash_deepseek-coder-6.7b-instruct.jsonl", "r") as f:
    with open("exp-cur50lines-bg5000tokens-500examples/results/mistral-7b-instruct/method_code_compressor_t512_rankonly/mistralai_slash_Mistral-7B-Instruct-v0.3.jsonl", "r") as f:
        data = [json.loads(line) for line in f]
    
    example = data[190]
    # print(example.keys()) # dict_keys(['id', 'gt', 'original_background_context', 'original_current_function_context', 'language', 'prompt', 'output', 'es', 'em'])

    context = example["original_background_context"]
    question = example["original_current_function_context"]
    ground_truth = example["gt"]

    # Initialize compressor
    logger.info("Initializing compressor...")
    model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
    compressor = CodeCompressor(model_name=model_name)
    
    # Test function-based code file compression with query
    logger.info("\nTesting function-based code file compression with query...")

    original_tokens = len(compressor.tokenizer.encode(context))
    target_token = 512
    target_ratio = min(1.0, max(0.0, target_token / original_tokens))
    logger.info(f"CodeCompressor: Original tokens={original_tokens}, Target tokens={target_token}, Calculated ratio={target_ratio:.4f}")

    result = compressor.compress_code_file(
        code=context,
        query=question, # Using current function context as query focus
        instruction="Complete the following code function given the context.",
        rate=target_ratio,
        rank_only=False, # Test fine-grained compression
        fine_grained_importance_method="contrastive_perplexity", # Explicitly test default
        min_lines_for_fine_grained=5, # New parameter
        importance_beta=0.5, # Sensitivity to importance score
        use_knapsack=True,
    )

    # show the compressed code
    logger.info(f"Compressed code (using {result['fine_grained_method_used']}): \n{result['compressed_code']}")
    logger.info(f"Current function context: \n{question}")
    # final prompt
    final_prompt = result['compressed_prompt']
    # get the completion
    try:
        tokenized_prompt = compressor.tokenizer(final_prompt, return_tensors="pt").to(compressor.device)
        # Increase max_new_tokens for potentially longer completions
        completion_ids = compressor.model.generate(**tokenized_prompt, max_new_tokens=128, pad_token_id=compressor.tokenizer.eos_token_id)
        # Decode only the generated part, skipping special tokens
        completion = compressor.tokenizer.decode(completion_ids[0][len(tokenized_prompt.input_ids[0]):], skip_special_tokens=True)

        # Basic cleanup: remove leading/trailing whitespace and potentially stop words if needed
        completion = completion.strip()
        # More robust cleanup: Find the first meaningful line if generation includes noise
        completion_lines = [line for line in completion.split("\n") if line.strip() and not line.strip().startswith(("#", "//"))] # Simple comment removal
        cleaned_completion = completion_lines[0] if completion_lines else completion # Take first non-comment line or original if none found

    except Exception as e:
        logger.error(f"Error during generation or decoding: {e}")
        cleaned_completion = "[ERROR DURING GENERATION]"

    logger.info(f"Cleaned Completion: {cleaned_completion}")
    logger.info(f"Ground truth: {ground_truth}")

    # Optional: Test with conditional_ppl method
    logger.info("\nTesting fine-grained compression with conditional_ppl...")
    result_cond = compressor.compress_code_file(
        code=context,
        query=question,
        instruction="Complete the following code function given the context.",
        rate=target_ratio,
        rank_only=False,
        fine_grained_importance_method="conditional_ppl",
        min_lines_for_fine_grained=5,
        importance_beta=0.5
    )
    logger.info(f"Compressed code (using {result_cond['fine_grained_method_used']}): \n{result_cond['compressed_code']}")