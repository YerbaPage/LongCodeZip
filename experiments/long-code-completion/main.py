import os
import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
from llmlingua import PromptCompressor
import fire
from utils import load_data, compute_EM, compute_ES
from vllm import LLM, SamplingParams
from loguru import logger
from code_compressor import CodeCompressor
import gc
from typing import List
import re


# Helper function for splitting code by functions (standalone version)
def split_code_by_functions_standalone(code: str, language: str = "python") -> List[str]:
    """
    Split code into chunks based on function and class definitions for various languages.
    Standalone version that doesn't require CodeCompressor instance.
    
    Args:
        code: The code to split
        language: Programming language of the code (python, cpp, java, typescript, rust, go)
        
    Returns:
        List of code chunks, each containing a function, class, or class method
    """
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
    
    function_pattern = re.compile(patterns[language.lower()], re.MULTILINE)
    matches = list(function_pattern.finditer(code))
    
    if not matches:
        return [code] if code.strip() else []  # No matches, return whole code if not empty
        
    result_chunks = []
    
    # Add code before first match if exists
    if matches[0].start() > 0:
        pre_code = code[:matches[0].start()].strip()
        if pre_code:
            result_chunks.append(pre_code)
    
    # Process each match
    for i, match in enumerate(matches):
        start = match.start()
        
        # End is either start of next match or end of code
        if i < len(matches) - 1:
            end = matches[i + 1].start()
        else:
            end = len(code)
        
        chunk = code[start:end].strip()
        if chunk:
            result_chunks.append(chunk)
    
    return result_chunks


# Helper function for function-level RAG retrieval
def function_rag_retrieve(background_code: str, query_code: str, model, tokenizer, device, language: str, top_k: int) -> str:
    """Uses function-level chunking and retrieves top_k similar functions."""
    if not background_code.strip():
        return ""  # Return empty if no background context

    # Split code into function-based chunks
    chunks = split_code_by_functions_standalone(background_code, language)
    if not chunks:
        return ""  # Return empty if chunking results in nothing

    query_embedding = compute_embedding(query_code, model, tokenizer, device)

    chunk_embeddings = []
    valid_chunks = []
    for chunk in chunks:
        if chunk.strip():
            chunk_embeddings.append(compute_embedding(chunk, model, tokenizer, device))
            valid_chunks.append(chunk)

    if not valid_chunks:
        return ""

    # Stack embeddings for efficient similarity calculation
    chunk_embeddings_tensor = torch.stack(chunk_embeddings)

    # Compute cosine similarity
    similarities = torch.cosine_similarity(query_embedding.unsqueeze(0), chunk_embeddings_tensor, dim=1)

    # Get top_k indices
    top_k_indices = torch.topk(similarities, k=min(top_k, len(valid_chunks)), dim=0).indices

    # Retrieve relevant chunks
    retrieved_chunks = [valid_chunks[i] for i in top_k_indices.tolist()]

    # Combine relevant chunks (maintain order by similarity score)
    combined_code = "\n\n".join(retrieved_chunks)

    return combined_code


# Helper function for sliding window chunking
def chunk_sliding_window(code: str, window_size: int, overlap: int) -> list[str]:
    """Splits code into overlapping chunks using a sliding window."""
    lines = code.splitlines()
    if not lines:
        return []

    chunks = []
    start = 0
    stride = window_size - overlap
    if stride <= 0:
        raise ValueError("Overlap size must be smaller than window size.")

    while True:
        end = min(start + window_size, len(lines))
        chunk_lines = lines[start:end]
        if not chunk_lines:  # Should not happen if lines is not empty, but safety check
            break
        chunks.append("\n".join(chunk_lines))
        if end == len(lines):
            break  # Exit loop if we reached the end
        next_start = start + stride
        # If the next window would go past the end, break
        if next_start >= len(lines):
            # Add the final overlapping chunk if needed
            final_start = max(0, len(lines) - window_size)
            if final_start > start:  # Ensure it's a new chunk not already added
                final_chunk_lines = lines[final_start:]
                chunks.append("\n".join(final_chunk_lines))
            break
        start = next_start

    # Handle case where code is shorter than window size
    if not chunks and lines:
        return ["\n".join(lines)]

    # Remove duplicates while preserving order (important for RAG)
    seen = set()
    unique_chunks = []
    for chunk in chunks:
        if chunk not in seen:
            seen.add(chunk)
            unique_chunks.append(chunk)

    return unique_chunks


# Helper function to compute embeddings (using mean pooling)
def compute_embedding(text: str, model, tokenizer, device) -> torch.Tensor:
    """Computes sentence embedding for a text using the provided model."""
    if not text.strip():  # Handle empty strings
        return torch.zeros(model.config.hidden_size)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean pool the last hidden state; move to CPU so callers can accumulate
    # many embeddings in a list without pinning GPU memory.
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().detach().cpu()
    return embedding

# Helper function for RAG retrieval


def rag_retrieve(background_code: str, query_code: str, model, tokenizer, device, window_size: int, overlap: int, top_k: int) -> str:
    """Chunks background, embeds chunks and query, retrieves top_k similar chunks."""
    if not background_code.strip():
        return ""  # Return empty if no background context

    chunks = chunk_sliding_window(background_code, window_size, overlap)
    if not chunks:
        return ""  # Return empty if chunking results in nothing

    query_embedding = compute_embedding(query_code, model, tokenizer, device)

    chunk_embeddings = []
    valid_chunks = []
    for chunk in chunks:
        if chunk.strip():
            chunk_embeddings.append(compute_embedding(chunk, model, tokenizer, device))
            valid_chunks.append(chunk)

    if not valid_chunks:
        return ""

    # Stack embeddings for efficient similarity calculation
    chunk_embeddings_tensor = torch.stack(chunk_embeddings)

    # Compute cosine similarity
    similarities = torch.cosine_similarity(query_embedding.unsqueeze(0), chunk_embeddings_tensor, dim=1)

    # Get top_k indices
    top_k_indices = torch.topk(similarities, k=min(top_k, len(valid_chunks)), dim=0).indices

    # Retrieve and sort chunks by their original position
    relevant_chunks_with_indices = []
    original_indices_map = {chunk_content: idx for idx, chunk_content in enumerate(chunks)}  # Map content back to original index

    retrieved_chunk_contents = [valid_chunks[i] for i in top_k_indices.tolist()]

    # Find original start lines to sort chronologically (approximate)
    chunk_start_lines = {}
    current_line = 0
    lines = background_code.splitlines()
    chunk_map_from_sliding = chunk_sliding_window(background_code, window_size, overlap)  # Re-chunk to get consistent indexing if needed
    start_line_num = 0
    stride = window_size - overlap
    for i, chunk_content in enumerate(chunk_map_from_sliding):
        # This assumes the chunking function returns chunks in order
        chunk_start_lines[chunk_content] = start_line_num
        start_line_num += stride
        # Rough approximation, doesn't perfectly handle edge cases/final chunks

    sorted_relevant_chunks = sorted(
        retrieved_chunk_contents,
        key=lambda content: chunk_start_lines.get(content, float('inf'))  # Sort by approximate start line
    )

    # Combine relevant chunks
    # Original implementation joined with \n, let's keep it simple
    combined_code = "\n\n".join(sorted_relevant_chunks)  # Separate chunks by double newline for clarity

    return combined_code


# Helper function for LLMLingua compression
def compress_llmlingua(context: str, query: str, compressor: PromptCompressor, target_token: int, instruction: str) -> str:
    """Compresses context using LLMLingua."""
    if not context.strip():
        return ""
    try:
        # Ensure no "<|endoftext|>"
        context_clean = context.replace("<|endoftext|>", "")
        compressed = compressor.compress_prompt(
            context_clean,
            instruction=instruction,
            question=query + "\n" + instruction,  # Combine query and instruction for question
            target_token=target_token
        )
        # Ensure result exists and is string
        result = compressed.get('compressed_prompt', '')
        return result if isinstance(result, str) else ""
    except Exception as e:
        logger.error(f"LLMLingua compression failed: {e}")
        # Fallback: Truncate based on target tokens (approximate)
        tokens = compressor.tokenizer.encode(context_clean)
        if len(tokens) > target_token:
            return compressor.tokenizer.decode(tokens[:target_token])
        return context_clean


# Helper function for LongLLMLingua compression
def compress_longllmlingua(context: str, query: str, compressor: PromptCompressor, target_token: int, instruction: str, chunk_size: int, overlap: int) -> str:
    """Compresses context using LongLLMLingua with sliding window chunks."""
    if not context.strip():
        return ""
    try:
        # Ensure no "<|endoftext|>"
        context_clean = context.replace("<|endoftext|>", "")
        # Use our sliding window chunker
        chunks = chunk_sliding_window(context_clean, chunk_size, overlap)
        if not chunks:
            return ""  # Handle case where context is too short or chunking fails

        compressed = compressor.compress_prompt(
            chunks,
            instruction=instruction,
            question=query + "\n" + instruction,  # Combine query and instruction for question
            target_token=target_token,
            rank_method="longllmlingua"  # Use the specified rank method
        )
        # Ensure result exists and is string
        result = compressed.get('compressed_prompt', '')
        return result if isinstance(result, str) else ""
    except Exception as e:
        logger.error(f"LongLLMLingua compression failed: {e}")
        # Fallback: Truncate based on target tokens (approximate)
        tokens = compressor.tokenizer.encode(context_clean)
        if len(tokens) > target_token:
            return compressor.tokenizer.decode(tokens[:target_token])
        return context_clean

# Helper function for CodeCompressor (Rank Only or Fine-grained)


def compress_code_compressor(context: str, query: str, compressor: CodeCompressor, target_token: int, instruction: str, language: str, rank_only: bool, fine_ratio: float, importance_beta: float) -> str:
    """Compresses context using CodeCompressor based on target tokens and rank_only flag."""
    if not context.strip():
        return ""
    try:
        # Ensure no "<|endoftext|>"
        context_clean = context.replace("<|endoftext|>", "")
        if not context_clean.strip():
            return ""  # Return empty if clean context is empty

        # Tokenize to get original length
        # Use the compressor's tokenizer
        original_tokens = len(compressor.tokenizer.encode(context_clean))
        if original_tokens == 0:
            return ""  # Avoid division by zero

        # Calculate target ratio
        target_ratio = min(1.0, max(0.0, target_token / original_tokens))
        logger.info(f"CodeCompressor: Original tokens={original_tokens}, Target tokens={target_token}, Calculated ratio={target_ratio:.4f}")

        # Pass rank_only and fine_ratio
        # Assuming compressor is already initialized with the correct model
        compressed_result = compressor.compress_code_file(
            code=context_clean,
            query=query,  # Using current function context as query focus
            instruction=instruction,
            rate=target_ratio,
            language=language,
            rank_only=rank_only,  # Ensure rank_only mode is set
            fine_ratio=fine_ratio if not rank_only else None,  # Pass fine_ratio only if not rank_only
            importance_beta=importance_beta if not rank_only else None,  # Pass importance_beta only if not rank_only
        )

        # Extract compressed content - check both possible keys
        compressed_context = compressed_result.get("compressed_code")

        if not isinstance(compressed_context, str):
            logger.error(f"CodeCompressor returned non-string: {type(compressed_context)}")
            compressed_context = ""  # Fallback

        # Log results
        compressed_tokens_count = len(compressor.tokenizer.encode(compressed_context))
        final_ratio = (compressed_tokens_count / original_tokens) if original_tokens > 0 else 0
        logger.info(f"CodeCompressor: Compressed tokens={compressed_tokens_count}, Actual ratio={final_ratio:.4f}")

        return compressed_context

    except Exception as e:
        logger.error(f"CodeCompressor compression failed: {e}", exc_info=True)
        # Fallback: Truncate approximately based on target tokens (less ideal for rank_only)
        tokens = compressor.tokenizer.encode(context_clean)
        if len(tokens) > target_token:
            logger.warning(f"CodeCompressor falling back to simple truncation.")
            return compressor.tokenizer.decode(tokens[:target_token])
        return context_clean

# Function to save scores


def save_json(data: dict, file_path: str):
    """Saves dictionary data to a JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def generate_completions(llm, batch_prompts, max_new_tokens=128):
    # Generate completions for batch
    sampling_params = SamplingParams(
        temperature=0,
        top_p=0.95,
        max_tokens=max_new_tokens
    )

    batch_outputs = llm.generate(
        batch_prompts,
        sampling_params,
        use_tqdm=False
    )

    return [x.outputs[0].text for x in batch_outputs]


def evaluate_completion(
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
    method: str = "full",
    result_dir: str = "results/completion_baselines",
    embed_model_name: str = "microsoft/unixcoder-base",
    compression_model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
    dataset_path: str = "microsoft/LCC_python",
    dataset_split: str = "test",
    num_examples: int = 200,
    max_new_tokens: int = 128,
    batch_size: int = 16,
    # RAG params
    rag_window_size: int = 80,
    rag_overlap: int = 40,
    rag_top_k: int = 3,
    # Function RAG params
    function_rag_language: str = "python",
    function_rag_top_k: int = 3,
    # LLMLingua params
    lingua_target_token: int = 500,
    lingua_instruction: str = "Complete the following code function given the context.",
    # LongLLMLingua params
    longlingua_chunk_size: int = 80,
    longlingua_overlap: int = 40,
    # CodeCompressor params (New)
    code_compressor_target_token: int = 500,
    # vLLM params
    tensor_parallel_size: int = 1,
    trust_remote_code: bool = True,
    gpu_memory_utilization: float = 0.9,
    filter_current_lines_max: int = 50,
    filter_background_tokens_min: int = 3000,
    # New CodeCompressor fine-grained param
    code_compressor_fine_ratio: float = 1.0,  # Default 1.0 means rank_only=True
    # New CodeCompressor importance beta param
    importance_beta: float = 0.0, # Default beta is 0.0
):
    """Evaluates code completion baselines with a specified context preparation method."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- 1. Load Data ---
    # Assuming python for now, might need modification if dataset has multiple languages
    # Note: Language info might be needed for CodeCompressor if not always python
    dataset, _ = load_data(path=dataset_path, split=dataset_split, num_examples=num_examples,
                           filter_current_lines_max=filter_current_lines_max, filter_background_tokens_min=filter_background_tokens_min)
    logger.info(f"Loaded {len(dataset)} examples from {dataset_path} ({dataset_split} split)")

    # --- 2. Initialize Models ---
    embed_model = None
    embed_tokenizer = None
    if method == "rag" or method == "function_rag":
        logger.info(f"Initializing embedding model: {embed_model_name}")
        embed_tokenizer = AutoTokenizer.from_pretrained(embed_model_name)
        embed_model = AutoModel.from_pretrained(embed_model_name).to(device)
        embed_model.eval()  # Set to evaluation mode
        logger.info(f"Embedding model {embed_model_name} initialized.")

    lingua_compressor = None
    if method == "llmlingua" or method == "longllmlingua":
        logger.info(f"Initializing LLMLingua compressor: {compression_model_name}")
        lingua_compressor = PromptCompressor(model_name=compression_model_name, device_map="auto")
        logger.info(f"LLMLingua compressor {compression_model_name} initialized.")

    code_compressor_instance = None  # Renamed to avoid conflict
    if method == "code_compressor":
        logger.info(f"Initializing CodeCompressor: {compression_model_name}")
        # Assuming CodeCompressor takes model name and potentially device
        # Pass device explicitly if needed by your CodeCompressor implementation
        code_compressor_instance = CodeCompressor(compression_model_name)
        logger.info(f"CodeCompressor {compression_model_name} initialized.")

    if method in ["full", "no_context"]:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # try to compress a dummy prompt to avoid cuda error when initializing the vllm (strange bug)
        code_compressor_instance = PromptCompressor(model_name=compression_model_name, device_map="auto")
        logger.info(f"CodeCompressor {compression_model_name} initialized.")
        dummy_prompt = "def hello_world():\n    print('Hello, World!')"*100
        compressed_prompt = code_compressor_instance.compress_prompt(dummy_prompt, instruction="Complete the following code function given the context.", question="Complete the following code function given the context.", target_token=500)
        logger.info(f"Compressed prompt: {compressed_prompt}")

    # --- 3. Process the Specified Method ---
    logger.info(f"--- Processing Method: {method} ---")

    # Modify result directory based on method and parameters
    method_suffix = f"method_{method}"
    if method == "rag":
        method_suffix += f"_w{rag_window_size}_o{rag_overlap}_k{rag_top_k}"
    elif method == "function_rag":
        method_suffix += f"_lang{function_rag_language}_k{function_rag_top_k}"
    elif method == "llmlingua":
        method_suffix += f"_t{lingua_target_token}"
    elif method == "longllmlingua":
        method_suffix += f"_t{lingua_target_token}_cs{longlingua_chunk_size}_o{longlingua_overlap}"
    elif method == "code_compressor":
        # Determine if rank_only based on fine_ratio
        rank_only_for_suffix = (code_compressor_fine_ratio == 1.0)
        suffix_detail = "_rankonly" if rank_only_for_suffix else f"fr{code_compressor_fine_ratio}"
        # Add importance_beta to suffix
        if importance_beta > 0:
            suffix_detail += f"_b{importance_beta}"
        # Use code_compressor_target_token for consistency
        method_suffix += f"_t{code_compressor_target_token}{suffix_detail}"  # Updated suffix

    method_result_dir = os.path.join(result_dir, method_suffix)
    os.makedirs(method_result_dir, exist_ok=True)

    model_output_path = os.path.join(
        method_result_dir,
        f"{model_name.replace('/', '_slash_')}.jsonl",
    )
    score_output_path = os.path.join(
        method_result_dir,
        f"{model_name.replace('/', '_slash_')}-SCORES.json",
    )

    all_prompts = []
    original_data = []  # Store original data to merge with results

    # Prepare prompts based on method
    for i, example in enumerate(tqdm(dataset, desc=f"Preparing prompts for {method}")):
        background_ctx = example['background_context']
        current_func_ctx = example['current_function_context']  # This is the prefix
        ground_truth = example['gt']  # This is the completion target
        # Determine language - assuming python for now based on dataset path
        language = "python"  # IMPORTANT: Make dynamic if dataset contains multiple languages

        context_for_prompt = ""
        try:
            if method == "full":
                context_for_prompt = background_ctx + "\n\n" + current_func_ctx

                # some models have max context length of 32768, so we truncate the context (from the head) if it exceeds that
                tokenized_context = tokenizer.encode(context_for_prompt)
                if len(tokenized_context) > 32768-256:
                    logger.warning(f"Context length exceeds 32768, truncating from the head. Original length: {len(tokenized_context)}, Truncated length: 32768")
                    context_for_prompt = tokenizer.decode(tokenized_context[-(32768-256):])
            elif method == "rag":
                if not embed_model or not embed_tokenizer:
                    raise ValueError("RAG method selected but embedding model not initialized.")
                retrieved_ctx = rag_retrieve(
                    background_ctx, current_func_ctx,
                    embed_model, embed_tokenizer, device,
                    rag_window_size, rag_overlap, rag_top_k
                )
                context_for_prompt = retrieved_ctx + "\n\n" + current_func_ctx
            elif method == "function_rag":
                if not embed_model or not embed_tokenizer:
                    raise ValueError("Function RAG method selected but embedding model not initialized.")
                retrieved_ctx = function_rag_retrieve(
                    background_ctx, current_func_ctx,
                    embed_model, embed_tokenizer, device,
                    function_rag_language, function_rag_top_k
                )
                context_for_prompt = retrieved_ctx + "\n\n" + current_func_ctx
            elif method == "llmlingua":
                if not lingua_compressor:
                    raise ValueError("LLMLingua method selected but compressor not initialized.")
                compressed_ctx = compress_llmlingua(
                    background_ctx, current_func_ctx,
                    lingua_compressor, lingua_target_token, lingua_instruction
                )
                context_for_prompt = compressed_ctx + "\n\n" + current_func_ctx
            elif method == "longllmlingua":
                if not lingua_compressor:
                    raise ValueError("LongLLMLingua method selected but compressor not initialized.")
                compressed_ctx = compress_longllmlingua(
                    background_ctx, current_func_ctx,
                    lingua_compressor, lingua_target_token, lingua_instruction,
                    longlingua_chunk_size, longlingua_overlap
                )
                context_for_prompt = compressed_ctx + "\n\n" + current_func_ctx
            elif method == "code_compressor":
                if not code_compressor_instance:
                    raise ValueError("CodeCompressor method selected but compressor not initialized.")
                # Determine rank_only based on fine_ratio
                rank_only = (code_compressor_fine_ratio == 1.0)
                logger.info(f"CodeCompressor mode: {'Rank Only' if rank_only else f'Fine-grained (ratio={code_compressor_fine_ratio})'}")
                # Use current_func_ctx as the query for CodeCompressor to focus retrieval
                compressed_ctx = compress_code_compressor(
                    context=background_ctx,
                    query=current_func_ctx,  # Query is the current function prefix
                    compressor=code_compressor_instance,
                    target_token=code_compressor_target_token,
                    instruction=lingua_instruction,  # Reusing lingua instruction
                    language=language,
                    rank_only=rank_only,  # Pass determined rank_only flag
                    fine_ratio=code_compressor_fine_ratio,  # Pass fine_ratio
                    importance_beta=importance_beta, # Pass importance_beta
                )
                # Combine the compressed background context with the original current function context
                context_for_prompt = compressed_ctx + "\n\n" + current_func_ctx
            elif method == "no_context":
                context_for_prompt = current_func_ctx
            else:
                raise ValueError(f"Unknown method: {method}")

            prompt = context_for_prompt.strip()
            all_prompts.append(prompt)
            original_data.append({
                "id": example.get("id", i),
                "gt": ground_truth,
                "original_background_context": background_ctx,
                "original_current_function_context": current_func_ctx,
                "language": language  # Store language if needed later
            })
        except Exception as e:
            logger.warning(f"Error processing example {i} (ID: {example.get('id', 'N/A')}) for method {method}: {e}", exc_info=True)
            continue  # Skip this example

    # --- 4. Clean up Compression/Embedding Models ---
    logger.info("Freeing up GPU memory from compression/embedding models")
    if embed_model:
        del embed_model
    if embed_tokenizer:
        del embed_tokenizer
    if lingua_compressor:
        del lingua_compressor
    if code_compressor_instance:
        del code_compressor_instance  # Clean up CodeCompressor
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("GPU memory freed")

    # --- 5. Initialize Generation LLM ---
    # Check if there are any prompts to process before initializing LLM
    if not all_prompts:
        logger.error(f"No valid prompts were prepared for method {method}. Skipping generation and scoring.")
        return

    logger.info(f"Initializing generation LLM: {model_name}")
    llm = LLM(
        model=model_name,
        trust_remote_code=trust_remote_code,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=32768
    )
    logger.info(f"Generation LLM {model_name} initialized.")

    # --- 6. Generate Completions ---
    all_outputs = []
    logger.info(f"Generating completions for {len(all_prompts)} prompts...")
    for i in tqdm(range(0, len(all_prompts), batch_size), desc=f"Generating completions for {method}"):
        batch_prompts = all_prompts[i:i + batch_size]
        if not batch_prompts:
            continue

        try:
            batch_outputs = generate_completions(llm, batch_prompts, max_new_tokens=max_new_tokens)
            all_outputs.extend(batch_outputs)
        except Exception as e:
            logger.error(f"Error during generation for batch starting at index {i}: {e}")
            all_outputs.extend(["ERROR_GENERATING"] * len(batch_prompts))

    # --- 7. Evaluate and Save Results ---
    model_outputs_data = []
    total_es = 0
    total_em = 0
    valid_scores = 0

    if len(all_outputs) != len(original_data):
        logger.warning(f"Warning: Mismatch between generated outputs ({len(all_outputs)}) and original data ({len(original_data)}). Scores might be inaccurate.")
        min_len = min(len(all_outputs), len(original_data))
        all_outputs = all_outputs[:min_len]
        original_data = original_data[:min_len]
        all_prompts = all_prompts[:min_len]

    logger.info(f"Calculating scores and saving results for {len(all_outputs)} examples...")
    # make sure that the path exists
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    with open(model_output_path, "w") as f_out:
        for i in range(len(all_outputs)):
            output = all_outputs[i]
            # Ensure index is valid for original_data and all_prompts
            if i >= len(original_data) or i >= len(all_prompts):
                logger.error(f"Index {i} out of bounds after potential mismatch alignment. Stopping result processing.")
                break
            orig_data = original_data[i]
            prompt = all_prompts[i]
            gt = orig_data['gt']

            result = {
                **orig_data,
                "prompt": prompt,
                "output": output,
            }

            es = 0
            em = 0
            if output != "ERROR_GENERATING" and gt is not None:
                try:
                    es = compute_ES(gt, output)
                    em = compute_EM(gt, output)
                    total_es += es
                    total_em += em
                    valid_scores += 1
                except Exception as e:
                    logger.error(f"Error scoring example {orig_data.get('id', i)}: {e}")

            result['es'] = es
            result['em'] = em
            model_outputs_data.append(result)
            f_out.write(json.dumps(result) + "\n")

    logger.info(f"Raw results saved to {model_output_path}")

    avg_es = (total_es / valid_scores) if valid_scores > 0 else 0
    avg_em = (total_em / valid_scores) if valid_scores > 0 else 0

    # Update the parameters dictionary in scores
    scores = {
        "model_name": model_name,
        "method": method,
        "num_examples_scored": valid_scores,
        "num_examples_total": len(original_data),  # Use length of original_data before potential alignment issues
        "average_es": avg_es,
        "average_em": avg_em,
        "parameters": {
            "dataset_path": dataset_path,
            "dataset_split": dataset_split,
            "filter_current_lines_max": filter_current_lines_max,
            "filter_background_tokens_min": filter_background_tokens_min,
            "embed_model_name": embed_model_name if method == "rag" or method == "function_rag" else None,
            # Combine compression model name reporting
            "compression_model_name": compression_model_name if method in ["llmlingua", "longllmlingua", "code_compressor"] else None,
            "max_new_tokens": max_new_tokens,
            "batch_size": batch_size,
            # RAG specific params
            "rag_window_size": rag_window_size if method == "rag" else None,
            "rag_overlap": rag_overlap if method == "rag" else None,
            "rag_top_k": rag_top_k if method == "rag" else None,
            # Function RAG params
            "function_rag_language": function_rag_language if method == "function_rag" else None,
            "function_rag_top_k": function_rag_top_k if method == "function_rag" else None,
            # Lingua specific params (shared target token name)
            "lingua_target_token": lingua_target_token if method == "llmlingua" or method == "longllmlingua" else None,
            # LongLingua specific params
            "longlingua_chunk_size": longlingua_chunk_size if method == "longllmlingua" else None,
            "longlingua_overlap": longlingua_overlap if method == "longllmlingua" else None,
            # CodeCompressor specific params
            "code_compressor_target_token": code_compressor_target_token if method == "code_compressor" else None,  # Added parameter
            "code_compressor_rank_only": (code_compressor_fine_ratio == 1.0) if method == "code_compressor" else None,  # Determined by fine_ratio
            "code_compressor_fine_ratio": code_compressor_fine_ratio if method == "code_compressor" else None,  # Added parameter
            "importance_beta": importance_beta if method == "code_compressor" else None, # Added parameter
        }
    }

    logger.info(f"Method {method}: Avg ES = {avg_es:.2f}, Avg EM = {avg_em:.2f} ({valid_scores}/{len(original_data)} scored)")
    save_json(scores, score_output_path)
    logger.info(f"Scores saved to {score_output_path}")

    logger.info("Evaluation complete.")
    # Clean up LLM explicitly
    if 'llm' in locals() and llm is not None:
        del llm
        logger.info("Generation LLM deleted.")
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    fire.Fire(evaluate_completion)
