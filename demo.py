from longcodezip import LongCodeZip
from loguru import logger
import argparse
import os
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI


def generate_completion(
    prompt: str,
    generation_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
    max_new_tokens: int = 32,
    use_openai: bool = False,
    api_key: str = None,
    hf_model=None,
    hf_tokenizer=None,
):

    if use_openai:
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key missing. "
                "Provide api_key or set OPENAI_API_KEY env var."
            )

        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model=generation_model,
            messages=[{
                "role": "system",
                "content": "You are a code generator. Output ONLY valid code without comments and explanations."
            },
            {"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=0
        )

        generated = response.choices[0].message.content.strip()

        #  Remove markdown code fences
        if generated.startswith("```"):
            generated = generated.replace("```python", "")
            generated = generated.replace("```", "")
            generated = generated.strip()

    else:
        # Reuse a single preloaded HF model/tokenizer to avoid loading the
        # weights into VRAM on every call (which otherwise leaks GPU memory).
        if hf_model is None or hf_tokenizer is None:
            raise ValueError(
                "hf_model and hf_tokenizer must be provided when use_openai=False"
            )
        tokenized = hf_tokenizer(prompt, return_tensors="pt").to(hf_model.device)
        with torch.inference_mode():
            output_ids = hf_model.generate(
                **tokenized,
                max_new_tokens=max_new_tokens,
                pad_token_id=hf_tokenizer.eos_token_id
            )

        generated = hf_tokenizer.decode(
            output_ids[0][len(tokenized.input_ids[0]):],
            skip_special_tokens=True
        ).strip()

    return generated.split("\n\n")[0].strip()


if __name__ == "__main__":
    load_dotenv() 
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_openai",
        action="store_true",
        help="Use OpenAI API"
    )
    parser.add_argument(
        "--generation_model",
        type=str,
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        help="HF Model to use for generation"
    )
    args = parser.parse_args()

    with open("assets/example_context.py", "r") as f:
        context = f.read()

    question = '''
    async def _finalize_step(
        self, step: "AgentStep", messages: list["LLMMessage"], execution: "AgentExecution"
    ) -> None:
        step.state = AgentStepState.COMPLETED
    '''
   
    # Initialize compressor
    logger.info("Initializing compressor...")
    compression_model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
    compressor = LongCodeZip(model_name=compression_model_name)

    # Preload the generation model once (HF path only) to avoid reloading
    # weights on every generate_completion call.
    hf_model = None
    hf_tokenizer = None
    if not args.use_openai:
        hf_tokenizer = AutoTokenizer.from_pretrained(args.generation_model)
        hf_model = AutoModelForCausalLM.from_pretrained(
            args.generation_model,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        hf_model.eval()
    
    # Test function-based code file compression with query
    logger.info("\nTesting function-based code file compression with query...")

    original_tokens = len(compressor.tokenizer.encode(context))
    target_token = 64
    target_ratio = min(1.0, max(0.0, target_token / original_tokens))
    logger.info(f"LongCodeZip: Original tokens={original_tokens}, Target tokens={target_token}, Calculated ratio={target_ratio:.4f}")

    logger.info("\nTesting compression with Coarse-grained compression only...")
    result_cond = compressor.compress_code_file(
        code=context,
        query=question,
        instruction="Complete the following code function given the context.",
        rate=target_ratio,
        rank_only=True # Coarse-grained compression
    )
    logger.info(f"Compressed prompt: \n{result_cond['compressed_prompt']}")
    logger.info(f"Compression ratio: {result_cond['compression_ratio']:.4f}") # Compression ratio: 0.3856

    completion = generate_completion(prompt=result_cond["compressed_prompt"], generation_model=args.generation_model, use_openai=args.use_openai, hf_model=hf_model, hf_tokenizer=hf_tokenizer)
    logger.info(f"Completion: {completion}")


    logger.info("\nTesting compression with Coarse-grained and Fine-grained compression...")
    result_cond = compressor.compress_code_file(
        code=context,
        query=question,
        instruction="Complete the following code function given the context.",
        rate=target_ratio,
        rank_only=False # Corase-grained and Fine-grained compression
    )
    logger.info(f"Compressed prompt: \n{result_cond['compressed_prompt']}")
    logger.info(f"Compression ratio: {result_cond['compression_ratio']:.4f}") # Compression ratio: 0.1468

    completion = generate_completion(prompt=result_cond["compressed_prompt"], generation_model=args.generation_model, use_openai=args.use_openai, hf_model=hf_model, hf_tokenizer=hf_tokenizer)
    logger.info(f"Completion: {completion}")