#!/usr/bin/env python3

# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import time
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from typing import Dict, List
import json

class LLMProfiler:
    def __init__(self):
        self.metrics = {}
        
    def profile_inference(self, model, processor, inputs, max_new_tokens=40, num_runs=3):
        """Profile LLM inference with detailed timing metrics"""
        
        results = []
        
        for run in range(num_runs):
            print(f"Run {run + 1}/{num_runs}")
            
            # Warm up
            if run == 0:
                print("Warming up...")
                with torch.no_grad():
                    _ = model.generate(**inputs, max_new_tokens=5, do_sample=False)
            
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Start timing
            start_time = time.perf_counter()
            
            # Generate with streaming to capture TTFT
            generated_tokens = []
            ttft = None
            token_times = []
            
            with torch.no_grad():
                # For TTFT measurement, we need to use a custom generation loop
                input_length = inputs["input_ids"].shape[-1]
                
                # First token generation (prefill phase)
                prefill_start = time.perf_counter()
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=1, 
                    do_sample=False,
                    pad_token_id=processor.tokenizer.eos_token_id
                )
                prefill_end = time.perf_counter()
                
                ttft = prefill_end - prefill_start
                generated_tokens.append(outputs[0, input_length:])
                
                # Decode phase - generate remaining tokens one by one
                decode_times = []
                current_input = outputs
                
                for i in range(1, max_new_tokens):
                    decode_start = time.perf_counter()
                    
                    next_outputs = model.generate(
                        input_ids=current_input,
                        max_new_tokens=1,
                        do_sample=False,
                        pad_token_id=processor.tokenizer.eos_token_id
                    )
                    
                    decode_end = time.perf_counter()
                    decode_time = decode_end - decode_start
                    decode_times.append(decode_time)
                    
                    # Check if we hit EOS
                    new_token = next_outputs[0, -1:]
                    if new_token.item() == processor.tokenizer.eos_token_id:
                        break
                        
                    current_input = next_outputs
                    generated_tokens.append(new_token)
                
                # Final outputs
                final_outputs = current_input
            
            end_time = time.perf_counter()
            
            # Calculate metrics
            total_time = end_time - start_time
            input_tokens = input_length
            output_tokens = len(generated_tokens)
            total_tokens = input_tokens + output_tokens
            
            # Decode the output
            generated_text = processor.decode(
                final_outputs[0][input_length:], 
                skip_special_tokens=True
            )
            
            metrics = {
                'run': run + 1,
                'total_time': total_time,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': total_tokens,
                'end_to_end_tokens_per_sec': total_tokens / total_time,
                'prefill_tokens_per_sec': input_tokens / ttft if ttft > 0 else 0,
                'decode_tokens_per_sec': output_tokens / sum(decode_times) if decode_times else 0,
                'ttft_ms': ttft * 1000,
                'average_tpot_ms': (sum(decode_times) / len(decode_times)) * 1000 if decode_times else 0,
                'generated_text': generated_text
            }
            
            results.append(metrics)
            
            # Print metrics for this run
            print(f"  Total time: {total_time:.3f}s")
            print(f"  Input tokens: {input_tokens}")
            print(f"  Output tokens: {output_tokens}")
            print(f"  End-to-end tokens/sec: {metrics['end_to_end_tokens_per_sec']:.2f}")
            print(f"  Prefill tokens/sec: {metrics['prefill_tokens_per_sec']:.2f}")
            print(f"  Decode tokens/sec: {metrics['decode_tokens_per_sec']:.2f}")
            print(f"  TTFT: {metrics['ttft_ms']:.2f}ms")
            print(f"  Average TPOT: {metrics['average_tpot_ms']:.2f}ms")
            print(f"  Generated: {generated_text}")
            print()
        
        return results
    
    def calculate_averages(self, results: List[Dict]) -> Dict:
        """Calculate average metrics across runs"""
        if not results:
            return {}
        
        avg_metrics = {}
        numeric_keys = [
            'total_time', 'end_to_end_tokens_per_sec', 'prefill_tokens_per_sec',
            'decode_tokens_per_sec', 'ttft_ms', 'average_tpot_ms'
        ]
        
        for key in numeric_keys:
            values = [r[key] for r in results if key in r and r[key] is not None]
            if values:
                avg_metrics[f'avg_{key}'] = sum(values) / len(values)
                avg_metrics[f'min_{key}'] = min(values)
                avg_metrics[f'max_{key}'] = max(values)
        
        return avg_metrics

def main():
    print("🚀 Starting Gemma3-4B Performance Profiling...")
    
    device = "cpu"
    
    # Load model and processor
    print("Loading model and processor...")
    processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")
    model = AutoModelForImageTextToText.from_pretrained("google/gemma-3-4b-it").to(device)
    
    # Prepare input
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": "/ryzers/data/toucan.jpg"},
                {"type": "text", "text": "What animal is this? Only state the name of the animal."}
            ]
        },
    ]
    
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)
    
    # Profile inference
    profiler = LLMProfiler()
    results = profiler.profile_inference(model, processor, inputs, max_new_tokens=40, num_runs=3)
    
    # Calculate and display averages
    avg_metrics = profiler.calculate_averages(results)
    
    print("=" * 60)
    print("📊 PERFORMANCE SUMMARY")
    print("=" * 60)
    
    for key, value in avg_metrics.items():
        if 'tokens_per_sec' in key:
            print(f"{key}: {value:.2f} tokens/sec")
        elif 'ms' in key:
            print(f"{key}: {value:.2f} ms")
        else:
            print(f"{key}: {value:.3f}")
    
    # Save results to JSON
    output_data = {
        'individual_runs': results,
        'averages': avg_metrics,
        'model_info': {
            'model_name': "google/gemma-3-4b-it",
            'device': device,
            'torch_version': torch.__version__
        }
    }
    
    with open('/ryzers/gemma3_performance_results.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Results saved to /ryzers/gemma3_performance_results.json")

if __name__ == "__main__":
    main()
