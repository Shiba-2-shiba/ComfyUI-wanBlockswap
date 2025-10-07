import torch
import gc
import psutil
import comfy.model_management

# --- メモリ管理ノードの定義 ---

class DisTorchMemoryCleaner:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "latent": ("LATENT",),
        }}
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "clean_memory"
    CATEGORY = "MemoryUtils"

    def clean_memory(self, latent):
        print("=== DisTorch Memory Cleaner: Start ===")
        # GPUキャッシュのクリア
        if torch.cuda.is_available():
            try:
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(f'cuda:{i}'):
                        torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print("All GPU caches cleared")
            except Exception as e:
                print(f"GPU cache clear failed: {e}")

        # Pythonのガベージコレクション
        gc.collect()
        
        # DisTorchの仮想メモリを解放
        if hasattr(comfy.model_management, 'free_memory'):
            try:
                for i in range(torch.cuda.device_count()):
                    comfy.model_management.free_memory(0, f'cuda:{i}')
                print("DisTorch virtual memory reset for all CUDA devices")
            except Exception as e:
                print(f"DisTorch virtual memory reset failed: {e}")
        
        print("=== DisTorch Memory Cleaner: Complete ===")
        return (latent,)

class DisTorchMemoryManager:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "clean_gpu": ("BOOLEAN", {"default": True}),
                "clean_cpu": ("BOOLEAN", {"default": False, "tooltip": "CPU memory cleanup (use with caution)"}),
                "force_gc": ("BOOLEAN", {"default": True}),
                "reset_virtual_memory": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "model_to_unload": ("MODEL",),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "manage_memory"
    CATEGORY = "MemoryUtils"

    def manage_memory(self, latent, clean_gpu, clean_cpu, force_gc, reset_virtual_memory, model_to_unload=None):
        if model_to_unload is not None:
            print("=== Unloading Model from Memory ===")
            try:
                model_to_unload.detach()
                print("Model unloaded successfully.")
            except Exception as e:
                print(f"Could not unload model: {e}")

        print("=== DisTorch Memory Management: Start ===")
        
        # メモリ使用量の表示（前）
        gpu_memory_before = 0
        if torch.cuda.is_available():
            gpu_memory_before = torch.cuda.memory_allocated() / 1024**3
            print(f"GPU Memory before: {gpu_memory_before:.2f} GB")
        
        cpu_memory_before = psutil.virtual_memory().used / 1024**3
        print(f"CPU Memory before: {cpu_memory_before:.2f} GB")
        
        if clean_gpu and torch.cuda.is_available():
            try:
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(f'cuda:{i}'):
                        torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print("All GPU caches cleared")
            except Exception as e:
                print(f"GPU cache clear failed: {e}")

        if clean_cpu:
            try:
                collected = gc.collect()
                print(f"CPU memory cleanup: {collected} objects collected")
            except Exception as e:
                print(f"CPU memory cleanup failed: {e}")
        
        if force_gc:
            try:
                collected = gc.collect()
                print(f"Garbage collected: {collected} objects")
            except Exception as e:
                print(f"Garbage collection failed: {e}")
        
        if reset_virtual_memory and hasattr(comfy.model_management, 'free_memory'):
            try:
                for i in range(torch.cuda.device_count()):
                    comfy.model_management.free_memory(0, f'cuda:{i}')
                print("DisTorch virtual memory reset for all CUDA devices")
            except Exception as e:
                print(f"DisTorch virtual memory reset failed: {e}")
        
        # メモリ使用量の表示（後）
        if torch.cuda.is_available():
            try:
                gpu_memory_after = torch.cuda.memory_allocated() / 1024**3
                gpu_freed = gpu_memory_before - gpu_memory_after
                print(f"GPU Memory after: {gpu_memory_after:.2f} GB (freed: {gpu_freed:.2f} GB)")
            except Exception as e:
                print(f"GPU memory measurement failed: {e}")
        
        try:
            cpu_memory_after = psutil.virtual_memory().used / 1024**3
            cpu_freed = cpu_memory_before - cpu_memory_after
            print(f"CPU Memory after: {cpu_memory_after:.2f} GB (freed: {cpu_freed:.2f} GB)")
        except Exception as e:
            print(f"CPU memory measurement failed: {e}")
        
        print("=== Memory Management: Complete ===")
        
        return (latent,)

class DisTorchSafeMemoryManager:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "clean_gpu": ("BOOLEAN", {"default": True}),
                "force_gc": ("BOOLEAN", {"default": True}),
                "reset_virtual_memory": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "model_to_unload": ("MODEL",),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "safe_manage_memory"
    CATEGORY = "MemoryUtils"

    def safe_manage_memory(self, latent, clean_gpu, force_gc, reset_virtual_memory, model_to_unload=None):
        if model_to_unload is not None:
            print("=== Unloading Model from Memory ===")
            try:
                model_to_unload.detach()
                print("Model unloaded successfully.")
            except Exception as e:
                print(f"Could not unload model: {e}")
        
        print("=== DisTorch Safe Memory Management: Start ===")
        
        gpu_memory_before = 0
        if torch.cuda.is_available():
            gpu_memory_before = torch.cuda.memory_allocated() / 1024**3
            print(f"GPU Memory before: {gpu_memory_before:.2f} GB")
        
        if clean_gpu and torch.cuda.is_available():
            try:
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(f'cuda:{i}'):
                        torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print("All GPU caches cleared")
            except Exception as e:
                print(f"GPU cache clear failed: {e}")
        
        if force_gc:
            try:
                collected = gc.collect()
                print(f"Garbage collected: {collected} objects")
            except Exception as e:
                print(f"Garbage collection failed: {e}")
        
        if reset_virtual_memory and hasattr(comfy.model_management, 'free_memory'):
            try:
                for i in range(torch.cuda.device_count()):
                    comfy.model_management.free_memory(0, f'cuda:{i}')
                print("DisTorch virtual memory reset for all CUDA devices")
            except Exception as e:
                print(f"DisTorch virtual memory reset failed: {e}")
        
        if torch.cuda.is_available():
            try:
                gpu_memory_after = torch.cuda.memory_allocated() / 1024**3
                gpu_freed = gpu_memory_before - gpu_memory_after
                print(f"GPU Memory after: {gpu_memory_after:.2f} GB (freed: {gpu_freed:.2f} GB)")
            except Exception as e:
                print(f"GPU memory measurement failed: {e}")
        
        print("=== Safe Memory Management: Complete ===")
        
        return (latent,)

# --- ComfyUIへのノード登録 ---

NODE_CLASS_MAPPINGS = {
    "MemoryCleaner": DisTorchMemoryCleaner,
    "MemoryManager": DisTorchMemoryManager,
    "SafeMemoryManager": DisTorchSafeMemoryManager,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MemoryCleaner": "🧹 Memory Cleaner (Simple)",
    "MemoryManager": "🧹 Memory Manager (Advanced)",
    "SafeMemoryManager": "🧹 Memory Manager (Safe)",
}