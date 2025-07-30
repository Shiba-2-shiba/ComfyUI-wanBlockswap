import comfy.model_management
import gc
import torch
from comfy.patcher_extension import CallbacksMP
from comfy.model_patcher import ModelPatcher
from comfy.model_base import WAN21
from tqdm import tqdm
import logging

# --- Version Compatibility Patches ---
if not hasattr(comfy.model_management, 'load_model_gpu'):
    logging.warning("Patching 'comfy.model_management.load_model_gpu' not found. Creating a wrapper for 'load_models_gpu'.")
    comfy.model_management.load_model_gpu = lambda model: comfy.model_management.load_models_gpu([model])

if not hasattr(comfy.model_management, 'cleanup_models_gc'):
    logging.warning("Patching 'comfy.model_management.cleanup_models_gc' not found. Using 'cleanup_models' as a fallback.")
    comfy.model_management.cleanup_models_gc = comfy.model_management.cleanup_models
# --- End Patches ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WanVideoBlockSwap:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "blocks_to_swap": ("INT", {"default": 20, "min": 0, "max": 40, "step": 1}),
                "offload_img_emb": ("BOOLEAN", {"default": False}),
                "offload_txt_emb": ("BOOLEAN", {"default": False}),
                "use_non_blocking": ("BOOLEAN", {"default": False}),
            },
        }
    RETURN_TYPES = ("MODEL",)
    CATEGORY = "ComfyUI-wanBlockswap"
    FUNCTION = "set_callback"

    def set_callback(self, model: ModelPatcher, blocks_to_swap, offload_txt_emb, offload_img_emb, use_non_blocking):
        
        instance_id = id(self)
        logging.info(f"WanVideoBlockSwap: Initializing for Patcher ID: {id(model)}")

        # [SOLUTION] This callback runs AFTER the KSampler is done with the model.
        # It fully unpatches the model (especially Lora) and cleans up VRAM.
        def cleanup_after_sampling(model_patcher: ModelPatcher):
            logging.info(f"-[ CLEANUP_AFTER_SAMPLING CALLBACK ({instance_id}) START | Patcher ID: {id(model_patcher)} ]-")
            
            # Unpatch all weights (like LoRA) and restore the original model state.
            model_patcher.unpatch_model(model_patcher.offload_device, unpatch_weights=True)
            
            # Force garbage collection and empty cache to defragment VRAM.
            comfy.model_management.soft_empty_cache()
            gc.collect()
            
            logging.info(f"-[ CLEANUP_AFTER_SAMPLING CALLBACK ({instance_id}) END ]-")

        # This callback runs AFTER the model is loaded into VRAM.
        # It moves the specified blocks to the offload device.
        def swap_blocks_after_load(model_patcher: ModelPatcher, device_to, lowvram_model_memory, force_patch_weights, full_load):
            logging.info(f"-[ SWAP_BLOCKS_AFTER_LOAD CALLBACK ({instance_id}) START | Patcher ID: {id(model_patcher)} ]-")
            
            base_model = model_patcher.model
            main_device = model_patcher.load_device
            offload_device = model_patcher.offload_device

            if isinstance(base_model, WAN21):
                unet = base_model.diffusion_model
                logging.info(f"  - Swapping {blocks_to_swap} blocks in WAN21 UNet.")
                for b, block in tqdm(enumerate(unet.blocks), total=len(unet.blocks), desc=f"BlockSwap ({instance_id})"):
                    target_dev = main_device if b > blocks_to_swap else offload_device
                    block.to(target_dev)
                        
                if offload_txt_emb:
                    logging.info(f"  - Offloading text_embedding to {offload_device}")
                    unet.text_embedding.to(offload_device, non_blocking=use_non_blocking)
                if offload_img_emb:
                    logging.info(f"  - Offloading img_emb to {offload_device}")
                    unet.img_emb.to(offload_device, non_blocking=use_non_blocking)
            else:
                logging.warning(f"  - Model is not a WAN21 instance, block swapping skipped.")

            comfy.model_management.soft_empty_cache()
            gc.collect()
            logging.info(f"-[ SWAP_BLOCKS_AFTER_LOAD CALLBACK ({instance_id}) END ]-")
        
        model = model.clone()
        
        # Register the swapping callback to run AFTER loading
        model.add_callback_with_key(CallbacksMP.ON_LOAD, f"wan_block_swap_{instance_id}", swap_blocks_after_load)
        
        # Register the cleanup callback to run AFTER the model has been used for sampling
        model.add_callback_with_key(CallbacksMP.ON_CLEANUP, f"wan_cleanup_{instance_id}", cleanup_after_sampling)

        return (model, )

NODE_CLASS_MAPPINGS = {
    "wanBlockSwap": WanVideoBlockSwap
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "wanBlockSwap": "WanVideoBlockSwap"
}
