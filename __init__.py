# 既存のnodes.pyからマッピングをインポート
from .nodes import NODE_CLASS_MAPPINGS as wan_class_mappings, NODE_DISPLAY_NAME_MAPPINGS as wan_display_mappings

# 新しく追加したmemory_nodes.pyからマッピングをインポート
from .memory_nodes import NODE_CLASS_MAPPINGS as mem_class_mappings, NODE_DISPLAY_NAME_MAPPINGS as mem_display_mappings

# 2つのCLASSマッピングを結合
NODE_CLASS_MAPPINGS = {**wan_class_mappings, **mem_class_mappings}

# 2つのDISPLAY_NAMEマッピングを結合
NODE_DISPLAY_NAME_MAPPINGS = {**wan_display_mappings, **mem_display_mappings}


# 結合したマッピングをComfyUIに公開
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
