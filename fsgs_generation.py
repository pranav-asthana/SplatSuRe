# import os
# import struct
# import subprocess
# from pathlib import Path
# from PIL import Image

# DATASET_ROOT = Path("/home/pranav-htic/3DGS-SR/RealSr4x_Swin")

# def get_camera_width(sparse_dir):
#     """Reads the exact width expected by COLMAP from cameras.bin or cameras.txt"""
#     bin_path = sparse_dir / "cameras.bin"
#     txt_path = sparse_dir / "cameras.txt"
    
#     if bin_path.exists():
#         with open(bin_path, 'rb') as f:
#             num_cameras = struct.unpack('<Q', f.read(8))[0]
#             camera_id = struct.unpack('<I', f.read(4))[0]
#             model_id = struct.unpack('<I', f.read(4))[0]
#             width = struct.unpack('<Q', f.read(8))[0]
#             return int(width)
#     elif txt_path.exists():
#         with open(txt_path, 'r') as f:
#             for line in f:
#                 if line.startswith("#"): continue
#                 parts = line.strip().split()
#                 if len(parts) >= 4:
#                     return int(parts[2])
#     return None

# def get_image_width(img_dir):
#     """Reads the width of the first image found in the directory"""
#     for ext in ['.jpg', '.JPG', '.png', '.PNG', '.jpeg', '.JPEG']:
#         imgs = list(img_dir.glob(f"*{ext}"))
#         if imgs:
#             with Image.open(imgs[0]) as img:
#                 return img.width
#     return None

# def main():
#     for scene_dir in DATASET_ROOT.iterdir():
#         if not scene_dir.is_dir(): continue
#         scene_name = scene_dir.name

#         # if scene_name not in ["aeroplane", "cycle", "face"]: continue
        
#         sparse_dir = scene_dir / "sparse" / "0"
        
#         if not sparse_dir.exists(): continue
        
#         print(f"\n{'='*60}\n[FSGS] Processing Scene: {scene_name}\n{'='*60}")
        
#         # 1. Determine what COLMAP expects
#         cam_width = get_camera_width(sparse_dir)
#         if not cam_width:
#             print(f"[ERROR] Could not read camera width for {scene_name}. Skipping.")
#             continue
            
#         # 2. Check available image directories
#         lr_dir = scene_dir / "images"
#         hr_dir = scene_dir / "images_SR" if (scene_dir / "images_SR").exists() else scene_dir / "images_sr"
            
#         lr_width = get_image_width(lr_dir) if lr_dir.exists() else None
#         hr_width = get_image_width(hr_dir) if hr_dir.exists() else None
        
#         # 3. Dynamically match the correct folder
#         if cam_width == hr_width:
#             target_img_dir = hr_dir
#             print(f"[INFO] Camera width ({cam_width}) matches HR images. Using {target_img_dir.name}")
#         elif cam_width == lr_width:
#             target_img_dir = lr_dir
#             print(f"[INFO] Camera width ({cam_width}) matches LR images. Using {target_img_dir.name}")
#         else:
#             print(f"[ERROR] Camera width ({cam_width}) matches neither LR ({lr_width}) nor HR ({hr_width}).")
#             continue
            
#         # 4. Clean workspace and run COLMAP

# # ==========================================================
#         # FIX: Convert all images to true JPEGs so OpenImageIO doesn't crash
#         # ==========================================================
#         colmap_img_dir = scene_dir / "images_fsgs_tmp"
#         os.system(f"rm -rf {colmap_img_dir}")
#         colmap_img_dir.mkdir(parents=True, exist_ok=True)
        
#         # The LR folder has the exact expected names
#         for base_img in lr_dir.iterdir():
#             if not base_img.is_file(): continue
            
#             # Find the matching HR/LR file regardless of extension (.png, .JPG, etc.)
#             target_files = list(target_img_dir.glob(f"{base_img.stem}.*"))
#             if target_files:
#                 source_path = target_files[0].absolute()
#                 target_path = colmap_img_dir / base_img.name
                
#                 # Actually convert the image to a real JPEG instead of just renaming/symlinking it
#                 with Image.open(source_path) as img:
#                     rgb_img = img.convert('RGB') # Strip alpha channels if PNG has them
#                     rgb_img.save(target_path, format="JPEG", quality=100)
                    
#         # Point COLMAP to our perfectly converted folder
#         target_img_dir = colmap_img_dir
#         # ==========================================================
#         # ==========================================================
        
#         # 4. Clean workspace and run COLMAP
#         dense_dir = scene_dir / "dense"
#         os.system(f"rm -rf {dense_dir}")
#         dense_dir.mkdir(parents=True, exist_ok=True)
        
#         try:
#             print(f"[{scene_name}] 1/3 Undistorting images...")
#             subprocess.run(["colmap", "image_undistorter", "--image_path", str(target_img_dir), "--input_path", str(sparse_dir), "--output_path", str(dense_dir), "--output_type", "COLMAP"], check=True)
            
#             print(f"[{scene_name}] 2/3 Patch Match Stereo...")
#             subprocess.run(["colmap", "patch_match_stereo", "--workspace_path", str(dense_dir), "--workspace_format", "COLMAP", "--PatchMatchStereo.geom_consistency", "true"], check=True)
            
#             print(f"[{scene_name}] 3/3 Stereo Fusion...")
#             subprocess.run(["colmap", "stereo_fusion", "--workspace_path", str(dense_dir), "--workspace_format", "COLMAP", "--input_type", "geometric", "--output_path", str(dense_dir / "fused.ply")], check=True)
            
#             #rint(f"[{scene_name}] 3/3 Stereo Fusion...")
#             #ubprocess.run(["colmap", "stereo_fusion", "--workspace_path", str(dense_dir), "--workspace_format", "COLMAP", "--input_type", "photometric", "--output_path", str(dense_dir / "fused.ply")], check=True)

#             print(f"[SUCCESS] Finished {scene_name}. Scaffold saved to {dense_dir}/fused.ply")
#         except subprocess.CalledProcessError as e:
#             print(f"[ERROR] COLMAP failed on {scene_name} at step: {e.cmd}")

# if __name__ == "__main__":
#     main()

# # import os
# # import subprocess
# # from pathlib import Path
# # import shutil

# # DATASET_ROOT = Path("/home/pranav-htic/3DGS-SR/RealSr4x_Swin")
# # TARGET_SCENES = ["aeroplane", "cycle", "face"]

# # def main():
# #     for scene in TARGET_SCENES:
# #         scene_dir = DATASET_ROOT / scene
# #         sparse_dir = scene_dir / "sparse" / "0"
# #         hr_dir = scene_dir / "images_sr"
# #         lr_dir = scene_dir / "images" # Original LR images have the correct filenames

# #         dense_dir = scene_dir / "dense"
# #         colmap_img_dir = scene_dir / "images_fsgs_tmp"
        
# #         # 1. Clean workspaces
# #         os.system(f"rm -rf {dense_dir}")
# #         os.system(f"rm -rf {colmap_img_dir}")
# #         dense_dir.mkdir(parents=True, exist_ok=True)
# #         colmap_img_dir.mkdir(parents=True, exist_ok=True)

# #         print(f"\n{'='*50}\n[FSGS] Processing {scene}\n{'='*50}")

# #         # 2. Fix Filenames: Map HR images to exact COLMAP expected names
# #         hr_images = sorted(list(hr_dir.glob("*.jpg")))
# #         lr_images = sorted(list(lr_dir.glob("*.jpg")))

# #         if len(hr_images) != len(lr_images):
# #             print(f"[FATAL] Mismatch in image counts for {scene}: HR({len(hr_images)}) vs LR({len(lr_images)})")
# #             continue

# #         # Create a temp folder with the exact expected filenames
# #         for hr_img, lr_img in zip(hr_images, lr_images):
# #             shutil.copy(hr_img, colmap_img_dir / lr_img.name)

# #         try:
# #             print(f"[{scene}] 1/3 Undistorting images...")
# #             subprocess.run([
# #                 "colmap", "image_undistorter", 
# #                 "--image_path", str(colmap_img_dir), # Use the fixed folder
# #                 "--input_path", str(sparse_dir), 
# #                 "--output_path", str(dense_dir), 
# #                 "--output_type", "COLMAP"
# #             ], check=True)
            
# #             # SANITY CHECK
# #             dense_img_dir = dense_dir / "images"
# #             copied_files = list(dense_img_dir.glob("*.jpg"))
# #             if len(copied_files) == 0:
# #                 print(f"[FATAL] image_undistorter failed to copy images to {dense_img_dir}!")
# #                 continue

# #             print(f"[{scene}] 2/3 Patch Match Stereo...")
# #             subprocess.run([
# #                 "colmap", "patch_match_stereo", 
# #                 "--workspace_path", str(dense_dir), 
# #                 "--workspace_format", "COLMAP", 
# #                 "--PatchMatchStereo.geom_consistency", "true"
# #             ], check=True)
            
# #             print(f"[{scene}] 3/3 Stereo Fusion...")
# #             subprocess.run([
# #                 "colmap", "stereo_fusion", 
# #                 "--workspace_path", str(dense_dir), 
# #                 "--workspace_format", "COLMAP", 
# #                 "--input_type", "photometric", 
# #                 "--output_path", str(dense_dir / "fused.ply")
# #             ], check=True)
            
# #             print(f"[SUCCESS] {scene} fused.ply generated!")

# #         except subprocess.CalledProcessError as e:
# #             print(f"[ERROR] Failed on {scene} at step: {e.cmd}")
# #         finally:
# #              # Cleanup temp folder
# #              os.system(f"rm -rf {colmap_img_dir}")

# # if __name__ == "__main__":
# #     main()


import os
import struct
import subprocess
import shutil
from pathlib import Path
from PIL import Image

DATASET_ROOT = Path("/home/suresh/Documents/dataset")

def get_camera_width(sparse_dir):
    """Reads the exact width expected by COLMAP from cameras.bin or cameras.txt"""
    bin_path = sparse_dir / "cameras.bin"
    txt_path = sparse_dir / "cameras.txt"
    
    if bin_path.exists():
        with open(bin_path, 'rb') as f:
            num_cameras = struct.unpack('<Q', f.read(8))[0]
            camera_id = struct.unpack('<I', f.read(4))[0]
            model_id = struct.unpack('<I', f.read(4))[0]
            width = struct.unpack('<Q', f.read(8))[0]
            return int(width)
    elif txt_path.exists():
        with open(txt_path, 'r') as f:
            for line in f:
                if line.startswith("#"): continue
                parts = line.strip().split()
                if len(parts) >= 4:
                    return int(parts[2])
    return None

def get_image_width(img_dir):
    """Reads the width of the first image found in the directory"""
    for ext in ['.jpg', '.JPG', '.png', '.PNG', '.jpeg', '.JPEG']:
        imgs = list(img_dir.glob(f"*{ext}"))
        if imgs:
            with Image.open(imgs[0]) as img:
                return img.width
    return None

def main():
    for scene_dir in DATASET_ROOT.iterdir():
        if not scene_dir.is_dir(): continue
        scene_name = scene_dir.name

        # ONLY RUN ON THESE 3 SCENES
        if scene_name not in ["aeroplane", "cycle", "face"]: continue
        
        sparse_dir = scene_dir / "sparse" / "0"
        if not sparse_dir.exists(): continue
        
        print(f"\n{'='*60}\n[FSGS] Processing Scene: {scene_name}\n{'='*60}")
        
        # 1. Determine what COLMAP expects
        cam_width = get_camera_width(sparse_dir)
        if not cam_width:
            print(f"[ERROR] Could not read camera width for {scene_name}. Skipping.")
            continue
            
        # 2. Check available image directories
        lr_dir = scene_dir / "images"
        hr_dir = scene_dir / "images_SR" if (scene_dir / "images_SR").exists() else scene_dir / "images_sr"
            
        lr_width = get_image_width(lr_dir) if lr_dir.exists() else None
        hr_width = get_image_width(hr_dir) if hr_dir.exists() else None
        
        # 3. Dynamically match the correct folder
        if cam_width == hr_width:
            target_img_dir = hr_dir
            print(f"[INFO] Camera width ({cam_width}) matches HR images. Using {target_img_dir.name}")
        elif cam_width == lr_width:
            target_img_dir = lr_dir
            print(f"[INFO] Camera width ({cam_width}) matches LR images. Using {target_img_dir.name}")
        else:
            print(f"[ERROR] Camera width ({cam_width}) matches neither LR ({lr_width}) nor HR ({hr_width}).")
            continue

        # ==========================================================
        # 4. Convert all images to true JPEGs so OpenImageIO doesn't crash
        # ==========================================================
        colmap_img_dir = scene_dir / "images_fsgs_tmp"
        if colmap_img_dir.exists():
            shutil.rmtree(colmap_img_dir)
        colmap_img_dir.mkdir(parents=True, exist_ok=True)
        
        base_img_dir = scene_dir / "images" # The LR folder has the exact expected names
        for base_img in base_img_dir.iterdir():
            if not base_img.is_file(): continue
            
            # Find the matching HR/LR file regardless of extension (.png, .JPG, etc.)
            target_files = list(target_img_dir.glob(f"{base_img.stem}.*"))
            if target_files:
                source_path = target_files[0].absolute()
                target_path = colmap_img_dir / base_img.name
                
                # Actually convert the image to a real JPEG instead of just renaming it
                with Image.open(source_path) as img:
                    rgb_img = img.convert('RGB') # Strip alpha channels if PNG has them
                    rgb_img.save(target_path, format="JPEG", quality=100)
                
        # Point COLMAP to our perfectly formatted folder
        target_img_dir = colmap_img_dir
        # ==========================================================
        
        # 5. Clean workspace and run COLMAP
        dense_dir = scene_dir / "dense"
        if dense_dir.exists():
            shutil.rmtree(dense_dir)
        dense_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            print(f"[{scene_name}] 1/3 Undistorting images...")
            subprocess.run(["colmap", "image_undistorter", "--image_path", str(target_img_dir), "--input_path", str(sparse_dir), "--output_path", str(dense_dir), "--output_type", "COLMAP"], check=True)
            
            print(f"[{scene_name}] 2/3 Patch Match Stereo...")
            subprocess.run(["colmap", "patch_match_stereo", "--workspace_path", str(dense_dir), "--workspace_format", "COLMAP", "--PatchMatchStereo.geom_consistency", "true"], check=True)
            
            print(f"[{scene_name}] 3/3 Stereo Fusion (Geometric)...")
            subprocess.run(["colmap", "stereo_fusion", "--workspace_path", str(dense_dir), "--workspace_format", "COLMAP", "--input_type", "geometric", "--output_path", str(dense_dir / "fused.ply")], check=True)

            print(f"[SUCCESS] Finished {scene_name}. Scaffold saved to {dense_dir}/fused.ply")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] COLMAP failed on {scene_name} at step: {e.cmd}")
        finally:
            # Clean up the temporary image directory to keep the workspace tidy
            if colmap_img_dir.exists():
                shutil.rmtree(colmap_img_dir)

if __name__ == "__main__":
    main()