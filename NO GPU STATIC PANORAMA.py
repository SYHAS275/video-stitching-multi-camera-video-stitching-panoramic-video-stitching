#!/usr/bin/env python3
"""
Video Panorama Stitcher (CPU Version)
Single-file version combining video processing pipeline with Stitcher class.

Features:
- LAB color space matching with luminance protection
- Parallax-aware blending
- FFMPEG video encoding
"""

import numpy as np
import cv2
import subprocess
from datetime import datetime
import time
import os


# =============================================================================
# STITCHER CLASS
# =============================================================================

class Stitcher:
    def __init__(self):
        self.cachedH = None
        self.blend_start = None
        self.blend_end = None
        self.output_width = None
        self.output_height = None
        self.crop_top = None
        self.crop_bottom = None
        
        # Feature detector - CPU SIFT
        self.detector = cv2.SIFT_create(nfeatures=3000)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        print("Using CPU SIFT for feature detection")

    def stitch(self, images, ratio=0.75, reprojThresh=4.0):
        (imageB, imageA) = images  # left = B, right = A

        if self.cachedH is not None:
            return self.applyWarp(imageA, imageB, self.cachedH)

        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)

        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB,
                                ratio, reprojThresh)
        if M is None:
            print("Not enough matches to compute homography")
            return None

        (matches, H, status) = M
        if H is None:
            print("Homography is None")
            return None

        H = self._constrainHomography(H, imageA.shape, imageB.shape)
        self.cachedH = H.astype("float32")

        return self.applyWarp(imageA, imageB, self.cachedH)
    
    def _constrainHomography(self, H, shapeA, shapeB):
        H = H / H[2, 2]

        perspective_threshold = 0.002
        if abs(H[2, 0]) > perspective_threshold or abs(H[2, 1]) > perspective_threshold:
            H[2, 0] *= 0.5
            H[2, 1] *= 0.5
            H = H / H[2, 2]

        scale_x = np.sqrt(H[0, 0]**2 + H[1, 0]**2)
        scale_y = np.sqrt(H[0, 1]**2 + H[1, 1]**2)

        if scale_x > 1.3 or scale_x < 0.77:
            H[0, 0] /= scale_x
            H[1, 0] /= scale_x
        if scale_y > 1.3 or scale_y < 0.77:
            H[0, 1] /= scale_y
            H[1, 1] /= scale_y

        H = H / H[2, 2]
        return H

    def applyWarp(self, imageA, imageB, H):
        """
        CPU warp and blend.
        """
        h, w = imageB.shape[:2]
        
        # Calculate canvas size
        corners = np.float32([[0, 0], [imageA.shape[1], 0], 
                              [imageA.shape[1], imageA.shape[0]], [0, imageA.shape[0]]])
        warped_corners = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H)
        max_x = int(np.max(warped_corners[:, 0, 0]))
        canvas_width = min(max_x + 50, imageA.shape[1] + imageB.shape[1])

        # Warp
        warped = cv2.warpPerspective(imageA, H, (canvas_width, h),
            flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        # Find blend region (first frame only)
        if self.blend_start is None:
            overlap_start = w
            threshold_pixels = int(h * 0.15)

            for x in range(w):
                if np.count_nonzero(warped_gray[:, x] > 15) >= threshold_pixels:
                    overlap_start = x
                    break

            if overlap_start >= w:
                threshold_pixels = int(h * 0.05)
                for x in range(w):
                    if np.count_nonzero(warped_gray[:, x] > 15) >= threshold_pixels:
                        overlap_start = x
                        break

            if overlap_start > w - 30:
                overlap_start = max(0, w - 100)

            overlap_width = w - overlap_start
            blend_width = 100
            
            blend_center = overlap_start + overlap_width // 2
            self.blend_start = max(0, blend_center - blend_width // 2)
            self.blend_end = min(w, blend_center + blend_width // 2)
            
            valid_cols = np.where(np.any(warped_gray > 15, axis=0))[0]
            if len(valid_cols) > 0:
                self.output_width = min(valid_cols[-1] + 10, canvas_width)
            else:
                self.output_width = canvas_width
            self.output_height = h
            
            # Pre-compute gradient mask
            actual_blend_width = self.blend_end - self.blend_start
            if actual_blend_width > 0:
                mask_1d = np.linspace(0, 1, actual_blend_width, dtype=np.float32)
                mask_1d = mask_1d * mask_1d * (3 - 2 * mask_1d)
                self.gradient_mask = np.tile(mask_1d, (h, 1))
                self.gradient_mask_3 = np.dstack([self.gradient_mask] * 3)
            
            print(f"Blend region: {self.blend_start} to {self.blend_end}")
            print(f"Output size: {self.output_width}x{self.output_height}")

        blend_start = self.blend_start
        blend_end = self.blend_end
        actual_blend_width = blend_end - blend_start

        result = warped.copy()

        # === COLOR MATCHING ===
        sample_width = 150
        sample_start = max(0, blend_start - sample_width)
        sample_end = min(w, blend_end + sample_width)
        
        sample_left = imageB[:, sample_start:sample_end].copy()
        sample_right = warped[:h, sample_start:sample_end].copy()
        
        right_gray_sample = cv2.cvtColor(sample_right, cv2.COLOR_BGR2GRAY)
        
        valid_mask = (right_gray_sample > 50) & (right_gray_sample < 220)
        
        if np.sum(valid_mask) > 500:
            # LAB conversion
            left_lab = cv2.cvtColor(sample_left, cv2.COLOR_BGR2LAB).astype(np.float32)
            right_lab = cv2.cvtColor(sample_right, cv2.COLOR_BGR2LAB).astype(np.float32)
            
            transfer_params = []
            for c in range(3):
                left_vals = left_lab[:, :, c][valid_mask]
                right_vals = right_lab[:, :, c][valid_mask]
                
                if len(left_vals) > 100 and len(right_vals) > 100:
                    left_mean, left_std = np.mean(left_vals), np.std(left_vals)
                    right_mean, right_std = np.mean(right_vals), np.std(right_vals)
                    
                    if right_std > 1:
                        scale = np.clip(left_std / right_std, 0.8, 1.2)
                    else:
                        scale = 1.0
                    transfer_params.append((scale, right_mean, left_mean))
                else:
                    transfer_params.append((1.0, 0.0, 0.0))
            
            # Apply color correction
            warped_lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB).astype(np.float32)
            
            L_channel = warped_lab[:, :, 0]
            luminance_protection = np.ones_like(L_channel)
            
            dark_mask = L_channel < 60
            luminance_protection[dark_mask] = L_channel[dark_mask] / 60.0
            
            bright_mask = L_channel > 200
            luminance_protection[bright_mask] = (255 - L_channel[bright_mask]) / 55.0
            
            luminance_protection = np.clip(luminance_protection, 0.1, 1.0)
            
            for c in range(3):
                scale, right_mean, left_mean = transfer_params[c]
                corrected = (warped_lab[:, :, c] - right_mean) * scale + left_mean
                original = warped_lab[:, :, c]
                warped_lab[:, :, c] = original * (1 - luminance_protection) + corrected * luminance_protection
                warped_lab[:, :, c] = np.clip(warped_lab[:, :, c], 0, 255)
            
            warped = cv2.cvtColor(warped_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

        # === BLENDING ===
        result = warped.copy()
        result[:h, :blend_start] = imageB[:h, :blend_start]

        if actual_blend_width > 0:
            left_region = imageB[:, blend_start:blend_end].copy()
            right_region = result[:h, blend_start:blend_end].copy()
            
            # Grayscale for diff
            left_gray = cv2.cvtColor(left_region, cv2.COLOR_BGR2GRAY).astype(np.float32)
            right_gray = cv2.cvtColor(right_region, cv2.COLOR_BGR2GRAY).astype(np.float32)
            
            diff = np.abs(left_gray - right_gray)
            parallax_mask = (diff > 15).astype(np.float32)
            parallax_mask = cv2.dilate(parallax_mask, np.ones((40, 40), np.uint8))
            
            # Gaussian blur for parallax mask
            parallax_mask = cv2.GaussianBlur(parallax_mask, (31, 31), 0)
            
            final_mask = self.gradient_mask.copy()
            final_mask[parallax_mask > 0.5] = 0.0
            
            # Gaussian blur for final mask
            final_mask = cv2.GaussianBlur(final_mask, (11, 11), 0)
            
            final_mask_3 = np.dstack([final_mask] * 3)
            
            left_float = left_region.astype(np.float32)
            right_float = right_region.astype(np.float32)
            blended = left_float * (1.0 - final_mask_3) + right_float * final_mask_3
            blended = np.clip(blended, 0, 255).astype(np.uint8)
            
            result[:h, blend_start:blend_end] = blended

        # Fill holes
        left_part = imageB[:h, :w]
        result_part = result[:h, :w]

        left_gray_full = cv2.cvtColor(left_part, cv2.COLOR_BGR2GRAY)
        result_gray_full = cv2.cvtColor(result_part, cv2.COLOR_BGR2GRAY)

        holes = (result_gray_full < 10) & (left_gray_full > 10)
        if np.any(holes):
            holes_3 = np.dstack([holes] * 3)
            result_part[holes_3] = left_part[holes_3]

        result[:h, :w] = result_part
        
        result = self.fillFromSourceImages(result, imageA, imageB, H)
        result = result[:self.output_height, :self.output_width]
        
        return result

    def fillFromSourceImages(self, result, imageA, imageB, H):
        """Fill black regions using source images."""
        h, w_left = imageB.shape[:2]
        h_right, w_right = imageA.shape[:2]
        result_h, result_w = result.shape[:2]
        
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        black_mask = gray < 10
        
        if not np.any(black_mask):
            return result
        
        # Fill left side from imageB
        left_region_mask = black_mask[:h, :w_left]
        if np.any(left_region_mask):
            left_gray = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
            
            valid_fill = left_region_mask & (left_gray > 10)
            if np.any(valid_fill):
                valid_fill_3 = np.dstack([valid_fill] * 3)
                result[:h, :w_left][valid_fill_3] = imageB[valid_fill_3]
        
        # Fill right side from imageA (inverse warp)
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        black_mask = gray < 10
        
        if np.any(black_mask):
            try:
                H_inv = np.linalg.inv(H)
                black_coords = np.where(black_mask)
                if len(black_coords[0]) > 0:
                    y_coords = black_coords[0]
                    x_coords = black_coords[1]
                    
                    pts = np.float32(np.column_stack([x_coords, y_coords])).reshape(-1, 1, 2)
                    pts_transformed = cv2.perspectiveTransform(pts, H_inv)
                    pts_transformed = pts_transformed.reshape(-1, 2)
                    
                    for i in range(len(y_coords)):
                        src_x = int(round(pts_transformed[i, 0]))
                        src_y = int(round(pts_transformed[i, 1]))
                        
                        if 0 <= src_x < w_right and 0 <= src_y < h_right:
                            if np.mean(imageA[src_y, src_x]) > 10:
                                result[y_coords[i], x_coords[i]] = imageA[src_y, src_x]
            except np.linalg.LinAlgError:
                print("Could not invert homography for fill")
        
        return result

    def detectAndDescribe(self, image):
        """Feature detection using CPU SIFT."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        (kps, features) = self.detector.detectAndCompute(gray, None)
        kps = np.float32([kp.pt for kp in kps])
        return (kps, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
                       ratio, reprojThresh):
        """Feature matching."""
        if featuresA is None or featuresB is None:
            return None
        
        if len(kpsA) < 5 or len(kpsB) < 5:
            return None

        rawMatches = self.matcher.knnMatch(featuresA, featuresB, 2)

        matches = []
        for m_n in rawMatches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < n.distance * ratio:
                    matches.append((m.trainIdx, m.queryIdx))

        if len(matches) > 4:
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            try:
                M_affine, inliers = cv2.estimateAffinePartial2D(
                    ptsA, ptsB,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=reprojThresh,
                    confidence=0.99,
                    maxIters=2000
                )
                if M_affine is not None and inliers is not None:
                    H = np.vstack([M_affine, [0, 0, 1]])
                    status = inliers.ravel().astype(np.uint8)
                    return (matches, H, status)
            except:
                pass

            (H, status) = cv2.findHomography(ptsA, ptsB,
                                             cv2.RANSAC, reprojThresh)
            return (matches, H, status)

        return None


# =============================================================================
# VIDEO READER CLASS
# =============================================================================

class VideoReader:
    """Simple video reader wrapper."""
    def __init__(self, path):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        print(f"  Opened: {path}")
    
    def read(self):
        return self.cap.read()
    
    def get(self, prop):
        return self.cap.get(prop)
    
    def isOpened(self):
        return self.cap.isOpened()
    
    def release(self):
        self.cap.release()


# =============================================================================
# SYSTEM CHECK FUNCTION
# =============================================================================

def check_system():
    """Check available features."""
    print("=" * 60)
    print("SYSTEM CHECK")
    print("=" * 60)

    # Check FFMPEG
    ffmpeg_available = False
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'], 
            capture_output=True, text=True
        )
        if result.returncode == 0:
            ffmpeg_available = True
            print("✓ FFMPEG: Available")
    except:
        print("✗ FFMPEG: Not found")

    print("=" * 60)
    
    return ffmpeg_available


# =============================================================================
# MAIN VIDEO STITCHING PIPELINE
# =============================================================================

def stitch_videos(left_video_path, right_video_path, output_name=None, show_preview=True):
    """
    Main function to stitch two video files into a panorama.
    
    Args:
        left_video_path: Path to left video file
        right_video_path: Path to right video file
        output_name: Optional output filename (auto-generated if None)
        show_preview: Whether to show live preview window
    
    Returns:
        Path to output file
    """
    # System check
    ffmpeg_available = check_system()
    
    # Open videos
    print("\nOpening videos...")
    left_video = VideoReader(left_video_path)
    right_video = VideoReader(right_video_path)

    if not left_video.isOpened():
        raise IOError(f"Could not open {left_video_path}")
    if not right_video.isOpened():
        raise IOError(f"Could not open {right_video_path}")

    # Get video properties
    fps_left = left_video.get(cv2.CAP_PROP_FPS)
    fps_right = right_video.get(cv2.CAP_PROP_FPS)
    out_fps = min(fps_left, fps_right) if fps_left > 0 and fps_right > 0 else 30.0

    cap_temp = cv2.VideoCapture(left_video_path)
    total_frames = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_temp.release()

    print(f"FPS: {out_fps:.2f}, Frames: {total_frames}")

    # Create stitcher
    stitcher = Stitcher()

    # Output setup
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    temp_output = f"temp_stitched_{timestamp}.avi"
    
    if output_name:
        final_output = output_name
    else:
        final_output = f"stitched_output_{timestamp}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = None
    output_size = None

    frame_count = 0
    total_time = 0
    fps_history = []

    print("\n" + "=" * 60)
    print("STITCHING (CPU)")
    print("=" * 60)
    print("Operations:")
    print("  • Video decoding: CPU")
    print("  • Feature detection: CPU SIFT")
    print("  • Warp/Transform: CPU")
    print("  • Color conversion: CPU")
    print("  • Gaussian blur: CPU")
    print("  • Encoding: CPU")
    print("-" * 60)

    while True:
        frame_start = time.perf_counter()
        
        retL, left = left_video.read()
        retR, right = right_video.read()

        if not retL or not retR:
            print("\n\nVideos finished.")
            break

        # Ensure same size
        if left.shape[:2] != right.shape[:2]:
            right = cv2.resize(right, (left.shape[1], left.shape[0]))

        # Stitch
        stitch_start = time.perf_counter()
        stitched = stitcher.stitch([left, right])
        stitch_time = time.perf_counter() - stitch_start

        if stitched is None:
            print("Stitch failed, skipping frame.")
            continue

        # Create writer on first frame
        if writer is None:
            h, w = stitched.shape[:2]
            output_size = (w, h)
            writer = cv2.VideoWriter(temp_output, fourcc, out_fps, output_size)
            if not writer.isOpened():
                raise IOError("Could not create video writer")
            print(f"Output size: {w}x{h}")
            print("-" * 60)

        writer.write(stitched)
        frame_count += 1

        # Performance tracking
        frame_time = time.perf_counter() - frame_start
        total_time += frame_time
        current_fps = 1.0 / frame_time if frame_time > 0 else 0
        fps_history.append(current_fps)
        if len(fps_history) > 30:
            fps_history.pop(0)
        avg_fps = sum(fps_history) / len(fps_history)

        if total_frames > 0:
            progress = frame_count / total_frames
            eta = (total_frames - frame_count) / avg_fps if avg_fps > 0 else 0
            
            # Progress bar
            bar_width = 30
            filled = int(bar_width * progress)
            bar = "█" * filled + "░" * (bar_width - filled)
            
            print(f"\r[{bar}] {progress*100:5.1f}% | "
                  f"Frame {frame_count}/{total_frames} | "
                  f"FPS: {avg_fps:.1f} | "
                  f"ETA: {eta:.0f}s", end="", flush=True)
        else:
            print(f"\rFrame {frame_count} | "
                  f"FPS: {current_fps:.1f} (avg: {avg_fps:.1f}) | "
                  f"Stitch: {stitch_time*1000:.1f}ms", end="", flush=True)

        # Preview
        if show_preview:
            preview_height = 300
            
            scale_left = preview_height / left.shape[0]
            left_preview = cv2.resize(left, (int(left.shape[1] * scale_left), preview_height))
            
            scale_right = preview_height / right.shape[0]
            right_preview = cv2.resize(right, (int(right.shape[1] * scale_right), preview_height))
            
            scale_stitch = preview_height / stitched.shape[0]
            stitch_preview = cv2.resize(stitched, (int(stitched.shape[1] * scale_stitch), preview_height))
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(left_preview, "LEFT", (10, 30), font, 0.8, (0, 255, 0), 2)
            cv2.putText(right_preview, "RIGHT", (10, 30), font, 0.8, (0, 255, 0), 2)
            cv2.putText(stitch_preview, "STITCHED", (10, 30), font, 0.8, (0, 255, 0), 2)
            
            top_row = np.hstack([left_preview, right_preview])
            
            if top_row.shape[1] != stitch_preview.shape[1]:
                if top_row.shape[1] < stitch_preview.shape[1]:
                    pad_width = stitch_preview.shape[1] - top_row.shape[1]
                    top_row = np.hstack([top_row, np.zeros((preview_height, pad_width, 3), dtype=np.uint8)])
                else:
                    pad_width = top_row.shape[1] - stitch_preview.shape[1]
                    stitch_preview = np.hstack([stitch_preview, np.zeros((preview_height, pad_width, 3), dtype=np.uint8)])
            
            combined_preview = np.vstack([top_row, stitch_preview])
            
            cv2.imshow("Panorama Preview (q to quit)", combined_preview)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n\nUser stopped.")
                break

    # Cleanup
    left_video.release()
    right_video.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print(f"\n\nPhase 1 complete: {frame_count} frames in {total_time:.1f}s ({frame_count/total_time:.1f} FPS)")

    # Phase 2: Re-encode with FFMPEG
    output_file = temp_output
    if ffmpeg_available and os.path.exists(temp_output):
        print("\n" + "=" * 60)
        print("ENCODING (Phase 2: FFMPEG)")
        print("=" * 60)
        
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-i', temp_output,
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            final_output
        ]
        
        print(f"Running: {' '.join(ffmpeg_cmd[:6])}...")
        
        encode_start = time.perf_counter()
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        encode_time = time.perf_counter() - encode_start
        
        if result.returncode == 0:
            print(f"✓ Encoding complete in {encode_time:.1f}s")
            os.remove(temp_output)
            output_file = final_output
        else:
            print(f"✗ Encoding failed, keeping MJPG output")
            print(f"Error: {result.stderr[-500:] if result.stderr else 'Unknown'}")
            output_file = temp_output
    else:
        print("\nSkipping FFMPEG encoding (not available or no temp file)")
        output_file = temp_output

    # Final stats
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Frames processed: {frame_count}")
    print(f"Total time: {total_time:.1f}s ({frame_count/total_time:.1f} FPS)")
    print(f"Output: {output_file}")

    if os.path.exists(output_file):
        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"File size: {size_mb:.1f} MB")

    print("=" * 60)
    
    return output_file


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # --- Video files (EDIT THESE) ---
    LEFT_VIDEO = "air0.mp4"
    RIGHT_VIDEO = "air1.mp4"
    
    stitch_videos(
        left_video_path=LEFT_VIDEO,
        right_video_path=RIGHT_VIDEO,
        output_name=None,  # Auto-generates timestamped name, or set like "output.mp4"
        show_preview=True
    )