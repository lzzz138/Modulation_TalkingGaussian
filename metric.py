import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
import insightface
from insightface.app import FaceAnalysis
from scipy import linalg
import os

class VideoMetricsCalculator:
    def __init__(self, device='cuda'):
        self.device = device
        # 初始化 LPIPS 模型
        self.loss_fn_alex = lpips.LPIPS(net='alex').to(device)
        # 初始化 FID 所需的 Inception V3 模型
        self.inception_model = self._get_inception_model().to(device)
        self.inception_model.eval()
        # 初始化 InsightFace (ArcFace) 用于 CSIM
        self.face_analyzer = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.face_analyzer.prepare(ctx_id=0 if device == 'cuda' else -1, det_size=(640, 640))

    def _get_inception_model(self):
        """加载预训练的 Inception V3 用于计算 FID"""
        model = models.inception_v3(pretrained=True, transform_input=False)
        model.fc = torch.nn.Identity()  # 移除分类层，获取特征
        return model

    def load_video_frames(self, video_path, max_frames=None):
        """读取视频帧，返回 BGR 格式的 numpy 数组列表"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # 转换为 RGB 用于后续处理
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            if max_frames and len(frames) >= max_frames:
                break
        cap.release()
        return frames

    def preprocess_for_lpips(self, frame):
        """将图像预处理为 LPIPS 需要的 Tensor (-1 到 1)"""
        img = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1
        return img.to(self.device)

    def preprocess_delta_for_lpips(self, frame_cur, frame_prev):
        """将相邻帧差分预处理为 LPIPS Tensor ([-1, 1])"""
        delta = (frame_cur.astype(np.float32) - frame_prev.astype(np.float32)) / 255.0
        img = torch.from_numpy(delta).permute(2, 0, 1).unsqueeze(0).float()
        return img.to(self.device)

    def preprocess_for_inception(self, frame):
        """将图像预处理为 Inception 需要的 Tensor (0-1 归一化，调整大小)"""
        img = cv2.resize(frame, (299, 299))
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        return img.to(self.device)

    def calculate_psnr_ssim(self, frames_gen, frames_real):
        """计算 PSNR 和 SSIM (逐帧平均)"""
        psnr_vals = []
        ssim_vals = []
        min_len = min(len(frames_gen), len(frames_real))
        
        for i in range(min_len):
            p = psnr(frames_real[i], frames_gen[i])
            s = ssim(frames_real[i], frames_gen[i], channel_axis=2, data_range=255)
            psnr_vals.append(p)
            ssim_vals.append(s)
            
        return np.mean(psnr_vals), np.mean(ssim_vals)

    def calculate_lpips(self, frames_gen, frames_real):
        """计算 LPIPS (逐帧平均)"""
        lpips_vals = []
        min_len = min(len(frames_gen), len(frames_real))
        
        with torch.no_grad():
            for i in range(min_len):
                img1 = self.preprocess_for_lpips(frames_real[i])
                img2 = self.preprocess_for_lpips(frames_gen[i])
                d = self.loss_fn_alex(img1, img2)
                lpips_vals.append(d.item())
                
        return np.mean(lpips_vals)

    def calculate_tlpips(self, frames_gen, frames_real):
        """计算 tLPIPS: 相邻帧差分的 LPIPS，越低越好"""
        min_len = min(len(frames_gen), len(frames_real))
        if min_len < 2:
            return 0.0

        tlpips_vals = []
        with torch.no_grad():
            for i in range(1, min_len):
                real_delta = self.preprocess_delta_for_lpips(frames_real[i], frames_real[i - 1])
                gen_delta = self.preprocess_delta_for_lpips(frames_gen[i], frames_gen[i - 1])
                d = self.loss_fn_alex(real_delta, gen_delta)
                tlpips_vals.append(d.item())

        return np.mean(tlpips_vals) if tlpips_vals else 0.0

    def _get_inception_features(self, frames):
        """提取一组图像的特征用于 FID"""
        features = []
        with torch.no_grad():
            for frame in frames:
                inp = self.preprocess_for_inception(frame)
                feat = self.inception_model(inp)
                features.append(feat.squeeze().cpu().numpy())
        return np.vstack(features)

    def calculate_fid(self, frames_gen, frames_real):
        """计算 FID (基于整个视频帧集合的分布)"""
        act1 = self._get_inception_features(frames_real)
        act2 = self._get_inception_features(frames_gen)
        
        mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
        
        ssdiff = np.sum((mu1 - mu2)**2)
        covmean = linalg.sqrtm(sigma1.dot(sigma2))
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)
        return fid

    def calculate_csim(self, frames_gen, frames_real):
        csim_vals = []
        min_len = min(len(frames_gen), len(frames_real))
        
        for i in range(min_len):
            faces_real = self.face_analyzer.get(frames_real[i])
            faces_gen = self.face_analyzer.get(frames_gen[i])
            
            if len(faces_real) > 0 and len(faces_gen) > 0:
                emb_real = faces_real[0]['embedding']
                emb_gen = faces_gen[0]['embedding']
                
                # 🔧 关键修复：L2 归一化后计算余弦相似度
                emb_real_norm = emb_real / (np.linalg.norm(emb_real) + 1e-8)
                emb_gen_norm = emb_gen / (np.linalg.norm(emb_gen) + 1e-8)
                
                # 点积 = 余弦相似度（因为已归一化）
                similarity = np.dot(emb_real_norm, emb_gen_norm)
                
                # 确保结果在 [-1, 1] 范围内，并映射到 [0, 1]（可选）
                similarity = np.clip(similarity, -1, 1)
                # 如果需要 [0,1] 范围：similarity = (similarity + 1) / 2
                
                csim_vals.append(similarity)
            else:
                continue
                
        return np.mean(csim_vals) if csim_vals else 0.0

    def calculate_csim_reference(self, frames_gen, frames_real_ref):
        csim_vals = []
        
        faces_ref = self.face_analyzer.get(frames_real_ref[0])
        if len(faces_ref) == 0:
            print("警告：参考帧未检测到人脸")
            return 0.0
        
        emb_ref = faces_ref[0]['embedding']
        # 🔧 归一化参考嵌入
        emb_ref_norm = emb_ref / (np.linalg.norm(emb_ref) + 1e-8)
        
        for frame in frames_gen:
            faces_gen = self.face_analyzer.get(frame)
            if len(faces_gen) > 0:
                emb_gen = faces_gen[0]['embedding']
                emb_gen_norm = emb_gen / (np.linalg.norm(emb_gen) + 1e-8)
                
                similarity = np.dot(emb_ref_norm, emb_gen_norm)
                similarity = np.clip(similarity, -1, 1)
                csim_vals.append(similarity)
        
        return np.mean(csim_vals) if csim_vals else 0.0

    def _compute_flow(self, frame_prev, frame_cur):
        prev_gray = cv2.cvtColor(frame_prev, cv2.COLOR_RGB2GRAY)
        cur_gray = cv2.cvtColor(frame_cur, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, cur_gray, None,
            pyr_scale=0.5, levels=3, winsize=15, iterations=3,
            poly_n=5, poly_sigma=1.2, flags=0
        )
        return flow

    def _warp_frame_with_flow(self, frame, flow):
        h, w = flow.shape[:2]
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x + flow[..., 0]).astype(np.float32)
        map_y = (grid_y + flow[..., 1]).astype(np.float32)
        warped = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return warped

    def calculate_flow_warp_error(self, frames_gen, frames_real):
        """计算基于真值光流的生成视频时间一致性误差，越低越好"""
        min_len = min(len(frames_gen), len(frames_real))
        if min_len < 2:
            return 0.0

        errs = []
        for i in range(1, min_len):
            flow_real = self._compute_flow(frames_real[i - 1], frames_real[i])
            gen_prev_warped = self._warp_frame_with_flow(frames_gen[i - 1], flow_real)
            err = np.mean(np.abs(frames_gen[i].astype(np.float32) - gen_prev_warped.astype(np.float32))) / 255.0
            errs.append(err)
        return np.mean(errs) if errs else 0.0

    def _extract_face_kps(self, frame):
        faces = self.face_analyzer.get(frame)
        if len(faces) == 0:
            return None
        face = faces[0]
        if hasattr(face, "kps") and face.kps is not None:
            kps = face.kps
        else:
            try:
                kps = face['kps']
            except Exception:
                kps = None
        if kps is None:
            return None
        return np.array(kps, dtype=np.float32)

    def _sequence_jitter(self, kps_seq):
        if len(kps_seq) < 3:
            return 0.0
        arr = np.stack(kps_seq, axis=0)  # [T, K, 2]
        vel = np.diff(arr, axis=0)
        acc = np.diff(vel, axis=0)
        return float(np.linalg.norm(acc, axis=-1).mean())

    def calculate_landmark_jitter(self, frames_gen, frames_real):
        """计算关键点抖动：返回(生成抖动, 真值抖动, 抖动差值)，越低越好"""
        min_len = min(len(frames_gen), len(frames_real))
        if min_len < 3:
            return 0.0, 0.0, 0.0

        gen_seq = []
        real_seq = []
        for i in range(min_len):
            kps_gen = self._extract_face_kps(frames_gen[i])
            kps_real = self._extract_face_kps(frames_real[i])
            if kps_gen is None or kps_real is None:
                continue
            gen_seq.append(kps_gen)
            real_seq.append(kps_real)

        if len(gen_seq) < 3 or len(real_seq) < 3:
            return 0.0, 0.0, 0.0

        gen_jitter = self._sequence_jitter(gen_seq)
        real_jitter = self._sequence_jitter(real_seq)
        jitter_gap = abs(gen_jitter - real_jitter)
        return gen_jitter, real_jitter, jitter_gap

def main():
    # 配置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    calculator = VideoMetricsCalculator(device=device)
    
    # 视频路径 (请替换为你的实际路径)
    real_video_path = "/home/lzq/paperCode/talkinghead/TalkingGaussian/output/exp1/test/ours_None/gt/out.mp4"
    gen_video_path = "/home/lzq/paperCode/talkinghead/TalkingGaussian/output/exp1/test/ours_None/renders/out.mp4"
    if not os.path.exists(real_video_path) or not os.path.exists(gen_video_path):
        print("错误：请修改代码中的视频路径！")
        return

    print("正在加载视频帧...")
    # 根据需求调整 max_frames
    # FID 需要较多帧才有统计意义，建议至少 50-100 帧
    frames_real = calculator.load_video_frames(real_video_path)
    frames_gen = calculator.load_video_frames(gen_video_path)
    
    if len(frames_real) == 0 or len(frames_gen) == 0:
        print("错误：无法读取视频帧")
        return

    print(f"加载完成。真实帧数：{len(frames_real)}, 生成帧数：{len(frames_gen)}")

    # 1. PSNR & SSIM
    print("\n" + "="*50)
    print("计算 PSNR 和 SSIM...")
    avg_psnr, avg_ssim = calculator.calculate_psnr_ssim(frames_gen, frames_real)
    print(f"PSNR: {avg_psnr:.4f} dB")
    print(f"SSIM: {avg_ssim:.4f}")

    # 2. LPIPS
    print("\n" + "="*50)
    print("计算 LPIPS...")
    avg_lpips = calculator.calculate_lpips(frames_gen, frames_real)
    print(f"LPIPS: {avg_lpips:.4f}")

    # 3. FID
    print("\n" + "="*50)
    print("计算 FID (这可能需要一点时间)...")
    fid_score = calculator.calculate_fid(frames_gen, frames_real)
    print(f"FID: {fid_score:.4f}")

    # 4. CSIM (两种方式)
    print("\n" + "="*50)
    print("计算 CSIM (使用 InsightFace ArcFace)...")
    
    # 方式 1: 逐帧比较 (生成帧 i vs 真实帧 i)
    avg_csim_frame = calculator.calculate_csim(frames_gen, frames_real)
    print(f"CSIM (帧间比较): {avg_csim_frame:.4f}")
    
    # 方式 2: 与参考身份比较 (所有生成帧 vs 真实视频第一帧)
    # 这种方式更能反映身份保持能力
    avg_csim_ref = calculator.calculate_csim_reference(frames_gen, frames_real)
    print(f"CSIM (参考身份): {avg_csim_ref:.4f}")

    # 5. tLPIPS (temporal)
    print("\n" + "="*50)
    print("计算 tLPIPS (时间感知 LPIPS)...")
    avg_tlpips = calculator.calculate_tlpips(frames_gen, frames_real)
    print(f"tLPIPS: {avg_tlpips:.4f}")

    # 6. Optical-flow warp error
    print("\n" + "="*50)
    print("计算光流时间一致性误差...")
    flow_warp_err = calculator.calculate_flow_warp_error(frames_gen, frames_real)
    print(f"Flow Warp Error: {flow_warp_err:.4f}")

    # 7. Landmark jitter
    print("\n" + "="*50)
    print("计算关键点抖动...")
    gen_jitter, real_jitter, jitter_gap = calculator.calculate_landmark_jitter(frames_gen, frames_real)
    print(f"Landmark Jitter (Gen): {gen_jitter:.4f}")
    print(f"Landmark Jitter (Real): {real_jitter:.4f}")
    print(f"Landmark Jitter Gap: {jitter_gap:.4f}")
    
    print("\n" + "="*50)
    print("评估完成！")
    print("="*50)
    print("指标说明:")
    print("  PSNR: 越高越好 (>30 较好)")
    print("  SSIM: 越高越好 (>0.9 较好)")
    print("  LPIPS: 越低越好 (<0.2 较好)")
    print("  FID: 越低越好 (<50 较好，取决于数据集)")
    print("  CSIM: 越高越好 (>0.6 较好，表示身份保持良好)")
    print("  tLPIPS: 越低越好（时间闪烁越少）")
    print("  Flow Warp Error: 越低越好（时间/运动一致性越强）")
    print("  Landmark Jitter Gap: 越低越好（越接近真值运动平滑度）")
    print("="*50)

if __name__ == "__main__":
    main()
