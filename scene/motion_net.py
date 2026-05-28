import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder


def linear_no_cublas(x, layer):
    y = (x.unsqueeze(-2) * layer.weight).sum(dim=-1)
    if layer.bias is not None:
        y = y + layer.bias
    return y


# Audio feature extractor
class AudioAttNet(nn.Module):
    def __init__(self, dim_aud=64, seq_len=8):
        super(AudioAttNet, self).__init__()
        self.seq_len = seq_len
        self.dim_aud = dim_aud
        self.attentionConvNet = nn.Sequential(  # b x subspace_dim x seq_len
            nn.Conv1d(self.dim_aud, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(8, 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(4, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(2, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True)
        )
        self.attentionNet = nn.Sequential(
            nn.Linear(in_features=self.seq_len, out_features=self.seq_len, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x: [1, seq_len, dim_aud]
        y = x.permute(0, 2, 1)  # [1, dim_aud, seq_len]
        y = self.attentionConvNet(y) 
        y = y.view(1, self.seq_len)
        y = linear_no_cublas(y, self.attentionNet[0])
        y = self.attentionNet[1](y).view(1, self.seq_len, 1)
        return torch.sum(y * x, dim=1) # [1, dim_aud]


# Audio feature extractor
class AudioNet(nn.Module):
    def __init__(self, dim_in=29, dim_aud=64, win_size=16):
        super(AudioNet, self).__init__()
        self.win_size = win_size
        self.dim_aud = dim_aud
        self.encoder_conv = nn.Sequential(  # n x 29 x 16
            nn.Conv1d(dim_in, 32, kernel_size=3, stride=2, padding=1, bias=True),  # n x 32 x 8
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1, bias=True),  # n x 32 x 4
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1, bias=True),  # n x 64 x 2
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),  # n x 64 x 1
            nn.LeakyReLU(0.02, True),
        )
        self.encoder_fc1 = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(0.02, True),
            nn.Linear(64, dim_aud),
        )

    def forward(self, x):
        half_w = int(self.win_size/2)
        x = x[:, :, 8-half_w:8+half_w]
        x = self.encoder_conv(x).squeeze(-1)
        x = linear_no_cublas(x, self.encoder_fc1[0])
        x = self.encoder_fc1[1](x)
        x = linear_no_cublas(x, self.encoder_fc1[2])
        return x


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=False))

        self.net = nn.ModuleList(net)
    
    def forward(self, x):
        for l in range(self.num_layers):
            x = linear_no_cublas(x, self.net[l])
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
                # x = F.dropout(x, p=0.1, training=self.training)
                
        return x


class NoCublasProjector(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super().__init__()
        self.net = nn.ModuleList([
            nn.Linear(dim_in, dim_hidden),
            nn.Linear(dim_hidden, dim_out),
        ])

    def forward(self, x):
        x = linear_no_cublas(x, self.net[0])
        x = F.relu(x, inplace=True)
        x = linear_no_cublas(x, self.net[1])
        return x


class AudioUpperFaceGeometryModulator(nn.Module):
    def __init__(self,
                 audio_dim,
                 upper_dim,
                 plane_dim,
                 map_res=16,
                 query_dim=32,
                 hidden_dim=64,
                 num_planes=3):
        super().__init__()

        self.num_planes = num_planes
        self.plane_dim = plane_dim
        self.map_res = map_res
        self.query_dim = query_dim

        self.plane_query = nn.Parameter(torch.randn(1, num_planes, query_dim, map_res, map_res) * 0.02)

        self.audio_proj = NoCublasProjector(audio_dim, hidden_dim, num_planes * query_dim)
        self.upper_proj = NoCublasProjector(upper_dim, hidden_dim, num_planes * query_dim)

        self.audio_conv = nn.Sequential(
            nn.Conv2d(query_dim, query_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(query_dim, query_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.upper_conv = nn.Sequential(
            nn.Conv2d(query_dim, query_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(query_dim, query_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.gate_conv = nn.Sequential(
            nn.Conv2d(query_dim * 3, query_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(query_dim, 2, kernel_size=1),
        )
        self.out_conv = nn.Conv2d(query_dim, 2 * plane_dim, kernel_size=1)

        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, audio_feat, upper_feat):
        if audio_feat.dim() == 1:
            audio_feat = audio_feat.unsqueeze(0)
        if upper_feat.dim() == 1:
            upper_feat = upper_feat.unsqueeze(0)

        batch_size = audio_feat.shape[0]
        query = self.plane_query.expand(batch_size, -1, -1, -1, -1)

        audio_bias = self.audio_proj(audio_feat).view(batch_size, self.num_planes, self.query_dim, 1, 1)
        upper_bias = self.upper_proj(upper_feat).view(batch_size, self.num_planes, self.query_dim, 1, 1)

        query_flat = query.reshape(batch_size * self.num_planes, self.query_dim, self.map_res, self.map_res)
        audio_query = (query + audio_bias).reshape(batch_size * self.num_planes, self.query_dim, self.map_res, self.map_res)
        upper_query = (query + upper_bias).reshape(batch_size * self.num_planes, self.query_dim, self.map_res, self.map_res)

        audio_candidate = self.audio_conv(audio_query)
        upper_candidate = self.upper_conv(upper_query)
        gate_logits = self.gate_conv(torch.cat([query_flat, audio_candidate, upper_candidate], dim=1))
        gate = torch.softmax(gate_logits, dim=1)

        fused = gate[:, 0:1] * audio_candidate + gate[:, 1:2] * upper_candidate
        gamma_beta = self.out_conv(fused)
        gamma_beta = gamma_beta.view(
            batch_size,
            self.num_planes,
            2,
            self.plane_dim,
            self.map_res,
            self.map_res,
        )
        gate = gate.view(batch_size, self.num_planes, 2, self.map_res, self.map_res)
        return gamma_beta, gate


class MotionNetwork(nn.Module):
    def __init__(self,
                 audio_dim = 32,
                 ind_dim = 0,
                 args = None,
                 ):
        super(MotionNetwork, self).__init__()

        if 'esperanto' in args.audio_extractor:
            self.audio_in_dim = 44
        elif 'deepspeech' in args.audio_extractor:
            self.audio_in_dim = 29
        elif 'hubert' in args.audio_extractor:
            self.audio_in_dim = 1024
        else:
            raise NotImplementedError
    
        self.bound = 0.15
        self.exp_eye = True

        
        self.individual_dim = ind_dim
        if self.individual_dim > 0:
            self.individual_codes = nn.Parameter(torch.randn(10000, self.individual_dim) * 0.1) 

        # audio network
        self.audio_dim = audio_dim
        self.audio_net = AudioNet(self.audio_in_dim, self.audio_dim)

        self.audio_att_net = AudioAttNet(self.audio_dim)

        # DYNAMIC PART
        self.num_levels = 12
        self.level_dim = 1
        self.encoder_xy, self.in_dim_xy = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=16, log2_hashmap_size=17, desired_resolution=256 * self.bound)
        self.encoder_yz, self.in_dim_yz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=16, log2_hashmap_size=17, desired_resolution=256 * self.bound)
        self.encoder_xz, self.in_dim_xz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=16, log2_hashmap_size=17, desired_resolution=256 * self.bound)

        self.in_dim = self.in_dim_xy + self.in_dim_yz + self.in_dim_xz


        self.num_layers = 3       
        self.hidden_dim = 64

        self.exp_in_dim = 6 - 1
        self.eye_dim = 6 if self.exp_eye else 0
        self.exp_encode_net = MLP(self.exp_in_dim, self.eye_dim - 1, 16, 2)

        self.eye_att_net = MLP(self.in_dim, self.eye_dim, 16, 2)
        self.geometry_modulator = AudioUpperFaceGeometryModulator(
            audio_dim=self.audio_dim,
            upper_dim=self.eye_dim,
            plane_dim=self.in_dim_xy,
            map_res=getattr(args, 'geometry_mod_map_res', 16),
        )
        self.geometry_modulation_strength = 1.0
        self.condition_residual_scale = getattr(args, 'geometry_mod_condition_scale', 0.1)

        # rot: 4   xyz: 3   opac: 1  scale: 3
        self.out_dim = 11
        self.sigma_net = MLP(self.in_dim + self.audio_dim + self.eye_dim + self.individual_dim, self.out_dim, self.hidden_dim, self.num_layers)
        
        self.aud_ch_att_net = MLP(self.in_dim, self.audio_dim, 32, 2)


    @staticmethod
    @torch.jit.script
    def split_xyz(x):
        xy, yz, xz = x[:, :-1], x[:, 1:], torch.cat([x[:,:1], x[:,-1:]], dim=-1)
        return xy, yz, xz


    def encode_planes(self, xyz, bound):
        # x: [N, 3], in [-bound, bound]
        xy, yz, xz = self.split_xyz(xyz)
        feat_xy = self.encoder_xy(xy, bound=bound)
        feat_yz = self.encoder_yz(yz, bound=bound)
        feat_xz = self.encoder_xz(xz, bound=bound)

        return (xy, yz, xz), (feat_xy, feat_yz, feat_xz)


    def encode_x(self, xyz, bound):
        _, plane_feats = self.encode_planes(xyz, bound)
        return torch.cat(plane_feats, dim=-1)
    

    def encode_audio(self, a):
        # a: [1, 29, 16] or [8, 29, 16], audio features from deepspeech
        # if emb, a should be: [1, 16] or [8, 16]

        # fix audio traininig
        if a is None: return None

        enc_a = self.audio_net(a) # [1/8, 64]
        enc_a = self.audio_att_net(enc_a.unsqueeze(0)) # [1, 64]
            
        return enc_a


    def encode_upper_face(self, e):
        enc_e = self.exp_encode_net(e[:-1])
        enc_e = torch.cat([enc_e, e[-1:]], dim=-1)
        return enc_e.unsqueeze(0)


    @staticmethod
    def modulation_tv_loss(gamma_beta):
        loss = (gamma_beta[..., 1:, :] - gamma_beta[..., :-1, :]).abs().mean()
        loss = loss + (gamma_beta[..., :, 1:] - gamma_beta[..., :, :-1]).abs().mean()
        return loss


    @staticmethod
    def gate_overlap_loss(gate):
        return (gate[:, :, 0] * gate[:, :, 1]).mean()


    @staticmethod
    def sample_modulation(modulation_map, coords, bound):
        # modulation_map: [1, 2, C, H, W], coords: [N, 2] in [-bound, bound]
        channels = modulation_map.shape[2]
        sample_map = modulation_map.reshape(1, 2 * channels, modulation_map.shape[-2], modulation_map.shape[-1])
        grid = (coords / bound).clamp(-1, 1).view(1, -1, 1, 2)
        samples = F.grid_sample(sample_map, grid, mode='bilinear', padding_mode='border', align_corners=True)
        samples = samples.squeeze(0).squeeze(-1).transpose(0, 1)
        gamma, beta = samples.split(channels, dim=-1)
        return gamma, beta


    def apply_geometry_modulation(self, plane_coords, plane_feats, gamma_beta):
        strength = self.geometry_modulation_strength
        modulated_feats = []
        for plane_idx, (coords, feat) in enumerate(zip(plane_coords, plane_feats)):
            gamma, beta = self.sample_modulation(gamma_beta[:, plane_idx], coords, self.bound)
            feat_dyn = (1 + strength * torch.tanh(gamma)) * feat + strength * beta
            modulated_feats.append(feat_dyn)
        return modulated_feats


    def set_geometry_modulation_strength(self, strength):
        self.geometry_modulation_strength = max(0.0, min(1.0, float(strength)))


    def forward(self, x, a, e=None, c=None):
        # x: [N, 3], in [-bound, bound]
        plane_coords, plane_feats = self.encode_planes(x, bound=self.bound)

        enc_a = self.encode_audio(a)
        enc_e_global = self.encode_upper_face(e)

        gamma_beta, gate = self.geometry_modulator(enc_a, enc_e_global)
        plane_feats_dyn = self.apply_geometry_modulation(plane_coords, plane_feats, gamma_beta)
        enc_x_dyn = torch.cat(plane_feats_dyn, dim=-1)

        enc_a = enc_a.repeat(enc_x_dyn.shape[0], 1)
        aud_ch_att = self.aud_ch_att_net(enc_x_dyn)
        enc_w = enc_a * aud_ch_att
        
        eye_att = torch.relu(self.eye_att_net(enc_x_dyn))
        enc_e = enc_e_global * eye_att
        if c is not None:
            c = c.repeat(enc_x_dyn.shape[0], 1)
            h = torch.cat([enc_x_dyn, self.condition_residual_scale * enc_w, self.condition_residual_scale * enc_e, c], dim=-1)
        else:
            h = torch.cat([enc_x_dyn, self.condition_residual_scale * enc_w, self.condition_residual_scale * enc_e], dim=-1)

        h = self.sigma_net(h)

        d_xyz = h[..., :3] * 1e-2
        d_rot = h[..., 3:7]
        d_opa = h[..., 7:8]
        d_scale = h[..., 8:11]
        return {
            'd_xyz': d_xyz,
            'd_rot': d_rot,
            'd_opa': d_opa,
            'd_scale': d_scale,
            'ambient_aud' : aud_ch_att.norm(dim=-1, keepdim=True),
            'ambient_eye' : eye_att.norm(dim=-1, keepdim=True),
            'modulation_tv': self.modulation_tv_loss(gamma_beta),
            'modulation_gate_overlap': self.gate_overlap_loss(gate),
        }


    # optimizer utils
    def get_params(self, lr, lr_net, wd=0):

        params = [
            {'params': self.audio_net.parameters(), 'lr': lr_net, 'weight_decay': wd}, 
            {'params': self.encoder_xy.parameters(), 'lr': lr},
            {'params': self.encoder_yz.parameters(), 'lr': lr},
            {'params': self.encoder_xz.parameters(), 'lr': lr},
            {'params': self.geometry_modulator.parameters(), 'lr': lr_net, 'weight_decay': wd},
            {'params': self.sigma_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
        ]
        params.append({'params': self.audio_att_net.parameters(), 'lr': lr_net * 5, 'weight_decay': 0.0001})
        if self.individual_dim > 0:
            params.append({'params': self.individual_codes, 'lr': lr_net, 'weight_decay': wd})
        
        params.append({'params': self.aud_ch_att_net.parameters(), 'lr': lr_net, 'weight_decay': wd})
        params.append({'params': self.eye_att_net.parameters(), 'lr': lr_net, 'weight_decay': wd})
        params.append({'params': self.exp_encode_net.parameters(), 'lr': lr_net, 'weight_decay': wd})

        return params




class MouthMotionNetwork(nn.Module):
    def __init__(self,
                 audio_dim = 32,
                 ind_dim = 0,
                 args = None,
                 ):
        super(MouthMotionNetwork, self).__init__()

        if 'esperanto' in args.audio_extractor:
            self.audio_in_dim = 44
        elif 'deepspeech' in args.audio_extractor:
            self.audio_in_dim = 29
        elif 'hubert' in args.audio_extractor:
            self.audio_in_dim = 1024
        else:
            raise NotImplementedError
        
        
        self.bound = 0.15

        
        self.individual_dim = ind_dim
        if self.individual_dim > 0:
            self.individual_codes = nn.Parameter(torch.randn(10000, self.individual_dim) * 0.1) 

        # audio network
        self.audio_dim = audio_dim
        self.audio_net = AudioNet(self.audio_in_dim, self.audio_dim)

        self.audio_att_net = AudioAttNet(self.audio_dim)

        # DYNAMIC PART
        self.num_levels = 12
        self.level_dim = 1
        self.encoder_xy, self.in_dim_xy = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=64, log2_hashmap_size=17, desired_resolution=384 * self.bound)
        self.encoder_yz, self.in_dim_yz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=64, log2_hashmap_size=17, desired_resolution=384 * self.bound)
        self.encoder_xz, self.in_dim_xz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=64, log2_hashmap_size=17, desired_resolution=384 * self.bound)

        self.in_dim = self.in_dim_xy + self.in_dim_yz + self.in_dim_xz

        ## sigma network
        self.num_layers = 3
        self.hidden_dim = 32

        self.out_dim = 3
        self.sigma_net = MLP(self.in_dim + self.audio_dim + self.individual_dim, self.out_dim, self.hidden_dim, self.num_layers)
        
        self.aud_ch_att_net = MLP(self.in_dim, self.audio_dim, 32, 2)
    

    def encode_audio(self, a):
        # a: [1, 29, 16] or [8, 29, 16], audio features from deepspeech
        # if emb, a should be: [1, 16] or [8, 16]

        # fix audio traininig
        if a is None: return None

        enc_a = self.audio_net(a) # [1/8, 64]
        enc_a = self.audio_att_net(enc_a.unsqueeze(0)) # [1, 64]
            
        return enc_a
    

    @staticmethod
    @torch.jit.script
    def split_xyz(x):
        xy, yz, xz = x[:, :-1], x[:, 1:], torch.cat([x[:,:1], x[:,-1:]], dim=-1)
        return xy, yz, xz


    def encode_x(self, xyz, bound):
        # x: [N, 3], in [-bound, bound]
        N, M = xyz.shape
        xy, yz, xz = self.split_xyz(xyz)
        feat_xy = self.encoder_xy(xy, bound=bound)
        feat_yz = self.encoder_yz(yz, bound=bound)
        feat_xz = self.encoder_xz(xz, bound=bound)
        
        return torch.cat([feat_xy, feat_yz, feat_xz], dim=-1)


    def forward(self, x, a):
        # x: [N, 3], in [-bound, bound]
        enc_x = self.encode_x(x, bound=self.bound)

        enc_a = self.encode_audio(a)
        enc_w = enc_a.repeat(enc_x.shape[0], 1)
        # aud_ch_att = self.aud_ch_att_net(enc_x)
        # enc_w = enc_a * aud_ch_att

        h = torch.cat([enc_x, enc_w], dim=-1)

        h = self.sigma_net(h)

        d_xyz = h * 1e-2
        d_xyz[..., 0] = d_xyz[..., 0] / 5
        d_xyz[..., 2] = d_xyz[..., 2] / 5
        return {
            'd_xyz': d_xyz,
            # 'ambient_aud' : aud_ch_att.norm(dim=-1, keepdim=True),
        }


    # optimizer utils
    def get_params(self, lr, lr_net, wd=0):

        params = [
            {'params': self.audio_net.parameters(), 'lr': lr_net, 'weight_decay': wd}, 
            {'params': self.encoder_xy.parameters(), 'lr': lr},
            {'params': self.encoder_yz.parameters(), 'lr': lr},
            {'params': self.encoder_xz.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
        ]
        params.append({'params': self.audio_att_net.parameters(), 'lr': lr_net * 5, 'weight_decay': 0.0001})
        if self.individual_dim > 0:
            params.append({'params': self.individual_codes, 'lr': lr_net, 'weight_decay': wd})
        
        params.append({'params': self.aud_ch_att_net.parameters(), 'lr': lr_net, 'weight_decay': wd})

        return params
