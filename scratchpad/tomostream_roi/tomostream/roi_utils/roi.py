import torch, os, time, h5py

import numpy as np
from torch.utils.data import DataLoader, Dataset
from skimage import measure
from skimage.measure import regionprops
import torch.nn.functional as F
import pandas as pd 

from roi_utils.model import unet

def rescale_vol_for_NN(vol, pmin=2, pmax=97):
    assert len(vol.shape)==3, 'only support 3D'
    _min, _max = np.percentile(vol, (pmin, pmax), axis=(1,2))
    _min = _min[:, None, None]
    _max = _max[:, None, None]
    _vols= (vol - _min) / (_max - _min)
    return _vols.astype(vol.dtype)

class SSegDataset(Dataset):
    def __init__(self, vol):
        self.images = vol
        self.len = self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx, np.newaxis].astype(np.float32)

    def __len__(self):
        return self.len

def coord_re_proj(c, stride, ksz, maxs):
    return min(round(c * stride + ksz // 2), maxs)

def load_seg_nn(mdl_fn):
    model = unet()
    model.load_state_dict(torch.load(mdl_fn, map_location=torch.device('cpu')))
    if torch.cuda.is_available():
        model = model.cuda()
    return model

# vol_t1, vol_t2 must be rescaled to [0, 255]
def roi_search(vol_t1, vol_t2, model, ksz=32, stride=24, mbsz=8):
    assert vol_t1.shape == vol_t2.shape, "must be identical shape"
    s, h, w = vol_t1.shape
    
    ph = int(np.ceil(h / 8) * 8) - h
    pw = int(np.ceil(w / 8) * 8) - w

    if pw > 0 or ph > 0:
        vol_t1p = np.pad(vol_t1, ((0, 0),(0, ph), (0, pw)), mode = 'edge')
        vol_t2p = np.pad(vol_t2, ((0, 0),(0, ph), (0, pw)), mode = 'edge')
    else:
        vol_t1p = vol_t1
        vol_t2p = vol_t2
    pw = -pw if pw!=0 else None
    ph = -ph if ph!=0 else None
        
    ds_t1 = SSegDataset(vol_t1p)
    dl_t1 = DataLoader(dataset=ds_t1, batch_size=mbsz, shuffle=False, \
                       num_workers=1, prefetch_factor=mbsz, drop_last=False, pin_memory=True)
    mask_t1 = []
    for X_mb in dl_t1:
        with torch.no_grad():
            pred  = model.forward(X_mb.cuda()).argmax(axis=1)
            mask_t1.append(pred.to(torch.uint8)[:, :ph, :pw])
    mask_t1 = torch.cat(mask_t1, axis=0)

    ds_t2 = SSegDataset(vol_t2p)
    dl_t2 = DataLoader(dataset=ds_t2, batch_size=mbsz, shuffle=False, \
                       num_workers=1, prefetch_factor=mbsz, drop_last=False, pin_memory=True)
    mask_t2 = []
    for X_mb in dl_t2:
        with torch.no_grad():
            pred  = model.forward(X_mb.cuda()).argmax(axis=1)
            mask_t2.append(pred.to(torch.uint8)[:, :ph, :pw])
    mask_t2 = torch.cat(mask_t2, axis=0)

    sand_mask = ~(mask_t1 | mask_t2)
    diff_mask = torch.abs((torch.from_numpy(vol_t1).cuda().to(torch.float32) - \
                           torch.from_numpy(vol_t2).cuda().to(torch.float32)) * sand_mask )
    
    pool4density= F.avg_pool3d(diff_mask[None, None], ksz, stride).cpu().numpy().squeeze()

    # try to search for 10-20 RoIs
    density_lo, density_hi = np.percentile(pool4density, (85, 100))
    mid = density_lo
    for _ in range(50):
        cc_labels = measure.label((pool4density > mid).astype(np.uint8))
        n_cc = cc_labels.max()
        if n_cc > 20: 
            density_lo = mid
            mid = (mid + density_hi) / 2
        elif n_cc < 8: 
            density_hi = mid
            mid = (mid + density_lo) / 2
        else:
            break

    cols = ("dim1", "dim2", "dim3", "importance", "d1_min", "d2_min", "d3_min", "d1_max", "d2_max", "d3_max")
    vals = []
    for region in regionprops(cc_labels):
        sid_o = coord_re_proj(region.centroid[0], stride, ksz, s-1)
        row_o = coord_re_proj(region.centroid[1], stride, ksz, h-1)
        col_o = coord_re_proj(region.centroid[2], stride, ksz, w-1)
        
        bbox_d1s = coord_re_proj(region.bbox[0], stride, ksz, s-1)
        bbox_d2s = coord_re_proj(region.bbox[1], stride, ksz, h-1)
        bbox_d3s = coord_re_proj(region.bbox[2], stride, ksz, w-1)
        bbox_d1e = coord_re_proj(region.bbox[3], stride, ksz, s-1)
        bbox_d2e = coord_re_proj(region.bbox[4], stride, ksz, h-1)
        bbox_d3e = coord_re_proj(region.bbox[5], stride, ksz, w-1)
        vals.append((sid_o, row_o, col_o, region.filled_area, bbox_d1s, bbox_d2s, bbox_d3s, bbox_d1e, bbox_d2e, bbox_d3e))
        
    RoIs    = pd.DataFrame(vals, columns=cols)
    RoIs = RoIs.sort_values(by='importance', ascending=False, ignore_index=True)
    centers = RoIs[['dim1', 'dim2', 'dim3']].values
    bbox_start = RoIs[['d1_min', 'd2_min', 'd3_min']].values
    bbox_width = RoIs[['d1_max', 'd2_max', 'd3_max']].values - RoIs[['d1_min', 'd2_min', 'd3_min']].values
    importance = RoIs.importance.values
    # print(RoIs)
    # RoIs.to_csv('RoIs-segm-%d.csv' % time.time(), index=False)
    return centers, importance, bbox_start, bbox_width, RoIs

def vol_3d_norm_gpu(vol):
    mean = vol.mean(axis=(-1,-2)).unsqueeze(dim=-1).unsqueeze(dim=-1)
    std  = vol.std (axis=(-1,-2)).unsqueeze(dim=-1).unsqueeze(dim=-1)
    norm = (vol - mean) / std
    return norm

def roi_search_subtraction(vol_t1, vol_t2, ksz=32, stride=24, qthr=0.9):
    assert vol_t1.shape == vol_t2.shape
    s, h, w = vol_t1.shape
    vt1 = torch.from_numpy(vol_t1.astype(np.float32))[None,None]
    vt2 = torch.from_numpy(vol_t2.astype(np.float32))[None,None]
    vol_t1_norm = vol_3d_norm_gpu(vt1)
    vol_t2_norm = vol_3d_norm_gpu(vt2)
    vol_diff = np.abs(vol_t1_norm - vol_t2_norm)
    thr = torch.quantile(vol_diff[0, 0, (s//2-5):(s//2+5)], qthr)
    vol_diff = torch.where(vol_diff > thr, vol_diff, torch.zeros(1))
    
    pool4density = F.avg_pool3d(vol_diff, ksz, stride).cpu().numpy().squeeze()
    density_lo, density_hi = np.percentile(pool4density, (85, 100))
    mid = density_lo
    for _ in range(50):
        cc_labels = measure.label((pool4density > mid).astype(np.uint8))
        n_cc = cc_labels.max()
        if n_cc > 10: 
            density_lo = mid
            mid = (mid + density_hi) / 2
        elif n_cc < 5: 
            density_hi = mid
            mid = (mid + density_lo) / 2
        else:
            break
            
    cols = ("dim1", "dim2", "dim3", "importance", "d1_min", "d2_min", "d3_min", "d1_max", "d2_max", "d3_max")
    vals = []
    for region in regionprops(cc_labels):
        sid_o = coord_re_proj(region.centroid[0], stride, ksz, s-1)
        row_o = coord_re_proj(region.centroid[1], stride, ksz, h-1)
        col_o = coord_re_proj(region.centroid[2], stride, ksz, w-1)
        
        bbox_d1s = coord_re_proj(region.bbox[0], stride, ksz, s-1)
        bbox_d2s = coord_re_proj(region.bbox[1], stride, ksz, h-1)
        bbox_d3s = coord_re_proj(region.bbox[2], stride, ksz, w-1)
        bbox_d1e = coord_re_proj(region.bbox[3], stride, ksz, s-1)
        bbox_d2e = coord_re_proj(region.bbox[4], stride, ksz, h-1)
        bbox_d3e = coord_re_proj(region.bbox[5], stride, ksz, w-1)
        vals.append((sid_o, row_o, col_o, region.filled_area, bbox_d1s, bbox_d2s, bbox_d3s, bbox_d1e, bbox_d2e, bbox_d3e))

    RoIs = pd.DataFrame(vals, columns=cols)
    RoIs = RoIs.sort_values(by='importance', ascending=False, ignore_index=True)
    centers = RoIs[['dim1', 'dim2', 'dim3']].values
    bbox_start = RoIs[['d1_min', 'd2_min', 'd3_min']].values
    bbox_width = RoIs[['d1_max', 'd2_max', 'd3_max']].values - RoIs[['d1_min', 'd2_min', 'd3_min']].values
    importance = RoIs.importance.values
    # print(RoIs)
    # RoIs.to_csv('RoIs-subs-%d.csv' % time.time(), index=False)
    return centers, importance, bbox_start, bbox_width, RoIs

if __name__ == "__main__":
    vol_t1 = h5py.File('SSeg/dataset/t1.h5', 'r')['data'][:]
    vol_t2 = h5py.File('SSeg/dataset/t2.h5', 'r')['data'][:]
    seg_mdl= load_seg_nn('mdl-ep00230.pth')
    
    tick = time.time()    
    centers, importance, bbox_start, bbox_width, RoIs = roi_searh(vol_t1, vol_t2, model=seg_mdl, mbsz=8)

    print("It took %.2f s to search RoIs" % (time.time() - tick))
    print(RoIs)
    
    tick = time.time()    
    centers, importance, bbox_start, bbox_width, RoIs = roi_search_subtraction(vol_t1, vol_t2)

    print("It took %.2f s to search RoIs" % (time.time() - tick))
    print(RoIs)
