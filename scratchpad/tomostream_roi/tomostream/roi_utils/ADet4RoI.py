import numpy as np
import torch, sys
import pandas as pd

# sys.path.append('/home/beams/TOMO/epics/synApps/support/tomostream_roi/tomostream/roi_utils')
from roi_utils.model import EncDec

def coord_reverse_proj(c, stride, ksz, maxs):
    return min(round(c * stride + ksz // 2), maxs)

def load_AE4ADet_NN(mdl):
    model = EncDec()
    model.load_state_dict(torch.load(mdl, map_location=torch.device('cpu')))
    return model.cuda()

def img2patch(img, psz=64, stride=48):
    h, w  = img.shape
    res = []
    for _r in range(0, h-psz, stride):
        for _c in range(0, w-psz, stride):
            res.append(img[_r:_r+psz, _c:_c+psz])
    return np.array(res)

def roi_search_ADet(vol, model='/home/beams/TOMO/gas_hydrates_3dzoom_Dec2021/models/mdl-ep0200.pth', psz=64, stride=48):
    if isinstance(model, str):
        model = load_AE4ADet_NN(model)
    else:
        model = model.cuda()
        
    s, h, w  = vol.shape
    rec_loss = np.zeros((s, (h-psz)//stride+1, (w-psz)//stride+1), dtype=np.float32)
    for sid in range(vol.shape[0]):
        patches = img2patch(vol[sid])[:, None]
        ss = int(np.sqrt(patches.shape[0]))
        inp = torch.from_numpy(patches).cuda()
        with torch.no_grad():
            pred = model.forward(inp)
            rec_loss[sid] = torch.nn.functional.mse_loss(inp, pred, reduction='none').cpu().numpy().mean(axis=(1,2,3)).reshape((ss, ss))
    
    rois = np.dstack(np.unravel_index(np.argsort(rec_loss.ravel()), rec_loss.shape))[0]
    vals = []
    for imp, (s, r, c) in enumerate(rois):
        _ro = coord_reverse_proj(r, stride, psz, h-1)
        _co = coord_reverse_proj(c, stride, psz, w-1)
        vals.append([s, _ro, _co, imp, s, max(0, _ro-psz//2), max(0, _co-psz//2), s, min(h, _ro+psz//2), min(w, _co+psz//2)])
    cols = ("dim1", "dim2", "dim3", "importance", "d1_min", "d2_min", "d3_min", "d1_max", "d2_max", "d3_max")
    
    RoIs    = pd.DataFrame(vals, columns=cols)
    RoIs = RoIs.sort_values(by='importance', ascending=False, ignore_index=True)
    centers = RoIs[['dim1', 'dim2', 'dim3']].values
    bbox_start = RoIs[['d1_min', 'd2_min', 'd3_min']].values
    bbox_width = RoIs[['d1_max', 'd2_max', 'd3_max']].values - RoIs[['d1_min', 'd2_min', 'd3_min']].values
    importance = RoIs.importance.values

    return centers, importance, bbox_start, bbox_width, RoIs

