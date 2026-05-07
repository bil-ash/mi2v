import cv2  # isort:skip

import argparse
import gc
import os
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm
from PIL import Image

#from tools.datasets.utils import extract_frames
#from .. import tools 
from unimatch import UniMatch

# torch.backends.cudnn.enabled = False # This line enables large batch, but the speed is similar

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")
VID_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

def extract_frames(
    video_path,
    frame_inds=None,
    points=None,
    backend="opencv",
    return_length=False,
    num_frames=None,
):
    """
    Args:
        video_path (str): path to video
        frame_inds (List[int]): indices of frames to extract
        points (List[float]): values within [0, 1); multiply #frames to get frame indices
    Return:
        List[PIL.Image]
    """
    assert backend in ["av", "opencv", "decord"]
    assert (frame_inds is None) or (points is None)

    if backend == "av":
        import av

        container = av.open(video_path)
        if num_frames is not None:
            total_frames = num_frames
        else:
            total_frames = container.streams.video[0].frames

        if points is not None:
            frame_inds = [int(p * total_frames) for p in points]

        frames = []
        for idx in frame_inds:
            if idx >= total_frames:
                idx = total_frames - 1
            target_timestamp = int(idx * av.time_base / container.streams.video[0].average_rate)
            container.seek(target_timestamp)
            frame = next(container.decode(video=0)).to_image()
            frames.append(frame)

        if return_length:
            return frames, total_frames
        return frames

    elif backend == "decord":
        import decord

        container = decord.VideoReader(video_path, num_threads=1)
        if num_frames is not None:
            total_frames = num_frames
        else:
            total_frames = len(container)

        if points is not None:
            frame_inds = [int(p * total_frames) for p in points]

        frame_inds = np.array(frame_inds).astype(np.int32)
        frame_inds[frame_inds >= total_frames] = total_frames - 1
        frames = container.get_batch(frame_inds).asnumpy()  # [N, H, W, C]
        frames = [Image.fromarray(x) for x in frames]

        if return_length:
            return frames, total_frames
        return frames

    elif backend == "opencv":
        cap = cv2.VideoCapture(video_path)
        if num_frames is not None:
            total_frames = num_frames
        else:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if points is not None:
            frame_inds = [int(p * total_frames) for p in points]

        frames = []
        for idx in frame_inds:
            if idx >= total_frames:
                idx = total_frames - 1

            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

            # HACK: sometimes OpenCV fails to read frames, return a black frame instead
            try:
                ret, frame = cap.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
            except Exception as e:
                print(f"[Warning] Error reading frame {idx} from {video_path}: {e}")
                # First, try to read the first frame
                try:
                    print(f"[Warning] Try reading first frame.")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = Image.fromarray(frame)
                # If that fails, return a black frame
                except Exception as e:
                    print(f"[Warning] Error in reading first frame from {video_path}: {e}")
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frame = Image.new("RGB", (width, height), (0, 0, 0))

            # HACK: if height or width is 0, return a black frame instead
            if frame.height == 0 or frame.width == 0:
                height = width = 256
                frame = Image.new("RGB", (width, height), (0, 0, 0))

            frames.append(frame)

        if return_length:
            return frames, total_frames
        return frames
    else:
        raise ValueError


def merge_scores(gathered_list: list, meta: pd.DataFrame, column):
    # reorder
    indices_list = list(map(lambda x: x[0], gathered_list))
    scores_list = list(map(lambda x: x[1], gathered_list))

    flat_indices = []
    for x in zip(*indices_list):
        flat_indices.extend(x)
    flat_scores = []
    for x in zip(*scores_list):
        flat_scores.extend(x)
    flat_indices = np.array(flat_indices)
    flat_scores = np.array(flat_scores)

    # filter duplicates
    unique_indices, unique_indices_idx = np.unique(flat_indices, return_index=True)
    meta.loc[unique_indices, column] = flat_scores[unique_indices_idx]

    # drop indices in meta not in unique_indices
    meta = meta.loc[unique_indices]
    return meta


class VideoTextDataset(torch.utils.data.Dataset):
    def __init__(self, meta_path, frame_inds=[0, 10, 20, 30]):
        self.meta_path = meta_path
        self.meta = pd.read_csv(meta_path)
        self.frame_inds = frame_inds

    def __getitem__(self, index):
        sample = self.meta.iloc[index]
        path = sample["video_path"]

        # extract frames
        images = extract_frames(path, frame_inds=self.frame_inds, backend="opencv")

        # transform
        images = torch.stack([pil_to_tensor(x) for x in images])

        # stack
        # shape: [N, C, H, W]; dtype: torch.uint8
        images = images.float()
        H, W = images.shape[-2:]
        if H > W:
            images = rearrange(images, "N C H W -> N C W H")
        images = F.interpolate(images, size=(320, 576), mode="bilinear", align_corners=True)

        ret = dict(index=index, images=images)
        return ret

    def __len__(self):
        return len(self.meta)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("meta_path", type=str, help="Path to the input CSV file")
    parser.add_argument("--bs", type=int, default=4, help="Batch size")  # don't use too large bs for unimatch
    parser.add_argument("--num_workers", type=int, default=16, help="Number of workers")
    parser.add_argument("--skip_if_existing", action="store_true")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    meta_path = args.meta_path
    if not os.path.exists(meta_path):
        print(f"Meta file '{meta_path}' not found. Exit.")
        exit()

    wo_ext, ext = os.path.splitext(meta_path)
    out_path = f"{wo_ext}_flow{ext}"
    if args.skip_if_existing and os.path.exists(out_path):
        print(f"Output meta file '{out_path}' already exists. Exit.")
        exit()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=24))
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())

    # build model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = UniMatch(
        feature_channels=128,
        num_scales=2,
        upsample_factor=4,
        num_head=1,
        ffn_dim_expansion=4,
        num_transformer_layers=6,
        reg_refine=True,
        task="flow",
    )
    ckpt = torch.load("/data/shuaizhang/MobileI2V/tools/scoring/optical_flow/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth")
    model.load_state_dict(ckpt["model"])
    model = model.to(device)

    # build dataset
    dataset = VideoTextDataset(meta_path=meta_path, frame_inds=[0, 10, 20, 30])
    dataloader = DataLoader(
        dataset,
        batch_size=args.bs,
        num_workers=args.num_workers,
        sampler=DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=False,
            drop_last=False,
        ),
    )

    # compute optical flow scores
    indices_list = []
    scores_list = []
    model.eval()
    for batch in tqdm(dataloader, disable=dist.get_rank() != 0):
        indices = batch["index"]
        images = batch["images"].to(device, non_blocking=True)

        B = images.shape[0]
        batch_0 = rearrange(images[:, :-1], "B N C H W -> (B N) C H W").contiguous()
        batch_1 = rearrange(images[:, 1:], "B N C H W -> (B N) C H W").contiguous()

        with torch.no_grad():
            res = model(
                batch_0,
                batch_1,
                attn_type="swin",
                attn_splits_list=[2, 8],
                corr_radius_list=[-1, 4],
                prop_radius_list=[-1, 1],
                num_reg_refine=6,
                task="flow",
                pred_bidir_flow=False,
            )
            flow_maps = res["flow_preds"][-1].cpu()  # [B * (N-1), 2, H, W]
            flow_maps = rearrange(flow_maps, "(B N) C H W -> B N H W C", B=B)
            flow_scores = flow_maps.abs().mean(dim=[1, 2, 3, 4])
            flow_scores = flow_scores.tolist()

        indices_list.extend(indices.tolist())
        scores_list.extend(flow_scores)

    # save local results
    meta_local = merge_scores([(indices_list, scores_list)], dataset.meta, column="flow")
    save_dir_local = os.path.join(os.path.dirname(out_path), "parts")
    os.makedirs(save_dir_local, exist_ok=True)
    out_path_local = os.path.join(
        save_dir_local, os.path.basename(out_path).replace(".csv", f"_part_{dist.get_rank()}.csv")
    )
    meta_local.to_csv(out_path_local, index=False)

    # wait for all ranks to finish data processing
    dist.barrier()

    torch.cuda.empty_cache()
    gc.collect()
    gathered_list = [None] * dist.get_world_size()
    dist.all_gather_object(gathered_list, (indices_list, scores_list))
    if dist.get_rank() == 0:
        meta_new = merge_scores(gathered_list, dataset.meta, column="flow")
        meta_new.to_csv(out_path, index=False)
        print(f"New meta with optical flow scores saved to '{out_path}'.")


if __name__ == "__main__":
    main()
