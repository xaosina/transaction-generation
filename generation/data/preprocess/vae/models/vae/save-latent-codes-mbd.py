# python -m tabsyn.vae.save-latent-codes-mbd --dataname mbd_debug - runner
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
from utils_train import preprocess, TabularDataset
from tabsyn.vae.model import Model_VAE, Encoder_model, Decoder_model
import argparse
import logging

torch.set_printoptions(sci_mode=False, precision=4)
np.set_printoptions(suppress=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

LR = 1e-3
WD = 0
D_TOKEN = 6
TOKEN_BIAS = True

N_HEAD = 2
FACTOR = 64
NUM_LAYERS = 2

DIRPATH = os.path.dirname(__file__)


def main(args):

    device = args.device
    ckpt_dir = args.ckpt_dir
    logger.info("Reading dataset. It may take some time.")
    X_num, X_cat, categories, d_numerical, _, cat_inverse  = preprocess(
        args.data_dir, task_type=info["task_type"], inverse=True
    )
    d_numerical = d_numerical - 1
    categories.pop(0)

    logger.info("Train/Test data preprocessing")

    X_train_num, X_test_num = X_num
    X_train_cat, X_test_cat = X_cat

    X_train_num, X_test_num = (
        torch.tensor(X_train_num).float(),
        torch.tensor(X_test_num).float(),
    )
    X_train_cat, X_test_cat = torch.tensor(X_train_cat), torch.tensor(X_test_cat)

    train_data = TabularDataset(X_train_num.float(), X_train_cat)
    test_data = TabularDataset(X_test_num.float(), X_test_cat)

    batch_size = 2**15
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=4,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        num_workers=4,
        drop_last=False,
    )

    logger.info("Model initialization")
    model = Model_VAE(
        NUM_LAYERS,
        d_numerical,
        categories,
        D_TOKEN,
        n_head=N_HEAD,
        factor=FACTOR,
        bias=True,
    )
    model = model.to(device)

    pre_encoder = Encoder_model(
        NUM_LAYERS, d_numerical, categories, D_TOKEN, n_head=N_HEAD, factor=FACTOR
    ).to(device)
    pre_decoder = Decoder_model(
        NUM_LAYERS, d_numerical, categories, D_TOKEN, n_head=N_HEAD, factor=FACTOR
    ).to(device)

    pre_encoder.eval()
    pre_decoder.eval()

    logger.info("Model loading...")

    ckpt = torch.load(f"{ckpt_dir}/model.pt")
    model.load_state_dict(ckpt["model_state_dict"])
    _ = model.eval()

    pre_encoder.load_weights(model)
    pre_decoder.load_weights(model)

    pre_encoder.eval()
    pre_decoder.to(device)

    def process_loader(loader, pre_encoder, device, cat_inverse=None):
        latent_codes = []
        client_ids = []
        abs_times = []
        orig_ids = [] if cat_inverse else None

        with torch.inference_mode():
            for batch_num, batch_cat in tqdm(loader):
                batch_num = batch_num.to(device)
                batch_cat = batch_cat.to(device)

                abs_time, batch_num = batch_num[:, 1], batch_num[:, [i for i in range(batch_num.shape[1]) if i != 1]]
                client_id, batch_cat_cut = batch_cat[:, 0], batch_cat[:, 1:]

                z = pre_encoder(batch_num, batch_cat_cut).cpu().numpy()
                latent_codes.append(z)
                client_ids.append(client_id.cpu().numpy())
                abs_times.append(abs_time.cpu().numpy())

                if cat_inverse:
                    orig_ids.append(cat_inverse(batch_cat.cpu().numpy())[:, 0])

        latent_codes = np.concatenate(latent_codes)
        client_ids = np.concatenate(client_ids)
        abs_times = np.concatenate(abs_times)
        orig_ids = np.concatenate(orig_ids) if orig_ids else None

        return latent_codes, client_ids, abs_times, orig_ids

    # Process training data
    logger.info("Create train latents")
    train_z, train_client_ids, train_abs_times, train_orig_ids = process_loader(train_loader, pre_encoder, device, cat_inverse)

    # Process test data
    logger.info("Create test latents")
    test_z, test_client_ids, test_abs_times, _ = process_loader(test_loader, pre_encoder, device)

    # Save the results
    logger.info(f"Create all latents to {ckpt_dir}")
    np.save(f"{ckpt_dir}/train_z.npy", train_z)
    np.save(f"{ckpt_dir}/train_client_ids.npy", train_client_ids)
    np.save(f"{ckpt_dir}/train_abs_times.npy", train_abs_times)
    np.save(f"{ckpt_dir}/train_orig_ids.npy", train_orig_ids)

    np.save(f"{ckpt_dir}/test_z.npy", test_z)
    np.save(f"{ckpt_dir}/test_client_ids.npy", test_client_ids)
    np.save(f"{ckpt_dir}/test_abs_times.npy", test_abs_times)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Save latent codes for mbd dataset")

    parser.add_argument("--dataname", type=str, required=True, help="Name of dataset")
    parser.add_argument("--gpu", type=int, help="Name of dataset")

    args = parser.parse_args()

    args.data_dir = DIRPATH + f"/../../data/{args.dataname}_with_id"
    args.ckpt_dir = DIRPATH + f"/../../tabsyn/vae/ckpt/{args.dataname}"
    args.info_path = DIRPATH + f"/../../data/{args.dataname}_with_id/info.json"

    with open(args.info_path, "r") as f:
        info = json.load(f)

    args.device = f"cuda:{args.gpu}" if args.gpu else "cuda:0"
    logger.info(f"Cuda device selected: {args.device}")

    logger.info("Process started")
    main(args)
    logger.info("Process successfully ended.")
