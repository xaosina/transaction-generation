# python -m tabsyn.vae.save-latent-codes-mbd --dataname mbd_debug - runner
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
from utils_train import preprocess, TabularDataset
# from tabsyn.vae.model import Model_VAE, Encoder_model, Decoder_model
from tabsyn.rvqvae.model import Model_VAE, Encoder_model, Decoder_model, VectorQuantizer
import argparse
import logging
import yaml

from torch.profiler import profiler

torch.set_printoptions(sci_mode=False, precision=4)
np.set_printoptions(suppress=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

DIRPATH = os.path.dirname(__file__)


def main(args, config):

    device = args.device
    ckpt_dir = args.ckpt_dir
    save_z, save_z_codes = args.latents, args.latent_codes

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

    if args.vae_type == 'rvqvae':
        model = Model_VAE(
            config['NUM_LAYERS'],
            d_numerical,
            categories,
            config['D_TOKEN'],
            n_head=config['N_HEAD'],
            factor=config['FACTOR'],
            bias=True,
            num_embeddings=config['NUM_EMBEDDINGS'],
            decay=config['DECAY'],
            commitment_cost=config['COMMITMENT_COST'],
            rvq_stages=config['RVQ_STAGES']
        )
    else:
        model = Model_VAE(
            config['NUM_LAYERS'],
            d_numerical,
            categories,
            config['D_TOKEN'],
            n_head=config['N_HEAD'],
            factor=config['FACTOR'],
            bias=True,
        )
    model = model.to(device)

    # # TODO DELETE

    # for i, vq_layer in enumerate(model.RVQ_VAE.rvq_ema.vq_layers):
    #     cb = vq_layer._embedding.weight.detach().cpu().numpy()
    #     np.save(f"codebooks_stage_{i}", cb)

    # # TODO DELETE

    pre_encoder = Encoder_model(
        config['NUM_LAYERS'], d_numerical, categories, config['D_TOKEN'], 
        n_head=config['N_HEAD'], factor=config['FACTOR']
    ).to(device)
    pre_decoder = Decoder_model(
        config['NUM_LAYERS'], d_numerical, categories, config['D_TOKEN'], 
        n_head=config['N_HEAD'], factor=config['FACTOR']
    ).to(device)
    pre_quantizer = VectorQuantizer(
        config['RVQ_STAGES'], config['NUM_EMBEDDINGS'], config['D_TOKEN'], 
        d_numerical, categories, config['COMMITMENT_COST'], config['DECAY']
    ).to(device)

    logger.info("Model loading...")

    ckpt = torch.load(f"{ckpt_dir}/model.pt")
    model.load_state_dict(ckpt["model_state_dict"])
    _ = model.eval()

    # # TODO DELETE

    # for i, vq_layer in enumerate(model.RVQ_VAE.rvq_ema.vq_layers):
    #     cb = vq_layer._embedding.weight.detach().cpu().numpy()
    #     np.save(f"after_codebooks_stage_{i}", cb)
                
    # # TODO DELETE

    pre_encoder.eval()
    pre_decoder.eval()
    pre_quantizer.eval()

    pre_encoder.load_weights(model)
    pre_decoder.load_weights(model)
    pre_quantizer.load_weights(model)



    def process_loader(loader, output_path, pre_encoder, device, cat_inverse=None, save_z=True, save_z_codes=False, part=1.0):
        # Оценка размера выходных данных (если возможно)
        # breakpoint()
        if part < 1.0:
            total_size = int(int((len(loader.dataset) // (2**15)) * part) * 2**15)
        else:
            total_size = len(loader.dataset)
        logger.info(f"Total size of train_z is {total_size}")

        # memmap_array = np.memmap(output_path, dtype='float32', mode='w+', shape=(total_size, 32, 48)) 
        latent_codes = np.zeros(shape=(total_size, 32), dtype=int)
        latents = np.zeros(shape=(total_size, 48), dtype=float)
        client_ids = np.zeros(total_size, dtype=int)
        abs_times = np.zeros(total_size, dtype=float)
        # orig_ids = np.zeros(total_size, dtype=int) if cat_inverse else None

        idx = 0
        with torch.inference_mode():
            for batch_num, batch_cat in tqdm(loader):
                batch_num = batch_num.to(device)
                batch_cat = batch_cat.to(device)

                abs_time, batch_num = batch_num[:, 1], batch_num[:, [i for i in range(batch_num.shape[1]) if i != 1]]
                client_id, batch_cat_cut = batch_cat[:, 0], batch_cat[:, 1:]

                z = pre_encoder(batch_num, batch_cat_cut)
                z, codes, zqs = pre_quantizer(z, save_z=True)

                # save_tensor = z.cpu().numpy()
                # save_tensor = torch.round(zqs.cpu(), decimals=4).numpy()
                # Записываем данные в предварительно выделенные тензоры
                batch_size = z.shape[0]
                if save_z:
                    latents[idx:idx+batch_size, ] = z.cpu().numpy()
                if save_z_codes:
                    latent_codes[idx:idx+batch_size, ] = codes.cpu().numpy()
                client_ids[idx:idx+batch_size] = client_id.cpu().numpy()
                abs_times[idx:idx+batch_size] = abs_time.cpu().numpy()

                # if cat_inverse:
                #     orig_ids[idx:idx+batch_size] = cat_inverse(batch_cat.cpu().numpy())[:, 0]
                del z, codes, zqs, batch_num, batch_cat
                torch.cuda.empty_cache()
                idx += batch_size
                if idx == total_size:
                    break

        # Конвертация в numpy после сбора всех данных
        return latents, latent_codes, client_ids, abs_times, None


    # Process training data
    logger.info("Create train latents")
    train_z, train_z_codes, train_client_ids, train_abs_times, _ = process_loader(train_loader, f"{ckpt_dir}/train_z.npy", 
                                                             pre_encoder, device, cat_inverse, save_z, save_z_codes, part=args.part)

    # Process test data
    logger.info("Create test latents")
    test_z, test_z_codes, test_client_ids, test_abs_times, _ = process_loader(test_loader, f"{ckpt_dir}/test_z.npy", 
                                                           pre_encoder, device, save_z, save_z_codes, part=args.part)

    # Save the results
    logger.info(f"Create all latents to {ckpt_dir}")
    np.save(f"{ckpt_dir}/train_z.npy", train_z)
    np.save(f"{ckpt_dir}/train_z_codes.npy", train_z_codes)
    np.save(f"{ckpt_dir}/train_client_ids.npy", train_client_ids)
    np.save(f"{ckpt_dir}/train_abs_times.npy", train_abs_times)
    # np.save(f"{ckpt_dir}/train_orig_ids.npy", train_orig_ids)

    np.save(f"{ckpt_dir}/test_z.npy", test_z)
    np.save(f"{ckpt_dir}/test_z_codes.npy", test_z_codes)
    np.save(f"{ckpt_dir}/test_client_ids.npy", test_client_ids)
    np.save(f"{ckpt_dir}/test_abs_times.npy", test_abs_times)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Save latent codes for mbd dataset")

    parser.add_argument("--dataname", type=str, required=True, help="Name of dataset")
    parser.add_argument("--gpu", type=int, help="Name of dataset")
    parser.add_argument("--vae-type", type=str, default='vae', required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("-z", "--latents", action="store_true", help='Save latents', required=True)
    parser.add_argument("-zc", "--latent_codes", action="store_true", help='Save latent codes', required=True)
    parser.add_argument('-p', '--part', type=float, default=1.0, help='Part of sample to use')
    args = parser.parse_args()

    args.data_dir = DIRPATH + f"/../../data/{args.dataname}_with_id"
    args.ckpt_dir = DIRPATH + f"/../../tabsyn/{args.vae_type}/ckpt/{args.ckpt}"
    args.info_path = DIRPATH + f"/../../data/{args.dataname}_with_id/info.json"


    with open(f"{args.ckpt_dir}/config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    with open(args.info_path, "r") as f:
        info = json.load(f)

    args.device = f"cuda:{args.gpu}" if args.gpu else "cuda:0"
    logger.info(f"Cuda device selected: {args.device}")

    logger.info("Process started")
    main(args, config)
    logger.info("Process successfully ended.")
