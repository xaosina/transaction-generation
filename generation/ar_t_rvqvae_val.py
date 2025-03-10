import os
from datetime import datetime 
import argparse
import logging
import json
import yaml
from tabsyn.utils.logger import Logger
import zoneinfo
import time
from tqdm import tqdm
import random
import string
from pathlib import Path
from torch.cuda.amp import autocast, GradScaler

import torch

from tabsyn.utils.training_utils import (
    DataParallelAttrAccess, 
    make_random_subset,
    multi_gpu,
    get_state_dict,
)

import tabsyn.transformer as ar_transformer
import tabsyn.rnn as rnn
from tabsyn.utils.latent_utils import SeqLatentPredictionDataset

from torch.utils.data import DataLoader

from tabsyn.utils.data_utils import (
    get_latent_data_info,
    encode_client_ids,
)

from tabsyn.utils.other_utils import (
    dictprettyprint,
)

from tabsyn.utils.testing_utils import SampleEvaluator

import delu

TIME_ZONE = zoneinfo.ZoneInfo("Europe/Moscow")

DIFFUSION_BACKBONES = ["mlp", "unet", "unetS", "unetM", "unetL", "unetXL"]

EMA_BETAS = [0.99, 0.999, 0.9999]

def config_init(config_dir, args):
    with open(config_dir, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    config['dataname'] = args.dataname
    config['device'] = args.device
    config['debug'] = args.debug
    config["rawhist_stdratio"] = args.rh_sr
    config["art_ckpt"] = args.eval_ckpt
    config["vae_type"] = args.vae_type

    match config["DATA_MODE"]:
        case "num2mnum":
            config["reconstruction_type"] = "quants"
        case "num2cat":
            config["reconstruction_type"] = "codes"
        case "num2num":
            config["reconstruction_type"] = "nothing"
        case _:
            raise NotImplementedError

    return config

def set_misc(args):

    logger = Logger(
        name=__name__,
        level=logging.INFO,
        log_to_stdout=True,
    )

    curr_dir = os.path.dirname(os.path.abspath(__file__))

    config = config_init(f"{curr_dir}/configs/config.yaml", args)

    info_path = f"data/{config['dataname']}/info.json"

    with open(info_path, "r") as f:
        dataset_info = json.load(f)

    config['data_dir'] = f"data/{config['dataname']}"

    config["timestamp"] = datetime.now(TIME_ZONE).strftime("%m-%d-%H:%M")

    config["exp_id"] = "".join(random.choices(string.ascii_lowercase, k=10)) 
    config["exp_name"] = f"sk_{config['exp_id']}"

    ckpt_dir = f"{curr_dir}/ckpt/eval/{config['timestamp']}_{config['exp_id']}"
    config["art_ckpt"] = f"{curr_dir}/ckpt/{config['art_ckpt']}"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    else:
        logger.info('Directory is not empty!')
        exit()

    logger.set_file(f"{ckpt_dir}/log.txt")

    logger.info(config)

    config['vae_ckpt_dir'] = f"{curr_dir}/{config['vae_type']}/ckpt/{args.vae_ckpt}"

    config['deb'] = args.debug

    config["multi_gpu_regime"] = len(args.gpu_ids) > 1

    config["n_est"] = config["VAL_FREQ"]

    config["artr_ckpt"] = Path(f"{ckpt_dir}/")

    return dataset_info, config, logger

def create_forecasting_model(config, logger, use_checkpoint=False):
    match config["MODEL"]:
        case "transformer":
            model = ar_transformer.get_model(config["TR_SIZE"])

            match config["DATA_MODE"]:
                case "num2mnum":
                    model.input_mode = "48"
                    model.output_mode = "16x48"
                case "num2cat":
                    model.input_mode = "48"
                    model.output_mode = "16x"
                case "num2num":
                    model.input_mode = "48"
                    model.output_mode = "48"
                case _:
                    raise NotImplementedError
                
            model.hidden_dim = config["VAE_HIDDEN_DIM"]
            model.codes_dim = config["NUM_EMBEDDINGS"]
            model.codes_number = config["RVQ_STAGES"]
            model.masked_training = config["MASKED_TRAINING"]
            model.init(loss_type="full" if config["TRAIN_MODE"] == "TA" else "last",
                       mask_params=(config["BATCH_SIZE"], config["EST_HIST_LEN"]) if config["MASKED_TRAINING"] else None)

            optimizer = torch.optim.Adam(model.parameters(), lr=config['LR'])

            if use_checkpoint:
                logger.info(f"Load_model from checkpoint: {config['art_ckpt']}")
                state_dict = torch.load(config["art_ckpt"] + "/artr_wholestate.pt", map_location=config["device"])
                model.load_state_dict(state_dict["model_state_dict"])
                config["ckpt_start_epoch"] = state_dict['epoch']
                config["ckpt_best_loss"] = state_dict["loss"]
                optimizer.load_state_dict(state_dict["optimizer_state_dict"])
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(config["device"])

        case "rnn":
            model = rnn.get_model("gru")
            if use_checkpoint:
                model.load_state_dict(torch.load(config["eval_ckpt"] + "/artr_wholestate.pt", map_location=config["device"])["model_state_dict"])
            optimizer = torch.optim.Adam(model.parameters(), lr=config['LR'])
        case _:
            NotImplementedError("Unknown model type in config['MODEL']")

    return model.to(config['device']), optimizer, None

def load_data(dataset_info, config, logger):
    # data loading
    logger.info("Start loading the data.")
    (
        train_info
    ) = get_latent_data_info(config, partition="train")

    (
        test_info
    ) = get_latent_data_info(config, partition="test")

    train_info['client_ids'], test_info['client_ids'] = encode_client_ids(train_info['client_ids'], test_info['client_ids'], config)


    logger.info("Creating train dataset...")

    train_dataset = SeqLatentPredictionDataset( #TODO: rework
        train_info["latents"],
        train_info["client_ids"],
        train_info["abs_times"],
        z_codes=train_info["latent_codes"],
        z_rq=train_info["latent_residual_qz"],
        min_hist_len=config["EST_HIST_LEN"],
        gen_len=config["EST_GEN_LEN"],
        # sort_data=True if config["deb"] else False
    )

    logger.info(f"(basic) train dataset size: {len(train_dataset)}")

    logger.info("Creating test dataset...")

    test_dataset = SeqLatentPredictionDataset(
        test_info["latents"],
        test_info["client_ids"],
        test_info["abs_times"],
        z_codes=test_info["latent_codes"],
        z_rq=test_info["latent_residual_qz"],
        min_hist_len=config["EST_HIST_LEN"],
        gen_len=config["EST_GEN_LEN"],
        mode="last",
        # sort_data=True if config["deb"] else False
    )

    logger.info(f"(basic) test dataset size: {len(test_dataset)}")

    total_train_samples = 1000 if config["deb"] else len(train_dataset)
    train_dataset = make_random_subset(train_dataset, total_train_samples)

    logger.info(f"actual train dataset size: {len(train_dataset)}")

    total_test_samples = 1000 if config["deb"] else len(test_dataset)  # TODO
    total_test_samples = min(len(test_dataset), total_test_samples)
    test_dataset = make_random_subset(test_dataset, total_test_samples)

    logger.info(f"actual test dataset size: {len(test_dataset)}" )

    config["num_train_samples"] = total_train_samples
    config["num_test_samples"] = total_test_samples

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["BATCH_SIZE"],
        collate_fn=SeqLatentPredictionDataset.dloader_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=config["NUM_WORKERS"],
    )

    test_loader = DataLoader(
        test_dataset,
        config["EVAL_BATCH_SIZE"],
        collate_fn=SeqLatentPredictionDataset.dloader_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=config["NUM_WORKERS"],
    )
    return train_loader, test_loader

def initialize_training(config, logger):
    # Initialize TrainState
    train_state = {
        "epoch": 0,
        "best_loss": float("inf"),
        "best_discri": float("inf"),
        "patience": 0,
        "CURR_TRAIN_STEP": 0,
        "glob_start_time": time.perf_counter()
    }

    # training
    # wandb.init(
    #     project="tabsyn",
    #     name=config["exp_name"],
    #     config=config,
    #     mode="disabled",
    # )

    logger.info("Final config:\n-----------------")
    logger.info(dictprettyprint(config))

    logger.info(
        "----------------------\n",
        ">>> START TRAINING <<<\n",
        "----------------------\n",
    )
    return train_state


def evaluate_model(train_loader, test_loader, 
                model, optimizer,
                config, logger):
    model.eval()
    train_state = initialize_training(config, logger)

    evaluator = SampleEvaluator(model, config, logger, train_state)
    evaluator.evaluate(train_loader, blim=config["VAL_N"], discr='train_')
    evaluator.evaluate(test_loader, blim=config["VAL_N"], discr='test_')

    logger.info("Validation is done.")


def main(args):
    dataset_info, config, logger = set_misc(args)
    train_loader, test_loader = load_data(dataset_info, config, logger)
    gen_model, optimizer, _ = create_forecasting_model(config, logger, use_checkpoint=True)
    evaluate_model(train_loader, test_loader,
                gen_model, optimizer,
                config, logger)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="AR transformer in RVQ-VAE latents")

    parser.add_argument("--dataname", type=str, required=True, help="Name of dataset.")
    parser.add_argument("--gpu-ids", type=int, nargs='*', required=True, help="GPU index.")
    parser.add_argument("--vae-ckpt", type=str, required=True, help="Checkpoint of RVQ-VAE")
    parser.add_argument("--vae-type", type=str, default='vae', help="vae/rvqvae")
    parser.add_argument("--debug", action='store_const', const=True, default=False, help='Run it with debug')
    parser.add_argument("--eval-ckpt", type=str, required=True, help="AR Transformer checkpoint")
    parser.add_argument('--rh-sr', type=float, default=0.0, help='std ratio to noise raw history data')

    args = parser.parse_args()

    assert torch.cuda.is_available()
    assert args.gpu_ids is not None
    assert len(args.gpu_ids) > 0
    args.device = f"cuda:{args.gpu_ids[0]}"

    main(args)