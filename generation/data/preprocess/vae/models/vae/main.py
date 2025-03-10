import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import warnings

import os
from tqdm import tqdm
import json
import time

from tabsyn.vae.model import Model_VAE, Encoder_model, Decoder_model
from utils_train import preprocess, TabularDataset

warnings.filterwarnings("ignore")


LR = 1e-3
WD = 0
D_TOKEN = 6
TOKEN_BIAS = True

N_HEAD = 2
FACTOR = 64
NUM_LAYERS = 2
NUM_EPOCHS = 1000


def compute_loss(X_num, X_cat, Recon_X_num, Recon_X_cat, params):
    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss = (X_num - Recon_X_num).pow(2).mean()
    ce_loss = 0
    acc = 0
    total_num = 0
    for idx, x_cat in enumerate(Recon_X_cat):
        if x_cat is not None:
            ce_loss += ce_loss_fn(x_cat, X_cat[:, idx])
            x_hat = x_cat.argmax(dim=-1)
        acc += (x_hat == X_cat[:, idx]).float().sum()
        total_num += x_hat.shape[0]

    ce_loss /= idx + 1
    acc /= total_num
    # loss = mse_loss + ce_loss
    latent_loss = 0.
    if "mu_z" in params:
        temp = 1 + params["std_z"] - params["mu_z"].pow(2) - params["std_z"].exp()

        latent_loss = -0.5 * torch.mean(temp.mean(-1).mean())
    else:
        posterior_loss = params["latent_loss"]
        latent_loss = posterior_loss


    return mse_loss, ce_loss, latent_loss, acc

    # return mse_loss, ce_loss, loss_kld, acc

def validate(model, test_loader, device):
    # Here we recode validation by batches
    pbar = tqdm(test_loader, total=len(test_loader))  # 5
    pbar.set_description(f"Validation")  # 6

    total_val_mse_loss = 0
    total_val_ce_loss = 0
    total_val_latent_loss = 0
    total_val_metric = 0
    total_val_perplexity = 0
    val_count = 0

    for batch_num, batch_cat in pbar:
        batch_num = batch_num.to(device)
        batch_cat = batch_cat.to(device)

        Recon_X_num, Recon_X_cat, params = model(batch_num, batch_cat)

        val_mse_loss, val_ce_loss, val_latent_loss, val_metric = compute_loss(
            batch_num, batch_cat, Recon_X_num, Recon_X_cat, params
        )
        val_loss = val_mse_loss.item() + val_ce_loss.item()

        val_batch_length = batch_num.shape[0]
        val_count += val_batch_length
        total_val_mse_loss += val_mse_loss.item() * val_batch_length
        total_val_ce_loss += val_ce_loss.item() * val_batch_length
        total_val_latent_loss += val_latent_loss.item() * val_batch_length
        total_val_metric += val_metric.item() * val_batch_length

    avg_val_mse_loss = total_val_mse_loss / val_count
    avg_val_ce_loss = total_val_ce_loss / val_count
    avg_val_latent_loss = total_val_latent_loss / val_count
    avg_val_acc = total_val_metric / val_count

    return avg_val_mse_loss, avg_val_ce_loss, avg_val_latent_loss, avg_val_acc

def main(args):
    dataname = args.dataname
    data_dir = f"data/{dataname}"

    max_beta = args.max_beta
    min_beta = args.min_beta
    lambd = args.lambd

    device = args.device

    info_path = f"data/{dataname}/info.json"

    with open(info_path, "r") as f:
        info = json.load(f)

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = f"{curr_dir}/ckpt/{dataname}"
    start_from_ckpt = False
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    else:
        # start_from_ckpt = True
        print(f"We start from checkpoint {ckpt_dir}")

    model_save_path = f"{ckpt_dir}/model.pt"
    encoder_save_path = f"{ckpt_dir}/encoder.pt"
    decoder_save_path = f"{ckpt_dir}/decoder.pt"

    X_num, X_cat, categories, d_numerical = preprocess(
        data_dir, task_type=info["task_type"]
    )

    X_train_num, _ = X_num
    X_train_cat, _ = X_cat

    X_train_num, X_test_num = X_num
    X_train_cat, X_test_cat = X_cat

    X_train_num, X_test_num = (
        torch.tensor(X_train_num).float(),
        torch.tensor(X_test_num).float(),
    )
    X_train_cat, X_test_cat = torch.tensor(X_train_cat), torch.tensor(X_test_cat)

    train_data = TabularDataset(X_train_num.float(), X_train_cat)
    test_data = TabularDataset(X_test_num.float(), X_test_cat)  # 1

    # X_test_num = X_test_num.float().to(device) #2
    # X_test_cat = X_test_cat.to(device) #3

    batch_size = 4096
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    test_loader = DataLoader(  # 4
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )
    if args.vq:
        model = Model_VAE(
            NUM_LAYERS,
            d_numerical,
            categories,
            D_TOKEN,
            n_head=N_HEAD,
            factor=FACTOR,
            bias=True,
            vq_vae=True,
            num_embeddings=512,
            decay=0.99,
            commitment_cost=0.25,
        )

    else:

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
    )
    pre_decoder = Decoder_model(
        NUM_LAYERS, d_numerical, categories, D_TOKEN, n_head=N_HEAD, factor=FACTOR
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.95, patience=10, verbose=True
    )
    beta = max_beta

    if start_from_ckpt:
        ckpt = torch.load(f"{ckpt_dir}/model.pt")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        beta = ckpt.get("beta", max_beta)  # Восстанавливаем beta, если она есть
        epoch = ckpt.get("epoch", 0)
        print(ckpt.get("beta"))
        pre_encoder.load_weights(model)
        pre_decoder.load_weights(model)
        

    pre_encoder.to(device)
    pre_decoder.to(device)

    pre_encoder.eval()
    pre_decoder.eval()

    num_epochs = NUM_EPOCHS
    best_train_loss = float("inf")

    current_lr = optimizer.param_groups[0]["lr"]
    patience = 0
    
    if start_from_ckpt:
        print('Check saved model, load set best_train_loss')
        
        model.eval()
        with torch.no_grad():
            avg_val_mse_loss, avg_val_ce_loss, avg_val_kl_loss, avg_val_acc = validate(model, test_loader, device)
        val_loss = avg_val_mse_loss + avg_val_ce_loss
        best_train_loss = val_loss
        
        print(
            "beta = {:.6f}, Val MSE:{:.6f}, Val CE:{:.6f}, Val ACC:{:6f}".format(
                beta,
                avg_val_mse_loss,
                avg_val_ce_loss,
                avg_val_acc,
            )
        )
    

    start_time = time.time()
    for epoch in range(num_epochs):
        pbar = tqdm(train_loader, total=len(train_loader))
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")

        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0
        curr_latent_loss = 0.0

        curr_count = 0

        for batch_num, batch_cat in pbar:
            model.train()
            optimizer.zero_grad()

            batch_num = batch_num.to(device)
            batch_cat = batch_cat.to(device)

            Recon_X_num, Recon_X_cat, params = model(batch_num, batch_cat)

            loss_mse, loss_ce, latent_loss, train_acc = compute_loss(
                batch_num, batch_cat, Recon_X_num, Recon_X_cat, params
            )
            beta = 1
            loss = loss_mse + loss_ce + beta * latent_loss
            loss.backward()
            optimizer.step()

            batch_length = batch_num.shape[0]
            curr_count += batch_length
            curr_loss_multi += loss_ce.item() * batch_length
            curr_loss_gauss += loss_mse.item() * batch_length
            curr_latent_loss += latent_loss.item() * batch_length

        num_loss = curr_loss_gauss / curr_count
        cat_loss = curr_loss_multi / curr_count
        latent_loss = curr_latent_loss / curr_count

        """
            Evaluation
        """
        # try:
        #     send_telegram_msg('vae still fitting!')
        # except:
        #     print('no tg messages due to no internet connection')

        model.eval()
        with torch.no_grad():
            avg_val_mse_loss, avg_val_ce_loss, avg_val_kl_loss, avg_val_acc = validate(model, test_loader, device)
            val_loss = avg_val_mse_loss + avg_val_ce_loss
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]["lr"]

            if new_lr != current_lr:
                current_lr = new_lr
                print(f"Learning rate updated: {current_lr}")

            train_loss = val_loss
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                patience = 0
                print(f'model was saved with loss = {best_train_loss}')
                # torch.save(model.state_dict(), model_save_path)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "beta": beta,
                        # "epoch": epoch,
                    },
                    model_save_path,
                )
            else:
                patience += 1
                if patience == 10:
                    if beta > min_beta:
                        beta = beta * lambd

        # print('epoch: {}, beta = {:.6f}, Train MSE: {:.6f}, Train CE:{:.6f}, Train KL:{:.6f}, Train ACC:{:6f}'.format(epoch, beta, num_loss, cat_loss, kl_loss, train_acc.item()))
        print(
            "epoch: {}, beta = {:.6f}, Train MSE: {:.6f}, Train CE:{:.6f}, Train KL:{:.6f}, Val MSE:{:.6f}, Val CE:{:.6f}, Train ACC:{:6f}, Val ACC:{:6f}".format(
                epoch,
                beta,
                num_loss,
                cat_loss,
                latent_loss,
                avg_val_mse_loss,
                avg_val_ce_loss,
                train_acc.item(),
                avg_val_acc,
            )
        )

    end_time = time.time()
    print("Training time: {:.4f} mins".format((end_time - start_time) / 60))

    # Saving latent embeddings
    with torch.no_grad():
        pre_encoder.load_weights(model)
        pre_decoder.load_weights(model)

        torch.save(pre_encoder.state_dict(), encoder_save_path)
        torch.save(pre_decoder.state_dict(), decoder_save_path)

        X_train_num = X_train_num.to(device)
        X_train_cat = X_train_cat.to(device)

        print("Successfully load and save the model!")

        train_z = pre_encoder(X_train_num, X_train_cat).detach().cpu().numpy()

        np.save(f"{ckpt_dir}/train_z.npy", train_z)

        print("Successfully save pretrained embeddings in disk!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Variational Autoencoder")

    parser.add_argument(
        "--dataname", type=str, default="adult", help="Name of dataset."
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU index.")
    parser.add_argument("--max_beta", type=float, default=1e-2, help="Initial Beta.")
    parser.add_argument("--min_beta", type=float, default=1e-5, help="Minimum Beta.")
    parser.add_argument("--lambd", type=float, default=0.7, help="Decay of Beta.")
    parser.add_argument("--vq", action="store_const", const=True, default=False)


    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = "cuda:{}".format(args.gpu)
    else:
        args.device = "cpu"
    print(f'GPU device: {args.device}')