from tqdm import tqdm

import torch

from tabsyn.utils.latent_utils import SeqLatentPredictionDataset

from tabsyn.utils.training_utils import (
    reshape_sample
)

from tabsyn.utils.latent_utils import (
    get_generate_info,
    split_num_cat_target,
    recover_data,
)

from .metric_utils import MetricEstimator

import delu


class SampleEvaluator:

    def __init__(self, model, config, logger, train_state):
        self.model = model
        self.config = config
        self.logger = logger
        self.train_state = train_state
        self.name_prefix : str = "",
        self.blim = None
        self.rec_info, self.num_inverse, self.cat_inverse = get_generate_info(self.config)

    def evaluate(self, loader, blim=None, discr=""):
        self.name_prefix = discr
        self.blim = blim
        gen_samples, gt_samples, gt_client_ids, has_samples = self.__generate_samples(loader)
        if has_samples:
            gt_df_save_path, gen_df_save_path  = self.__recover_and_save_samples(gen_samples, gt_samples, gt_client_ids)
            self.__evaluate_and_save(gt_df_save_path, gen_df_save_path)
        else:
            self.logger.info("Нет сэмплов для оценки.")



    def __generate_samples(self, data_loader):
        self.model.eval()
        
        pbar = tqdm(data_loader, total=len(data_loader))

        gen_samples = []
        gt_samples = []
        gt_client_ids = []

        for batch_idx, samp_batch in enumerate(pbar):
            if self.blim is not None and batch_idx > self.blim:
                break
            samp_inp = samp_batch

            B = samp_inp.size(0)

            with torch.no_grad():
                samp_res = self.model(
                    samp_inp, 
                    max_steps=self.config["EST_GEN_LEN"], # Это будет в батче?
                    rvqvae_decoder=self.rec_info['rvq_ema'] # We need to decode our data before generate next token from generated
                    ) 
            
            # TODO: удалить, но учесть в батче
            # match self.config["DATA_MODE"]:
            #     case "num2cat":
            #         stacked = [
            #             reshape_sample(sample, desired_rows=self.config["EST_GEN_LEN"], group_size=self.config["RVQ_STAGES"])     
            #             for sample in samp_res
            #         ]
            #     case "num2mnum":
            #         stacked = [
            #             reshape_sample(
            #                 sample, 
            #                 desired_rows=self.config["EST_GEN_LEN"], 
            #                 group_size=self.config["RVQ_STAGES"] * self.config["VAE_HIDDEN_DIM"]
            #                 ).reshape(
            #                     self.config["EST_GEN_LEN"], self.config["RVQ_STAGES"], self.config["VAE_HIDDEN_DIM"]
            #                 )
            #             for sample in samp_res
            #         ]
            #     case "num2num":
            #         stacked = [
            #             reshape_sample(sample, desired_rows=self.config["EST_GEN_LEN"], group_size=self.config["VAE_HIDDEN_DIM"])     
            #             for sample in samp_res
            #         ]
            #     case _:
            #         raise NotImplementedError

 
            # stacked = torch.stack(samp_res)
            
            # TODO: удалить, но учесть в батче            
            # match self.config["DATA_MODE"]:
            #     case "num2cat":
            #         samp_batch_info["hist_seqs"] = samp_batch_info["hist_seq_codes"]
            #         samp_batch_info["gen_seqs"] = samp_batch_info["gen_seq_codes"]
            #     case "num2mnum":
            #         samp_batch_info["hist_seqs"] = samp_batch_info["hist_seq_residual_codes"]
            #         samp_batch_info["gen_seqs"] = samp_batch_info["gen_seq_residual_codes"]
            #     case "num2num":
            #         samp_batch_info["hist_seqs"] = samp_batch_info["hist_seqs"]
            #         samp_batch_info["gen_seqs"] = samp_batch_info["gen_seqs"]
            #     case _:
            #         raise NotImplementedError
                
            # TODO: Функция в батче для возврата hist + gt и hist + gen
            # (
            #     gt_batch_z,
            #     gt_batch_ids,
            # ) = SeqLatentPredictionDataset.batch2raw_data(samp_batch)

            # gt_samples.append(gt_batch_z.cpu())
            # gt_client_ids.append(gt_batch_ids.cpu())

            # samp_batch_info["gen_seqs"] = stacked.cpu()
            # samp_batch_info["gen_seqs"] = stacked

            # (
            #     gen_batch_z,
            #     _,
            # ) = SeqLatentPredictionDataset.batch2raw_data(samp_batch_info)
            # gen_samples.append(gen_batch_z.cpu())

            del samp_batch

        delu.cuda.free_memory()
        return gen_samples, gt_samples, gt_client_ids
    

    def __recover_and_save_samples(self, gen_samples, gt_samples, gt_client_ids):
    
        gt_client_ids = torch.cat(gt_client_ids)

        _dfs = dict(
            gen=None,
            gt=None,
        )

        gen_df_save_path = self.config["artr_ckpt"] / "validation_gen.csv"
        gt_df_save_path = self.config["artr_ckpt"] / "validation_gt.csv"

        for parti, samples, df_save_path in zip(
            ["gen", "gt"],
            [gen_samples, gt_samples],
            [gen_df_save_path, gt_df_save_path],
        ):

            # TODO: Make class Reconstructor - будет делать Latent to Data в зависимости от encoder/decoder.
            # TODO: Функция просто получает выход модели, а возвращает реальные данные
            
            samples = torch.cat(samples).detach().cpu().numpy()

            syn_num, syn_cat, syn_target = split_num_cat_target(
                samples,
                self.rec_info,
                self.num_inverse,
                self.cat_inverse,
                client_ids=gt_client_ids.numpy(),
                config=self.config
            )
            syn_df = recover_data(syn_num, syn_cat, syn_target, self.rec_info)

            idx_name_mapping = self.rec_info["idx_name_mapping"]
            idx_name_mapping = {
                int(key): value for key, value in idx_name_mapping.items()
            }

            syn_df.rename(columns=idx_name_mapping, inplace=True)
            syn_df["event_time"] = syn_df.groupby("client_id")["time_diff_days"].cumsum()
            # syn_df = recover_client_ids(syn_df, gt_client_ids.numpy(), "client_id")

            syn_df.to_csv(df_save_path, index=False)
            _dfs[parti] = syn_df
            self.logger.info(f"Saving validation {parti} data to {df_save_path}")

        return gt_df_save_path, gen_df_save_path

    def __evaluate_and_save(self, gt_df_save_path, gen_df_save_path):
        # TODO: функция принимает путь до gt и gen. 

        self.logger.info(f"Epoch: {self.train_state['epoch']}; metric eval started")

        metric_estimator = MetricEstimator(gt_df_save_path, gen_df_save_path, self.config, self.logger, self.train_state)
        metric_estimator.set_name_fixes(name_prefix=self.name_prefix)
        metric_estimator.estimate()