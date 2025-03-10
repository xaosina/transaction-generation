import os
import json
import numpy as np
import pandas as pd
import torch
from typing import Any, Dict, Literal, Tuple
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from einops import rearrange
from utils_train import preprocess
from tabsyn.rvqvae.model import Model_VAE as RVQVAE_Model
from tabsyn.rvqvae.model import Decoder_model as RVQVAE_Decoder
from tabsyn.vae.model import Model_VAE as VAE_Model
from tabsyn.vae.model import Decoder_model as VAE_Decoder
from torch.profiler import record_function


##------------------------------------------------------------
#               Код, который нужно переписать
# ##------------------------------------------------------------
def get_generate_info(config):

    dataset_dir = f"data/{config['dataname']}_with_id"

    with open(f"{dataset_dir}/info.json", "r") as f:
        info = json.load(f)

    task_type = info["task_type"]

    _, _, categories, d_numerical, num_inverse, cat_inverse = preprocess(
        dataset_dir, task_type=task_type, inverse=True
    )

    d_numerical -= 1
    categories.pop(0)


    # decoder_save_path = f"{config['vae_ckpt_dir']}/decoder.pt"
    
    # try:
    #     pre_decoder.load_state_dict(torch.load(decoder_save_path))
    # except:

    if config["vae_type"] == 'rvqvae':
        
        pre_decoder = RVQVAE_Decoder(
            config['NUM_LAYERS'], d_numerical, categories, config['D_TOKEN'], 
            n_head=config['N_HEAD'], factor=config['FACTOR']
        ).to(config['device'])

        model = RVQVAE_Model(
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
        pre_decoder = VAE_Decoder(
            config['NUM_LAYERS'], d_numerical, categories, config['D_TOKEN'], 
            n_head=config['N_HEAD'], factor=config['FACTOR']
        ).to(config['device'])

        model = VAE_Model(
            config['NUM_LAYERS'],
            d_numerical,
            categories,
            config['D_TOKEN'],
            n_head=config['N_HEAD'],
            factor=config['FACTOR'],
            bias=True,
        )
    ckpt = torch.load(f"{config['vae_ckpt_dir']}/model.pt")
    model.load_state_dict(ckpt["model_state_dict"])

    pre_decoder.load_weights(model)
    # pre_decoder.load_state_dict(torch.load(decoder_save_path))

    info["pre_decoder"] = pre_decoder
    info["token_dim"] = config['D_TOKEN']
    info["rvq_ema"] = model.RVQ_VAE.rvq_ema if config["vae_type"] == "rvqvae" else None

    return (
        info,
        num_inverse,
        cat_inverse,
    )


class Reconstructor:
    """
    We could reconstruct the target from three types data
    codes - 16 codes of 16th codebooks
    quants - 16th aprox vectors of rvq
    no_recovers - ready vector - just return source
    """
    def __init__(self, type="no_recovers", dequant=None):
        self.dequantizer = dequant
        match type:
            case "nothing":
                self.recover = self.__just_return
            case "quants":
                self.recover = self.__sum_quants
            case "codes":
                self.recover = self.__from_codes
            case _:
                raise NotImplementedError
        
    def __just_return(self, data):
        return data
    
    def __sum_quants(self, data):
        return np.sum(data, axis=1)
    
    def __from_codes(self, data):
        return self.dequantizer.decode_codes(torch.tensor(data))
        


@torch.no_grad()
def split_num_cat_target(syn_data, info, num_inverse, cat_inverse, client_ids=None, save_logits=None, config=None,):

    task_type = info["task_type"]
    # assert task_type == 'binclass', 'currently only "binclass" is implemented corectly!'

    num_col_idx = info["num_col_idx"]
    cat_col_idx = info["cat_col_idx"]
    target_col_idx = info["target_col_idx"]

    n_num_feat = len(num_col_idx)
    n_cat_feat = len(cat_col_idx)

    if task_type == "regression":
        n_num_feat += len(target_col_idx)
    else:
        n_cat_feat += len(target_col_idx)

    pre_decoder = info["pre_decoder"]
    token_dim = info["token_dim"]
    dequant = info["rvq_ema"]
    
    rec = Reconstructor(config["reconstruction_type"], dequant)
    syn_data = rec.recover(syn_data)
    
    # syn_data = dequant.decode_codes(torch.tensor(syn_data))
    syn_data = syn_data.reshape(syn_data.shape[0], -1, token_dim)

    def get_decoded_data(data):
        from torch.utils.data import DataLoader,  TensorDataset
        if isinstance(data, torch.Tensor):
            dataset = TensorDataset(data)
        else:
            dataset = TensorDataset(torch.tensor(data))

        B = 1024

        dataloader = DataLoader(dataset, batch_size=B, shuffle=False)

        num_result = []
        cat_result = [torch.zeros(0)] * 4
        
        with torch.no_grad():
            for batch in dataloader:
                
                batch_data = batch[0].to(config['device'])
                
                norm_batch = pre_decoder(batch_data)
                
                num_result.extend(norm_batch[0])
                cat_features = [el.argmax(dim=-1) for el in norm_batch[1]]
                
                # cat_result.extend([el.argmax(dim=-1) for el in norm_batch[1]])
                for i in range(4):
                    cat_result[i] = torch.cat([cat_result[i], cat_features[i].cpu()])
                    ...

        return (num_result, cat_result)

    # norm_input = pre_decoder(torch.tensor(syn_data).to(config['device']))
    norm_input = get_decoded_data(syn_data)
    x_hat_num, x_hat_cat = norm_input

    # TODO: Save x_hat_cat[0] for test and train.
    if save_logits is not None:
        np.save(save_logits, x_hat_cat[0])
    
    syn_cat = x_hat_cat
    
    syn_num = torch.stack(x_hat_num).cpu().numpy()
    if task_type == 'binclass':
        syn_num = np.insert(syn_num, 0, 0.0, axis=1)  # dummy abs_time
        syn_cat = torch.stack(syn_cat).t().cpu().numpy()
    elif task_type == "regression":
        syn_num = np.insert(syn_num, 1, 0.0, axis=1)  # dummy abs_time 1
        # if client_ids is not None:
        #     syn_cat = np.insert(torch.stack(syn_cat).t().cpu().numpy(), 0, client_ids, axis=1)
        # else:
        #     syn_cat = np.insert(torch.stack(syn_cat).t().cpu().numpy(), 0, 0, axis=1)  # dummy client_id
        syn_cat = np.insert(torch.stack(syn_cat).t().cpu().numpy(), 0, 0, axis=1)  # dummy client_id

    syn_num = num_inverse(syn_num)
    # syn_cat[:, 2] = np.minimum(syn_cat[:, 2], 326)
    syn_cat = cat_inverse(syn_cat)
    syn_cat[:, 0] = client_ids

    if info["task_type"] == "regression":
        assert syn_num.shape[1] == len(num_col_idx) + len(target_col_idx)
        syn_target = syn_num[:, : len(target_col_idx)]
        syn_num = syn_num[:, len(target_col_idx) :]

    else:
        assert syn_cat.shape[1] == len(cat_col_idx) + len(target_col_idx)
        syn_target = syn_cat[:, : len(target_col_idx)]
        syn_cat = syn_cat[:, len(target_col_idx) :]

    return syn_num, syn_cat, syn_target


def recover_data(syn_num, syn_cat, syn_target, info):

    num_col_idx = info["num_col_idx"]
    cat_col_idx = info["cat_col_idx"]
    target_col_idx = info["target_col_idx"]

    idx_mapping = info["idx_mapping"]
    idx_mapping = {int(key): value for key, value in idx_mapping.items()}

    syn_df = pd.DataFrame()

    if info["task_type"] == "regression":
        for i in range(len(num_col_idx) + len(cat_col_idx) + len(target_col_idx)):
            if i in set(num_col_idx):
                syn_df[i] = syn_num[:, idx_mapping[i]]
            elif i in set(cat_col_idx):
                syn_df[i] = syn_cat[:, idx_mapping[i] - len(num_col_idx)]
            else:
                syn_df[i] = syn_target[
                    # :, idx_mapping[i] - len(num_col_idx) - len(cat_col_idx)
                    :, 0
                ]

    else:
        for i in range(len(num_col_idx) + len(cat_col_idx) + len(target_col_idx)):
            if i in set(num_col_idx):
                syn_df[i] = syn_num[:, idx_mapping[i]]
            elif i in set(cat_col_idx):
                syn_df[i] = syn_cat[:, idx_mapping[i] - len(num_col_idx)]
            else:
                syn_df[i] = syn_target[
                    :, idx_mapping[i] - len(num_col_idx) - len(cat_col_idx)
                ]

    return syn_df


class SeqLatentPredictionDataset(Dataset):

    def __init__(
            self,
            z,
            client_ids, 
            abs_times,
            z_codes = None,
            z_rq= None,
            prior_hstates : np.ndarray | None = None,
            min_hist_len : int = 16,
            gen_len : int = 16,
            mode: Literal['all', 'last', 'interval']  = 'all',
            sort_data: bool = False,
            carve_interval : int = 1,
        ) -> None:

        '''
        prior_hstates are assumed to be sorted (with order of ascending client IDs)!
        '''
        if len(z.shape) > 2:
            z = z[:, 1:, :] # If VAE. Rewrite save-latents.
            z = z.reshape(z.shape[0], -1)

        super().__init__()
        assert z.shape[0] == client_ids.shape[0], (z.shape[0], client_ids.shape[0])
        assert client_ids.shape[0] == abs_times.shape[0]

        df_seqs = pd.DataFrame(
            dict(
                client_id=client_ids.long(), 
                order=abs_times
            )
        )
        
        if sort_data:
            df_seqs = df_seqs.sort_values(
                ["client_id", "order"]
            )
            self.z = z[df_seqs.index.values]
            if z_codes is not None:
                self.z_codes = z_codes[df_seqs.index.values]
            else:
                self.z_codes = None
            if z_rq is not None:
                self.z_residual_codes = z_rq[df_seqs.index.values]
            else:
                self.z_residual_codes = z_rq
            self.abs_times = abs_times[df_seqs.index.values] 

        else:
            
            assert (
                df_seqs
                .groupby('client_id')
                .filter(lambda _df : not _df.order.is_monotonic_increasing)
                .empty
            )

            self.z = z
            self.z_codes = z_codes
            self.z_residual_codes = z_rq
            # self.age_cats = age_cats #TODO
            self.abs_times = abs_times

        #############################
        # lookup_df 
        # Columns:
        # - client_id
        # - seq_lens : length of the time seq for the client
        # - start_t : starting position in self.z tensor

        lookup_df = (
            df_seqs
            .groupby('client_id', as_index=False)
            .size()
            .rename(columns={"size": "seq_lens"})
        )

        lookup_df['start_t'] = (
            lookup_df['seq_lens']
            .cumsum()
            .shift(periods=1, fill_value=0)
        )
        self.lookup_df = lookup_df

        # hstates
        self.rets_phstates = prior_hstates is not None
        if self.rets_phstates:
            assert len(prior_hstates) == len(lookup_df)
            self.phstates = prior_hstates
            self.cid2phstate_idx = dict((v,k) for k,v in lookup_df.client_id.to_dict().items())

        ##############################
        # idx_df iterates over available training seqs
        # Columns:
        # - peruser_idx : idx of sequense corresponding to the user
        # - client_id
        # - hist_t : idx of user history in self.z
        # - gen_t : idx of gen seq in self.z
        
        self.idx_mode = mode
        self.gen_len = gen_len
        self.carve_interval = carve_interval
        min_length = min_hist_len + gen_len

        def tseqs_carver(x, min_hist_len=min_hist_len, gen_len=gen_len):
            assert len(x) == 1
            _start_t = x.start_t.iloc[0]
            _seq_len = x.seq_lens.iloc[0]
            assert _seq_len >= min_hist_len + gen_len

            if self.idx_mode == 'all':
                gen_idxs = np.arange(
                    _start_t + min_hist_len, 
                    _start_t + _seq_len - gen_len + 1)
            elif self.idx_mode == 'last':
                gen_idxs = [_start_t + _seq_len - gen_len,]
            elif self.idx_mode == 'last_x500':
                gen_idxs = [_start_t + _seq_len - gen_len,] * 500
            elif self.idx_mode == 'interval':
                _whole_len = _seq_len - gen_len - min_hist_len + 1
                gen_idxs = np.arange(
                    0, 
                    _whole_len, 
                    self.carve_interval
                )
                _diff = (_whole_len - 1 - gen_idxs[-1]) % self.carve_interval // 2
                gen_idxs += _diff
                gen_idxs += _start_t + min_hist_len
                assert gen_idxs[-1] <= _start_t + _seq_len - gen_len
            else:
                raise Exception()
            
            _df = pd.DataFrame(
                dict(
                    client_id=x['client_id'].values[0], 
                    hist_t=_start_t, 
                    gen_t=gen_idxs))
            return _df


        self.idx_df = (
            lookup_df
            .drop(lookup_df[lookup_df.seq_lens < min_length].index)
            .groupby('client_id', group_keys=False)
            .apply(tseqs_carver, include_groups=True)
            .reset_index()
            .rename(columns={"index": "peruser_idx"})
        )

        self.induced_client_ids = mode != "last"
        if self.induced_client_ids:
            self.idx_df['client_id'] = self.idx_df.index
        print('Length: ', len(self.idx_df))
    
    @staticmethod
    def batch2raw_data(
            batch: Dict,
        ) -> Tuple[
            torch.Tensor, # z
            torch.Tensor, # client ids
        ]:
        '''
        returns (z, client ids)
        '''
        hist_lens = batch['hist_lens']
        hist_seqs = batch['hist_seqs']
        client_ids = batch['client_ids']
        gen_seqs = batch['gen_seqs']
        
        bs = gen_seqs.size(0)
        gen_len = gen_seqs.size(1)
        _size = sum(hist_lens).item() + bs * gen_len
        z = torch.zeros((_size, *gen_seqs.shape[2:])).to(gen_seqs)
        client_ids_pop = torch.zeros((_size)).to(client_ids)
        _start_idx = 0
        for i in range(bs):
            _len = hist_lens[i]
            client_ids_pop[_start_idx:_start_idx + _len + gen_len] = client_ids[i]
            z[_start_idx:_start_idx + _len] = torch.tensor(np.array(hist_seqs[i][0:_len]))
            _start_idx += _len
            z[_start_idx:_start_idx + gen_len] = gen_seqs[i]
            _start_idx += gen_len
        
        return z, client_ids_pop


    @staticmethod
    def get_rawhist_batch(batch: Dict, config, max_len) -> torch.Tensor:
        with record_function("get_rawhist_batch"):
            
            gen = batch['gen_seqs']
            gen_codes = None
            hist = batch['hist_seqs']
            hist_codes = None

            def hist_process(data, max_len, long=False):
                tensors = [
                    torch.from_numpy(np.array(seq[-max_len:, :]))
                    for seq in data
                ]
                return torch.stack(tensors).long() if long else  torch.stack(tensors).float()

            with record_function("hist_process"):
                proms_list = hist_process(hist, max_len)

            match config["DATA_MODE"]:
                case "num2cat":
                    target = batch['gen_seq_codes']
                    hist_codes = batch["hist_seq_codes"]
                    proms_codes_list = hist_process(hist_codes, max_len, long=True)
                case "num2mnum":
                    with record_function("target ="):
                        target = batch['gen_seq_residual_codes']
                    with record_function("hist_codes ="):
                        hist_codes = batch["hist_seq_residual_codes"]
                    with record_function("proms_codes_list, _ = "):
                        proms_codes_list = hist_process(hist_codes, max_len, long=False)

                        # seq = seq.long().to(config['device']) if long else seq.float().to(config['device'])
                case "num2num":
                    target = gen
                    proms_codes_list = proms_list


            match config["TRAIN_MODE"]:
                case "TA":
                            
                    return (
                        proms_list.to(config['device']), 
                        proms_codes_list.to(config['device']),
                        )
                case "AL":
                    return (
                        proms_list.to(config['device']),
                        target.to(config['device'])
                    )
                case _:
                    NotImplementedError()

    @staticmethod
    def dloader_collate_fn(elems) -> Dict:
        with record_function("collator"):
            max_hist_len = max(map(lambda elem: len(elem['hist_seq']), elems))
            one_lat_dim = elems[0]['hist_seq'].shape[1:]
            
            client_ids = torch.tensor([elem['client_id'] for elem in elems]).long()
            last_seen_abs_times = torch.tensor([elem['last_seen_abs_time'] for elem in elems])
            gen_len = elems[0]['gen_seq'].shape[0]

            gen_seqs = torch.tensor(np.array([elem['gen_seq'] for elem in elems])).float()
            
            assert gen_seqs.shape == (len(elems), gen_len, *one_lat_dim) # (bs, gen_len, one_lat_dim)

            hist_seqs_list = [elem['hist_seq'] for elem in elems]
            hist_lens = torch.tensor(list(map(len, hist_seqs_list))).long()

            gen_seq_codes = None
            hist_seq_codes = None

            gen_seq_residual_codes = None
            hist_seq_residual_codes = None

            if elems[0]["gen_seq_codes"] is not None:
                gen_seq_codes = torch.tensor(np.array([elem["gen_seq_codes"] for elem in elems])).long()
                hist_seq_codes = [elem['hist_seq_codes'] for elem in elems]

            if elems[0]["gen_seq_residual_codes"] is not None:
                gen_seq_residual_codes = torch.tensor(np.array([elem["gen_seq_residual_codes"] for elem in elems])).float()
                hist_seq_residual_codes = [elem["hist_seq_residual_codes"] for elem in elems]

            res = dict(
                gen_seqs = gen_seqs,
                gen_seq_codes = gen_seq_codes,
                gen_seq_residual_codes = gen_seq_residual_codes,
                hist_seqs = hist_seqs_list,
                hist_seq_codes = hist_seq_codes,
                hist_seq_residual_codes = hist_seq_residual_codes,
                hist_lens = hist_lens,
                last_seen_abs_times = last_seen_abs_times,
                client_ids = client_ids,
            )


            return res

    @property
    def num_unique_clients(self) -> int:
        return self.idx_df.client_id.nunique()
    
    def __getitem__(self, index) -> Any:
        seq_info = self.idx_df.loc[index]
        client_id = seq_info.client_id

        hist_t, gen_t = seq_info.hist_t, seq_info.gen_t
        
        last_seen_abs_time = self.abs_times[gen_t - 1] # Last seen timestamp

        res = dict(
            client_id = client_id,
            gen_seq = self.z[gen_t:gen_t + self.gen_len],
            gen_seq_codes = self.z_codes[gen_t:gen_t + self.gen_len] if self.z_codes is not None else None,
            gen_seq_residual_codes = self.z_residual_codes[gen_t:gen_t + self.gen_len] if self.z_residual_codes is not None else None,
            hist_seq = self.z[hist_t:gen_t],
            hist_seq_codes = self.z_codes[hist_t:gen_t] if self.z_codes is not None else None,
            hist_seq_residual_codes = self.z_residual_codes[hist_t:gen_t] if self.z_residual_codes is not None else None,
            last_seen_abs_time = last_seen_abs_time
        )
        
        if self.rets_phstates:
            res['phstate'] = self.phstates[self.cid2phstate_idx[client_id]]
        return res
        
    def __len__(self) -> int:
        return len(self.idx_df)
