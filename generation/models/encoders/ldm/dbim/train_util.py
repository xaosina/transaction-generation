from typing import Mapping, Any, Dict
from .karras_diffusion import (
    KarrasDenoiser,
    VPNoiseSchedule,
    VENoiseSchedule,
    I2SBNoiseSchedule,
    DDBMPreCond,
    I2SBPreCond,
)
import logging

logger = logging.getLogger(__name__)

def get_sampling_defaults(params: Mapping[str, Any]) -> Dict:
    s_params = dict(
        mask=None,
        rho=7.0,
        sampler="heun",
        churn_step_ratio=0.0,
        eta=0.0,
        order=2,
    )
    gen_sampler = params['gen_sampler']
    nfe = params['sampling_nfe']
    s_params['gen_sampler'] = gen_sampler

    s_params['sampler'] = gen_sampler
    if gen_sampler == "heun":
        s_params['steps'] = int((nfe + 1) / 3.)
        s_params['churn_step_ratio'] = 0.33
    elif gen_sampler == "dbim":
        s_params['steps'] = nfe - 1
        s_params['eta'] = 0.0
    elif gen_sampler == 'dbim_high_order':
        s_params['steps'] = nfe - 1
        s_params['order'] = 2
    else:
        raise Exception(f'Unsupported gen_sampler "{gen_sampler}"')
    return s_params

def get_sampling_params(params: Mapping[str, Any]) -> Dict:
    s_params = get_sampling_defaults(params)
    keys = list(s_params.keys())
    s_params = s_params | dict(params)
    return {k:s_params[k] for k in keys}


def data_defaults() -> Dict:
    return dict(
        sigma_data=0.5,
        cov_xy=0.0,
    )

def noise_schedule_defaults(noise_schedule: str) -> Dict:
    if noise_schedule.startswith("ve"):
        return dict(
            sigma_max=80.0,
            sigma_min=0.002,
        )
    if noise_schedule.startswith("vp"):
        return dict(
            beta_d=2,
            beta_min=0.1,
            sigma_max=1,
            sigma_min=0.0001,
        )
    if noise_schedule.startswith('i2sb'):
        return dict(
            beta_max=1.0,
            beta_min=0.1,
            sigma_max=1,
            sigma_min=0.0001,
        )
    else:
        raise Exception(f'Unsupported noise schedule "{noise_schedule}"')
    

def get_diffusion(params: Mapping[str, Any]) -> KarrasDenoiser:

    # noise_schedule

    noise_schedule = params['noise_schedule']

    ns_params = noise_schedule_defaults(noise_schedule) | dict(params)
    data_params = data_defaults() | dict(params)

    upd_data_params = dict(set(data_params.items()) - set(dict(params).items()))
    if len(upd_data_params) > 0:
        logger.warning(
            f'Bridge diffusion utilizes default data params: "{upd_data_params}"'
        )

    if noise_schedule.startswith("vp"):
        ns = VPNoiseSchedule(beta_d=ns_params['beta_d'], beta_min=ns_params['beta_min'])
        precond = DDBMPreCond(ns, sigma_data=data_params['sigma_data'], cov_xy=data_params['cov_xy'])
    elif noise_schedule == "ve":
        ns = VENoiseSchedule(sigma_max=ns_params['sigma_max'])
        precond = DDBMPreCond(ns, sigma_data=data_params['sigma_data'], cov_xy=data_params['cov_xy'])
    elif noise_schedule.startswith("i2sb"):
        ns = I2SBNoiseSchedule(beta_max=ns_params['beta_max'], beta_min=ns_params['beta_min'])
        precond = I2SBPreCond(ns)
    


    return KarrasDenoiser(
        noise_schedule=ns,
        precond=precond,
        t_max=ns_params['sigma_max'],
        t_min=ns_params['sigma_min'],
    )