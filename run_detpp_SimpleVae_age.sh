#!/bin/bash
python main.py --config_factory "[start,datasets/age/age,methods/detpp_SimpleVAE_frozen_bts,best_params/age/detpp1.0]" --run_name "detpp_SimpleVAE_frozen_bts"
python main.py --config_factory "[start,datasets/age/age,methods/detpp_SimpleVAE_frozen,best_params/age/detpp1.0]" --run_name "detpp_SimpleVAE_frozen"
python main.py --config_factory "[start,datasets/age/age,methods/detpp_SimpleVAE_train_bts,best_params/age/detpp1.0]" --run_name "detpp_SimpleVAE_train_bts"
python main.py --config_factory "[start,datasets/age/age,methods/detpp_SimpleVAE_train,best_params/age/detpp1.0]" --run_name "detpp_SimpleVAE_train"
