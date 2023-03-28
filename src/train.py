import argparse
import os
import shutil
from pprint import pprint
import yaml

from autogluon.multimodal import MultiModalPredictor


def get_input_path(path):
    file = os.listdir(path)[0]
    if len(os.listdir(path)) > 1:
        print(f"WARN: more than one file is found in {channel} directory")
    print(f"Using {file}")
    filename = f"{path}/{file}"
    return filename


def get_env_if_present(name):
    result = None
    if name in os.environ:
        result = os.environ[name]
    return result


if __name__ == "__main__":
    # Disable Autotune
    os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"

    # ------------------------------------------------------------ Args parsing
    print("Starting AG")
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    parser.add_argument(
        "--output-data-dir", type=str, default=get_env_if_present("SM_OUTPUT_DATA_DIR")
    )
    parser.add_argument(
        "--model-dir", type=str, default=get_env_if_present("SM_MODEL_DIR")
    )
    parser.add_argument("--n_gpus", type=str, default=get_env_if_present("SM_NUM_GPUS"))
    parser.add_argument(
        "--train_dir", type=str, default=get_env_if_present("SM_CHANNEL_TRAIN")
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        required=False,
        default=get_env_if_present("SM_CHANNEL_TEST"),
    )
    parser.add_argument(
        "--ag_config", type=str, default=get_env_if_present("SM_CHANNEL_CONFIG")
    )
    parser.add_argument(
        "--serving_script", type=str, default=get_env_if_present("SM_CHANNEL_SERVING")
    )

    args, _ = parser.parse_known_args()

    print(f"Args: {args}")

    # See SageMaker-specific environment variables: https://sagemaker.readthedocs.io/en/stable/overview.html#prepare-a-training-script
    os.makedirs(args.output_data_dir, mode=0o777, exist_ok=True)

#     config_file = get_input_path(args.ag_config)
#     with open(config_file) as f:
#         config = yaml.safe_load(f)  # AutoGluon-specific config

#     if args.n_gpus:
#         config["num_gpus"] = int(args.n_gpus)

#     print("Running training job with the config:")
#     pprint(config)

    # ---------------------------------------------------------------- Training
    
    train_dir = args.train_dir
    
    # config 명시로 변경필요
    train_path = os.path.join(train_dir, "Annotations", "usersplit_train_cocoformat.json")
    
    model_path = args.model_dir
        
    # config 명시로 변경필요
    checkpoint_name = "yolox_x_8x8_300e_coco"  
    num_gpus = -1  # use all GPUs
    val_metric = "map"
    
    predictor = MultiModalPredictor(
        hyperparameters={
            "model.mmdet_image.checkpoint_name": checkpoint_name,
            "env.num_gpus": num_gpus,
            "optimization.val_metric": val_metric,
        },
        problem_type="object_detection",
        sample_data_path=train_path,
        path=model_path
    )
    
    # hyperparam 또한 config명시로 변경필요
    predictor.fit(
        train_path,
        hyperparameters={
            "optimization.learning_rate": 5e-6, # we use two stage and detection head has 100x lr
            "optimization.max_epochs": 1,   # for the real use case, at least 50
            "optimization.check_val_every_n_epoch": 1, # make sure there is at least one validation
            "env.per_gpu_batch_size": 2,  # decrease it when model is large
        },
    )

#     train_file = get_input_path(args.train_dir)
#     train_data = TabularDataset(train_file)

#     save_path = os.path.normpath(args.model_dir)

#     ag_predictor_args = config["ag_predictor_args"]
#     ag_predictor_args["path"] = save_path
#     ag_fit_args = config["ag_fit_args"]
    
#     predictor = TabularPredictor(**ag_predictor_args).fit(train_data, **ag_fit_args)

    # --------------------------------------------------------------- Inference

    # if args.test_dir:
    #     pass
    # else:
    #     if config.get("leaderboard", False):
    #         lb = predictor.leaderboard(silent=False)
    #         lb.to_csv(f"{args.output_data_dir}/leaderboard.csv")

    if args.serving_script:
        print("Saving serving script")
        serving_script_saving_path = os.path.join(save_path, "code")
        os.mkdir(serving_script_saving_path)
        serving_script_path = get_input_path(args.serving_script)
        shutil.move(
            serving_script_path,
            os.path.join(
                serving_script_saving_path, os.path.basename(serving_script_path)
            ),
        )
