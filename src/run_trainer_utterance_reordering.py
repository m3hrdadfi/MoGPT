import datetime
import os
import pprint
from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
import torch
from huggingface_hub import Repository
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer
from transformers.utils import get_full_repo_name

import wandb
# from data_modules.binary_utterance_reordering import UtteranceReorderingDataModule
from data_modules.token_utterance_reordering import UtteranceReorderingDataModule as TokenUtteranceReorderingDataModule
from data_modules.binary_utterance_reordering import UtteranceReorderingDataModule as BinaryUtteranceReorderingDataModule
from models.lite_gpt2_dh import LiteGPT2DoubleHeadsModel
from utils.generation import generation
from utils.generation_utils import generation_kwargs
from utils.helpers import get_gpu_memory

SEED = os.environ.get("SEED", None)
try:
    SEED = int(SEED)
except TypeError:
    SEED = None
NO_DECAY = ["bias", "LayerNorm.weight"]


def main():
    print(f"\n***\nSEED: {SEED}\n***\n")

    if SEED:
        # pl.seed_everything(SEED, workers=True)
        pl.seed_everything(SEED)

    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Sanity checking")
    parser.add_argument("--with_persona", action="store_true", help="With persona")

    parser.add_argument("--project_name", type=str)
    parser.add_argument("--run_name", type=str)

    # DataModule
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--validation_file", type=str, default=None)
    parser.add_argument("--test_file", type=str, default=None)
    parser.add_argument("--predict_file", type=str, default=None)
    parser.add_argument("--ignore_index", type=int, default=-100)
    parser.add_argument("--label_name", type=str, default="labels")
    parser.add_argument("--aux_name", type=str, default="aux_labels")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--preprocessing_num_workers", type=int, default=1)
    parser.add_argument("--load_from_cache_file", action="store_false")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--limit_train_samples", type=int, default=0)
    parser.add_argument("--limit_val_samples", type=int, default=0)
    parser.add_argument("--limit_test_samples", type=int, default=0)
    parser.add_argument("--limit_predict_samples", type=int, default=0)

    # ModelModule
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="gpt2")
    parser.add_argument("--load_weights", action="store_false")
    parser.add_argument("--deepspeed_sharding", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--do_prob_ratio", type=float, default=0.15)
    parser.add_argument("--reorder_prob_ratio", type=float, default=0.20)
    parser.add_argument("--do_on", type=str, default="RANDOM")
    parser.add_argument("--do_type", type=str, default="BINARY")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--num_freezing_layers", type=int, default=0, help="Number of freezing layers."
    )

    # Callbacks
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`.")
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--save_checkpoints", action="store_true")
    parser.add_argument("--wandb_key", type=str, default=None)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # GPUs checking
    if args.accelerator.lower() == "gpu" and not args.debug:
        gpus = get_gpu_memory()
        print(f"Number of GPUs: {gpus}")
        # if not all([True if gpu > 35 else False for gpu in gpus]):
        #     raise Exception("There is something wrong with GPU allocation on the server side!")

    # Sanity checking
    if (args.train_file is not None and os.path.exists(args.train_file)) or (args.validation_file is not None and os.path.exists(args.validation_file)):
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, json or txt file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, json or txt file."
    else:
        raise ValueError("Need either a dataset name or a training/validation file.")

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    if args.wandb_key == "None" or args.wandb_key == "":
        args.wandb_key = None

    now = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M")
    if args.output_dir:
        app_name = f"{args.run_name}_{args.do_type.lower()}_alpha-{args.alpha}_do_prob-{args.do_prob_ratio}_reorder_prob-{args.reorder_prob_ratio}_{now}"
        ckpt_dir = os.path.join(args.output_dir, args.project_name, app_name)
        log_dir = os.path.join(ckpt_dir, "logs")
        args.output_dir = ckpt_dir

        if not args.push_to_hub:
            os.makedirs(ckpt_dir, exist_ok=True)
            os.makedirs(log_dir, exist_ok=True)

    repo = None
    if args.push_to_hub:
        if args.hub_model_id is None:
            repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
        else:
            repo_name = args.hub_model_id

        repo = Repository(args.output_dir, clone_from=repo_name, use_auth_token=args.hub_token)

        with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
            if "step_*" not in gitignore:
                gitignore.write("step*\n")
            if "epoch_*" not in gitignore:
                gitignore.write("epoch*\n")
    elif args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # Args overview
    pprint.pp(args.__dict__)
    model_name_or_path = "gpt2" if "gpt2" in args.pretrained_model_name_or_path else args.pretrained_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, add_prefix_space=True)

    persona_token = "<persona>"
    context_token = "<context>"
    dialog_token = "<dialog>"
    mask_token = "<mask>"
    sep_token = "<sep>"
    bos_token = "<bos>"
    eos_token = "<eos>"
    pad_token = "<pad>"

    special_tokens = {
        "bos_token": bos_token,
        "eos_token": eos_token,
        "pad_token": pad_token,
        "mask_token": mask_token,
        'additional_special_tokens': [sep_token, persona_token, context_token, dialog_token]
    }
    tokenizer.add_special_tokens(special_tokens)

    features_kwargs = {
        "persona_token": "<persona>",
        "context_token": "<context>",
        "dialog_token": "<dialog>",
        "sep_token": "<sep>",
        "mask_token": "<mask>",
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
        "ignore_keys": ["labels"],
        "zero_keys": [],
        "ignore_index": args.ignore_index,
    }
    args.do_type = args.do_type.upper()

    if args.do_type == "TOKEN":
        features_kwargs["ignore_keys"].append("aux_labels")

        dm = TokenUtteranceReorderingDataModule(
            tokenizer=tokenizer,
            train_file=args.train_file,
            validation_file=args.validation_file,
            test_file=args.test_file,
            predict_file=args.predict_file,
            ignore_index=args.ignore_index,
            label_name=args.label_name,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            preprocessing_num_workers=args.preprocessing_num_workers,
            load_from_cache_file=args.load_from_cache_file,
            cache_dir=args.cache_dir,
            limit_train_samples=args.limit_train_samples if args.limit_train_samples > 0 else None,
            limit_val_samples=args.limit_val_samples if args.limit_val_samples > 0 else None,
            limit_test_samples=args.limit_test_samples if args.limit_test_samples > 0 else None,
            limit_predict_samples=args.limit_predict_samples if args.limit_predict_samples > 0 else None,
            with_persona=args.with_persona,
            convert_to_features_kwargs=features_kwargs,
            do_on=args.do_on,
            do_prob=args.do_prob_ratio,
            reorder_prob=args.reorder_prob_ratio,
        )
    else:
        features_kwargs["zero_keys"].append("aux_labels")
        dm = BinaryUtteranceReorderingDataModule(
            tokenizer=tokenizer,
            train_file=args.train_file,
            validation_file=args.validation_file,
            test_file=args.test_file,
            predict_file=args.predict_file,
            ignore_index=args.ignore_index,
            label_name=args.label_name,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            preprocessing_num_workers=args.preprocessing_num_workers,
            load_from_cache_file=args.load_from_cache_file,
            cache_dir=args.cache_dir,
            limit_train_samples=args.limit_train_samples if args.limit_train_samples > 0 else None,
            limit_val_samples=args.limit_val_samples if args.limit_val_samples > 0 else None,
            limit_test_samples=args.limit_test_samples if args.limit_test_samples > 0 else None,
            limit_predict_samples=args.limit_predict_samples if args.limit_predict_samples > 0 else None,
            with_persona=args.with_persona,
            convert_to_features_kwargs=features_kwargs,
            do_on=args.do_on,
            do_prob=args.do_prob_ratio,
            reorder_prob=args.reorder_prob_ratio,
        )
    dm.setup()

    print("===" * 20, "TRAIN")

    for batch in dm.train_dataloader():
        print({k: v.shape for k, v in batch.items()})

        print("INPUT_IDS", batch["input_ids"][0])
        print("LABELS", batch["labels"][0])

        if "aux_labels" in batch:
            print("INPUT", tokenizer.decode(batch["input_ids"][0]))
            print("OUTPUT", tokenizer.decode([i for i in batch["labels"][0] if i > -1]))
            # print("AUXI", tokenizer.decode([i for i, j in zip(batch["input_ids"][0], batch["aux_labels"][0]) if j == 1]))
            print("AUX_LABELS", batch["aux_labels"][0])

        if batch["length"] is None:
            raise Exception("TRAINING BUG")

        if not args.debug:
            break

    # raise
    print("===" * 20, "VALIDATION")

    for batch in dm.val_dataloader():
        print({k: v.shape for k, v in batch.items()})

        print("INPUT_IDS", batch["input_ids"][0])
        print("LABELS", batch["labels"][0])

        if "aux_labels" in batch:
            print("AUX_LABELS", batch["aux_labels"][0])

        # if batch["length"] is None:
        #     raise Exception("TRAINING BUG")

        if not args.debug:
            break

    print("===" * 20, "TEST")

    for batch in dm.test_dataloader():
        print({k: v.shape for k, v in batch.items()})

        print("INPUT_IDS", batch["input_ids"][0])
        print("LABELS", batch["labels"][0])

        # if batch["length"] is None:
        #     raise Exception("TRAINING BUG")

        if not args.debug:
            break

    print("DATA PROCESSING IS FINISHED \n")
    print(args.do_type)
    # print(features_kwargs)
    # raise

    num_labels, aux_num_labels = 1, len(tokenizer) if args.do_type == "TOKEN" else 2
    model = LiteGPT2DoubleHeadsModel(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        tokenizer=tokenizer,
        lr=args.lr,
        no_decay=NO_DECAY,
        weight_decay=args.weight_decay,
        num_warmup_steps=args.num_warmup_steps,
        num_labels=num_labels,
        aux_num_labels=aux_num_labels,
        alpha=args.alpha
    )
    print(model)
    # raise

    # Setup wandb
    loggers = []
    wandb_logger = None
    if isinstance(args.wandb_key, str) and len(args.wandb_key) > 0:
        # run_name = f"{args.run_name}_alpha-{args.alpha}_prob-{args.prob_ratio}_clsH-{args.classifier_head_num_layers}_{now}"
        wandb.login(key=args.wandb_key, relogin=True)
        # wandb_logger = WandbLogger(name=app_name, project=args.project_name, log_model=args.report_to)
        wandb_logger = WandbLogger(name=app_name, project=args.project_name, log_model=False)
        loggers += [wandb_logger]

    if args.tensorboard and log_dir:
        loggers += [
            pl.loggers.TensorBoardLogger(
                save_dir=log_dir,
                version="version_" + datetime.datetime.now().strftime("%d-%m-%Y--%H-%M-%S"),
                name=f"{args.project_name}-{args.run_name}"
            )
        ]

    callbacks = [
        pl.callbacks.RichProgressBar(),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        # GenerateResponseAndPushModelToHubCallback(
        #     tokenizer=tokenizer,
        #     data_loader=dm.predict_dataloader(),
        #     wandb_logger=wandb_logger,
        #     args=args,
        #     repo=repo,
        #     **generation_kwargs
        # ),
    ]
    if args.early_stopping:
        callbacks += [
            pl.callbacks.EarlyStopping(
                monitor="val_lm_loss_epoch",
                min_delta=0.0,
                patience=10,
                verbose=True,
                mode="min"
            )
        ]

    if args.save_checkpoints and ckpt_dir:
        callbacks += [
            pl.callbacks.ModelCheckpoint(
                dirpath=ckpt_dir,
                filename="{epoch:02d}-{val_loss:.2f}",
                save_top_k=2,
                verbose=True,
                monitor="val_lm_loss_epoch",
                mode="min"
            )
        ]

    training_args = {
        "enable_progress_bar": True,
        "num_sanity_val_steps": 0,
        "logger": loggers if len(loggers) > 0 else True,
        # "log_every_n_steps": args.log_every_n_steps,
        # "val_check_interval": args.val_check_interval,
        # "gpus": args.gpus,
        "val_check_interval": 1.0,
        "check_val_every_n_epoch": 1,
        "accelerator": args.accelerator,
        "devices": args.devices,
        "max_epochs": args.max_epochs,
        "callbacks": callbacks,
    }
    if args.precision and args.precision != "None":
        training_args["precision"] = args.precision
    if args.accumulate_grad_batches and int(args.accumulate_grad_batches) > 1:
        training_args["accumulate_grad_batches"] = args.accumulate_grad_batches
    if args.gradient_clip_val and int(args.gradient_clip_val) > 0:
        training_args["gradient_clip_val"] = args.gradient_clip_val
    if args.strategy != "None":
        training_args["strategy"] = args.strategy

    pprint.pp(training_args)
    trainer = pl.Trainer(**training_args)
    trainer.fit(model, datamodule=dm)
    trainer.test(model, dm.test_dataloader())
    generation(
        data_loader=dm.predict_dataloader(),
        model=model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        tokenizer=tokenizer,
        generation_kwargs=generation_kwargs,
        args=args,
        wandb_logger=wandb_logger,
    )
    trainer.save_checkpoint(os.path.join(ckpt_dir, "epoch_last.ckpt"))
    model.save_hf_checkpoint(args.output_dir)
    model.tokenizer.save_pretrained(args.output_dir)


if __name__ == '__main__':
    main()
