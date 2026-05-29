# SPDX-License-Identifier: MIT
import argparse
from pathlib import Path


PRESETS = {
    "rat": {
        "cfg": "cfg/release/rat.yml",
        "ckpt_path": "weights/rat/checkpoint_best.pt",
        "sampling_steps": 10,
        "solver": "euler",
    },
    "babel": {
        "cfg": "cfg/release/babel.yml",
        "ckpt_path": "weights/babel/checkpoint_best.pt",
        "sampling_steps": 10,
        "solver": "euler",
    },
    "rat_bnd": {
        "cfg": "cfg/release/rat_bnd.yml",
        "ckpt_path": "weights/rat_bnd/checkpoint_best.pt",
        "sampling_steps": 100,
        "solver": "lin_poly",
        "lin_poly_p": 5,
        "lin_poly_long_step": 1000,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Public evaluation presets for CogFlow release.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="rat")
    parser.add_argument("--cfg", type=str, default=None, help="Optional config override.")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Optional checkpoint override.")
    parser.add_argument("--data_dir", type=str, default=None, help="Optional dataset root override.")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--sampling_steps", type=int, default=None)
    parser.add_argument("--solver", choices=["euler", "lin_poly"], default=None)
    parser.add_argument("--lin_poly_p", type=int, default=None)
    parser.add_argument("--lin_poly_long_step", type=int, default=None)
    parser.add_argument("--save_samples", action="store_true", default=False)
    parser.add_argument("--eval_on_train", action="store_true", default=False)
    parser.add_argument("--fix_random_seed", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_eval_namespace(args):
    preset = PRESETS[args.preset]
    ckpt_path = args.ckpt_path or preset["ckpt_path"]
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. "
            "Download the public weight bundle and place it under the default weights directory, "
            "or pass --ckpt_path explicitly."
        )

    return argparse.Namespace(
        ckpt_path=ckpt_path,
        cfg=args.cfg or preset["cfg"],
        exp="public_release",
        save_samples=args.save_samples,
        eval_on_train=args.eval_on_train,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        n_train=None,
        n_test=None,
        rotate=False,
        data_norm=None,
        data_source=None,
        subset=None,
        rotate_time_frame=None,
        num_workers=args.num_workers,
        fix_random_seed=args.fix_random_seed,
        seed=args.seed,
        sampling_steps=args.sampling_steps or preset["sampling_steps"],
        solver=args.solver or preset["solver"],
        lin_poly_p=args.lin_poly_p or preset.get("lin_poly_p", 2),
        lin_poly_long_step=args.lin_poly_long_step or preset.get("lin_poly_long_step", 1000),
        method="cogflow",
        variant=None,
        decoder=None,
        action_fusion=None,
        num_regime=None,
        m2_decoder_style="historical_pre_film",
        sde_control_style="encoded",
        enable_dissipativity=False,
        dissipativity_weight=None,
    )


def main():
    args = parse_args()
    from eval_utils import run_evaluation

    eval_args = build_eval_namespace(args)
    run_evaluation(eval_args)


if __name__ == "__main__":
    main()

