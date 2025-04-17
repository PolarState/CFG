import argparse
import logging
import os
import struct
import sys

from cfg import cfg_datasets, cfg_defines, cfg_generator

import transformers

parser = argparse.ArgumentParser(
    prog="generate dataset",
    description="saves token ids from a tokenized dataset to a file for faster loading.",
)

parser.add_argument("-o", "--output_path", type=str, default=None)
parser.add_argument("--overwrite", type=bool, default=False)
parser.add_argument("--cfg", default="cfg3b")
parser.add_argument("--context_length", default=512, type=int)
parser.add_argument("--num_generations", default=100, type=int)

args = parser.parse_args()


def generate_dataset_from_cfg(cfg, output_file_path, context_length, num_generations):

    cfg_start_symbols = list(cfg_generator.get_start_symbols(cfg))[0]

    tokenizer = transformers.GPTNeoXTokenizerFast.from_pretrained(
        "openai-community/gpt2"
    )

    new_dataset = cfg_datasets.CFGRandomGenerationDataset(
        cfg,
        cfg_start_symbols,
        num_generations=num_generations * 96 * 512,
        tokenizer=tokenizer,
        window_length=context_length,
    )

    with open(output_file_path, "wb") as f:
        for i, sample in enumerate(new_dataset):
            token_list = sample.cpu().detach().numpy().tolist()
            format_string = "!" + "i" * len(token_list)
            binary_data = struct.pack(format_string, *token_list)

            f.write(binary_data)
            sys.stdout.write("\rDoing thing %i" % i)
            sys.stdout.flush()


if __name__ == "__main__":

    if os.path.exists(args.output_path) and not args.overwrite:
        raise ValueError(
            "Cannot overwrite existing dataset without explicit overwrite flag set."
        )

    cfg = cfg_defines.get_cfg(args.cfg)

    generate_dataset_from_cfg(
        cfg=cfg,
        output_file_path=args.output_path,
        context_length=args.context_length,
        num_generations=args.num_generations,
    )
    print("COMPLETE")
