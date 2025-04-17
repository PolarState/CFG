import os
import struct
import sys

from cfg import cfg_datasets, cfg_defines, cfg_generator

import transformers

OUTPUT_FILE_PATH = "../datasets/cfg3f_validation_dataset.bin"
OUTPUT_FILE_PATH = "../datasets/cfg3f_train_dataset.bin"

CFG = cfg_defines.cfg3f
CONTEXT_LENGTH = 512

cfg_start_symbols = list(cfg_generator.get_start_symbols(CFG))[0]

tokenizer = transformers.GPTNeoXTokenizerFast.from_pretrained("openai-community/gpt2")

new_dataset = cfg_datasets.CFGRandomGenerationDataset(
    CFG,
    cfg_start_symbols,
    num_generations=100000 * 96 * 512,
    tokenizer=tokenizer,
    window_length=CONTEXT_LENGTH,
)

with open(OUTPUT_FILE_PATH, "wb") as f:
    for i, sample in enumerate(new_dataset):
        token_list = sample.cpu().detach().numpy().tolist()
        format_string = "!" + "i" * len(token_list)
        binary_data = struct.pack(format_string, *token_list)

        f.write(binary_data)
        sys.stdout.write("\rDoing thing %i" % i)
        sys.stdout.flush()

print("COMPLETE")
