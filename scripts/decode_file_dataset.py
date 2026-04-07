import struct

import transformers

tokenizer = transformers.GPTNeoXTokenizerFast.from_pretrained(
    "openai-community/gpt2"
)
# load pretrained model.
model = transformers.GPTNeoXForCausalLM(gpt_config)


filename = "../datasets/test.bin"
window_length = 512

with open(filename, "rb") as f:
    while True:
        bytes_to_read = window_length * struct.calcsize("i")
        binary_chunk = f.read(bytes_to_read)
        if not binary_chunk:
            break  # End of file

        # Unpack the binary data into a tuple of integers
        format_string = "!" + "i" * window_length
        if len(binary_chunk) != struct.calcsize(format_string):
            raise ValueError(
                f"Unexpected end of file or corrupted data. Expected {struct.calcsize(format_string)} bytes, got {len(binary_chunk)}."
            )

        token_list = list(struct.unpack(format_string, binary_chunk))
        
        print(token_list)
        print(tokenizer.decode(token_list))
        
        # test generation of pretrained model.
        
        break
