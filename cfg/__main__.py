# TODO: Create an actual test file.

from cfg import cfg_generator
from cfg import cfg_defines

cfg_rules = cfg_defines.cfg3b

# Example Usage
generated_string = cfg_generator.generate_from_cfg("22", cfg_rules)
print("Generated CFG String:", generated_string)

# Validate a given sequence
print(
    f"Is '{generated_string}' in the CFG language? {cfg_generator.validate_string(generated_string, '22', cfg_rules)}"
)

# Generate strings
for i in range(10):
    generated_string = cfg_generator.generate_from_cfg("15", cfg_rules)
    print(len(generated_string))

# Get longest sequence
print(cfg_generator.get_longest_sequence("22", cfg_rules))
print(cfg_generator.get_longest_sequence("15", cfg_rules))