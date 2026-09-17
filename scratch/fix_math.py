with open("scratch/test_prototype.py") as f:
    code = f.read()

# Replace problematic math strings
code = code.replace(r"\mathbf{Planck\ \&\ T5:\ ", r"\mathbf{Planck\ and\ T5:\ ")
with open("scratch/test_prototype.py", "w") as f:
    f.write(code)

