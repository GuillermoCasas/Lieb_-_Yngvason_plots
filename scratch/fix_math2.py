with open("scratch/test_prototype.py") as f:
    code = f.read()

code = code.replace(r"\implies", r"\Rightarrow")
with open("scratch/test_prototype.py", "w") as f:
    f.write(code)
