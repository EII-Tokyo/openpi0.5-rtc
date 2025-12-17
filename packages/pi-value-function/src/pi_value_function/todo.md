[X] siglip
    [X] migrate siglip2 into siglip.py
[X] gemma
    [X] use gemma package for gemma3 270M
    [X] use gemma 
[X] value head
[X] weight loader
[X] training script
[X] utils/failure labeling
[] data/web data
[] try both freezing backbone and smaller backbone and compare performances



QUESTIONS:
- why not use the same backbone with a different head?
- why use jax nnx when it doesn't have support for gemma3?
- 