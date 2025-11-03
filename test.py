import tensorflow_datasets as tfds

subset = tfds.load("maestro", split="train[:5%]", as_supervised=False)
print(subset)