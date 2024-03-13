Here are some thoughts and considerations on setting the tokenizer's vocabulary size.

The size of the tokenizer's vocabulary affects the size of the embedding layer and the language model head (the linear layer at the end of the model used for predicting the next token). This is a consideration for computational resources.

More importantly, the size of the vocabulary impacts the quality of the model. Increasing the number of parameters in the embedding layer means there's a higher chance of undertraining many vocabularies. A larger vocabulary size means we have more unique tokens, which often results in fewer occurrences of these tokens in the training set, increasing the chance of undertraining the embedding vectors associated with them.

Also, as we increase the vocabulary size, we tend to have shorter sequence lengths. This means we compress more text into single tokens, giving the model less time to process and understand the information during the forward pass.

Regarding specific configurations which is good to double check with them:

- byte_fallback=True: This option allows the tokenizer to handle unrecognized characters by using their byte codes instead of zero-padding. This helps in processing unknown characters effectively. More details can be found here: (https://github.com/google/sentencepiece/blob/master/src/sentencepiece_model.proto#L194).

- add_dummy_prefix=True: This adds a dummy space before each token, helping maintain consistency in token representation whether they occur at the beginning or in the middle of a sentence. More information is available here: (https://github.com/google/sentencepiece/blob/master/src/sentencepiece_model.proto#L256).


This is Llama2 tokenizer settings 

tokenizer setting, https://gist.github.com/HamidShojanazeri/0b92941ff1506162b54a8170d4b6a788
