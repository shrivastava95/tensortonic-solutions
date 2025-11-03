import numpy as np

"""
Summarization:
Causal masking implemented in transformers, allows one to compute forward passes very efficiently. "Causal" here is treating the sequence as one where the occurence of a token is conditioned on the other tokens before it in a given sequence. How is this done?

Let's consider this from the perspective of this interesting propery of text, which is that it is causal in nature. We can see that if we number the words in a sequence with the index of their occurence like so:

["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
[  0       1        2       3       4       5       6      7       8  ]

We can see that, if the causal property is to hold, then token at index i is conditioned on all tokens with index less than i.

How can we achieve this property within the attention layer?

Lets look at a tokenized sequence now. consider the above discussed sentence, each tokenized into vectors of shape 512. If we were to make predictions using these embeddings directly, then it would make sense to model the task of autoregression such that the token at index i is being predicted using the tokens before it (0 ... i-1).

Now, we need a layer that propagates this causal constraint.

Lets see how a causal masked transformer self-attention layer achieves this.
    0 1 2 3 4 5 6 7 8
0: [Y X X X X X X X X]
1: [Y Y X X X X X X X]
2: [Y Y Y X X X X X X]
3: [Y Y Y Y X X X X X]
4: [Y Y Y Y Y X X X X]
5: [Y Y Y Y Y Y X X X]
6: [Y Y Y Y Y Y Y X X]
7: [Y Y Y Y Y Y Y Y X]
8: [Y Y Y Y Y Y Y Y Y]

Rows: queries
Columns: keys
(numbers added to help track indices)
It is seen that queries of lower token index than a given key do not attend with it. And remember, the values in SDPA are essentially row-vectors that are combined in a linear combination to get outputs. They are weighted according to the alignment of a given Query with all the keys it is allowed to attend to. These weights happen to sum to 1 since they are obtained via a softmax along the row dimension. And, before the application of softmax, the positions marked X are replaced with -np.inf so that their attention weight becomes zero, allowing for the causal constraint. 

Now we can see clearly how, in a causal self-attention layer, the query embedding for each position in the sequence is mapped to a linearly weighted combination of those values that have indices less than or equal to that of the position of the query.

How is this significant for autoregressive generation?

Lets consider the fact that every embedding in each layer depends on the embeddings of layers before it, which have positions less than or equal to it.

Embedding(layer_id, token_pos_i) depends on Embedding(layer_id-1, token_pos_j) where i >= j

One can see that, via induction, embeddings in a layer depend on embeddings at positions before or equal to it in all layers prior to that layer.

This means that, if you stack multiple of such causal self-attention layers, then the output at position i will depend on the input tokens at position i and earlier!!

This sounds oddly familiar to the causal constraints we were discussing in the beginning. The prediction at position i is conditioned on tokens <= i... This means that we can model the prediction of the word belonging to position (i+1), at the position i of the transformer, and this can be done for all the tokens in the sequence simultaneously. So, it is as if we are computing the forward passes for all next-token prediction tasks simultaneously for the entire sequence! in one forward pass only! And THAT is the magic of causal self-attention.

In GPT-2, causal self-attention is most strikingly used in exactly the manner described above. The implementation follows the style of a decoder-only transformer as it is quite conducive to model pretty much any natural language task, since you can practically serialize anything, and thus model it as a causal autoregression task.


"""

def causal_mask(seq_len):
    """
    Return a boolean mask of shape (seq_len, seq_len)
    where mask[i,j] = True if position j is visible to position i.
    """
    return np.where(np.tri(seq_len), False, True)
