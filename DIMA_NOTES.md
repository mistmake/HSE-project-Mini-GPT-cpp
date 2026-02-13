# Comments on bigram and one-layer NN
[link]([url](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part1_bigrams.ipynb))
- We need to pick light datatset with simple word or short sentences.
- this part is quite easy to make and doesn't require any advanced structures, so we can create a class for BiGram and One-Layer_NN. 
- Moreover, the results of these models are too bad for being shown as a project, so we must create more advanced models.
## Pipeline for bigram
create dicts with string->int and int->string in dataset
create 2by2 tensor with all bigrams converted into the numbers. don't forget to add beggining and the end of each word.
create tensor of probabilites of N add 1 to normalize
write simple word generator

### create negativelog_likelyhood function:
in loop by all words and by each word:
1. take probability of each bigram in the word
2. take it's logarithm
3. add it to the overall likelyhood
mulitpy it by -1.

## Pipeline for one-layer NN
create training set of bigrams by creating xs and ys.
generate random weights
### Gradient descent in the loop:
1. create tensor of encoded vectors for all xs. it creates the tensor of vectors, where all entities are 0 except for 1, corresponding to the number, we are encoding
2. create logits, that are x_encoded @ W
3. **Softmax**
- create counts by exponentiating logits. they are equivalent to N in bigram
- create a tensor of probabilites that is a counts divided by the sum of the row


4. evaluate loss
5. set weight gradients to None
6. do backward pass
7. update weights 

### Sample from the model
in while True repeat 1-3 from GD and pick with multinomial the next charachter

# Commets on MLP (multilayer perceptron)
Main idea: present words/chars not as integers, but as "embeddings" in n-dimensional vector space of knowledge. 
To research:
- article "a neural probabalistic language model"
- cross-entropy
- 
## Steps
1. create a vocabulary and mappings string->int int->string
2. build a dataset, with X, _which consist ~block_size~ chars. empty spaces are filled with starts or ends_, Y, which consists of the next character after corresponding element from X.
3. Randomly generate len_vocabulary*n-dimensional of weights. each embedding can be accessed by C[X]
4. 
