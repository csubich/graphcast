# Copyright 2024 Crown in Right of Canada
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def zero_grad_like(grad):
    '''Compute a tree structure of all zeros, matching the composition of an input
    structure – intended to initialize gradient updates given a sample gradient'''
    import tree
    return tree.map_structure(lambda gr: 0*gr, grad)
    # if (isinstance(grad,dict)):
    #     return {k : zero_grad_like(grad[k]) for k in grad.keys()}
    # elif (grad is None): return 0
    # else:
    #     return grad*0

# def grad_accumulate(grad,accum,weight):
#     if (accum is None): return(grad_accumulate(grad,zero_grad_like(grad),weight))
#     if (isinstance(grad,dict)):
#         return {k : grad_accumulate(grad[k],accum[k],weight) for k in grad.keys()}
#     else:
#         return accum + weight*grad

minibatch_warned = False
def grad_sum_minibatch(grad):
    '''Return the sum of a structure over the first axis, intended to accumulate the
    gradients over a prepended 'minibatch' dimension'''
    import tree
    global minibatch_warned
    # Check to see if any parameter gradients have three or more dimensions, indicative of
    # actually having a prepended minibatched dimension
    if (not minibatch_warned and max(tree.flatten(tree.map_structure(lambda g: g.ndim,grad))) < 3):
        minibatch_warned = True
        import warnings
        warnings.warn('CAUTION: executing grad_sum_minibatch when gradient may not have a minibatch dimension')

    return tree.map_structure(lambda gr: gr.sum(axis=0),grad)

def grad_accumulate(grad,accum,weight):
    '''Return accum + weight*grad, intended to accumulate gradients over several independent
    examples of a batch'''
    import tree
    # Accumulate the gradient
    gsm = grad_sum_minibatch(grad)
    if (accum is None):
        accum = zero_grad_like(gsm)
    return tree.map_structure(lambda gr, acc : acc + gr*weight, gsm, accum)


def weight_mask(params,default=False):
    '''Retrun a structure of True/False, depending on whether the leaf key has name 'w' (False) or
    anything else (True); intended to initialize the mask for weight decay in the AdamW optimizer'''
    if (isinstance(params,dict)):
        return {k : weight_mask(params[k],True if (k=='w') else False) for k in params.keys()}
    else:
        return default
