from algorithm import basis_algorithm_collection

def IRCR(args):
    # The algorithmic components of IRCR is implemented in the replay buffer.
    return basis_algorithm_collection[args.basis_alg](args)
