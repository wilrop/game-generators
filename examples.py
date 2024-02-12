import game_generators as gage

# Generate a batch of Bach-Stravinsky games directly.
batch_size = 10
payoff_matrices = gage.nfg.bach_stravinsky(batch_size)
print(payoff_matrices.shape)

# Or alternatively through the generic interface.
payoff_matrices = gage.generate_nfg("bach_stravinsky", batch_size)
print(payoff_matrices.shape)
