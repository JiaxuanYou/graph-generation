from train import *

# model configuration
input_size = 16 # input vector size
hidden_size = 64 # hidden vector size (for a single layer)
n_layers = 1
# train configuration
lr = 0.0001
decoder_lr_ratio = 1

encoder = EncoderRNN(input_size, hidden_size, n_layers)
decoder = DecoderRNN(input_size, hidden_size, n_layers)

encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)