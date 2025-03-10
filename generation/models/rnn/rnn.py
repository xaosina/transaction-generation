

import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(RNN, self).__init__()
        
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def __forward(self, 
                text_list,
                train_seq,
                target=None):

        B = train_seq.shape[0]

        gru_out, h_n = self.gru(train_seq) 

        out = self.fc(gru_out[:, -1, :])  
        
        if target is not None:
            self.loss = {'nll': F.mse_loss(out, target.view(B, -1))}

        return out
    
    def forward(self, 
                text_list,
                train_seq,
                target=None,
                max_steps=None):
        
        if target is not None:
            
            return self.__forward(
                text_list,
                train_seq, 
                target,
            )
        else:
            return self.generate(
                text_list,
                train_seq,
                max_steps=max_steps
            )

        return out

    def generate(self, text_list, initial_sequence, max_steps):
        generated_sequence = initial_sequence
        generated_start_index = generated_sequence.shape[1]

        for _ in range(max_steps):

            next_step = self.__forward(text_list, generated_sequence)

            next_step = next_step.unsqueeze(1)
            generated_sequence = torch.cat((generated_sequence, next_step), dim=1) 
        
        return generated_sequence[:, generated_start_index:, :]