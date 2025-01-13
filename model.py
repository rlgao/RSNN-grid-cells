import torch
import torch.nn as nn
import torch.nn.functional as F

# from RSNN_layers.spike_neuron import *
# from RSNN_layers.spike_dense import *
# from RSNN_layers.spike_rnn import *


class RNN(nn.Module):
    def __init__(self, options, place_cells):
        super().__init__()
        
        self.Np = options.Np
        self.Ng = options.Ng
        
        self.sequence_length = options.sequence_length
        self.weight_decay = options.weight_decay
        self.place_cells = place_cells

        # input weights
        self.encoder = nn.Linear(self.Np, self.Ng, bias=False)
        
        self.RNN = nn.RNN(
            input_size=2,
            hidden_size=self.Ng,
            nonlinearity=options.nonlinearity,
            bias=False
        )
        
        # linear read-out weights
        self.decoder = nn.Linear(self.Ng, self.Np, bias=False)
        
        self.softmax = nn.Softmax(dim=-1)

    def get_grids(self, inputs):
        '''
        Compute grid cell activations.
        Args:
            inputs: Batch of 2d velocity inputs with shape [sequence_length, batch_size, 2].
        Returns: 
            g: Batch of grid cell activations with shape [sequence_length, batch_size, Ng].
        '''
        v, p0 = inputs
        # v: (sequence_length, batch_size, 2)
        # p0: (batch_size, Np)
        
        init_state = self.encoder(p0)[None]  # (1, batch_size, Ng)
        
        g, _ = self.RNN(v, init_state)  # (sequence_length, batch_size, Ng)
        
        return g

    def predict(self, inputs):
        '''
        Predict place cell code.
        Args:
            inputs: Batch of 2d velocity inputs with shape [sequence_length, batch_size, 2].
        Returns: 
            place_preds: Predicted place cell activations with shape [sequence_length, batch_size, Np].
        '''
        place_preds = self.decoder(self.get_grids(inputs))
        
        return place_preds

    def compute_loss(self, inputs, pc_outputs, pos):
        '''
        Compute avg. loss and decoding error.
        Args:
            inputs: Batch of 2d velocity inputs with shape [sequence_length, batch_size, 2].
            pc_outputs: Ground truth place cell activations with shape [sequence_length, batch_size, Np].
            pos: Ground truth 2d position with shape [sequence_length, batch_size, 2].
        Returns:
            loss: Avg. loss for this training batch.
            err: Avg. decoded position error in m.
        '''
        y = pc_outputs
        
        preds = self.predict(inputs)
        yhat = self.softmax(preds)
        
        loss = -(y * torch.log(yhat)).sum(-1).mean()

        # Weight regularization 
        loss += self.weight_decay * (self.RNN.weight_hh_l0 ** 2).sum()

        # Compute decoding error (m)
        pred_pos = self.place_cells.get_nearest_cell_pos(preds)
        err = torch.sqrt(((pos - pred_pos)**2).sum(-1)).mean()

        return loss, err


class LSTM(nn.Module):
    def __init__(self, options, place_cells):
        super().__init__()
        
        self.Np = options.Np
        self.Nlstm = options.Nlstm  # LSTM hidden size
        self.Ng = options.Ng
        
        self.sequence_length = options.sequence_length
        
        self.dropout_rate = options.dropout_rate
        self.weight_decay = options.weight_decay
        
        self.place_cells = place_cells

        # input weights
        self.encoder1 = nn.Linear(self.Np, self.Nlstm) #, bias=False)
        self.encoder2 = nn.Linear(self.Np, self.Nlstm) #, bias=False)
        
        # LSTM cell
        self.LSTM = nn.LSTMCell(input_size=2, hidden_size=self.Nlstm)
        
        # grid activation
        self.glayer = nn.Linear(self.Nlstm, self.Ng, bias=False)
        
        if options.nonlinearity == 'relu':
            self.nonlin = nn.ReLU()
        elif options.nonlinearity == 'tanh':
            self.nonlin = nn.Tanh()
        elif options.nonlinearity == 'sigmoid':
            self.nonlin = nn.Sigmoid()
        else:
            self.nonlin = nn.Identity()
        
        # linear read-out weights
        self.decoder = nn.Linear(self.Ng, self.Np) #, bias=False)
        
        self.softmax = nn.Softmax(dim=-1)

    def get_grids(self, inputs):
        '''
        Compute grid cell activations.
        Args:
            inputs: Batch of 2d velocity inputs with shape [sequence_length, batch_size, 2].
        Returns: 
            g: Batch of grid cell activations with shape [sequence_length, batch_size, Ng].
        '''
        v, p0 = inputs
        # v: (sequence_length, batch_size, 2)
        # p0: (batch_size, Np)
        
        hx = self.encoder1(p0)  # (batch_size, Nlstm)
        cx = self.encoder2(p0)  # (batch_size, Nlstm)
        
        g = []
        for i in range(v.size()[0]):  # sequence_length
            # one time step in the sequence
            hx, cx = self.LSTM(v[i], (hx, cx))
            
            g_step = self.glayer(hx)  # (batch_size, Ng)
            
            # non-negativity constraint
            g_step = self.nonlin(g_step)
            
            if self.training and self.dropout_rate > 0:
                g_step = F.dropout(g_step, self.dropout_rate)
                
            g.append(g_step)
            
        g = torch.stack(g, dim=0)  # (sequence_length, batch_size, Ng)
        
        return g

    def predict(self, inputs):
        '''
        Predict place cell code.
        Args:
            inputs: Batch of 2d velocity inputs with shape [sequence_length, batch_size, 2].
        Returns: 
            place_preds: Predicted place cell activations with shape [sequence_length, batch_size, Np].
        '''
        place_preds = self.decoder(self.get_grids(inputs))  # (sequence_length, batch_size, Np)
        
        return place_preds

    def compute_loss(self, inputs, pc_outputs, pos):
        '''
        Compute avg. loss and decoding error.
        Args:
            inputs: Batch of 2d velocity inputs with shape [sequence_length, batch_size, 2].
            pc_outputs: Ground truth place cell activations with shape [sequence_length, batch_size, Np].
            pos: Ground truth 2d position with shape [sequence_length, batch_size, 2].
        Returns:
            loss: Avg. loss for this training batch.
            err: Avg. decoded position error in m.
        '''
        y = pc_outputs
        
        preds = self.predict(inputs)
        yhat = self.softmax(preds)
        
        loss = -(y * torch.log(yhat)).sum(-1).mean()

        # Weight regularization 
        loss += self.weight_decay * (self.glayer.weight.norm(2)).sum()

        # Compute decoding error (m)
        pred_pos = self.place_cells.get_nearest_cell_pos(preds)
        err = torch.sqrt(((pos - pred_pos)**2).sum(-1)).mean()

        return loss, err



# class RSNN(nn.Module):
#     def __init__(self, options, place_cells):
#         super().__init__()
        
#         self.sequence_length = options.sequence_length
#         self.Np = options.Np
#         self.Nrsnn = options.Nrsnn  # RSNN hidden size
#         self.Ng = options.Ng
        
#         self.dropout_rate = options.dropout_rate
#         self.weight_decay = options.weight_decay
        
#         self.place_cells = place_cells
#         self.device = options.device

#         self.is_bias = True

#         # input weights, proj p0 -> mems
#         self.encoder1 = nn.Linear(self.Np, self.Nrsnn)
#         self.encoder2 = nn.Linear(self.Np, self.Nrsnn)
#         self.encoder3 = nn.Linear(self.Np, self.Nrsnn)
        
#         # =================================
#         # RSNN layers
#         self.dense_1 = spike_dense(
#             2, self.Nrsnn,
#             tauM_inital=20, tauM_inital_std=5, 
#             tauAdp_inital=200, tauAdp_inital_std=50, 
#             device=self.device, bias=self.is_bias
#         )
#         self.rnn_1 = spike_rnn(
#             self.Nrsnn, self.Nrsnn,
#             tauM_inital=20, tauM_inital_std=5, 
#             tauAdp_inital=200, tauAdp_inital_std=50, 
#             device=self.device, bias=self.is_bias
#         )
#         self.dense_2 = readout_integrator(
#             self.Nrsnn, self.Nrsnn,
#             tauM_inital=10, tauM_inital_std=1, 
#             device=self.device, bias=self.is_bias
#         )
#         # RSNN layers initialization
#         nn.init.xavier_normal_(self.dense_1.dense.weight)
#         nn.init.kaiming_normal_(self.rnn_1.recurrent.weight)
#         nn.init.xavier_normal_(self.dense_2.dense.weight)
#         if self.is_bias:
#             nn.init.constant_(self.dense_1.dense.bias, 0)
#             nn.init.constant_(self.rnn_1.recurrent.bias, 0)
#             nn.init.constant_(self.dense_2.dense.bias, 0)
#         # =================================
        
#         # grid activation
#         self.glayer = nn.Linear(self.Nrsnn, self.Ng, bias=False)
        
#         if options.nonlinearity == 'relu':
#             self.nonlin = nn.ReLU()
#         elif options.nonlinearity == 'tanh':
#             self.nonlin = nn.Tanh()
#         elif options.nonlinearity == 'sigmoid':
#             self.nonlin = nn.Sigmoid()
#         else:
#             self.nonlin = nn.Identity()
        
#         # linear read-out weights
#         self.decoder = nn.Linear(self.Ng, self.Np)
        
#         self.softmax = nn.Softmax(dim=-1)

#     def get_grids(self, inputs):
#         '''
#         Compute grid cell activations.
#         Args:
#             inputs: Batch of 2d velocity inputs with shape [sequence_length, batch_size, 2].
#         Returns: 
#             g: Batch of grid cell activations with shape [sequence_length, batch_size, Ng].
#         '''
#         v, p0 = inputs
#         # v: (sequence_length, batch_size, 2)
#         # p0: (batch_size, Np)
        
#         seq_length, batch_size, input_dim = v.size()
        
#         # initialize RSNN neuron states
#         self.dense_1.set_neuron_state(batch_size)
#         self.rnn_1.set_neuron_state(batch_size)
#         self.dense_2.set_neuron_state(batch_size)
        
#         # proj p0 -> mems
#         self.dense_1.mem = self.encoder1(p0)  # (batch_size, Nrsnn)
#         self.rnn_1.mem = self.encoder2(p0)    # (batch_size, Nrsnn)
#         self.dense_2.mem = self.encoder3(p0)  # (batch_size, Nrsnn)
        
#         g = []
#         for i in range(v.size()[0]):  # sequence_length
#             # one time step in the sequence
#             rsnn_input = v[i]
#             mem_layer1, spike_layer1 = self.dense_1.forward(rsnn_input)
#             mem_layer2, spike_layer2 = self.rnn_1.forward(spike_layer1)
#             mem_layer3 = self.dense_2.forward(spike_layer2)
            
#             g_step = self.glayer(mem_layer3)  # (batch_size, Ng)
            
#             # non-negativity constraint
#             g_step = self.nonlin(g_step)
            
#             if self.training and self.dropout_rate > 0:
#                 g_step = F.dropout(g_step, self.dropout_rate)
                
#             g.append(g_step)
            
#         g = torch.stack(g, dim=0)  # (sequence_length, batch_size, Ng)
        
#         return g

#     def predict(self, inputs):
#         '''
#         Predict place cell code.
#         Args:
#             inputs: Batch of 2d velocity inputs with shape [sequence_length, batch_size, 2].
#         Returns: 
#             place_preds: Predicted place cell activations with shape [sequence_length, batch_size, Np].
#         '''
#         place_preds = self.decoder(self.get_grids(inputs))  # (sequence_length, batch_size, Np)
        
#         return place_preds

#     def compute_loss(self, inputs, pc_outputs, pos):
#         '''
#         Compute avg. loss and decoding error.
#         Args:
#             inputs: Batch of 2d velocity inputs with shape [sequence_length, batch_size, 2].
#             pc_outputs: Ground truth place cell activations with shape [sequence_length, batch_size, Np].
#             pos: Ground truth 2d position with shape [sequence_length, batch_size, 2].
#         Returns:
#             loss: Avg. loss for this training batch.
#             err: Avg. decoded position error in m.
#         '''
#         y = pc_outputs
#         preds = self.predict(inputs)
#         yhat = self.softmax(preds)
#         loss = -(y * torch.log(yhat)).sum(-1).mean()
        
#         # Weight regularization
#         loss += self.weight_decay * (self.glayer.weight.norm(2)).sum()

#         # Compute decoding error (m)
#         pred_pos = self.place_cells.get_nearest_cell_pos(preds)
#         err = torch.sqrt(((pos - pred_pos)**2).sum(-1)).mean()

#         return loss, err
    
#     def parameters(self):
#         base_params = self.get_params()['base_params']
#         tau_params = self.get_params()['tau_params']
#         other_params = self.get_params()['other_params']

#         all_params = base_params + tau_params + other_params
#         return all_params
    
#     def get_params(self):
#         if self.is_bias:
#             base_params = [
#                 self.dense_1.dense.weight,
#                 self.dense_1.dense.bias,

#                 self.rnn_1.dense.weight,
#                 self.rnn_1.dense.bias,
#                 self.rnn_1.recurrent.weight,
#                 self.rnn_1.recurrent.bias,

#                 self.dense_2.dense.weight,
#                 self.dense_2.dense.bias
#             ]
#         else:
#             base_params = [
#                 self.dense_1.dense.weight,
#                 self.rnn_1.dense.weight,
#                 self.rnn_1.recurrent.weight,
#                 self.dense_2.dense.weight,
#             ]
#         tau_params = [
#             self.dense_1.tau_m,
#             self.rnn_1.tau_m,
#             self.dense_2.tau_m,

#             self.dense_1.tau_adp,
#             self.rnn_1.tau_adp
#         ]
#         other_params = [
#             self.encoder1.weight,
#             self.encoder1.bias,
#             self.encoder2.weight,
#             self.encoder2.bias,
#             self.encoder3.weight,
#             self.encoder3.bias,

#             self.glayer.weight,

#             self.decoder.weight,
#             self.decoder.bias
#         ]
#         all_params = {
#             'base_params': base_params,
#             'tau_params': tau_params,
#             'other_params': other_params
#         }
#         return all_params
