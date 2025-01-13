import numpy as np
import scipy
import datetime as dt
import os
import json

from scores import GridScorer

class Ratemap:
    def __init__(self, options, res, ratemaps):
        self.options = options
        self.ratemaps = ratemaps  # (Ng, res, res)
        
        # =======================================
        # compute grid scores and sacs
        
        # DeepMind param
        starts = [0.2] * 10
        ends = np.linspace(0.4, 1.0, num=10)
        # Dehong Xu param
        # starts = [0.1] * 20
        # ends = np.linspace(0.2, 1.2, num=20)
        
        mask_parameters = zip(starts, ends.tolist())
        
        box_width = options.box_width
        box_height = options.box_height
        coord_range = ((-box_width / 2., box_width / 2.), (-box_height / 2., box_height / 2.))
        
        scorer = GridScorer(res, coord_range, mask_parameters)

        score_60, score_90, max_60_mask, max_90_mask, sacs, max_60_ind = zip(
            *[scorer.get_scores(rm) for rm in ratemaps]
        )
        # =======================================
        
        self.score_60 = score_60
        self.score_90 = score_90
        self.max_60_mask = max_60_mask
        self.max_90_mask = max_90_mask
        self.sacs = sacs
        self.max_60_ind = max_60_ind
        
        self.grid_thresh = 0.37
        
    def get_grid_scale(self):
        grid_scale_in_m = []
        
        for i in range(len(self.max_60_mask)):  # Ng
            if self.score_60[i] >= self.grid_thresh:
                mask = self.max_60_mask[i]
                scale = mask[1] * (self.options.box_width) / 2.
                grid_scale_in_m.append(scale)

        return grid_scale_in_m
    

def save_options(options):
    filename = os.path.join(options.save_dir, options.run_ID)
    if not os.path.isdir(filename):
        os.makedirs(filename, exist_ok=True)
    filename = os.path.join(filename, 'options.json')
    
    # with open(filename, 'w') as file:
    #     json.dump(options.__dict__, file)
    
    data = options.__dict__
    with open(filename, 'w') as json_file:
        json_file.write('{\n')
        for key, value in data.items():
            json_file.write(f'"{key}": {json.dumps(value)},\n')
        # Remove the last comma and add closing brace
        json_file.seek(json_file.tell() - 2, 0)  # Move back to remove the last comma
        json_file.write('\n}')  # Write the closing brace
    

def generate_run_ID(options):
    ''' 
    Create a unique run ID from the most relevant parameters. 
    Remaining parameters can be found in params.npy file. 
    '''
    date_time = dt.datetime.now().strftime("%m%d_%H%M%S")
    # date_time = dt.datetime.now().strftime("%m%d_%H%M")
    run_ID = options.RNN_type + '_' + date_time + '_' + options.neuron_type

    return run_ID


def get_2d_sort(x1, x2):
    """
    Reshapes x1 and x2 into square arrays, and then sorts
    them such that x1 increases downward and x2 increases
    rightward. Returns the order.
    """
    n = int(np.round(np.sqrt(len(x1))))
    total_order = x1.argsort()
    total_order = total_order.reshape(n, n)
    for i in range(n):
        row_order = x2[total_order.ravel()].reshape(n, n)[i].argsort()
        total_order[i] = total_order[i, row_order]
    total_order = total_order.ravel()
    return total_order


def dft(N,real=False,scale='sqrtn'):
    if not real:
        return scipy.linalg.dft(N,scale)
    else:
        cosines = np.cos(2*np.pi*np.arange(N//2+1)[None,:]/N*np.arange(N)[:,None])
        sines = np.sin(2*np.pi*np.arange(1,(N-1)//2+1)[None,:]/N*np.arange(N)[:,None])
        if N%2==0:
            cosines[:,-1] /= np.sqrt(2)
        F = np.concatenate((cosines,sines[:,::-1]),1)
        F[:,0] /= np.sqrt(N)
        F[:,1:] /= np.sqrt(N/2)
        return F


# def skaggs_power(Jsort):
#     F = dft(int(np.sqrt(N)), real=True)
#     F2d = F[:,None,:,None]*F[None,:,None,:]

#     F2d_unroll = np.reshape(F2d, (N, N))

#     F2d_inv = F2d_unroll.conj().T
#     Jtilde = F2d_inv.dot(Jsort).dot(F2d_unroll)

#     return (Jtilde[1,1]**2 + Jtilde[-1,-1]**2) / (Jtilde**2).sum()


# def skaggs_power_2(Jsort):
#     J_square = np.reshape(Jsort, (n,n,n,n))
#     Jmean = np.zeros([n,n])
#     for i in range(n):
#         for j in range(n):
#             Jmean += np.roll(np.roll(J_square[i,j], -i, axis=0), -j, axis=1)

# #     Jmean[0,0] = np.max(Jmean[1:,1:])
#     Jmean = np.roll(np.roll(Jmean, n//2, axis=0), n//2, axis=1)
#     Jtilde = np.real(np.fft.fft2(Jmean))
    
#     Jtilde[0,0] = 0
#     sk_power = Jtilde[1,1]**2 + Jtilde[0,1]**2 + Jtilde[1,0]**2
#     sk_power += Jtilde[-1,-1]**2 + Jtilde[0,-1]**2 + Jtilde[-1,0]**2
#     sk_power /= (Jtilde**2).sum()
    
#     return sk_power


# def calc_err():
#     inputs, _, pos = next(gen)
#     pred = model(inputs)
#     pred_pos = place_cells.get_nearest_cell_pos(pred)
#     return tf.reduce_mean(tf.sqrt(tf.reduce_sum((pos - pred_pos)**2, axis=-1)))


# from visualize import compute_ratemaps, plot_ratemaps


# def compute_variance(res, n_avg):
    
#     activations, rate_map, g, pos = compute_ratemaps(model, data_manager, options, res=res, n_avg=n_avg)

#     counts = np.zeros([res,res])
#     variance = np.zeros([res,res])

#     x_all = (pos[:,0] + options['box_width']/2) / options['box_width'] * res
#     y_all = (pos[:,1] + options['box_height']/2) / options['box_height'] * res
#     for i in tqdm(range(len(g))):
#         x = int(x_all[i])
#         y = int(y_all[i])
#         if x >=0 and x < res and y >=0 and y < res:
#             counts[x, y] += 1
#             variance[x, y] += np.linalg.norm(g[i] - activations[:, x, y]) / np.linalg.norm(g[i]) / np.linalg.norm(activations[:,x,y])

#     for x in range(res):
#         for y in range(res):
#             if counts[x, y] > 0:
#                 variance[x, y] /= counts[x, y]
                
#     return variance


def load_trained_weights(model, trainer, weight_dir):
    ''' Load weights stored as a .npy file (for github)'''

    # Train for a single step to initialize weights
    trainer.train(n_epochs=1, n_steps=1, save=False)

    # Load weights from npy array
    weights = np.load(weight_dir, allow_pickle=True)
    model.set_weights(weights)
    print('Loaded trained weights.')

