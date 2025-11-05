##### Test implementation of differentiable trajectory reweighting #####

import jax as jp
import jax.numpy as jnp

class DiffTRE:
    def __init__(self, traj_data):
        self.traj_data = traj_data
        self.num_frames = traj_data.shape[0]
        self.reference_energy = 
        self.current_energy = 
        self.delta_energy = self.current_energy - self.reference_energy
        self.beta = 1

    def load_experimental_observable(self):
        """
        Loads in experimental observables from numpy file
        according to config file
        tkl_exp is the flattened upper triangle of the experimental Tkl matrix
        which contains sums of average HiC contact probabilities between each type-type pair

        HiC contact probabilities are calculated by KR-normalizing the HiC contact counts
        and then dividing by the max contact count for each row.
        """

        tkl_exp = np.load(self.tkl_exp_path)
        tkl_exp = tkl_exp.flatten()
        return tkl_exp

    

    def calculate_traj_weights(self):
        """
        Weight each frame based on the probability of the frame being sampled
        from the reference distribution given the current energy.
        Parameters
        ----------
        delta_energy: array of delta energies for each frame

        Returns
        -------
        w: array of weights for each frame
        """

        w = jnp.exp(-self.beta * delta_energy)
        w /= jnp.sum(w)
        return w
    

    def load_positions(self, frame):

    def loss(self, params, data):
        """
        Loss function for the differentiable trajectory reweighting.
        """
        w = self.calculate_traj_weights()
        return -jnp.sum(w * data)

