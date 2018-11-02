import numpy as np
from sklearn import mixture

import torch
import torchvision
from torch import nn
from torch.distributions import MultivariateNormal
import math

def js_divergence(mu_1: torch.tensor, cov_1: torch.tensor, 
                 mu_2: torch.tensor, cov_2: torch.tensor):
    assert mu_1.shape == mu_2.shape, 'mu shape mismatch'
    assert cov_1.shape == cov_2.shape, 'cov shape mismatch'
    
    # Monte Carlo samples
    MC_samples = 1000
    
    Pd = MultivariateNormal(loc = mu_1, covariance_matrix=cov_1)
    Qd = MultivariateNormal(loc = mu_2, covariance_matrix=cov_2)
    P_samples = Pd.sample((MC_samples,))
    Q_samples = Qd.sample((MC_samples,))
    
    P = lambda x: torch.tensor(np.power(2, Pd.log_prob(x).numpy()))
    Q = lambda x: torch.tensor(np.power(2, Qd.log_prob(x).numpy()))
    M = lambda x: 0.5 * P(x) + 0.5 * Q(x)
    
    P_div_M = lambda x: P(x) / M(x)
    Q_div_M = lambda x: Q(x) / M(x)

    D_KL_approx_PM = lambda x: (1 / MC_samples) * sum(torch.log2(P_div_M(x)))
    D_KL_approx_QM = lambda x: (1 / MC_samples) * sum(torch.log2(Q_div_M(x)))

    return 0.5 * D_KL_approx_PM(P_samples) + 0.5 * D_KL_approx_QM(Q_samples)

class GMM_ECO(object):
    def __init__(self, init_components, max_components, learning_rate = 0.005, pi_threshold = 0.001, covariance_type = None, init = False):
        assert covariance_type is None, 'Not implemented!'
        self.n_components = init_components
        self.max_components = max_components
        self.components = {'features': [], 'heatmaps': []}
        self.learning_rate = learning_rate
        self.dataloader = None
        self.var_size = None
        self.covariance_matrix = None
        self.pi_threshold = pi_threshold
        self.__init = init
        self.means_ = torch.tensor([], dtype=torch.float)
        self.weights_ = torch.tensor([], dtype=torch.float)
        self.distance_vector = torch.tensor([], dtype=torch.float)
        self.distance_matric = torch.tensor([], dtype=torch.float)
        self.fitted = False

    # GMM_ECO.fit:
    # fit the size of data, which is parameter 'var_size', and it's a number, 
    # and if init = True when a GMM_ECO object was created, use parameter 'data'
    # to initialize the original GMM distribution
    def fit(self, data):
        _, D, H, W = data.shape
        self.data = data
        self.var_size = D * H * W
#         self.covariance_matrix = torch.eye(self.var_size, dtype=torch.float) # lack of memory
        self.__initialization(data)
        self.fitted = True
            
    def __update_dis_vector(self, new_sample):
        if self.n_components == 0:
            pass
        else:
            t = self.means_ - new_sample
            self.distance_vector = torch.pow(t, 2).sum(1)

    def __update_dis_matric(self):
        if self.n_components == 0:
            pass
        else:
            a_sqr = torch.pow(self.means_, 2).sum(1)
            b_sqr = torch.pow(self.means_, 2).sum(1).unsqueeze(1)
            ab = torch.mm(self.means_, torch.transpose(self.means_, 0, 1))
            self.distance_matric = a_sqr + b_sqr - 2 * ab + 1e12 * torch.eye(self.means_.shape[0], dtype=torch.float)
    
    # GMM_ECO.__get_matric_min_ele_and_idx:
    # get the min element in a 2D matrix 
    # return both value and index(a tuple)
    def __get_matric_min_ele_and_idx(self, mat):
        x = torch.min(torch.min(mat, 1)[0], 0)[1]
        y = torch.min(torch.min(mat, 0)[0], 0)[1]
        return mat[x,y], (x, y)
    
    def __get_vector_min_ele_and_idx(self, vec):
        return torch.min(vec, 0)[0], torch.min(vec, 0)[1]

    # GMM_ECO.update:
    # simplified version
    # initialize a component with pi = gamma and u = x where x is a sample
    # merge two components if distance is less than a threshold
    # otherwise if number of components exceeds n_threshold then discard one whose pi is less than threshold
    def update(self, sample):
        # sample: a feature and heatmap tuple of one sample
        # feature: shape of (F, H, W)
        feature, heatmap = sample
        F, H, W = feature.shape
        pi_new = torch.tensor(self.learning_rate, dtype=torch.float)
        mean_new = feature.view(F * H * W).float()
#         print(mean_new.shape)
        self.__update_dis_vector(mean_new)
#         if self.n_components == self.max_components and self.distance_vector.min() > 8 * 1e6:
#             print(self.distance_vector.min())
#             self.max_components += 1
        
        if self.n_components == self.max_components:
            min_val, min_idx = torch.min(self.weights_, 0)
            if min_val.item() < self.pi_threshold:
                self.weights_[min_idx.item()] = 0
                self.weights_ = self.weights_ * (1 - pi_new) / torch.sum(self.weights_)
                self.weights_[min_idx.item()] = pi_new
                self.means_[min_idx.item()] = mean_new
                self.components['features'][min_idx.item()].clear()
                self.components['heatmaps'][min_idx.item()].clear()
                self.components['features'][min_idx.item()].append(feature)
                self.components['heatmaps'][min_idx.item()].append(heatmap)
                
            else:
                new_sample_min_dist, closest_sample_to_new_sample = self.__get_vector_min_ele_and_idx(self.distance_vector)
                existing_samples_min_dist, closest_existing_sample_pair = self.__get_matric_min_ele_and_idx(self.distance_matric)
                if new_sample_min_dist < existing_samples_min_dist:
                    self.weights_ = self.weights_ * (1 - pi_new)
                    alpha1 = self.weights_[closest_sample_to_new_sample] / (self.weights_[closest_sample_to_new_sample] + pi_new)
                    alpha2 = 1 - alpha1
                    self.means_[closest_sample_to_new_sample] = alpha1 * self.means_[closest_sample_to_new_sample] + alpha2 * mean_new
                    self.weights_[closest_sample_to_new_sample] += pi_new
                    self.components['features'][min_idx.item()].append(feature)
                    self.components['heatmaps'][min_idx.item()].append(heatmap)
                else:
                    s_1, s_2 = closest_existing_sample_pair
                    self.weights_ = self.weights_ * (1 - pi_new)
                    alpha1 = self.weights_[s_1] / (self.weights_[s_1] + self.weights_[s_2])
                    alpha2 = 1 - alpha1
                    self.means_[s_1] = alpha1 * self.means_[s_1] + alpha2 * self.means_[s_2]
                    self.weights_[s_1] += self.weights_[s_2]
                    self.components['features'][s_1].extend(self.components['features'][s_2])
                    self.components['heatmaps'][s_1].extend(self.components['heatmaps'][s_2])
                    
                    self.weights_[s_2] = pi_new
                    self.means_[s_2] = mean_new
                    self.components['features'][s_2].clear()
                    self.components['heatmaps'][s_2].clear()
                    self.components['features'][s_2].append(feature)
                    self.components['heatmaps'][s_2].append(heatmap)

        else:
            if self.n_components == 0:
                self.weights_ = torch.tensor(1.).unsqueeze(0)
                self.means_ = mean_new.unsqueeze(0)
                self.components['features'].append([feature])
                self.components['heatmaps'].append([heatmap])
                self.n_components = 1
            else:
                self.weights_ = self.weights_ * (1 - pi_new)
                self.weights_ = torch.cat((self.weights_, pi_new.unsqueeze(0)))
                self.means_ = torch.cat((self.means_, mean_new.unsqueeze(0)))
                self.components['features'].append([feature])
                self.components['heatmaps'].append([heatmap])
                self.n_components += 1

        self.__update_dis_matric()
        
    def insert(self, features, heatmaps):
        # features: (N, , , ,)
        assert features.shape[0] == heatmaps.shape[0], 'shape mismatch'
        N = features.shape[0]
        for i in range(N):
            self.update((features[i], heatmaps[i]))

    def predict(self, var):
        px = 0.0
        t = self.means_ - var
        p = (-1 / 2) * torch.pow(t / 255, 2).sum(1)
        p = torch.exp(p)
        for i in range(self.n_components):
            p[i] = self.weights_[i] * p[i]
#         print(p / p.sum())
        t = p / p.sum()
                
        return t.max(0)
    
    def __js_divergence(self, mu_1, cov_1, mu_2, cov_2):
        assert mu_1.shape == mu_2.shape, 'mu shape mismatch'
        assert cov_1.shape == cov_2.shape, 'cov shape mismatch'

        # Monte Carlo samples
        MC_samples = 1000

        Pd = MultivariateNormal(loc = mu_1, covariance_matrix=cov_1)
        Qd = MultivariateNormal(loc = mu_2, covariance_matrix=cov_2)
        P_samples = Pd.sample((MC_samples,))
        Q_samples = Qd.sample((MC_samples,))

    #     print(Pd.log_prob(torch.tensor([1.,1.])).type())
        P = lambda x: torch.tensor(np.power(2, Pd.log_prob(x).numpy()))
        Q = lambda x: torch.tensor(np.power(2, Qd.log_prob(x).numpy()))
        M = lambda x: 0.5 * P(x) + 0.5 * Q(x)

        P_div_M = lambda x: P(x) / M(x)
        Q_div_M = lambda x: Q(x) / M(x)

        D_KL_approx_PM = lambda x: (1 / MC_samples) * sum(torch.log2(P_div_M(x)))
        D_KL_approx_QM = lambda x: (1 / MC_samples) * sum(torch.log2(Q_div_M(x)))

        return 0.5 * D_KL_approx_PM(P_samples) + 0.5 * D_KL_approx_QM(Q_samples)
    
    # require:
    # data: shape of (N, D, H, W)
    def __initialization(self, data):
        if self.__init:
            gmm_sk = mixture.GaussianMixture(n_components=self.n_components, covariance_type='diag', max_iter=100)
            n, d, h, w = data.size()[0], data.size()[1], data.size()[2], data.size()[3]
            if n > 1000 :
                gmm_sk.fit(data[np.random.choice(n, 1000, replace=False)].view(-1, d * h * w).numpy())
            else:
                gmm_sk.fit(data.view(-1, d * h * w).numpy())
            
            self.means_ = torch.tensor(gmm_sk.means_, dtype=torch.float)
            self.weights_ = torch.tensor(gmm_sk.weights_, dtype=torch.float)
            print(self.means_.shape)
            self.__update_dis_matric()
        else:
            self.n_components = 0