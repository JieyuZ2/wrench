import math
from typing import Optional, Any, Union
import numpy as np
import scipy.sparse as ssp
import torch
from torch import digamma
from tqdm import trange

from wrench.basemodel import BaseLabelModel
from wrench.dataset import BaseDataset


def create_tuples(dataset: Union[BaseDataset, np.ndarray]):
    ids = np.repeat(np.array(range(len(dataset))), len(dataset.weak_labels[0]))
    workers = np.repeat(
        np.array([i for i in range(len(dataset.weak_labels[0]))]), len(dataset.weak_labels)
    ).reshape(len(dataset.weak_labels[0]), -1).T.reshape(-1)
    classes = np.array(dataset.weak_labels).reshape(-1)
    tuples = np.vstack((ids, workers, classes)).astype(np.int32)

    return tuples.T


def scale(data):
    _max = torch.max(data)
    _min = torch.min(data)
    return (data - _min) / (_max - _min)


def lanczos_tridiag(
        matmul_closure,
        max_iter,
        dtype,
        device,
        matrix_shape,
        batch_shape=torch.Size(),
        init_vecs=None,
        num_init_vecs=1,
        tol=1e-5,
):
    # Determine batch mode
    multiple_init_vecs = False

    if not callable(matmul_closure):
        raise RuntimeError(
            "matmul_closure should be a function callable object that multiples a (Lazy)Tensor "
            "by a vector. Got a {} instead.".format(matmul_closure.__class__.__name__)
        )

    # Get initial probe ectors - and define if not available
    if init_vecs is None:
        init_vecs = torch.randn(matrix_shape[-1], num_init_vecs, dtype=dtype, device=device)
        init_vecs = init_vecs.expand(*batch_shape, matrix_shape[-1], num_init_vecs)

    else:
        num_init_vecs = init_vecs.size(-1)

    # Define some constants
    num_iter = min(max_iter, matrix_shape[-1])
    dim_dimension = -2

    # Create storage for q_mat, alpha,and beta
    # q_mat - batch version of Q - orthogonal matrix of decomp
    # alpha - batch version main diagonal of T
    # beta - batch version of off diagonal of T
    q_mat = torch.zeros(num_iter, *batch_shape, matrix_shape[-1], num_init_vecs, dtype=dtype, device=device)
    t_mat = torch.zeros(num_iter, num_iter, *batch_shape, num_init_vecs, dtype=dtype, device=device)

    # Begin algorithm
    # Initial Q vector: q_0_vec
    q_0_vec = init_vecs / torch.norm(init_vecs, 2, dim=dim_dimension).unsqueeze(dim_dimension)
    q_mat[0].copy_(q_0_vec)

    # Initial alpha value: alpha_0
    r_vec = matmul_closure(q_0_vec)
    alpha_0 = q_0_vec.mul(r_vec).sum(dim_dimension)

    # Initial beta value: beta_0
    r_vec.sub_(alpha_0.unsqueeze(dim_dimension).mul(q_0_vec))
    beta_0 = torch.norm(r_vec, 2, dim=dim_dimension)

    # Copy over alpha_0 and beta_0 to t_mat
    t_mat[0, 0].copy_(alpha_0)
    t_mat[0, 1].copy_(beta_0)
    t_mat[1, 0].copy_(beta_0)

    # Compute the first new vector
    q_mat[1].copy_(r_vec.div_(beta_0.unsqueeze(dim_dimension)))

    # Now we start the iteration
    for k in range(1, num_iter):
        # Get previous values
        q_prev_vec = q_mat[k - 1]
        q_curr_vec = q_mat[k]
        beta_prev = t_mat[k, k - 1].unsqueeze(dim_dimension)

        # Compute next alpha value
        r_vec = matmul_closure(q_curr_vec) - q_prev_vec.mul(beta_prev)
        alpha_curr = q_curr_vec.mul(r_vec).sum(dim_dimension, keepdim=True)
        # Copy over to t_mat
        t_mat[k, k].copy_(alpha_curr.squeeze(dim_dimension))

        # Copy over alpha_curr, beta_curr to t_mat
        if (k + 1) < num_iter:
            # Compute next residual value
            r_vec.sub_(alpha_curr.mul(q_curr_vec))
            # Full reorthogonalization: r <- r - Q (Q^T r)
            correction = r_vec.unsqueeze(0).mul(q_mat[: k + 1]).sum(dim_dimension, keepdim=True)
            correction = q_mat[: k + 1].mul(correction).sum(0)
            r_vec.sub_(correction)
            r_vec_norm = torch.norm(r_vec, 2, dim=dim_dimension, keepdim=True)
            r_vec.div_(r_vec_norm)

            # Get next beta value
            beta_curr = r_vec_norm.squeeze_(dim_dimension)
            # Update t_mat with new beta value
            t_mat[k, k + 1].copy_(beta_curr)
            t_mat[k + 1, k].copy_(beta_curr)

            # Run more reorthoganilzation if necessary
            inner_products = q_mat[: k + 1].mul(r_vec.unsqueeze(0)).sum(dim_dimension)
            could_reorthogonalize = False
            for _ in range(10):
                if not torch.sum(inner_products > tol):
                    could_reorthogonalize = True
                    break
                correction = r_vec.unsqueeze(0).mul(q_mat[: k + 1]).sum(dim_dimension, keepdim=True)
                correction = q_mat[: k + 1].mul(correction).sum(0)
                r_vec.sub_(correction)
                r_vec_norm = torch.norm(r_vec, 2, dim=dim_dimension, keepdim=True)
                r_vec.div_(r_vec_norm)
                inner_products = q_mat[: k + 1].mul(r_vec.unsqueeze(0)).sum(dim_dimension)

            # Update q_mat with new q value
            q_mat[k + 1].copy_(r_vec)

            if torch.sum(beta_curr.abs() > 1e-6) == 0 or not could_reorthogonalize:
                break

    # Now let's transpose q_mat, t_mat intot the correct shape
    num_iter = k + 1

    # num_init_vecs x batch_shape x matrix_shape[-1] x num_iter
    q_mat = q_mat[:num_iter].permute(-1, *range(1, 1 + len(batch_shape)), -2, 0).contiguous()
    # num_init_vecs x batch_shape x num_iter x num_iter
    t_mat = t_mat[:num_iter, :num_iter].permute(-1, *range(2, 2 + len(batch_shape)), 0, 1).contiguous()

    # If we weren't in batch mode, remove batch dimension
    if not multiple_init_vecs:
        q_mat.squeeze_(0)
        t_mat.squeeze_(0)

    # We're done!
    return q_mat, t_mat


def fable_vb(tuples,
             X,  # (num_items, num_features)
             device=None,
             kernel_function=None,
             num_groups=10,  # M
             alpha=1,  # alpha_k, it can be 1 or \sum_i gamma_ik depended on empirical_prior
             a_v=4,  # beta_kk
             b_v=1,  # beta_kk', k neq k'
             nu_k_learned=None,
             mu_jkml_learned=None,
             eval=False,
             seed=1234,
             inference_iter=500,
             desired_rank=128,
             empirical_prior=False,
             disable_tqdm=False, ):
    torch.cuda.empty_cache()
    num_items, num_workers, num_classes = tuples.max(axis=0) + 1

    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # initialize alpha_ga_i, m_hat with non-information prior
    alpha_ga_i = torch.rand(num_items) + 1e-3
    m_hat = torch.rand((num_items, num_classes * num_groups)) - 0.5

    y_is_one_lij = []
    y_is_one_lji = []
    for k in list(range(num_classes)) + [-1]:
        selected = (tuples[:, 2] == k)
        coo_ij = ssp.coo_matrix((torch.ones(selected.sum()), tuples[selected, :2].T),
                                shape=(num_items, num_workers),
                                dtype=np.bool)
        y_is_one_lij.append(coo_ij.tocsr())
        y_is_one_lji.append(coo_ij.T.tocsr())

    # initialize c_ik, ga_ik, beta_mu_kl
    beta_mu_kl = torch.zeros(num_classes, num_classes + 1)

    for k in range(num_classes):
        for l in list(range(num_classes)) + [-1]:
            if k == l:
                beta_mu_kl[k, l] = a_v - b_v
            elif l != -1:
                beta_mu_kl[k, l] = b_v
            else:
                beta_mu_kl[k, l] = y_is_one_lji[l].sum()

    # initialize z_ik, zg_ikm, sigma_ik
    if kernel_function is None:
        sigma_init_temp = torch.eye(num_items).to(device, non_blocking=True)
    else:
        sigma_init_temp = torch.tensor(kernel_function(X)).to(device)
    sigma_init_temp = sigma_init_temp + torch.eye(num_items).to(device) * 1e-3

    # Options for initializing other inverse matrix calculation methods
    sigma_inv = torch.linalg.inv(sigma_init_temp)
    sigma_hat = []
    for l in range(num_classes * num_groups):
        sigma_hat.append(sigma_init_temp.cpu())

    z_ik = torch.zeros((num_items, num_classes))
    for l in range(num_classes):
        z_ik[:, [l]] += torch.tensor(y_is_one_lij[l].sum(axis=-1) + 1e-5)
    z_ik /= z_ik.sum(dim=-1, keepdim=True)
    z_ik = z_ik.double()

    if empirical_prior:
        alpha = z_ik.sum(dim=0)

    # q(G, Z) = zg_ikm (paper ver)
    # q(upsilon, omega): f_ik, ga_ik
    # q(lambda): alpha_ga_i, beta_ga_i
    # q(f): m_hat, sigma_hat
    # q(pi): a_ga_ik, b_ga_ik
    # q(Tau): nu_k
    # q(V): mu_jkml
    # ################################
    # q(Z): z_ik (prediction)
    # ============================== Inference =====================================
    zg_ikm = torch.tensor(np.random.dirichlet(np.ones(num_groups), z_ik.shape)) * z_ik[:, :, None]
    for it in trange(0, inference_iter, unit='iter', disable=disable_tqdm):

        if eval is False:
            # update rules for q(Tau)
            nu_k = alpha + z_ik.sum(dim=0)
            mu_jkml = torch.zeros((num_workers, num_classes, num_groups, num_classes + 1)) + beta_mu_kl[None, :, None, :]
            # update rules for q(V)
            for l in list(range(num_classes)) + [-1]:
                for k in range(num_classes):
                    tmp = torch.tensor(y_is_one_lji[l].dot(zg_ikm[:, k, :].numpy()))
                    if l == -1:
                        mu_jkml[:, k, :, num_classes] += tmp
                    else:
                        mu_jkml[:, k, :, l] += tmp
        else:
            nu_k = torch.tensor(nu_k_learned)
            mu_jkml = torch.tensor(mu_jkml_learned)

        # update rules for q(Upsilon, Omega)
        gamma_update = torch.exp(digamma(alpha_ga_i)) / num_classes

        c_ik = torch.zeros((num_items, num_classes * num_groups))
        ga_ik = torch.zeros((num_items, num_classes * num_groups))
        f_ik = torch.zeros((num_items, num_classes * num_groups))
        for k in range(num_classes * num_groups):
            f_ik[:, k] = torch.sqrt(torch.abs(torch.pow(m_hat[:, k], 2) + torch.diag(sigma_hat[k])))
            f_ik[:, k] = f_ik[:, k].clamp(max=1e2)  # prevent overflow
            tem_exp = -0.5 * m_hat[:, k]
            tem_exp = tem_exp.clamp(max=1e2)  # prevent overflow
            ga_ik[:, k] = gamma_update * torch.exp(tem_exp) / torch.cosh(f_ik[:, k])
            c_ik[:, k] = f_ik[:, k]

        # update rules for q(Lambda)
        alpha_ga_i = ga_ik.sum(dim=-1) + 1
        a_ga_ikm = (zg_ikm + 1.0).double()
        b_ga_ikm = math.log(2.0) - m_hat / 2.0

        # update rules for m_hat and sigma_hat
        divide_ab = a_ga_ikm.reshape((num_items, num_classes * num_groups)) / b_ga_ikm.reshape(
            (num_items, num_classes * num_groups))
        divide_ab = divide_ab.to(device, non_blocking=True, dtype=torch.float64)
        ga_ik = ga_ik.to(device, non_blocking=True, dtype=torch.float64)
        c_ik = c_ik.to(device, non_blocking=True)
        diag = ((divide_ab + ga_ik) / (2 * c_ik) * torch.tanh(c_ik * 0.5))
        for k in range(num_classes * num_groups):
            if desired_rank == None:
                sigma_hat_tmp = torch.linalg.inv(sigma_inv + torch.diag(diag[:, k]))  # non-singular
            else:
                sigma_inv_hat = sigma_inv + torch.diag(diag[:, k])
                q_mat, t_mat = lanczos_tridiag(sigma_inv_hat.matmul,
                                               max_iter=desired_rank,
                                               dtype=sigma_inv_hat.dtype,
                                               device=device,
                                               matrix_shape=sigma_inv.shape)
                sigma_hat_tmp = q_mat @ torch.linalg.inv(t_mat) @ q_mat.T  # Lanczos
                sigma_hat_tmp = sigma_hat_tmp.double()

            m_hat[:, k] = (0.5 * sigma_hat_tmp.to(device) @ (divide_ab[:, k] - ga_ik[:, k])).to('cpu')
            sigma_hat[k] = sigma_hat_tmp.to('cpu')  # save gpu memory
            del sigma_hat_tmp  # save gpu memory

        # q(G, Z)
        Eq_log_pi_ikm = digamma(a_ga_ikm.reshape(num_items, num_classes, num_groups)) - torch.log(
            b_ga_ikm.reshape(num_items, num_classes, num_groups))
        Eq_log_tau_k = digamma(nu_k) - digamma(nu_k.sum())
        Eq_log_v_jkml = digamma(mu_jkml) - digamma(mu_jkml.sum(dim=-1, keepdim=True))

        zg_ikm = Eq_log_pi_ikm + Eq_log_tau_k[None, :, None]
        for l in list(range(num_classes)) + [-1]:
            for k in range(num_classes):
                if l == -1:
                    zg_ikm[:, k, :] += y_is_one_lij[l].dot(Eq_log_v_jkml[:, k, :, num_classes].numpy())
                else:
                    zg_ikm[:, k, :] += y_is_one_lij[l].dot(Eq_log_v_jkml[:, k, :, l].numpy())

        # update rules for q(Z)
        zg_ikm = torch.exp(zg_ikm)
        zg_ikm /= zg_ikm.reshape(num_items, -1).sum(dim=-1)[:, None, None]
        if torch.isnan(zg_ikm).any():
            print('stop')
            break

        last_z_ik = z_ik

        # prediction
        z_ik = zg_ikm.sum(dim=-1)

        if torch.allclose(last_z_ik, z_ik, atol=1e-2):
            print(f'gp-ebcc: convergent at step {it}')
            break

    return z_ik.cpu().numpy(), nu_k.cpu().numpy(), mu_jkml.cpu().numpy()


class Fable(BaseLabelModel):
    """Fable

        Usage:

            fable = FABLE(num_groups, a_pi, a_v, b_v, inference_iter, empirical_prior, kernel_function, desired_rank)
            fable.fit(data)
            fable.test(data)

        Parameters:

            num_groups: number of subtypes
            a_pi: The parameter of dirichlet distribution for generate mixture weight.
            a_v: b_kk, number of corrected labeled items under every class.
            b_v: b_kk', all kind of miss has made b_kk' times.
            inference_iter: Iterations of variational inference.
            empirical_prior: The empirical prior of alpha.
            kernel_function: The kernel function of Gaussian process.
            desired_rank: Param for reduced rank approximation (Lanczos), which reduce the rank of matrix to desired_rank.
            device: The torch.device to use.
            seed: Random seed.
    """

    def __init__(self,
                 kernel_function,
                 num_groups: Optional[int] = 10,
                 alpha: Optional[float] = 1,
                 a_v: Optional[float] = 4,
                 b_v: Optional[float] = 1,
                 inference_iter: Optional[int] = 1000,
                 seed: Optional[int] = None,
                 empirical_prior=False,
                 device: Optional[torch.device] = None,
                 **kwargs: Any):
        super().__init__()
        self.hyperparas = {
            'seed': np.random.randint(1e8) if seed is None else seed,
            'kernel_function': kernel_function,
            'num_groups': num_groups,
            'alpha': alpha,
            'a_v': a_v,
            'b_v': b_v,
            'empirical_prior': empirical_prior,
            'inference_iter': inference_iter,
            **kwargs
        }
        self.params = {
            'nu_k_learned': None,
            'mu_jkml_learned': None
        }
        self.device = device

    def fit(self,
            dataset_train: Union[BaseDataset, np.ndarray],
            dataset_valid: Optional[Union[BaseDataset, np.ndarray]] = None,
            y_valid: Optional[np.ndarray] = None,
            n_class: Optional[int] = None,
            verbose: Optional[bool] = False,
            *args: Any,
            **kwargs: Any):
        train_tuples = create_tuples(dataset_train)
        inputs = np.array(dataset_train.features)
        nan_index = np.unique(np.argwhere(np.isnan(inputs))[:, 0])
        inputs[nan_index] = np.zeros(inputs.shape[1])
        print(f'NaN values included: {nan_index.tolist()}')

        pred, nu_k, mu_jkml = fable_vb(train_tuples, inputs,
                                       device=self.device,
                                       **self.params, **self.hyperparas)
        self.params.update({
            'nu_k_learned': nu_k,
            'mu_jkml_learned': mu_jkml
        })
        return pred

    def predict_proba(self,
                      dataset: Union[BaseDataset, np.ndarray],
                      batch_learning: Optional[bool] = False,
                      **kwargs: Any):
        test_batch = dataset

        tuples = create_tuples(test_batch)
        inputs = np.array(test_batch.features)
        nan_index = np.unique(np.argwhere(np.isnan(inputs))[:, 0])
        inputs[nan_index] = np.zeros(inputs.shape[1])
        eval = True

        if self.params['nu_k_learned'] is None or self.params['mu_jkml_learned'] is None:
            eval = False

        disable_tqdm = False
        if batch_learning:
            disable_tqdm = True
        pred, _, _ = fable_vb(tuples, inputs,
                              eval=eval,
                              device=self.device,
                              disable_tqdm=disable_tqdm,
                              **self.hyperparas, **self.params)
        return pred
