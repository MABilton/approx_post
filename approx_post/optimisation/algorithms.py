
# See: https://www4.stat.ncsu.edu/~lu/ST7901/reading%20materials/Adam_algorithm.pdf
def adam(grad, optim_params):
    # Update number of iterations:
    optim_params['num_iter'] += 1
    # Unpack parameters:
    beta_1, beta_2, lr, eps, num_iter = \
        (optim_params[key] for key in ('beta_1', 'beta_2', 'lr', 'eps', 'num_iter'))
    # Unpack velocity and momentum terms from previous iterations:
    m_tm1 = optim_params["m_tm1"] if "m_tm1" in optim_params else 0.
    v_tm1 = optim_params["v_tm1"] if "v_tm1" in optim_params else 0.
    # Compute ADAM update:
    m_t = (1-beta_1)*grad + beta_1*m_tm1
    v_t = (1-beta_2)*grad**2 + beta_2*v_tm1
    tilde_m_t = m_t/(1-beta_1**num_iter)
    tilde_v_t = v_t/(1-beta_2**num_iter)
    update = lr*tilde_m_t/(tilde_v_t**(1/2) + eps)
    # Update m_tm1 and v_tm1 for next iteration:
    optim_params["m_tm1"], optim_params["v_tm1"] = m_t, v_t
    return (update, optim_params)

# See: https://machinelearningjourney.com/index.php/2021/01/05/adagrad-optimizer/
def adagrad(grad, optim_params):
    # Unpack parameters:
    lr, eps = optim_params["lr"], optim_params["eps"]
    # Unpack squared gradient:
    s = optim_params["s"] if "s" in optim_params else 0*grad
    # Update squared gradient and store for next iteration:
    s += grad**2
    optim_params["s"] = s
    # Perform adagrad update:
    update = lr*grad/((s + eps)**0.5)
    return (update, optim_params)