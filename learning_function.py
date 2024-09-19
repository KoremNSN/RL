import pytensor.tensor as pt

def update_single(stim, reward, As, vec, alpha, n_subj):
    """
    
    """
    
    PE = reward - As[pt.arange(n_subj), stim]
    As = pt.set_subtensor(As[pt.arange(n_subj), stim], 
                          As[pt.arange(n_subj), stim] + alpha * PE)
    
    vec = pt.set_subtensor(vec[pt.arange(n_subj), 0], 
                           pt.switch(pt.eq(stim, 1), 
                                     As[pt.arange(n_subj), 1], 
                                     As[pt.arange(n_subj), 0]))
    return As, vec


def update_valence(stim, reward,
                      As, vec,
                      alpha_p, alpha_n, n_subj):
    """
    
    """
     
    PE = reward - As[pt.arange(n_subj), stim]
    As = pt.set_subtensor(As[pt.arange(n_subj),stim], 
                          (pt.switch(pt.eq(reward,1),
                                     As[pt.arange(n_subj),stim] + alpha_p * PE,
                                     As[pt.arange(n_subj),stim] + alpha_n * PE)))
    
    # in order to get a vector of expected outcome (dependent on the stimulus presentes [CS+, CS-] 
    # we us if statement (switch in theano)
    vec = pt.set_subtensor(vec[pt.arange(n_subj),0], 
                           (pt.switch(pt.eq(stim,1), 
                                      As[pt.arange(n_subj),1], 
                                      As[pt.arange(n_subj),0])))
    
    return As, vec

def update_halves(stim, reward, trial,
                   As, vec, 
                   alpha_1, alpha_2,  n_subj):
    """
    
    """
    PE = reward - As[pt.arange(n_subj), stim]
    As = pt.set_subtensor(As[pt.arange(n_subj),stim], 
                          (pt.switch(pt.eq(trial,1),
                                     As[pt.arange(n_subj),stim] + alpha_1 * PE,
                                     As[pt.arange(n_subj),stim] + alpha_2 * PE)))
    
    # in order to get a vector of expected outcome (dependent on the stimulus presentes [CS+, CS-] 
    # we us if statement (switch in theano)
    vec = pt.set_subtensor(vec[pt.arange(n_subj),0], 
                           (pt.switch(pt.eq(stim,1),
                                      As[pt.arange(n_subj),1], 
                                      As[pt.arange(n_subj),0])))
    
    return As, vec
 
def update_valence_halves(stim, reward, trial,
             As, vec, 
             alpha_p1, alpha_n1, alpha_p2, alpha_n2,  n_subj):
    """
    
    """
    PE = reward - As[pt.arange(n_subj), stim]
    As = pt.set_subtensor(As[pt.arange(n_subj),stim], 
                          (pt.switch(pt.eq(trial,1),
                                     (pt.switch(pt.eq(reward,1),
                                                As[pt.arange(n_subj),stim] + alpha_p1 * PE,
                                                As[pt.arange(n_subj),stim] + alpha_n1 * PE)),
                                                (pt.switch(pt.eq(reward,1),
                                                           As[pt.arange(n_subj),stim] + alpha_p2 * PE,
                                                           As[pt.arange(n_subj),stim] + alpha_n2 * PE)))))
                                                                 
    
    # in order to get a vector of expected outcome (dependent on the stimulus presentes [CS+, CS-] 
    # we us if statement (switch in theano)
    vec = pt.set_subtensor(vec[pt.arange(n_subj),0], 
                           (pt.switch(pt.eq(stim,1),
                                      As[pt.arange(n_subj),1], 
                                      As[pt.arange(n_subj),0])))
    
    return As, vec

def update_decay(stim, reward,
                   As,vec,
                   alpha, decay, n_subj):
    """
    
    """
     
    PE = reward - As[pt.arange(n_subj), stim]
    As = pt.set_subtensor(As[pt.arange(n_subj),stim], 
                          As[pt.arange(n_subj),stim] * decay + alpha * PE)
    
    # in order to get a vector of expected outcome (dependent on the stimulus presentes [CS+, CS-] 
    # we us if statement (switch in theano)
    vec = pt.set_subtensor(vec[pt.arange(n_subj),0], 
                           (pt.switch(pt.eq(stim,1), 
                                      As[pt.arange(n_subj),1], 
                                      As[pt.arange(n_subj),0])))
    
    return As, vec

def update_counterfactual(stim, reward, As, vec, alpha_chosen, alpha_counterfactual, n_subj):
    """
    Update Q table for both chosen and unchosen stimuli (counterfactual learning) using separate learning rates.
    
    The chosen stimulus is updated based on the actual reward with the learning rate alpha_chosen.
    The unchosen stimulus is updated with a counterfactual reward (what could have happened) using alpha_counterfactual.
    """
    
    # Prediction error for the chosen stimulus
    PE_chosen = reward - As[pt.arange(n_subj), stim]
    
    # Index of the unchosen stimulus (opposite of stim)
    stim_opposite = 1 - stim
    
    # Counterfactual prediction error for the unchosen stimulus
    PE_unchosen = (1 - reward) - As[pt.arange(n_subj), stim_opposite]
    
    # Update chosen stimulus value with its own learning rate
    As = pt.set_subtensor(As[pt.arange(n_subj), stim], 
                          As[pt.arange(n_subj), stim] + alpha_chosen * PE_chosen)
    
    # Update unchosen stimulus value with a separate counterfactual learning rate
    As = pt.set_subtensor(As[pt.arange(n_subj), stim_opposite], 
                          As[pt.arange(n_subj), stim_opposite] + alpha_counterfactual * PE_unchosen)
    
    # Update vec based on the new As values
    vec = pt.set_subtensor(vec[pt.arange(n_subj), 0], 
                           pt.switch(pt.eq(stim, 1), 
                                     As[pt.arange(n_subj), 1], 
                                     As[pt.arange(n_subj), 0]))
    return As, vec

def update_phall(stim, reward,
                As, vec, alpha, assoc,
                eta, kappa, n_subj):
    """
    This function updates the Q table according to Hybrid PH model
    For information, please see this paper: https://www.sciencedirect.com/science/article/pii/S0896627316305840?via%3Dihub
  
    """
      
    delta = reward - As[pt.arange(n_subj), stim]
    alpha = pt.set_subtensor(alpha[pt.arange(n_subj), stim], 
                             eta * abs(delta) + (1 - eta) * alpha[pt.arange(n_subj), stim])
    As = pt.set_subtensor(As[pt.arange(n_subj),stim], 
                          As[pt.arange(n_subj),stim] + kappa * alpha[pt.arange(n_subj), stim] * delta)
    
    # in order to get a vector of expected outcome (dependent on the stimulus presentes [CS+, CS-] 
    # we us if statement (switch in theano)
    vec = pt.set_subtensor(vec[pt.arange(n_subj),0], 
                           (pt.switch(pt.eq(stim,1), 
                                      As[pt.arange(n_subj),1], 
                                      As[pt.arange(n_subj),0])))
    
    # we use the same idea to get the associability per trial
    assoc = pt.set_subtensor(assoc[pt.arange(n_subj),0],
                              (pt.switch(pt.eq(stim,1), 
                                         alpha[pt.arange(n_subj),1],
                                         alpha[pt.arange(n_subj),0])))
    
    return As, vec, alpha, assoc


# Rouhani et al. version https://elifesciences.org/articles/61077
# generate functions to run
def update_hall_rounai(stim, reward,
                       As, vec,
                       eta, kappa, n_subj):
    """
    This function updates the Q table according to Hybrid PH model
    For information, please see this paper: https://www.sciencedirect.com/science/article/pii/S0896627316305840?via%3Dihub
  
    """
    
    
    delta = reward - As[pt.arange(n_subj), stim]
    alpha = eta + (kappa * abs(delta))
    alpha = 1 / (1 + pm.math.exp(-alpha)) # sigmoid function
    As = pt.set_subtensor(As[pt.arange(n_subj),stim], 
                          As[pt.arange(n_subj),stim] + alpha * delta)
    
  
    vec = pt.set_subtensor(vec[pt.arange(n_subj),0], (pt.switch(pt.eq(stim,1), 
                                                                As[pt.arange(n_subj),1], 
                                                                As[pt.arange(n_subj),0])))
        
    return As, vec, alpha

def update_RW_age(stim, reward, ageT,
             As, vec,
             alpha, age, n_subj):
    """
    This function updates the Q table according to the RL update rule.
    It will be called by theano.scan to do so recursevely, given the observed data and the alpha parameter
    This could have been replaced be the following lamba expression in the theano.scan fn argument:
        fn=lamba action, reward, As, alpha: pt.set_subtensor(As[action], As[action] + alpha * (reward - As[action]))
    """
     
    PE = reward - As[pt.arange(n_subj), stim]
    learnRate = alpha + age * As[pt.arange(n_subj),ageT]
    
    As = pt.set_subtensor(As[pt.arange(n_subj),stim], 
                          As[pt.arange(n_subj),stim] + learnRate * PE)
    
    vec = pt.set_subtensor(vec[pt.arange(n_subj),0], (pt.switch(pt.eq(stim,1), 
                                                                As[pt.arange(n_subj),1], 
                                                                As[pt.arange(n_subj),0])))
    
    return As, vec