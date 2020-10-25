
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    
     """
        Implementation of the Logistic Regression method 
        
        ARGUMENTS :
        _____________________________________________________
        |                |                                  |
        |    y           |    Predictions                   |
        |    tx          |    Samples                       |
        |    initial_w   |    Initial weights               |
        |    max_iters   |    Maximum number of iterations  |
        |    gamma       |    Step size                     |
        |________________|__________________________________|    
            
        RETURN VALUES :
        _____________________________________________________
        |                |                                  |
        |    w           |    Optimal weights               |
        |    loss        |    Final loss value              |
        |________________|__________________________________|
          
    """

    losses = []                   #loss array

    ws = [initial_w]              #weight array
    w = initial_w                 #current weight
    
    threshold = 1e-8
    
    """ --- ITERATIONS --- """
        
    for iter in range(max_iter):
        
        # gradient L_n formula: x_n * (sigma(x_n * w) - y_n)
        tx_t=tx.T
        grad=tx_t.dot( sigmoid (tx.dot(w))-y)

        # loss function
        loss = np.sum(np.log(1. + np.exp(np.dot(tx, w)))) - np.dot(y.T, np.dot(tx, w))

        # weight
        w = w - gamma * grad
        
        # array filling
        losses.append(loss)
        ws.append(w)
        
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        
        # converge criterion
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    
    """ --- RETURN VALUES --- """
    return ws[-1], losses[-1]


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    
     """
        Implementation of the Regularized Logistic Regression method 
        
        ARGUMENTS :
        _____________________________________________________
        |                |                                  |
        |    y           |    Predictions                   |
        |    tx          |    Samples                       |
        |    lambda_     |    Penalty factor                |
        |    initial_w   |    Initial weights               |
        |    max_iters   |    Maximum number of iterations  |
        |    gamma       |    Step size                     |
        |________________|__________________________________|    
            
        RETURN VALUES :
        _____________________________________________________
        |                |                                  |
        |    w           |    Optimal weights               |
        |    loss        |    Final loss value              |
        |________________|__________________________________|
          
    """
    
    losses = []                   #loss array
    
    ws = [initial_w]              #weight array
    w = initial_w                 #current weight
    
    threshold = 1e-8
    
    """ --- ITERATIONS --- """
        
    for iter in range(max_iter):
        
        # gradient L_n formula: x_n * (sigma(x_n * w) - y_n)
        tx_t=tx.T
        grad=tx_t.dot( sigmoid (tx.dot(w))-y) + 2 * lambda_ * w

        # loss function
        loss = np.sum(np.log(1. + np.exp(np.dot(tx, w)))) - np.dot(y.T, np.dot(tx, w)) + lambda_ * np.linalg.norm(w) ** 2

        # weight
        w = w - gamma * grad
        
        # array filling
        losses.append(loss)
        ws.append(w)
        
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        
        # converge criterion
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
            
    
    """ --- RETURN VALUES --- """
    return ws[-1], losses[-1]
"""
