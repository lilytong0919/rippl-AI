# helper functions (copied from aux_fcn to avoid dependency issue)
import matplotlib.pyplot as plt
import numpy as np

def get_performance(pred_events, true_events, threshold=0, exclude_matched_trues=False, verbose=True):

    '''
    [precision, recall, F1, TP, FN, IOU] = get_performance(pred_events, true_events)
    Computes all these measures given a cell array with boundaries of pred_events

    Inputs:
    -------
        pred_events		 Nx2 matrix with start and end of pred events (seconds)
        true_events		 Mx2 matrix with start and end of true events (seconds)
        threshold		   Threshold to IoU. By default is 0
        exclude_matched_trues False by defaut (one true can match many predictions)

    Output:
    -------
        precision		   Metric indicating the percentage of correct 
                            predictions out of total predictions
        recall			  Metric indicating the percentage of true events 
                            predicted correctly
        F1				  Metric with a measure that combines precision and recall.
        TP				  Nx1 matrix indicating which pred event detected a true event, and
                            a true positive (True) or was false negative (False)
        FN				  Mx1 matrix indicating which true event has been detected (False)
                            or not (True)
        IOU				 NxM array with intersections over unions

    A Navas-Olive, LCN 2020
    '''

    # Check similarity of pred and true events by computing if intersection over union > 0
    # Compute IOU
    [IOU, IOU_pred, IOU_true] = intersection_over_union(pred_events, true_events)
    # IOU_pred-> IOU de cada evento predecido
    # IOU_true-> IOU de la GT


 # Excluye los true coincidentes
    if exclude_matched_trues:
        # Take maximal IOUs, and make the rest be zero
        pred_with_maxIOU = np.argmax(IOU, axis=0)
        IOU_pred_one_true_match = np.zeros_like(IOU)
        for itrue, ipred in enumerate(pred_with_maxIOU):
            IOU_pred_one_true_match[ipred, itrue] = IOU[ipred, itrue]
        # True positive: Predicted event that has a IoU with any true > 0
        TP = (IOU_pred_one_true_match.sum(axis=1) > threshold)
        # False negative: Predicted event that has not a IoU with any true
        FN = (IOU_true <= threshold)

    else:
        # True positive: Predicted event that has a IoU with any true > 0
        # 
        TP = (IOU_pred>threshold) 
        # False negative: Predicted event that has not a IoU with any true
        FN = (IOU_true<=threshold)  
    
    # Precision and recall
    precision = np.mean(TP)      # Media de verdaderos positivos: 1 si todas las predicciones son aciertos
    recall = 1. - np.mean(FN)    # 1-media de falsos negativos: 1 si toda la GT está presente en los aciertos
    F1 = 2. * (precision * recall) / (precision + recall)  
    if (precision + recall) == 0:
        F1 = 0.
    else:
        F1 = 2. * (precision * recall) / (precision + recall)

    if verbose:
        print('precision =', precision)
        print('recall =', recall)
        print('F1 =', F1)
    
    # Variable outputs
    return precision, recall, F1, TP, FN, IOU

def intersection_over_union(x, y):
    '''
    IOU = intersection_over_union(x, y) computes the percentage of 
    intersection over their union between every two pair of intervals 
    x and y.
    
    Inputs:
    -------
      x	 Nx2 array with beginnings and ends of 1D events
      y	 Mx2 array with beginnings and ends of 1D events
    
    Output:
    -------
      IOU   NxM array with intersections over unions
      IOUx  (optional) Nx1 array with indexes of y events with maximal IOU 
            It's zero if IOU=0 for all y events
      IOUy  (optional) Mx1 array with indexes of x events with maximal IOU 
            It's zero if IOU=0 for all x events
    
    A Navas-Olive, LCN 2020
    '''

    if (len(x)>0) and (len(y)>0):

        # Initialize
        Intersection = np.zeros((x.shape[0],  y.shape[0]),dtype=np.float32)
        Union = np.ones((x.shape[0],  y.shape[0]),dtype=np.float32)
        # Go through every y (beginning-end) pair
        for iy in range(y.shape[0]):
            # Intersection duration: difference between minimum end and maximum ini
            Intersection[:, iy] = np.maximum( np.minimum(x[:, 1], y[iy, 1]) - np.maximum(x[:, 0], y[iy, 0]), 0)
            # Union duration: sum of durations of both events minus its intersection
            Union[:, iy, None] = np.diff(x, axis=1) + np.diff(y[iy, :]) - Intersection[:, iy, None]

        # Compute intersection over union
        IOU = Intersection / Union

        # Compute which events from y have maximal IOU with x
        IOUx = np.max(IOU, axis=1, keepdims=True)

        # Compute which events from x have maximal IOU with y
        IOUy = np.max(IOU, axis=0, keepdims=True)

        # Optional outputs
        return IOU, IOUx, IOUy
        
    elif len(x)==0:
        
        print('x is empty. Cant perform IoU')
        return np.array([]), np.array([]), np.zeros((y.shape[0], 1))
        
    elif len(y)==0:
        
        print('y is empty. Cant perform IoU')
        return np.array([]), np.zeros((1, x.shape[0])), np.array([])
    
def plot_all_events(t_events, lfp, sf, win=0.100, title='', savefig=''):
    '''
    plot_all_events(t_events, lfp, sf, win=0.100, title='', savefig='')
    
    Mandatory inputs:
    -----------------
        events (numpy array):
            Array of size (#events, 2) with all times of events
        lfp (numpy array):
            formated lfp with all channels
        sf (int): 
            sampling frequency of the 'lfp' variable

    Optional inputs:
    ----------------
        win (float): 
            window size at each side of the center of the ripple
        title (string):
            if provided, displays this title
        savefig (string):
            if provided, saves the image in the savefig directory.
            It has to be the full name: e.g. images/session1_events.png
        
    '''
    # find center of events
    bounds = t_events.copy()
    t_events = (t_events[:, 1] + t_events[:, 0]) / 2
    bounds = bounds[:, 1] - t_events # convert to time relative to center
    # Convert to indexes
    id_events = (t_events*sf).astype(int)
    # Make window array
    ids_win = np.arange(-win*sf , win*sf+1).astype(int)

    # Plot curated events
    n_cols = int(np.sqrt(len(t_events)))*1.5
    plt.figure(figsize=(18,12))
    dx, dy = 0, 0
    list_events = []
    for ii,id_event in enumerate(id_events):
        event = lfp[id_event+ids_win,:]
        x = dx + np.linspace(.05,.95,len(event))
        y = dy+(event/3-np.arange(lfp.shape[1]))/lfp.shape[1]*0.8
        swr_start = np.max((0,len(event)/2-bounds[ii]*sf)).astype(int)
        swr_end = np.min((len(event), len(event)/2+bounds[ii]*sf)).astype(int)
        # plot first the data within the window(gray lines), while leaving the true swr range blank
        plt.plot(x[:swr_start], y[:swr_start], 
                 linewidth=0.7, color="gray", alpha=0.6)
        plt.plot(x[swr_end:], y[swr_end:], 
                 linewidth=0.7, color="gray", alpha=0.6)
        # overlay the bounds of events as colored, thicker, alpha=1 lines
        plt.plot(x[swr_start:swr_end], y[swr_start:swr_end], 
                 linewidth=0.8, color=np.random.rand(3), alpha=1)
        dx = dx+1
        if dx >= n_cols:
            dx = 0
            dy = dy-1
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    # Title and save
    if len(title) > 0:
        plt.title(title)
    plt.tight_layout()
    if len(savefig) > 0:
        plt.savefig(savefig)
    plt.show()

