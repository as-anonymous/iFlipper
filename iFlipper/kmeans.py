from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import copy

from sklearn.cluster import KMeans 
from utils import measure_violations

def kMeans(data, label, m, edge, w_edge):
    """         
        Applies k-means clustering, and flips labels so that the examples within the same cluster have the same label.
        If the output does not satisfy the violations limit m, it adjusts the number of clusters k.

        Args: 
            data: Features of the data
            label: Labels of the data
            m: The violations limit
            edge: Indices of similar pairs
            w_edge: Similarity values of similar pairs
            
        Return:
            flipped_label: Flipped labels for a given m
    """

    k_high, k_low = min(label.shape[0], 10000), 1 # set upper limit

    init_violations = measure_violations(label, edge, w_edge)
    best_flips, best_violations = label.shape[0], init_violations
    best_flips_fail, best_violations_fail = label.shape[0], init_violations

    while (k_high - k_low > 1):
        k = int((k_high + k_low) / 2)

        kmeans_label = copy.deepcopy(label)
        kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
        for i in range(k):
            idx = (kmeans.labels_ == i)
            labels, num_labels = np.unique(label[idx], return_counts=True)

            if len(num_labels) > 0:
                max_idx = np.argmax(num_labels)
                major_label = int(labels[max_idx])
                kmeans_label[idx] = major_label

        num_violations = measure_violations(kmeans_label, edge, w_edge)
        num_flips = np.sum(label != kmeans_label)

        if num_violations <= m: 
            if num_flips < best_flips:
                best_flips = num_flips
                best_violations = num_violations
                best_flipped_label = copy.deepcopy(kmeans_label)
            k_low = k
        else:
            if (best_violations > m) and (num_violations < best_violations_fail):
                best_flips_fail = num_flips
                best_violations_fail = num_violations
                best_flipped_label_fail = copy.deepcopy(kmeans_label)
            k_high = k

        if best_violations <= m:
            flipped_label = copy.deepcopy(best_flipped_label)
        else:
            flipped_label = copy.deepcopy(best_flipped_label_fail)

    return flipped_label