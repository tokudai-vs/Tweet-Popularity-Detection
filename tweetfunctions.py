import numpy as np
from scipy.stats import chi2
from math import log, exp
import sys


def memory_pdf(t, theta=0.2314843, cutoff=300, c=0.0006265725):
    if t < cutoff:
        return c
    else:
        return c * exp((log(t) - log(cutoff)) * ((-1) * (1 + theta)))


def memory_ccdf(t, theta=0.2314843, cutoff=300, c=0.0006265725):
    for _ in range(len(t)):
        if t[_] < 0:
            t[_] = 0
        else:
            pass
        index1 = list(np.where(t <= cutoff))
        index2 = list(np.where(t > cutoff))
        ccdf = [0] * len(t)
        for i in index1:
            ccdf[index1] = 1 - c * t[index1]
        for i in index2:
            ccdf[index2] = c * cutoff ** (1 + theta) / theta * t[index2] ** (-theta)
        return ccdf


def linear_kernel(t1, t2, p_time, slope, c=0.0006265725):
    return c * (t2 - p_time * slope * t2 + slope * t2 ** 2 / 2) - c * (
        t1 - p_time * slope * t1 + slope * t1 ** 2 / 2
    )


def power_kernel(
    t1, t2, p_time, share_time, slope, theta=0.2314843, cutoff=300, c=0.0006265725
):
    return c * cutoff ** (1 + theta) * (t2 - share_time) ** ((-1) * theta) * (
        share_time * slope
        - theta
        + (theta - 1) * p_time * slope
        - theta * slope * t2
        + 1
    ) / ((theta - 1) * theta) - c * cutoff ** (1 + theta) * (t1 - share_time) ** (
        (-1) * theta
    ) * (
        share_time * slope
        - theta
        + (theta - 1) * p_time * slope
        - theta * slope * t1
        + 1
    ) / (
        (theta - 1) * theta
    )


def integral_memory_kernel(
    p_time, share_time, slope, window, theta=0.2314843, cutoff=300, c=0.0006265725
):
    index1 = np.where(p_time <= share_time)
    index2 = np.where(p_time > share_time and p_time <= share_time + window)
    index3 = np.where(p_time > share_time + cutoff and p_time <= share_time + window)
    index4 = np.where(
        p_time > share_time + window and p_time <= share_time + window + cutoff
    )
    index5 = np.where(p_time > share_time + window + cutoff)

    # check this part
    integral = [0] * len(share_time)
    for i in index1:
        integral[i] = 0

    stime = []
    for i in index2:
        stime.append(share_time[i])
    index2_val = linear_kernel(stime, p_time, p_time, slope)
    for i in index2:
        integral[i] = index2_val

    stime = []
    for i in index3:
        stime.append(share_time[i])
    index3_val = linear_kernel(stime, stime + cutoff, p_time, slope) + power_kernel(
        stime + cutoff, p_time, p_time, stime, slope
    )
    for i in index3:
        integral[i] = index3_val

    stime = []
    for i in index4:
        stime.append(share_time[i])
    index4_val = linear_kernel(
        p_time - window, stime + cutoff, p_time, slope
    ) + power_kernel(stime + cutoff, p_time, p_time, stime, slope)
    for i in index4:
        integral[i] = index4_val

    stime = []
    for i in index5:
        stime.append(share_time[i])
    index5_val = power_kernel(p_time - window, p_time, p_time, stime, slope)
    for i in index5:
        integral[i] = index5_val

    return integral


# Parameters
# share_time: observed resharing times, sorted, share_time[0] = 0
# degree: observed node degrees
# p_time: equally spaced vector of time to estimate the infectiousness, p_time[0] = 0
# max_window: maximum span of locally weight kernel
# min_window: minimum span of locally weight kernel
# min_count: minimum number of resharings included in the window
#
# Parameters type
# share_time: array
# degree: integer
# p_time: array
# max_window: integer
# min_windows: integer
# min_count: integer


def get_infectiousness(
    share_time, degree, p_time, max_window=2 * 60 * 60, min_window=300, min_count=5
):
    share_time = np.array(share_time)
    ix = sorted(share_time)  # sort share_time

    p_time = np.array(p_time)
    slopes = 1 / (p_time / 2)
    for _ in range(len(slopes)):
        if slopes[_] < 1 / max_window:
            slopes[_] = 1 / max_window
        elif slopes[_] > 1 / min_window:
            slopes[_] = 1 / min_window
        else:
            pass

    windows = p_time / 2
    for _ in range(len(windows)):
        if windows[_] > max_window:
            windows[_] = max_window
        elif windows[_] < min_window:
            windows[_] = min_window
        else:
            pass

    for j in range(len(p_time)):
        ind = np.where(
            share_time >= (p_time[j] - windows[j]) and share_time < p_time[j]
        )
        # index of share_time >= p_time[j] - windows[j] and share_time < p_time[j]
        if len(ind) < min_count:
            # index of share_time < p_time[j]
            ind2 = np.where(share_time < p_time[j])
            lcv = len(ind2)
            ind = ind2[max((lcv - min_count), 1) : lcv]
            slopes[j] = 1 / (p_time[j] - share_time[ind[0]])
            windows[j] = p_time[j] - share_time[ind[0]]

    M_I = np.zeros(len(share_time), len(p_time))
    for j in range(len(p_time)):
        M_I[:, j] = degree * integral_memory_kernel(
            p_time[j], share_time, slopes[j], windows[j]
        )

    infectiousness_seq = [0] * len(p_time)
    p_low_seq = [0] * len(p_time)
    p_up_seq = [0] * len(p_time)
    share_time.pop()  # remove original tweet, first in this array

    for j in range(len(p_time)):
        share_time_tri = list(
            share_time[i]
            for i in [
                np.where(
                    share_time >= (p_time[j] - windows[j]) and share_time < p_time[j]
                )
            ]
        )
        rt_count_weighted = np.sum(slopes[j] * (share_time_tri - p_time[j]) + 1)

        I = np.sum(M_I[:, j])
        rt_num = len(share_time_tri)

        if rt_count_weighted == 0:
            pass
        else:
            infectiousness_seq[j] = rt_count_weighted / I
            p_low_seq[j] = (
                infectiousness_seq[j] * chi2.ppf(0.05, 2 * rt_num) / (2 * rt_num)
            )
            p_up_seq = infectiousness_seq[j] * chi2.ppf(0.95, 2 * rt_num) / (2 * rt_num)

    return [infectiousness_seq, p_up_seq, p_low_seq]


def pred_cascade(
    p_time, infectiousness, share_time, degree, n_star=100, features_return=False
):
    if len(n_star) == 1:
        n_star = [n_star] * len(p_time)

    features = np.zeros(len(p_time), 3)

    prediction = np.zeros(len(p_time), 1)

    for i in len(p_time):
        share_time_now = [share_time[x] for x in np.where(share_time <= p_time[i])]
        nf_now = [degree[x] for x in np.where(share_time <= p_time[i])]
        rt0 = np.sum([share_time[x] for x in np.where(share_time <= p_time[i])]) - 1
        rt1 = np.sum(
            nf_now * infectiousness[i] * memory_ccdf(p_time[i] - share_time_now)
        )
        prediction[i] = rt0 + rt1 / (1 - infectiousness[i] * n_star[i])
        features[i, :] = [rt0, rt1, infectiousness[i]]

        if infectiousness[i] > 1 / n_star:
            prediction[i] = sys.maxsize
    features = ["current.rt", "numerator", "infectiousness"]

    if not features_return:
        return prediction
    else:
        return (prediction, features)
