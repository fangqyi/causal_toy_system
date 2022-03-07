import numpy as np

q = 2
eps = 1e1  # TODO: DEBUG

# measure the information-theoretic granger causality (on alice's action to bob's reward at time t)
def measure_causality(data, alice, bob, t, weighted=True, normalized_w=False):
    a_data, r_data = data
    # gather data
    a_t = a_data[alice][t][:, alice, t]  # a_t
    r_t = r_data[alice][t][:, bob, t]  # r_t
    r_prev_t = r_data[alice][t][:, bob, t-1]  # r_t-1

    # measure causality GC
    # GC = I(r_prev_t, a_t; r_t) - I(r_prev_t; r_t)
    #    = I(a_t; r_t) - I(a_t; r_prev_t; r_t)
    #    = -H(r_prev_t) + H(r_prev_t, r_t) + H(r_prev_t, a_t) - H(r_prev_t, r_t, a_t)
    #    = C(r_prev_t) - C(r_prev_t, r_t) - C(r_prev_t, a_t) + C(r_prev_t, r_t, a_t)

    if weighted:
        w = r_t
    else:
        w = None
    c_1 = compute_c([r_prev_t], weighted, w, normalized_w)
    c_2 = compute_c([r_prev_t, r_t], weighted, w, normalized_w)
    c_3 = compute_c([r_prev_t, a_t], weighted, w, normalized_w)
    c_4 = compute_c([r_prev_t, r_t, a_t], weighted, w, normalized_w)

    return c_1 - c_2 - c_3 + c_4


# measure correlation integrals (can be weighted such as reward)
def compute_c(v_list, weighted=True, w=None, normalized_w = False):
    def heavside(x):
        return int(x>0)

    def normalize(v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    n = v_list[0].shape[0]  # sample num
    v = np.dstack(v_list).reshape((n, len(v_list)))  # size [sample_num][v_list]

    if weighted and normalized_w:
        w = normalize(w)

    c = .0
    for i in range(n):
        s = 0
        for j in range(n):
            if i == j:
                pass
            s += heavside(np.linalg.norm(v[i]-v[j])-eps)
        if not weighted:
            c += s**q
        else:
            c += w[i]

    c /= n*(n-1)**(q-1)
    return c





