import numpy as np

import globals
import env
import agents
import causality
from prettytable import PrettyTable

All = None  # measure_agent


def rollout_once(env, agents, measured_t, measured_agent):
    env.reset()
    r = None
    obs = None  # no obs needs to be used here
    a_s = np.zeros((globals.N_AGENTS, globals.EPS_LEN))
    r_s = np.zeros((globals.N_AGENTS, globals.EPS_LEN))

    for t_idx in range(globals.EPS_LEN):
        a = agents.act(t_idx, r, obs, measured_t == t_idx, measured_agent)
        obs, r = env.step(a)
        a_s[:, t_idx] = a
        r_s[:, t_idx] = r
    return a_s, r_s


env = env.toy_env(globals.N_AGENTS,
                  globals.EPS_LEN,
                  globals.EPS_STATIC,
                  globals.IS_RANDOM)

agents = agents.agents(globals.N_AGENTS)

print("Created env and agents")
print("agent num: {}".format(globals.N_AGENTS))
print("episode length: {}".format(globals.EPS_LEN))
print("sample num: {}".format(globals.SAMPLE_NUM))
total = globals.SAMPLE_NUM * (globals.N_AGENTS * globals.EPS_LEN) ** 2
d_a = np.arange(total, dtype=np.int64).reshape(globals.N_AGENTS,
                                               globals.EPS_LEN,
                                               globals.SAMPLE_NUM,
                                               globals.N_AGENTS,
                                               globals.EPS_LEN)
d_r = np.empty_like(d_a)

# acquire samples

print("Gathering samples...")
for measured_agent in range(globals.N_AGENTS):
    for measured_t in range(globals.EPS_LEN):
        for cnt in range(globals.SAMPLE_NUM):
            ep_a, ep_r = rollout_once(env, agents, measured_t, measured_agent)
            d_a[measured_agent][measured_t][cnt] = np.array(ep_a)
            d_r[measured_agent][measured_t][cnt] = np.array(ep_r)
        print("Completed for time step {}".format(measured_t))
    print("Completed for agent {}".format(measured_agent))
print("finished")

# action samples: d_a[agent_num i][eps_len t][sample_num][agent_num][eps_num]
total = globals.N_AGENTS * globals.N_AGENTS * globals.EPS_LEN
weighted_ci = np.arange(total, dtype=np.int64).reshape(globals.N_AGENTS, globals.N_AGENTS, globals.EPS_LEN)
normalized_weighted_ci = np.arange(total, dtype=np.int64).reshape(globals.N_AGENTS, globals.N_AGENTS, globals.EPS_LEN)
ci = np.arange(total, dtype=np.int64).reshape(globals.N_AGENTS, globals.N_AGENTS, globals.EPS_LEN)

# compute causal influence
print("Computing causal influence ...")
for alice in range(globals.N_AGENTS):
    for t in range(globals.EPS_LEN):
        for bob in range(globals.N_AGENTS):
            # measure the causal influence of alice's actions to bob's rewards at time t
            ci[alice][bob][t] = causality.measure_causality((d_a, d_r), alice, bob, t, weighted=False)
            weighted_ci[alice][bob][t] = causality.measure_causality((d_a, d_r), alice, bob, t, weighted=True,
                                                                     normalized_w=False)
            normalized_weighted_ci[alice][bob][t] = causality.measure_causality((d_a, d_r), alice, bob, t, weighted=True,
                                                                                normalized_w=True)
print("finished")

# show expected vs computed causality
expected = env.get_inf_table()  # [agent_num][agent_num][eps_len]
print("Printing all result")


def print_table(t_exp, t_ci, t_w_ci, t_nw_ci, agent_num):
    heading = ['expected/ci/w_ci/nw_ci']
    for i in range(agent_num):
        heading.append('{}'.format(i+1))
    t = PrettyTable(heading)
    for alice in range(agent_num):
        r = [''.format(alice+1)]
        for bob in range(agent_num):
            r.append('{}/ {:.2f}/ {:.2f}/ {:.2f}'.format(t_exp[alice][bob], t_ci[alice][bob], t_w_ci[alice][bob], t_nw_ci[alice][bob]))
        t.add_row(r)
    print(t)


for t in range(globals.EPS_LEN):
    print("at time {}".format(t))
    print_table(expected[:, :, t], ci[:, :, t], weighted_ci[:, :, t], normalized_weighted_ci[:, :, t], globals.N_AGENTS)
print("finished")


