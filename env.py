# control the reward generation process
import globals
import random
import numpy as np

class toy_env:
    def __init__(self, n_agents, eps_len, eps_static=True, is_random=True, rel_file=None):
        self.n_agents = n_agents
        self.eps_len = eps_len
        self.t = 0  # time step
        if not is_random and rel_file is None:
            raise ValueError("Must input rel_file")
        if is_random and rel_file is not None:
            raise ValueError("Conflict: is_random and rel_file")
        self.agents_rel = []
        if is_random:
            self._gen_rand_game(eps_static)
        else:
            self._gen_game_from_file(rel_file)

    def reset(self):
        self.t = 0

    def step(self, actions):
        if len(actions) != self.n_agents:
            raise ValueError("Not enough actions")
        r_s = []
        for i in range(self.n_agents):
            r_s.append(self._compute_r(actions, self.agents_rel[self.t][i]))
        obs = []
        self.t += 1
        return obs, r_s

    def get_inf_table(self):
        # agent_rel [eps_len][agent_num][num_inf]
        # convert to adjacent table
        shape = (globals.N_AGENTS, globals.N_AGENTS, globals.EPS_LEN)
        infl_t = np.zeros(shape)
        for alice in range(globals.N_AGENTS):
            for t in range(globals.EPS_LEN):
                for bob in self.agents_rel[t][alice]:
                    if bob != globals.N_AGENTS:
                        infl_t[bob][alice][t] += 1
        return infl_t

    def _gen_rand_game(self, is_time_static):
        # [0, 1, ..., agent_num] (0 represents env)
        def __gen_agents_rel(n_agents, num_inf):
            rel_s = []
            for i in range(self.n_agents):
                rel = random.choices(list(range(n_agents + 1)), k=num_inf)
                rel_s.append(rel)
            return rel_s

        if is_time_static:
            rel_s = __gen_agents_rel(self.n_agents, globals.MAX_NUM_INF)
            for t in range(self.eps_len):
                self.agents_rel.append(rel_s)
        else:
            for t in range(self.eps_len):
                rel_s = __gen_agents_rel(self.n_agents, globals.MAX_NUM_INF)
                self.agents_rel.append(rel_s)

    def _gen_game_from_file(self, rel_file):
        pass

    def _compute_r(self, actions, infl):
        div_len = globals.SIG_LEN / globals.MAX_NUM_INF
        r = 0
        for src_id in range(len(infl)):
            src = infl[src_id]
            if src == globals.N_AGENTS:  # from env
                r += int(random.randint(0, 2 ** div_len - 1)) << int(div_len * src_id)
            else:  # determined by agents
                r += int(actions[src]) & (int(2 ** div_len - 1) << int(div_len * src_id))
        return r
