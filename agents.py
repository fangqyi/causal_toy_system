import globals
import random

div_len = globals.SIG_LEN / globals.MAX_NUM_INF
all = None

class agents:
    def __init__(self, n_agents, is_lr_random=True, lr_file=None):
        self.n_agents = n_agents
        self.lr_rand = is_lr_random
        if not is_lr_random and lr_file is None:
            raise ValueError("Must input lr_file")
        if is_lr_random and lr_file is not None:
            raise ValueError("Conflict: is_lr_random and lr_file")
        self.agents_locs = []
        self._gen_lr_rand()

    def act(self, t, r, obs=None, uniform=False, uniform_agent=None):
        def _sel_fill_rand_and_convert(s, loc_s):
            # if idx in loc_s, keep; otherwise, replace it with a random bit
            s = "{0:b}".format(s)  # from int to bin str
            s = [s[idx] if idx in loc_s else random.choice(['0', '1']) for idx in range(len(s))]
            return int("".join(s), 2)  # from bin str to int

        a_s = []
        for i in range(self.n_agents):
            if t == 0 or (uniform and (uniform_agent is all or uniform_agent == i)):
                # act uniformly in the initial step or due to sampling need
                a = random.randint(0, 2 ** div_len - 1)
            else:
                a = _sel_fill_rand_and_convert(r[i], self.agents_locs[i])
            a_s.append(a)
        return a_s

    # generate the numbers and locations of bits in agent actions that are determined by reward signals
    def _gen_lr_rand(self, fixed_lrs=None):
        if fixed_lrs is None:
            lrs = [random.randint(0, div_len) for _ in range(self.n_agents)]
        else:
            lrs = [fixed_lrs] * self.n_agents
        self.agents_locs = [random.choices(list(range(globals.SIG_LEN)), k=lr) for lr in lrs]

    def _gen_lr_from_file(self):
        pass
