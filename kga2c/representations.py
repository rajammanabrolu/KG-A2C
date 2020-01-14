import networkx as nx
import numpy as np
import openie
from fuzzywuzzy import fuzz
from jericho.util import clean


class StateAction(object):

    def __init__(self, spm, vocab, vocab_rev, tsv_file, max_word_len):
        self.graph_state = nx.DiGraph()
        self.max_word_len = max_word_len
        self.graph_state_rep = []
        self.visible_state = ""
        self.drqa_input = ""
        self.vis_pruned_actions = []
        self.pruned_actions_rep = []
        self.sp = spm
        self.vocab_act = vocab
        self.vocab_act_rev = vocab_rev
        self.vocab_kge = self.load_vocab_kge(tsv_file)
        self.adj_matrix = np.zeros((len(self.vocab_kge['entity']), len(self.vocab_kge['entity'])))
        self.room = ""

    def visualize(self):
        # import matplotlib.pyplot as plt
        pos = nx.spring_layout(self.graph_state)
        edge_labels = {e: self.graph_state.edges[e]['rel'] for e in self.graph_state.edges}
        print(edge_labels)
        nx.draw_networkx_edge_labels(self.graph_state, pos, edge_labels)
        nx.draw(self.graph_state, pos=pos, with_labels=True, node_size=200, font_size=10)
        #plt.show()

    def load_vocab_kge(self, tsv_file):
        ent = {}
        with open(tsv_file, 'r') as f:
            for line in f:
                e, eid = line.split('\t')
                ent[e.strip()] = int(eid.strip())
        rel = {}
        with open(tsv_file, 'r') as f:
            for line in f:
                r, rid = line.split('\t')
                rel[r.strip()] = int(rid.strip())

        return {'entity': ent, 'relation': rel}

    def update_state(self, visible_state, inventory_state, objs, prev_action=None, cache=None):

        prev_room = self.room

        graph_copy = self.graph_state.copy()
        con_cs = [graph_copy.subgraph(c) for c in nx.weakly_connected_components(graph_copy)]

        prev_room_subgraph = None
        prev_you_subgraph = None

        for con_c in con_cs:
            for node in con_c.nodes:
                node = set(str(node).split())
                if set(prev_room.split()).issubset(node):
                    prev_room_subgraph = nx.induced_subgraph(graph_copy, con_c.nodes)

        for edge in self.graph_state.edges:
            if 'you' in edge[0]:
                graph_copy.remove_edge(*edge)

        self.graph_state = graph_copy

        visible_state = visible_state.split('\n')
        room = visible_state[0]
        visible_state = clean(' '.join(visible_state[1:]))

        dirs = ['north', 'south', 'east', 'west', 'southeast', 'southwest', 'northeast', 'northwest', 'up', 'down']

        self.visible_state = str(visible_state)
        rules = []

        if cache is None:
            sents = openie.call_stanford_openie(self.visible_state)['sentences']
        else:
            sents = cache

        if sents == "":
            return []

        in_aliases = ['are in', 'are facing', 'are standing', 'are behind', 'are above', 'are below', 'are in front']

        in_rl = []
        in_flag = False

        for i, ov in enumerate(sents):
            sent = ' '.join([a['word'] for a in ov['tokens']])
            triple = ov['openie']
            for d in dirs:
                if d in sent and i != 0:
                    rules.append((room, 'has', 'exit to ' + d))

            for tr in triple:
                h, r, t = tr['subject'].lower(), tr['relation'].lower(), tr['object'].lower()

                if h == 'you':
                    for rp in in_aliases:
                        if fuzz.token_set_ratio(r, rp) > 80:
                            r = "in"
                            in_rl.append((h, r, t))
                            in_flag = True
                            break

                if h == 'it':
                    break
                if not in_flag:
                    rules.append((h, r, t))

        if in_flag:
            cur_t = in_rl[0]
            for h, r, t in in_rl:
                if set(cur_t[2].split()).issubset(set(t.split())):
                    cur_t = h, r, t
            rules.append(cur_t)
            room = cur_t[2]

        try:
            items = inventory_state.split(':')[1].split('\n')[1:]
            for item in items:
                rules.append(('you', 'have', str(' ' .join(item.split()[1:]))))
        except:
            pass

        if prev_action is not None:
            for d in dirs:
                if d in prev_action and self.room != "":
                    rules.append((prev_room, d + ' of', room))
                    if prev_room_subgraph is not None:
                        for ed in prev_room_subgraph.edges:
                            rules.append((ed[0], prev_room_subgraph[ed]['rel'], ed[1]))
                    break

        for o in objs:
            #if o != 'all':
            rules.append((str(o), 'in', room))

        add_rules = rules

        for rule in add_rules:
            u = '_'.join(str(rule[0]).split())
            v = '_'.join(str(rule[2]).split())
            if u in self.vocab_kge['entity'].keys() and v in self.vocab_kge['entity'].keys():
                if u != 'it' and v != 'it':
                    self.graph_state.add_edge(rule[0], rule[2], rel=rule[1])

        return add_rules, sents

    def get_state_rep_kge(self):
        ret = []
        self.adj_matrix = np.zeros((len(self.vocab_kge['entity']), len(self.vocab_kge['entity'])))

        for u, v in self.graph_state.edges:
            u = '_'.join(str(u).split())
            v = '_'.join(str(v).split())

            if u not in self.vocab_kge['entity'].keys() or v not in self.vocab_kge['entity'].keys():
                break

            u_idx = self.vocab_kge['entity'][u]
            v_idx = self.vocab_kge['entity'][v]
            self.adj_matrix[u_idx][v_idx] = 1

            ret.append(self.vocab_kge['entity'][u])
            ret.append(self.vocab_kge['entity'][v])

        return list(set(ret))

    def get_state_kge(self):
        ret = []
        self.adj_matrix = np.zeros((len(self.vocab_kge['entity']), len(self.vocab_kge['entity'])))

        for u, v in self.graph_state.edges:
            u = '_'.join(str(u).split())
            v = '_'.join(str(v).split())

            if u not in self.vocab_kge['entity'].keys() or v not in self.vocab_kge['entity'].keys():
                break

            u_idx = self.vocab_kge['entity'][u]
            v_idx = self.vocab_kge['entity'][v]
            self.adj_matrix[u_idx][v_idx] = 1

            ret.append(u)
            ret.append(v)

        return list(set(ret))

    def get_obs_rep(self, *args):
        ret = [self.get_visible_state_rep_drqa(ob) for ob in args]
        return pad_sequences(ret, maxlen=300)

    def get_visible_state_rep_drqa(self, state_description):
        remove = ['=', '-', '\'', ':', '[', ']', 'eos', 'EOS', 'SOS', 'UNK', 'unk', 'sos', '<', '>']

        for rm in remove:
            state_description = state_description.replace(rm, '')

        return self.sp.encode_as_ids(state_description)

    def get_action_rep_drqa(self, action):

        action_desc_num = 20 * [0]
        action = str(action)

        for i, token in enumerate(action.split()[:20]):
            short_tok = token[:self.max_word_len]
            action_desc_num[i] = self.vocab_act_rev[short_tok] if short_tok in self.vocab_act_rev else 0

        return action_desc_num

    def step(self, visible_state, inventory_state, objs, prev_action=None, cache=None, gat=True):
        ret, ret_cache = self.update_state(visible_state, inventory_state, objs, prev_action, cache)

        self.pruned_actions_rep = [self.get_action_rep_drqa(a) for a in self.vis_pruned_actions]

        inter = self.visible_state #+ "The actions are:" + ",".join(self.vis_pruned_actions) + "."
        self.drqa_input = self.get_visible_state_rep_drqa(inter)

        self.graph_state_rep = self.get_state_rep_kge(), self.adj_matrix

        return ret, ret_cache


def pad_sequences(sequences, maxlen=None, dtype='int32', value=0.):
    '''
    Partially borrowed from Keras
    # Arguments
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        value: float, value to pad the sequences to the desired value.
    # Returns
        x: numpy array with dimensions (number_of_sequences, maxlen)
    '''
    lengths = [len(s) for s in sequences]
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)
    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break
    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        # pre truncating
        trunc = s[-maxlen:]
        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))
        # post padding
        x[idx, :len(trunc)] = trunc
    return x


