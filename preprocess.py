# import tensorflow as tf
from nltk.tokenize.moses import MosesTokenizer

cap_trn = "../data/train.txt"
cap_vld = "../data/valid.txt"

tk = MosesTokenizer()


def trn_cap(ifname):
    with open(ifname, 'r') as f:
        caps = []
        img_caps = []
        for line in f:
            line = line.rstrip('\n')
            try:
                int(line)
                if img_caps:
                    caps.append(img_caps)
            except ValueError:
                img_caps.append(tk.tokenize(line))


if __name__ == '__main__':
    trn_cap(cap_trn)
