
class Vocab():
    def __init__(self, hp, special_labels=None):
        self.idxToLabel = {}
        self.labelToIdx = {}
        self.hp = hp
        self.unk_freq = hp.unk_freq

        if special_labels is not None:
            for label in special_labels:
                self.add(label, self.unk_freq+1)

    def add(self, label, freq):
        if freq > self.unk_freq:
            idx = len(self.labelToIdx)
            self.idxToLabel[idx] = label
            self.labelToIdx[label] = idx

    def getIdx(self, label):
        if label in self.labelToIdx:
            return self.labelToIdx[label]
        else:
            return self.labelToIdx[self.hp.UNK_WORD]

    def getLabel(self, idx):
        if idx in self.idxToLabel:
            return self.idxToLabel[idx]
        else:
            return self.idxToLabel[self.hp.UNK]

    def __len__(self):
        return len(self.idxToLabel)
