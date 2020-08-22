import math
import random
import subprocess
import numpy as np
from scipy.stats import zscore
from sklearn.cluster import MiniBatchKMeans


lon = np.linspace(-179.95, 179.95, 3600)
lat = np.linspace(-59.95, 59.95, 1200)
average_prec = np.transpose(np.loadtxt('data/average_prec.txt'))[300:1500]
percentile_25 = np.loadtxt('data/extreme_prec_50.txt')[300:1500]
percentile_50 = np.loadtxt('data/extreme_prec_75.txt')[300:1500]
percentile_75 = np.loadtxt('data/extreme_prec_95.txt')[300:1500]
percentile_95 = np.loadtxt('data/extreme_prec_95.txt')[300:1500]
topography = np.transpose(np.loadtxt('data/Topography.txt'))[300:1500]

lon_data, lat_data = np.meshgrid(lon, lat)
assert all(x.shape == (1200, 3600) for x in (lon_data, lat_data, average_prec, percentile_25, percentile_50,
                                             percentile_75, percentile_95, topography))
clust_data = np.concatenate([
    zscore(lon_data.reshape(-1, 1)),
    zscore(lat_data.reshape(-1, 1)),
    zscore(average_prec.reshape(-1, 1)),
    zscore(percentile_25.reshape(-1, 1)),
    zscore(percentile_50.reshape(-1, 1)),
    zscore(percentile_75.reshape(-1, 1)),
    zscore(percentile_95.reshape(-1, 1)),
    zscore(topography.reshape(-1, 1))
], axis=1)


class BooleanGene:
    def __init__(self, name, value=None):
        self.name = name
        self.value = random.choice((True, False)) if value is None else value

    def mutated(self):
        return BooleanGene(self.name, value=not self.value)


class NumericGene:
    def __init__(self, name, low, high, value=None, dtype=None):
        self.name = name
        self.dtype = dtype
        if dtype == int:
            self.low = int(low)
            self.high = int(high)
            self.value = random.randint(self.low, self.high) if value is None else value
        elif dtype is None or dtype == float:
            self.low = float(low)
            self.high = float(high)
            self.value = random.uniform(self.low, self.high) if value is None else value
        else:
            raise ValueError(f'dtype must be int or float but has value {dtype}')

    def perturbed(self, sigma):
        if self.dtype == int:
            val = min(max(self.low, round(random.gauss(self.value, sigma))), self.high)
            return NumericGene(self.name, self.low, self.high, value=val, dtype=int)
        else:
            val = min(max(self.low, random.gauss(self.value, sigma)), self.high)
            return NumericGene(self.name, self.low, self.high, value=val, dtype=float)

    def mutated(self):
        if self.dtype == int:
            return NumericGene(self.name, self.low, self.high, value=random.randint(self.low, self.high), dtype=int)
        else:
            return NumericGene(self.name, self.low, self.high, value=random.uniform(self.low, self.high), dtype=float)


class Genome:
    CATEGORICAL_MUTATE_CHANCE = 0.01
    NUMERICAL_PERTURB_CHANCE = 0.19
    PERTURB_SIGMA_RATIO = 0.1
    NUMERICAL_MUTATE_CHANCE = 0.01

    def __init__(self, genes: dict):
        self.genes = genes
        g = self.genes
        self.clust_data_mask = (True, True, g['avg prec use'], g['25 use'], g['50 use'],
                                g['75 use'], g['95 use'], g['topo use'])
        self.clust_data_wts = np.array((g['coord wt'], g['coord wt'], g['avg prec wt'], g['25 wt'], g['50 wt'],
                                        g['75 wt'], g['95 wt'], g['topo wt']))[self.clust_data_mask]

    @staticmethod
    def rand_genome():
        genes = [
            NumericGene('avg size', 10, 50, dtype=int),
            BooleanGene('adjust size'),
            NumericGene('high lvl exponent', 1, 5, dtype=int),
            NumericGene('coord wt', 0, 1, dtype=float),
            BooleanGene('avg prec use'),
            NumericGene('avg prec wt', 0, 1, dtype=float),
            BooleanGene('25 use'),
            NumericGene('25 wt', 0, 1, dtype=float),
            BooleanGene('50 use'),
            NumericGene('50 wt', 0, 1, dtype=float),
            BooleanGene('75 use'),
            NumericGene('75 wt', 0, 1, dtype=float),
            BooleanGene('95 use'),
            NumericGene('95 wt', 0, 1, dtype=float),
            BooleanGene('topo use'),
            NumericGene('topo weight', 0, 1, dtype=float)
        ]
        return Genome(genes={g.name: g for g in genes})

    def mutated(self):
        new_genes = {}
        for gene in self.genes.values():
            if isinstance(gene, BooleanGene):
                if random.random() < self.CATEGORICAL_MUTATE_CHANCE:
                    new_genes[gene.name] = gene.mutated()
            else:
                x = random.random()
                if x < self.NUMERICAL_PERTURB_CHANCE:
                    new_genes[gene.name] = gene.perturbed(sigma=(gene.high - gene.low) * self.PERTURB_SIGMA_RATIO)
                elif x < self.NUMERICAL_PERTURB_CHANCE + self.NUMERICAL_MUTATE_CHANCE:
                    new_genes[gene.name] = gene.mutated()
        return Genome(genes=new_genes)

    def crossed(self, other):
        new_genes = {}
        for gene in self.genes.values():
            new_genes[gene.name] = random.choice((self.genes[gene.name], other.genes[gene.name]))
        return Genome(genes=new_genes)

    def fitness(self):
        wtd_data = clust_data[:, self.clust_data_mask] * self.clust_data_wts
        high_labels = MiniBatchKMeans(
            n_clusters=round(math.pow(10, self.genes['high lvl exponent']))
        ).fit_predict(wtd_data)
        new_labels = np.zeros_like(high_labels)

        for high_label in np.unique(high_labels):
            mask = high_labels == high_label
            sz = np.count_nonzero(mask)
            clust_subset_data = clust_data[mask]
            if self.genes['adjust size']:
                k = round(clust_data.shape[0] / self.genes['avg size'])
            else:
                k = round(sz / self.genes['avg size'])
            if sz == 0:
                continue
            elif sz < k:
                subset_labels = np.zeros(sz)
            else:
                subset_labels = MiniBatchKMeans(n_clusters=k).fit_predict(clust_subset_data)
            subset_labels += np.max(new_labels) + 1
            new_labels[mask] = subset_labels

        with open('C:\\Users\\Jerry Xiong\\Desktop\\NASA\\ga_clusters.txt', 'w') as f:
            for row in new_labels.reshape(1200, 3600):
                f.write(' '.join(str(x) for x in row))

        cmd = ['Rscript', 'C:\\Users\\Jerry Xiong\\Desktop\\NASA\\My R Code\\SampleNCR.R']
        # -50 was chosen so that 0.02 better avg NCR means that chance to be chosen is increased by e (2.718...)
        return -50 * float(subprocess.check_output(cmd, universal_newlines=True))


POP_SIZE = 100
NUM_FROM_CROSS = 75

population = [Genome.rand_genome() for _ in range(POP_SIZE)]
fitnesses = np.array([genome.fitness() for genome in population])
softmax_weights = np.exp(fitnesses) / np.sum(np.exp(fitnesses))
for generation in range(25):
    new_population = []
    for _ in range(NUM_FROM_CROSS):
        parents = random.choices(population, weights=softmax_weights, k=2)
        new_population.append(parents[0].crossed(parents[1]))
    for _ in range(POP_SIZE - NUM_FROM_CROSS):
        original = random.choices(population, weights=softmax_weights)[0]
        new_population.append(original.mutated())
