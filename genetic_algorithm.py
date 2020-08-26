"""Tunes recursive k-means clustering parameters using a genetic algorithm.

k-means clustering schemes are represented with a genome. Inside the genome there are genes that represent things
like the weight of a certain clustering variable, average cluster size, or whether a certain variable is used at all.
For each genome, a recursive k-means clustering is performed with parameters based on the genes in the genome. The
"fitness" of the genome represents how low the average NCR from 100 randomly sampled clusters is. This python script
calls a separate R script to calculate the NCR values. Higher fitness is better. Then, the genomes are "crossed" and
"mutated" to create a new population of genomes, where genomes with better fitness are more likely to pass on their
genes. Eventually, after a few generations, an optimal clustering scheme should be created. The idea is to use these
parameters and perform a slower HDBSCAN clustering to see if that will further improve average NCR.
"""


import math
import random
import subprocess
import numpy as np
from scipy.stats import zscore
from sklearn.cluster import MiniBatchKMeans


class BooleanGene:
    def __init__(self, name, value=None):
        self.name = name
        self.value = random.choice((True, False)) if value is None else value

    def mutated(self):
        return BooleanGene(self.name, value=not self.value)

    def copy(self):
        return BooleanGene(self.name, self.value)

    def __str__(self):
        return f'{self.value!s: <5}'
    
    def __repr__(self):
        return f'{self.name}={self.value!s: <5}'


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

    def copy(self):
        return NumericGene(self.name, self.low, self.high, self.value, self.dtype)

    def __str__(self):
        if self.dtype == int:
            return f'{self.value: <3}'
        return f'{self.value:.4f}'

    def __repr__(self):
        if self.dtype == int:
            return f'{self.name}={self.value: <3}'
        return f'{self.name}={self.value:.4f}'


class Genome:
    CATEGORICAL_MUTATE_CHANCE = 0.01
    NUMERICAL_PERTURB_CHANCE = 0.19
    PERTURB_SIGMA_RATIO = 0.1
    NUMERICAL_MUTATE_CHANCE = 0.01

    def __init__(self, genes: dict, fitness=None):
        self.genes = genes
        g = self.genes
        self.clust_data_mask = np.array((True, True, g['avg_prec_use'].value, g['25_use'].value, g['50_use'].value,
                                         g['75_use'].value, g['95_use'].value, g['topo_use'].value))
        self.clust_data_wts = np.array((g['coord_wt'].value, g['coord_wt'].value, g['avg_prec_wt'].value,
                                        g['25_wt'].value, g['50_wt'].value, g['75_wt'].value, g['95_wt'].value,
                                        g['topo_wt'].value))[self.clust_data_mask]
        self.ftns = fitness

    @staticmethod
    def rand_genome():
        genes = [
            NumericGene('avg_size', 50, 150, dtype=int),
            BooleanGene('adjust_size'),
            NumericGene('num_high_pow', 1, 5, dtype=float),
            NumericGene('coord_wt', 0, 1, dtype=float),
            BooleanGene('avg_prec_use'),
            NumericGene('avg_prec_wt', 0, 1, dtype=float),
            BooleanGene('25_use'),
            NumericGene('25_wt', 0, 1, dtype=float),
            BooleanGene('50_use'),
            NumericGene('50_wt', 0, 1, dtype=float),
            BooleanGene('75_use'),
            NumericGene('75_wt', 0, 1, dtype=float),
            BooleanGene('95_use'),
            NumericGene('95_wt', 0, 1, dtype=float),
            BooleanGene('topo_use'),
            NumericGene('topo_wt', 0, 1, dtype=float)
        ]
        return Genome(genes={g.name: g for g in genes})

    def mutated(self):
        new_genes = {}
        for gene in self.genes.values():
            if isinstance(gene, BooleanGene):
                if random.random() < self.CATEGORICAL_MUTATE_CHANCE:
                    new_genes[gene.name] = gene.mutated()
                else:
                    new_genes[gene.name] = gene.copy()
            else:
                x = random.random()
                if x < self.NUMERICAL_PERTURB_CHANCE:
                    new_genes[gene.name] = gene.perturbed(sigma=(gene.high - gene.low) * self.PERTURB_SIGMA_RATIO)
                elif x < self.NUMERICAL_PERTURB_CHANCE + self.NUMERICAL_MUTATE_CHANCE:
                    new_genes[gene.name] = gene.mutated()
                else:
                    new_genes[gene.name] = gene.copy()
        return Genome(genes=new_genes)

    def crossed(self, other):
        new_genes = {}
        for gene in self.genes.values():
            new_genes[gene.name] = random.choice((self.genes[gene.name].copy(), other.genes[gene.name].copy()))
        return Genome(genes=new_genes)

    @property
    def fitness(self):
        if self.ftns is not None:
            print(f'{self!r} has (precomputed) average NCR {self.ftns / -50:.4f}')
            return self.ftns
        wtd_data = clust_data[:, self.clust_data_mask] * self.clust_data_wts
        # print(f'Starting high level clustering with {round(math.pow(10, self.genes["num_high_pow"].value))} centers')
        high_labels = MiniBatchKMeans(
            n_clusters=round(math.pow(10, self.genes['num_high_pow'].value))
        ).fit_predict(wtd_data)
        num_high_clusters = np.unique(high_labels).shape[0]
        new_labels = np.full_like(high_labels, fill_value=-1)

        unique_labels = np.unique(high_labels)
        np.random.default_rng().shuffle(unique_labels)
        for i, high_label in enumerate(unique_labels):
            mask = high_labels == high_label
            sz = np.count_nonzero(mask)
            clust_subset_data = clust_data[mask]
            if self.genes['adjust_size'].value:
                k = round(sz / self.genes['avg_size'].value)
            else:
                k = round(clust_data.shape[0] / num_high_clusters / self.genes['avg_size'].value)
            # print(f'Starting low level clustering {i + 1} / {num_high_clusters} of size {sz} with {k} centers')
            if sz == 0:
                continue
            elif sz < k or k <= 1:
                subset_labels = np.zeros(sz)
            else:
                subset_labels = MiniBatchKMeans(n_clusters=k).fit_predict(clust_subset_data)
            subset_labels += np.max(new_labels) + 1
            new_labels[mask] = subset_labels
            if np.max(new_labels) > 1000 and i >= 4:
                break

        with open('C:\\Users\\Jerry Xiong\\Desktop\\NASA\\ga_clusters.txt', 'w') as f:
            for row in new_labels.reshape(1200, 3600):
                f.write(' '.join(str(x) for x in row) + '\n')

        # -50 was chosen so that 0.02 better avg NCR means that chance to be chosen is increased by e (2.718...)
        fitness = -50 * float(subprocess.check_output(
            'Rscript "C:/Users/Jerry Xiong/Desktop/NASA/My R Code/SampleNCRs.R"',
            stderr=subprocess.DEVNULL,
            universal_newlines=True).strip()[4:])
        print(f'{self!r} has average NCR {fitness / -50:.4f}')
        self.ftns = fitness
        return fitness

    def __str__(self):
        return ' '.join(str(g) for g in self.genes.values())

    def __repr__(self):
        return f'{self.__class__.__name__}({" ".join(repr(g) for g in self.genes.values())})'


POP_SIZE = 30
NUM_FROM_CROSS = 20
NUM_FROM_MUTATION = 8
NUM_RANDOM = 1
NUM_ELITE = 1
assert POP_SIZE == NUM_FROM_CROSS + NUM_FROM_MUTATION + NUM_RANDOM + NUM_ELITE

if __name__ == '__main__':
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

    open('data/best_genomes.txt', 'w').close()

    population = np.array([Genome.rand_genome() for _ in range(POP_SIZE)])
    for generation in range(25):
        print(f'Starting generation {generation}')
        fitnesses = np.array([genome.fitness for genome in population])
        softmax_weights = np.exp(fitnesses) / np.sum(np.exp(fitnesses))
        best_ind = np.argmax(softmax_weights)
        with open('data/best_genomes.txt', 'a') as f:
            f.write(f'{population[best_ind]!r} has average NCR {fitnesses[best_ind] / -50:.4f}\n')
        new_population = []
        for _ in range(NUM_FROM_CROSS):
            parents = np.random.choice(population, 2, replace=False, p=softmax_weights)
            new_population.append(parents[0].crossed(parents[1]))
        for _ in range(NUM_FROM_MUTATION):
            original = random.choices(population, weights=softmax_weights)[0]
            new_population.append(original.mutated())
        new_population.extend([Genome.rand_genome() for _ in range(NUM_RANDOM)])
        new_population.extend(sorted(population, key=lambda g: g.fitness, reverse=True)[:NUM_ELITE])

        population = np.array(new_population)
        assert population.shape[0] == POP_SIZE
