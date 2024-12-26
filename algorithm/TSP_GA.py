from __future__ import annotations
import numpy as np

class tsp_GeneticAlgorithm:
    def find_index_kernel(self, roulette, nums):
        ret = list()
        if len(nums.shape) == 1:
            for num in nums:
                left, right = 0, len(roulette) - 1
                while left < right:                
                    mid = (left + right) // 2
                    if num > roulette[mid]:
                        left = mid + 1
                    elif num < roulette[mid]:
                        right = mid
                
                ret.append(left)
        elif len(nums.shape) == 2 and nums.shape[1] == 2:
            for num in nums:
                ret.append(list())
                for n in num:
                    left, right = 0, len(roulette) - 1
                    while left < right:                
                        mid = (left + right) // 2
                        if n > roulette[mid]:
                            left = mid + 1
                        elif n < roulette[mid]:
                            right = mid
                
                    ret[-1].append(left)
        else:
            raise Exception('Wrong arrayshape')

        return np.array(ret, dtype=np.int32)

    def __init__(
        self,
        gen_size: int, ggap: float, mutation_probability: float,
        dmat: np.ndarray,
        threshold: float, max_iter: int
    ):
        self.gen_size = gen_size
        self.n_parents = int(gen_size * (1 - ggap))
        self.mut_prob = mutation_probability

        self.threshold = threshold
        self.max_iter = max_iter
        self.prev_best_fitness = np.sum(np.max(dmat, axis=-1)) # dmat의 row 당 최장거리의 합을 prev값으로 설정


        self.dmat = dmat
        self.n = self.dmat.shape[0]
        self.len_chrono = self.n

        self.generation = self.__make_initial_generation()

        self.mode = False
        self.is_shaked = False


        self.fitnesses = None
        self.roulette = None
        self.parent_indice = None
        self.next_gen_parent_indice = None

    def __make_initial_generation(self):
        ret = np.tile(np.arange(0, self.len_chrono), reps=(self.gen_size, 1))
        for chromo in ret:
            np.random.shuffle(chromo)

        return ret

    def run(self) -> tuple[list, int | float]:
        self.__update_fitness()

        iter = 0
        total_iter = 0
        while True:
            self.__sort_generation() \
                .__make_roulette() \
                .__selection() \
                .__crossover() \
                .__mutation() \
                .__update_fitness()

            increase = self.__is_increase()
            iter = (iter + 1) if increase else 0

            total_iter += 1

            end, iter = self.__is_end(iter)
            if end:
                break

        mn_ind = np.argmin(self.fitnesses)
        
        return self.generation[mn_ind].tolist(), self.fitnesses[mn_ind].item()
    
    def __update_fitness(self) -> tsp_GeneticAlgorithm:
        u = self.generation[:, :-1] # 출발지
        v = self.generation[:, 1:] # 목적지

        self.fitnesses = np.sum(self.dmat[(u, v)], axis=-1)

        return self

    def __sort_generation(self) -> tsp_GeneticAlgorithm:
        self.next_gen_parent_indice = np.argsort(self.fitnesses)[:self.n_parents]

        return self

    def __make_roulette(self) -> tsp_GeneticAlgorithm:
        mx = np.max(self.fitnesses)
        mn = np.min(self.fitnesses)
        arranged_fitnesses = mx + mn - self.fitnesses
        self.roulette = np.cumsum(arranged_fitnesses)

        return self

    def __selection(self) -> tsp_GeneticAlgorithm:
        size = self.gen_size - self.n_parents # 새로 생길 child의 개수

        nums = np.random.uniform(0, self.roulette[-1], size=size)
        self.parent_indice = self.find_index_kernel(self.roulette, nums)

        return self

    def __crossover(self) -> tsp_GeneticAlgorithm:
        self.__swap_crossver()

        return self

    def __swap_crossver(self) -> None:
        targets = self.generation[self.parent_indice]
        self.generation = self.generation[self.next_gen_parent_indice]

        rows = np.arange(targets.shape[0])
        cols01 = np.random.randint(0, targets.shape[-1], size=targets.shape[0])
        cols02 = np.random.randint(0, targets.shape[-1], size=targets.shape[0])
        
        targets[(rows, cols01)], targets[(rows, cols02)] = \
            targets[(rows, cols02)], targets[(rows, cols01)]        
        
        self.generation = np.vstack((self.generation, targets))

    def __crossover_rotation(self) -> None:
        targets = self.generation[self.parent_indice]
        self.generation = self.generation[self.next_gen_parent_indice]

        x_points = np.random.randint(0, self.len_chrono, size=(self.parent_indice.shape[0], 1))
        indice = np.mod(np.arange(self.len_chrono) - x_points, self.len_chrono)
        targets = targets[np.arange(targets.shape[0])[:, None], indice]
        
        self.generation = np.vstack((self.generation, targets))


    def __mutation(self) -> tsp_GeneticAlgorithm:
        mask = np.random.random(size=self.gen_size) < self.mut_prob
        
        rows = np.arange(self.gen_size)[mask]
    
        cols01 = np.random.randint(0, self.len_chrono, size=self.gen_size)[mask]
        cols02 = np.random.randint(0, self.len_chrono, size=self.gen_size)[mask]
        
        self.generation[(rows, cols01)], self.generation[(rows, cols02)] = \
            self.generation[(rows, cols02)], self.generation[(rows, cols01)]
        
        return self
    
    def __get_best_fitness(self) -> np.float64:
        return np.min(self.fitnesses)
    
    def __is_end(self, iter: int) -> tuple[bool, int]:
        return iter > self.max_iter, iter
    
    def __is_increase(self) -> bool:
        best_fitness = self.__get_best_fitness()
        if self.prev_best_fitness - best_fitness < self.threshold:
            return True
        else:
            self.prev_best_fitness = best_fitness
            return False