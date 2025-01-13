import os
import time
from itertools import product

import numpy as np
import pandas as pd
import torch

from aepsych import Strategy
from aepsych.acquisition import EAVC
from aepsych.generators import OptimizeAcqfGenerator, SobolGenerator
from aepsych.models import GPClassificationModel
from aepsych.utils_logging import getLogger
from threadpoolctl import threadpool_limits

logger = getLogger()
logger.setLevel("ERROR")


def time_model(n, induc, d, lb, ub, aqcf, use_gpu):
    lbArr = torch.tensor([lb] * d).double()
    ubArr = torch.tensor([ub] * d).double()

    model = GPClassificationModel(dim=d, inducing_size=induc)

    generator = OptimizeAcqfGenerator(
        lb=lbArr,
        ub=ubArr,
        acqf=aqcf,
        acqf_kwargs={"target": 0.75},
    )

    stimuliPerTrial = 1
    outcomeTypes = ["binary"]

    strat = Strategy(
        lb=lbArr,
        ub=ubArr,
        model=model,
        generator=generator,
        stimuli_per_trial=stimuliPerTrial,
        outcome_types=outcomeTypes,
        min_asks=1,
        use_gpu_modeling=use_gpu,
        use_gpu_generating=use_gpu,
    )

    sobols = SobolGenerator(lb=lbArr, ub=ubArr).gen(n)
    response = simulateBinary(sobols)

    strat.add_data(sobols, response)

    start = time.time()
    strat.fit()
    end = time.time()
    fitTime = end - start

    start = time.time()
    strat.gen(1)
    end = time.time()
    genTime = end - start

    return fitTime, genTime


def simulateBinary(x):
    rands = np.random.rand(*x.shape)
    rands = rands < x.cpu().numpy()
    rands = np.all(rands, axis=1)
    rands = rands.astype(float)
    return torch.tensor(rands)


if __name__ == "__main__":
    lb = 0
    ub = 1
    threads = 32
    inducs = [100]
    ns = [500]
    ds = [4]
    nReps = 3

    acqfs = [EAVC]

    filePath = "acqf_gpu_benchmark.csv"

    if os.path.exists(filePath):
        df = pd.read_csv(filePath)
    else:
        df = pd.DataFrame(
            columns=["n", "induc", "d", "acqf", "rep", "use_gpu", "fitTime", "genTime"]
        )
    for n, induc, d, acqf in product(ns, inducs, ds, acqfs):
        if n < induc:
            continue

        print(f"n: {n}, induc: {induc}, d: {d}, acqf: {acqf.__name__}")
        print("====")
        fitTimes = np.array([])
        genTimes = np.array([])
        rep = 0
        while len(fitTimes) < nReps:
            rep += 1
            with threadpool_limits(limits=threads):
                fitTime, genTime = time_model(n, induc, d, lb, ub, acqf, use_gpu=True)

            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {
                            "n": [n],
                            "induc": [induc],
                            "d": [d],
                            "acqf": [acqf.__name__],
                            "rep": [rep],
                            "use_gpu": [True],
                            "fitTime": [fitTime],
                            "genTime": [genTime],
                        }
                    ),
                ],
                ignore_index=True,
            )

            fitTimes = np.append(fitTimes, fitTime)
            genTimes = np.append(genTimes, genTime)

            print(
                f"GPU = n: {n}, induc: {induc}, d: {d}; Median fit time: {np.round(np.median(fitTimes), 2)}, Median gen time: {np.round(np.median(genTimes), 2)}",
                end="\r",
            )
        print()

        fitTimes = np.array([])
        genTimes = np.array([])
        rep = 0
        while len(fitTimes) < nReps:
            rep += 1
            with threadpool_limits(limits=threads):
                fitTime, genTime = time_model(n, induc, d, lb, ub, acqf, use_gpu=False)

            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {
                            "n": [n],
                            "induc": [induc],
                            "d": [d],
                            "acqf": [acqf.__name__],
                            "rep": [rep],
                            "use_gpu": [False],
                            "fitTime": [fitTime],
                            "genTime": [genTime],
                        }
                    ),
                ],
                ignore_index=True,
            )

            fitTimes = np.append(fitTimes, fitTime)
            genTimes = np.append(genTimes, genTime)

            print(
                f"CPU = n: {n}, induc: {induc}, d: {d}; Median fit time: {np.round(np.median(fitTimes), 2)}, Median gen time: {np.round(np.median(genTimes), 2)}",
                end="\r",
            )

        df.to_csv(filePath, index=False)

        print()
