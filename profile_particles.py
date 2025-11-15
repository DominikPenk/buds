import argparse
import math
import random
from dataclasses import dataclass
from typing import Literal
import numpy as np

import buds

try:
    import buds.extras

    WITH_EXTRAS = True
except ImportError:
    print("Could not import extras module")
    WITH_EXTRAS = False


@dataclass
class Arguments:
    num_particles: int
    world_type: Literal["sparse", "archetypes"]
    num_iterations: int
    size: tuple[int, int]


@buds.trait
class Particle:
    pos: buds.extras.FixedSizeArray[(2,), float]
    vel: buds.extras.FixedSizeArray[(2,), float]
    radius: float


WORLD_TYPE_MAP = {"sparse": buds.SparseWorld, "archetypes": buds.ArchetypeWorld}
if WITH_EXTRAS:
    WORLD_TYPE_MAP.update({"numpy": buds.extras.NumpyArchetypeWorld})


def random_velocity(speed: float):
    theta = 2 * math.pi * random.random()
    return speed * np.array([math.cos(theta), math.sin(theta)])


@buds.system
def move_particles(particle: Particle, domain: tuple[int, int]) -> None:
    particle.pos += particle.vel
    if particle.pos[0] < 0 or particle.pos[0] > domain[0]:
        particle.vel[0] *= -1
    if particle.pos[1] < 0 or particle.pos[1] > domain[1]:
        particle.vel[1] *= -1


def main(args: Arguments):
    print("Running a particle simulation:")
    print(f"  Number of particles    {args.num_particles}")
    print(f"  ECS world type         {args.world_type}")

    print("Setting up world ... ", end="")
    world: buds.World = WORLD_TYPE_MAP[args.world_type]()
    width, height = args.size
    for _ in range(args.num_particles):
        world.create_entity(
            Particle(
                np.array([random.uniform(0, width), random.uniform(0, height)]),
                random_velocity(5),
                7,
            )
        )
    print("Done!")

    N = args.num_particles
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)

    for iter in range(args.num_iterations):
        print(f"\rIteration: {iter + 1:4d}/{args.num_iterations}", end="")
        if args.world_type == "numpy":
            (p,) = world.get_vectorized_traits(Particle)
            dpos = p.pos[:, None, :] - p.pos[None, :, :]
            dvel = p.vel[:, None, :] - p.vel[None, :, :]
            square_dists = np.sum(dpos**2, axis=-1)
            isct_radius = p.radius[:, None] + p.radius[None, :]

            interaction_mask = (square_dists <= isct_radius**2) & (square_dists > 0)
            interaction_mask &= mask

            dv_dot_dp = np.einsum("ijk,ijk->ij", dvel, dpos)
            correction = (dv_dot_dp / np.maximum(square_dists, 1e-6))[:, :, None] * dpos
            correction[~interaction_mask] = 0.0

            p.vel += correction.sum(axis=0)
            p.vel -= correction.sum(axis=1)
            p.write_back()
        else:
            for (p0,), (p1,) in buds.itertools.trait_combinations(world, 2, Particle):
                dpos = p1.pos - p0.pos
                dvel = p1.vel - p0.vel
                square_dist = dpos[0] * dpos[0] + dpos[1] * dpos[1]
                isct_radius = p0.radius + p1.radius
                if square_dist > isct_radius**2:
                    continue

                dv_dot_dp = dpos[0] * dvel[0] + dpos[1] * dvel[1]
                correction = dv_dot_dp / square_dist * dpos

                p0.vel = p0.vel + correction
                p1.vel = p1.vel - correction
            move_particles(world, args.size)
    print()


if __name__ == "__main__":
    import cProfile
    import pstats
    import io

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-particles", type=int, default=500)
    parser.add_argument(
        "--world-type", choices=["sparse", "archetypes", "numpy"], default="archetypes"
    )
    parser.add_argument("--size", nargs=2, type=int, default=(600, 600))
    parser.add_argument("--strength", type=float, default=1000)
    parser.add_argument("--num_iterations", type=int, default=25)
    args = parser.parse_args()

    pr = cProfile.Profile()
    pr.enable()
    main(args)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
    ps.print_stats(20)
    print(s.getvalue())
