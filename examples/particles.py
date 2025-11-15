import argparse
import math
import random
from dataclasses import dataclass
from typing import Literal

import buds
import numpy as np

try:
    import buds.extras

    WITH_EXTRAS = True
except ImportError:
    print("Could not import extras module")
    WITH_EXTRAS = False


import pygame


@dataclass
class Arguments:
    num_particles: int
    world_type: Literal["sparse", "archetypes"]
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
def draw_particles(screen: pygame.Surface, particle: Particle):
    pygame.draw.circle(screen, (255, 255, 255), particle.pos, particle.radius)


@buds.system
def move_particles(
    particle: Particle, domain: tuple[int, int], gamma: float, dt: float
) -> None:
    particle.vel *= 1.0 - gamma * dt
    particle.pos += dt * particle.vel
    if particle.pos[0] < 0 or particle.pos[0] > domain[0]:
        particle.vel[0] *= -1
    if particle.pos[1] < 0 or particle.pos[1] > domain[1]:
        particle.vel[1] *= -1


def main(args: Arguments):
    print("Running a particle simulation:")
    print(f"  Number of particles    {args.num_particles}")
    print(f"  ECS world type         {args.world_type}")

    pygame.init()
    screen = pygame.display.set_mode(args.size)
    clock = pygame.time.Clock()
    width, height = screen.get_size()

    print("Setting up world ... ", end="")
    world: buds.World = WORLD_TYPE_MAP[args.world_type]()
    for _ in range(args.num_particles):
        world.create_entity(
            Particle(
                np.array([random.uniform(0, width), random.uniform(0, height)]),
                random_velocity(5),
                7,
            )
        )
    print("Done!")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((0, 0, 0))

        if args.world_type == "numpy":
            (p,) = world.get_vectorized_traits(Particle)
            dpos = p.pos[:, None, :] - p.pos[None, :, :]
            dist = np.linalg.norm(dpos, axis=-1)
            sum_r = p.radius[:, None] + p.radius[None, :]
            collide = (dist > 0) & (dist < sum_r)

            n = np.zeros_like(dpos)
            n[collide] = dpos[collide] / dist[collide][:, None]

            vrel = p.vel[:, None, :] - p.vel[None, :, :]
            vrel_n = np.sum(vrel * n, axis=-1)
            approaching = collide & (vrel_n < 0)

            J = -0.5 * vrel_n[approaching][:, None] * n[approaching]  # impulse
            i, j = np.where(approaching)
            np.add.at(p.vel, i, J)
            np.add.at(p.vel, j, -J)

            overlap = sum_r - dist
            overlap[~collide] = 0
            corr = (overlap / 2)[:, :, None] * n
            np.add.at(p.pos, np.arange(len(p.pos)), np.sum(corr, axis=1))
            np.add.at(p.pos, np.arange(len(p.pos)), -np.sum(corr, axis=0))
            p.write_back()

        else:
            for (a,), (b,) in buds.itertools.trait_combinations(world, 2, Particle):
                dpos = a.pos - b.pos
                dist = math.hypot(dpos[0], dpos[1])
                min_dist = a.radius + b.radius
                if dist == 0 or dist >= min_dist:
                    continue

                n = dpos / dist
                vrel = a.vel - b.vel
                vrel_n = vrel[0] * n[0] + vrel[1] * n[1]

                if vrel_n < 0:
                    J = -vrel_n * n
                    a.vel += 0.5 * J
                    b.vel -= 0.5 * J

                overlap = min_dist - dist
                correction = 0.5 * overlap * n
                a.pos += correction
                b.pos -= correction

        move_particles(world, screen.get_size(), gamma=0.01, dt=1.0)
        draw_particles(world, screen)

        pygame.display.flip()
        clock.tick(args.fps)

    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-particles", type=int, default=500)
    parser.add_argument(
        "--world-type", choices=["sparse", "archetypes", "numpy"], default="archetypes"
    )
    parser.add_argument("--size", nargs=2, type=int, default=(600, 600))
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--strength", type=float, default=1000)
    args = parser.parse_args()

    main(args)
