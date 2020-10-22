from pathlib import Path
import subprocess as subp
import sys

p = Path("main.cpp")
for p2 in Path().glob("corsika*.so"):
    if p2.stat().st_mtime > p.stat().st_mtime:
        break
else:
    subp.check_call("make")

from corsika import (
    ProcessList,
    Move,
    Decay,
    PairProduction,
    Bremsstrahlung,
    Stack,
    Particle,
    ParticleId,
    RNG,
)

if len(sys.argv) == 2:
    seed = int(sys.argv[1])
else:
    seed = 12345678

rng = RNG(seed)

stack = Stack()

p = Particle(ParticleId.Photon, 0, 10)

stack.append(p)

pl = ProcessList()
m = Move(100, 1e-3, 1e3)
pl.append(m)
pl.append(Decay(rng))
pl.append(PairProduction(rng))
pl.append(Bremsstrahlung(rng))

n = 0
while pl.run(stack):
    n += 1
    print("iteration", n, "energy deposit", m.energy_deposit)
    for p in stack:
        print("  ", p)
    if n == 15:
        break

esum = m.energy_deposit
for p in stack:
    esum += p.energy
print(f"energy {esum}")
