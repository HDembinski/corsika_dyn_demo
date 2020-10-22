#include <utility>
#include <vector>
#include <cassert>
#include <bitset>
#include <random>
#include <unordered_map>
#include <cmath>
#include <iostream>
#include <iomanip>

namespace u {
constexpr float m = 1.f;
constexpr float s = 1.f;
constexpr float kg = 1.f;
constexpr float GeV = 1.f;
constexpr float MeV = 1e-3 * GeV;
constexpr float cm = 1e-2 * m;
constexpr float km = 1e3 * m;
constexpr float g = 1e-3 * kg;
}

using rng_t = std::default_random_engine; // should be switched to PCG
rng_t global_rng(1); // some seed

constexpr auto kinf = std::numeric_limits<float>::infinity();
constexpr auto knan = std::numeric_limits<float>::quiet_NaN();
constexpr auto keps = std::numeric_limits<float>::epsilon();

struct ParticleId {
    enum {
        Photon = 0,
        Electron = -1,
        Positron = 1,
        Muon = -2,
        AntiMuon = 2
    };
    short pid;
};

ParticleId abs(ParticleId i) {
    return ParticleId{static_cast<short>(std::abs(i.pid))};
}

struct Particle : ParticleId {
    float x, ct, p; // x = position, ct = time * speed of light, p = momentum
};

template <class T>
inline T sqr(T x) { return x * x; }

inline float mass(const ParticleId& p) {
    constexpr float kmass[3] = {
        0, // Photon
        0.511 * u::MeV, // Electron
        105.65 * u::MeV // Muon
    };
    return kmass[std::abs(p.pid)];
}

inline float energy(const Particle& p) {
    const float m = mass(p);
    return std::sqrt(sqr(p.p) + sqr(m));
}

inline float kin_energy(const Particle& p) {
    return energy(p) - mass(p);
}

// beta = tanh(y)
// gamma = cosh(y)
// betagamma = sinh(y)

inline float betagamma(const Particle& p) {
    const float m = mass(p);
    return p.p / m; // beta_gamma
}

inline float beta(const Particle& p)  {
    const float x = betagamma(p);
    return x / std::sqrt(x * x + 1);
}

inline float gamma(const Particle& p) {
    const float x = betagamma(p);
    return std::sqrt(x * x + 1);
}

inline int charge(const ParticleId& p) {
    if (p.pid == ParticleId::Photon)
        return 0;
    return p.pid > 0 ? 1 : -1;
}

struct ProcessBase;

// Particle entry with metadata needed for propagation
struct StackParticle : Particle {
    float step = kinf; // current step size
    ProcessBase* handler = nullptr; // winning process that handles this step

    StackParticle() = default;
    StackParticle(const Particle& p) : Particle{p} {}
};

using Stack = std::vector<StackParticle>;

struct StackRange {
    using pointer = StackParticle*;
    pointer begin_, end_;

    pointer begin() const { return begin_; }
    pointer end() const { return end_; }

    StackRange(pointer b, pointer e) : begin_{b}, end_{e} {}
    StackRange(Stack::iterator b, Stack::iterator e) : begin_{&*b}, end_{&*e} {}
};

struct ProcessBase {
    using todo_t = std::vector<StackParticle*>;
    todo_t todo_;
    void handle(StackParticle& p) { todo_.push_back(&p); }

    virtual void step(StackRange r) = 0;
    virtual void run(Stack&) = 0;

    void run_and_reset(Stack& s) {
        run(s);
        todo_.clear();
    }
};

// propagate particle with continuous ionisation loss
struct Move : ProcessBase {
    float max_step_;
    float min_energy_;
    float obs_level_;
    float energy_deposit_ = 0;

    Move(float mstep, float emin, float obsl) :
        max_step_{mstep},
        min_energy_{emin},
        obs_level_{obsl} {}

    virtual void step(StackRange r) override {
        // optimization opportunity not implemented here:
        // we can run SIMD code to compute next N particles
        for (auto&& p : r) {
            if (p.step > 0) {
                p.step = std::min(max_step_, obs_level_ - p.x) + keps;
                p.handler = this;
            }
        }
    };

    virtual void run(Stack&) override {
        // optimization opportunity not implemented here:
        // we can run SIMD code to compute next N particles
        for (auto p : ProcessBase::todo_) {
            if (p->step > 0) {
                p->x += p->step;
                p->ct += p->step / beta(*p);
                if (charge(*p) != 0) {
                    const float de_dx = 2.127E-03 * u::MeV / u::cm; // Nitrogen 1 atm
                    const float eloss = de_dx * p->step;
                    const float m = mass(*p);
                    const float e = energy(*p);
                    const float ek = e - m;
                    if ((ek - eloss) > min_energy_) {
                        energy_deposit_ += eloss;
                        p->p = std::sqrt(sqr(e - eloss) - sqr(m));
                    } else {
                        energy_deposit_ += ek;
                        p->p = 0;
                    }
                }
                if (p->x >= obs_level_)
                    p->step = -1;
            }
        }
    }
};

template <class T>
struct DecayOrInteraction : ProcessBase {
    using exp_dist = std::exponential_distribution<float>;
    exp_dist dist_;

    float sample_length(const Particle& p) {
        const auto dl = static_cast<T*>(this)->length(p);
        if (dl == kinf)
            return kinf;
        const auto l = dist_(global_rng) * dl;
        std::cout << "dl = " << dl <<  " l = " << l << std::endl;
        return l;
    }

    virtual void step(StackRange r) override {
        // optimization opportunity not implemented here:
        // we can run SIMD code that compute next N particles
        for (auto&& p : r) {
            const auto l = sample_length(p);
            if (l < p.step) {
                p.step = l;
                p.handler = this;
            }
        }
    };
};

// decay process with static decay table, alternative could load table from file
struct Decay : DecayOrInteraction<Decay> {

    float length(const Particle& p) const {
        if (abs(p.pid) == ParticleId::Muon) {
            const float ctau = 659.54 * u::m;
            return ctau * gamma(p);
        }
        return kinf;
    }

    virtual void run(Stack& s) override {
        // convert muon to electron, neutrinos could be append to stack, but we ignore them
        for (auto* p : this->todo_) {
            p->pid = ParticleId::Electron;
            // ignored: energy loss through decay
        }
    }
};

// demo interaction process
struct PairProduction : DecayOrInteraction<PairProduction> {

    float length(const Particle& p) const {
        if (p.pid == ParticleId::Photon) {
            return  3.260E+04 * u::cm; // Nitrogen, 1 atm
        }
        return kinf;
    }

    virtual void run(Stack& s) override {
        for (auto* p : this->todo_) {
            // convert photon already on stack to electron
            p->pid = ParticleId::Electron;
            p->p = std::sqrt(p->p * p->p / 2 - mass(*p));

            // add other electron to stack
            Particle p2{*p};
            p2.pid = ParticleId::Positron;
            s.push_back(p2);
        }
    }
};

int main() {
    // initial stack
    Stack stack = { Particle{ParticleId::Photon, 0.f, 0.f, 1 * u::GeV } };

    Move m(1 * u::m, 3 * u::MeV, 10 * u::km );
    Decay d;
    PairProduction pp;

    std::vector<ProcessBase*> processes = { &m, &d, &pp };

    for (int i = 0; i < 3; ++i) {
        // compute all step sizes
        for (auto&& p : processes)
            p->step(StackRange(stack.begin(), stack.end()));

        // attach particles to winning process with shortest step size
        for (auto&& p : stack) {
            if (p.handler) {
                // all particles need to be moved
                if (p.handler != &m)
                    m.handle(p);
                p.handler->handle(p);
            }
        }

        // run processes and fill stack with new particles
        for (auto&& p : processes)
            p->run_and_reset(stack);

        std::cout << "iteration " << i << " energy deposit " << m.energy_deposit_ << std::endl;
        for (auto&& p : stack) {
            std::cout << "  pid " << std::setw(2) << p.pid << " x/m " << p.x/u::m << " ct/m "
                      << p.ct/u::m << " p/GeV " << p.p / u::GeV << std::endl;
        }
    }
}
