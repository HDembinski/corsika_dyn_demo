#include <utility>
#include <vector>
#include <cassert>
#include <random>
#include <cmath>
#include <sstream>
#include <memory>
#include <cstdint>
#include <stdexcept>
#include <iostream>
#include <algorithm>

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

constexpr auto kinf = std::numeric_limits<float>::infinity();
constexpr auto knan = std::numeric_limits<float>::quiet_NaN();
constexpr auto keps = std::numeric_limits<float>::epsilon();

struct ParticleId {
    enum Kind {
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
    float x, p; // x = position, p = momentum

    Particle() = default;
    Particle(short i, float xx, float pp) :
      ParticleId{i}, x{xx}, p{pp}
    {}
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

// Particle entry with metadata needed for propagation
struct StackParticle : Particle {
    float step = kinf; // current step size
    std::uint8_t pidx = 0; // index of process that handles this step

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
    std::uint8_t pidx_ = 0;

    using todo_t = std::vector<StackParticle*>;
    todo_t todo_;
    void handle(StackParticle& p) { todo_.push_back(&p); }

    virtual void step(StackRange) = 0;
    virtual void run(Stack&) = 0;
    virtual ~ProcessBase() {}

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
            p.pidx = 0;
            p.step = std::min(max_step_, obs_level_ - p.x) + keps;
        }
    };

    virtual void run(Stack&) override {
        // optimization opportunity not implemented here:
        // we can run SIMD code to compute next N particles
        for (auto p : ProcessBase::todo_) {
            if (p->step > 0) {
                p->x += p->step;
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

    std::shared_ptr<rng_t> rng_;
    exp_dist dist_;

    DecayOrInteraction(std::shared_ptr<rng_t> rng) : rng_{rng} {}

    float sample_length(const Particle& p) {
        const auto dl = static_cast<T*>(this)->length(p);
        if (dl == kinf)
            return kinf;
        return dist_(*rng_) * dl;
    }

    virtual void step(StackRange r) override {
        // optimization opportunity not implemented here:
        // we can run SIMD code that compute next N particles
        for (auto&& p : r) {
            const auto l = sample_length(p);
            if (l < p.step) {
                p.step = l;
                p.pidx = this->pidx_;
            }
        }
    };
};

// decay process with static decay table, alternative could load table from file
struct Decay : DecayOrInteraction<Decay> {

    using DecayOrInteraction<Decay>::DecayOrInteraction;

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

    using DecayOrInteraction<PairProduction>::DecayOrInteraction;

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
            p->p = std::sqrt(sqr(p->p) / 4 - sqr(mass(*p)));

            // add other electron to stack
            Particle p2{*p};
            p2.pid = ParticleId::Positron;
            s.push_back(p2);
        }
    }
};

struct Bremsstrahlung : DecayOrInteraction<Bremsstrahlung> {

    using DecayOrInteraction<Bremsstrahlung>::DecayOrInteraction;

    float length(const Particle& p) const {
        if (abs(p.pid) == ParticleId::Positron) {
            return  3.260E+04 * u::cm; // Nitrogen, 1 atm
        }
        return kinf;
    }

    virtual void run(Stack& s) override {
        for (auto* p : this->todo_) {
            // apply eloss to electron/positron already on stack
            p->p /= 2;
            // add photon to stack
            Particle p2{*p};
            p2.pid = ParticleId::Photon;
            s.push_back(p2);
        }
    }
};

struct ProcessList {
  std::vector<std::shared_ptr<ProcessBase>> procs_;

  bool run(Stack& s) {
    // finished particles are at front
    auto b = s.begin();
    for (auto e = s.end(); b != e && b->step < 0; ++b);
    if (b == s.end())
      return false;

    StackRange r(b, s.end());

    // compute all step sizes
    for (auto&& p : procs_)
        p->step(r);

    for (auto&& p : s) {
        // all particles need to be moved; Move is at front
        procs_.front()->handle(p);
        if (p.pidx)
          // attach particles to process with shortest step size
          procs_[p.pidx]->handle(p);
    }

    // run processes and fill stack with new particles
    for (auto&& p : procs_)
        p->run_and_reset(s);

    // move finished to front
    std::remove_if(s.begin(), s.end(), [](StackParticle& p) {
      return p.step > 0;
    });

    // delete particles below energy threshold
    auto end = std::remove_if(s.begin(), s.end(), [](StackParticle& p) {
      return p.p == 0;
    });
    s.resize(end - s.begin());

    return true;
  }

  void append(std::shared_ptr<ProcessBase> p) {
    if (procs_.empty()) {
      if (!dynamic_cast<Move*>(p.get()))
        throw std::runtime_error("Move must be first");
    }
    p->pidx_ = procs_.size();
    procs_.push_back(p);
  }
};

std::ostream& operator<<(std::ostream& os, const Particle& p) {
  os << "Particle(" << p.pid << ", " << p.x << ", " << p.p << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, const StackParticle& p) {
  os << "StackParticle(" << p.pid << ", " << p.x << ", " << p.p << ", " << p.step << ", " << static_cast<int>(p.pidx) << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, const Stack& s) {
  os << "[";
  for (auto&& p : s)
    os << "\n  " << p;
  os << (s.empty() ? "]" : "\n]");
  return os;
}

template <class T>
std::string str(const T& t) {
  std::ostringstream os;
  os << t;
  return os.str();
}

#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(Stack);

PYBIND11_MODULE(corsika, m) {

  py::class_<rng_t, std::shared_ptr<rng_t>>(m, "RNG")
    .def(py::init<unsigned long long>())
    ;

  py::enum_<ParticleId::Kind>(m, "ParticleId")
    .value("Photon", ParticleId::Photon)
    .value("Eletron", ParticleId::Electron)
    .value("Positron", ParticleId::Positron)
    .value("Muon", ParticleId::Muon)
    .value("AntiMuon", ParticleId::AntiMuon)
    .export_values()
    ;

  py::class_<Particle>(m, "Particle")
    .def(py::init<short, float, float>())
    .def_readwrite("pid", &Particle::pid)
    .def_readwrite("x", &Particle::x)
    .def_readwrite("p", &Particle::x)
    .def_property_readonly("charge", [](Particle& p) { return charge(p); })
    .def_property_readonly("mass", [](Particle& p) { return mass(p); })
    .def_property_readonly("energy", [](Particle& p) { return energy(p); })
    .def("__repr__", &str<Particle>)
    ;

  py::class_<StackParticle, Particle>(m, "StackParticle")
    .def(py::init<Particle>())
    .def_readwrite("step", &StackParticle::step)
    .def_property_readonly("pidx", [](StackParticle& p )->int { return p.pidx; })
    .def("__repr__", &str<StackParticle>)
    ;

  py::implicitly_convertible<Particle, StackParticle>();

  py::bind_vector<Stack>(m, "Stack")
    .def("__str__", &str<Stack>)
    ;

  py::class_<ProcessList>(m, "ProcessList")
    .def(py::init<>())
    .def("append", [](ProcessList& p, py::object o) {
      p.append(py::cast<std::shared_ptr<ProcessBase>>(o));
    })
    .def("run", &ProcessList::run)
    .def("__getitem__", [](ProcessList& pl, int i) { return pl.procs_[i]; })
    .def("__len__", [](const ProcessList& pl) { return pl.procs_.size(); })
    ;

  py::class_<ProcessBase, std::shared_ptr<ProcessBase>>(m, "ProcessBase");

  py::class_<Move, std::shared_ptr<Move>, ProcessBase>(m, "Move")
    .def(py::init<float, float, float>())
    .def_readwrite("energy_deposit", &Move::energy_deposit_)
    ;

  py::class_<Decay, std::shared_ptr<Decay>, ProcessBase>(m, "Decay")
    .def(py::init<std::shared_ptr<rng_t>>())
    .def("length", &Decay::length)
    .def("sample_length", &Decay::sample_length)
    ;

  py::class_<PairProduction, std::shared_ptr<PairProduction>, ProcessBase>(m, "PairProduction")
    .def(py::init<std::shared_ptr<rng_t>>())
    .def("length", &PairProduction::length)
    .def("sample_length", &PairProduction::sample_length)
    ;

  py::class_<Bremsstrahlung, std::shared_ptr<Bremsstrahlung>, ProcessBase>(m, "Bremsstrahlung")
    .def(py::init<std::shared_ptr<rng_t>>())
    .def("length", &Bremsstrahlung::length)
    .def("sample_length", &Bremsstrahlung::sample_length)
    ;
}
