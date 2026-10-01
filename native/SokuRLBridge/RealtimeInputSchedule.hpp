#pragma once
#include "ControlBlock.hpp"
#include <array>

namespace SokuRLBridge {
// A command is valid on [target, expires). Deadlines are absolute simulation
// frames: late arrival never extends a held key's lifetime.
struct RealtimeCommand {
    std::uint32_t sequence, match, round, seat;
    std::uint64_t observed, target, expires;
    LogicalInput input;
};

enum class RealtimeResult : std::uint32_t {
    Idle, Accepted, WrongContext, Invalid, Stale, Expired, QueueFull
};

class RealtimeInputSchedule {
public:
    static constexpr unsigned capacity = 32;
    explicit RealtimeInputSchedule(unsigned seat) : seat_(seat) {}

    void advance(std::uint32_t match, std::uint32_t round, std::uint64_t frame) {
        if (!active_ || match != match_ || round != round_ || frame < frame_) {
            count_ = 0;
            held_ = {};
            expires_ = 0;
            hasObservation_ = false;
            appliedSequence_ = 0;
            appliedAt_ = NO_FRAME;
        }
        active_ = true;
        match_ = match;
        round_ = round;
        frame_ = frame;
        if (frame >= expires_) held_ = {};
        unsigned consumed = 0;
        // Bounded by capacity, regardless of the producer's speed or state.
        while (consumed < count_ && pending_[consumed].target <= frame) {
            const auto &command = pending_[consumed++];
            if (command.expires <= frame) continue;
            held_ = command.input;
            expires_ = command.expires;
            appliedSequence_ = command.sequence;
            appliedAt_ = frame;
        }
        for (unsigned i = consumed; i < count_; ++i) pending_[i-consumed] = pending_[i];
        count_ -= consumed;
    }

    RealtimeResult submit(const RealtimeCommand &command) {
        if (!active_ || command.seat != seat_ || command.match != match_ || command.round != round_)
            return RealtimeResult::WrongContext;
        const auto &v = command.input;
        const auto boolean = [](std::int32_t value) { return value == 0 || value == 1; };
        if (!command.sequence || command.observed > frame_ || command.target < command.observed ||
            command.expires <= command.target || command.target - command.observed > MAX_DURATION_FRAMES ||
            command.expires - command.target > MAX_DURATION_FRAMES ||
            v.horizontalAxis < -1 || v.horizontalAxis > 1 || v.verticalAxis < -1 || v.verticalAxis > 1 ||
            !boolean(v.a) || !boolean(v.b) || !boolean(v.c) || !boolean(v.d) ||
            !boolean(v.changeCard) || !boolean(v.spellcard))
            return RealtimeResult::Invalid;
        if (hasObservation_ && (command.observed <= observed_ || command.target < target_))
            return RealtimeResult::Stale;
        if (command.expires <= frame_) return RealtimeResult::Expired;
        if (count_ == capacity) return RealtimeResult::QueueFull;
        pending_[count_++] = command;
        observed_ = command.observed;
        target_ = command.target;
        hasObservation_ = true;
        return RealtimeResult::Accepted;
    }

    LogicalInput input(unsigned seat, const LogicalInput &physical) const {
        return seat == seat_ ? held_ : physical;
    }
    unsigned size() const { return count_; }
    std::uint32_t appliedSequence() const { return appliedSequence_; }
    std::uint64_t appliedAt() const { return appliedAt_; }

private:
    unsigned seat_, count_ = 0;
    bool active_ = false, hasObservation_ = false;
    std::uint32_t match_ = 0, round_ = 0, appliedSequence_ = 0;
    std::uint64_t frame_ = 0, observed_ = 0, target_ = 0, expires_ = 0, appliedAt_ = NO_FRAME;
    LogicalInput held_{};
    std::array<RealtimeCommand, capacity> pending_{};
};
}
