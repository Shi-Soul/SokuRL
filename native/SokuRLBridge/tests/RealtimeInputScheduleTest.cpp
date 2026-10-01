#include "../RealtimeInputSchedule.hpp"
#include <iostream>
#include <stdexcept>

using namespace SokuRLBridge;
void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
RealtimeCommand command(unsigned sequence, unsigned seat, std::uint64_t observed,
    std::uint64_t target, std::uint64_t expires) {
    return {sequence, 7, 2, seat, observed, target, expires, {1, 0, 1, 0, 0, 0, 0, 0}};
}

int main() {
    try {
        for (unsigned seat = 0; seat != 2; ++seat) {
            RealtimeInputSchedule schedule(seat);
            const LogicalInput human{-9, 3, 12, 7, 0, 0, 0, 0};
            schedule.advance(7, 2, 10);
            require(schedule.submit(command(1, 1-seat, 10, 12, 15)) == RealtimeResult::WrongContext,
                "AI cannot acquire the human seat");
            require(schedule.submit(command(1, seat, 10, 12, 15)) == RealtimeResult::Accepted, "accept");
            schedule.advance(7, 2, 11);
            require(schedule.input(seat, human).a == 0, "no early injection");
            schedule.advance(7, 2, 12);
            require(schedule.input(seat, human).a == 1 && schedule.appliedAt() == 12, "exact injection");
            require(schedule.input(1-seat, human).a == 12, "physical human input preserved");
            // No producer calls: the game keeps advancing and releases AI keys.
            for (unsigned frame = 13; frame < 100000; ++frame) {
                schedule.advance(7, 2, frame);
                require(schedule.input(seat, human).a == (frame < 15 ? 1 : 0), "absolute expiration");
            }
            require(schedule.submit(command(2, seat, 11, 12, 15)) == RealtimeResult::Expired,
                "stalled producer cannot revive an expired action");
            schedule.advance(8, 0, 0);
            require(schedule.appliedSequence() == 0 && schedule.size() == 0, "match reset clears state");
            require(schedule.submit(command(3, seat, 0, 1, 10)) == RealtimeResult::WrongContext,
                "previous match rejected");
        }
        RealtimeInputSchedule queue(0);
        queue.advance(7, 2, 100);
        for (unsigned i = 0; i < queue.capacity; ++i)
            require(queue.submit(command(i+1, 0, i, 101+i, 150+i)) == RealtimeResult::Accepted, "fill");
        require(queue.submit(command(33, 0, 32, 133, 180)) == RealtimeResult::QueueFull, "bounded queue");
        queue.advance(7, 2, 105);
        require(queue.appliedSequence() == 5 && queue.appliedAt() == 105 && queue.size() == 27,
            "catch up to newest due command without replaying a burst");
        require(queue.submit(command(34, 0, 31, 132, 181)) == RealtimeResult::Stale, "old observations rejected");
        queue.advance(7, 3, 106);
        require(queue.size() == 0 && queue.input(0, {}).a == 0, "new round cancels held and future inputs");
        auto malformed = command(35, 0, 106, 106, 108);
        malformed.round = 3;
        malformed.input.a = 2;
        require(queue.submit(malformed) == RealtimeResult::Invalid, "only logical intents accepted");
        malformed.input.a = 1;
        malformed.observed = 107;
        require(queue.submit(malformed) == RealtimeResult::Invalid, "future observation rejected");
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
