# Bound PPIFlow child-call concurrency

PPIFlow candidate-wide stage coordinators submit child app calls concurrently with a conservative per-stage limit instead of processing serially or fanning out without bounds. This keeps expensive Modal stages efficient while preventing accidental over-submission; stages may override the default limit through configuration.
